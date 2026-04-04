#!/usr/bin/env python3
import argparse
import ast
import copy
import json
from pathlib import Path
from typing import List, Optional


STACK_IMPORTS = {
    "ms": "from mindspore.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler",
    "pta": "from torch_npu.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler",
}

STACK_SKIP_FIRST = {
    "ms": 0,
    "pta": 1,
}

ENTRYPOINT_NAMES = {
    "main",
    "run",
    "train",
    "infer",
    "inference",
    "evaluate",
    "eval",
    "predict",
    "serve",
}

LOOP_HINTS = {
    "train",
    "train_step",
    "optimizer",
    "backward",
    "generate",
    "infer",
    "inference",
    "forward",
    "predict",
    "step",
}


class InjectionError(RuntimeError):
    pass


class LoopCandidate:
    def __init__(
        self,
        node_id: int,
        score: int,
        reason: str,
        line: int,
        function_name: Optional[str],
        inside_main_guard: bool,
    ) -> None:
        self.node_id = node_id
        self.score = score
        self.reason = reason
        self.line = line
        self.function_name = function_name
        self.inside_main_guard = inside_main_guard


def has_existing_profiler_hooks(source: str) -> bool:
    signals = (
        "tensorboard_trace_handler(",
        "from mindspore.profiler import",
        "from torch_npu.profiler import",
        "mindspore.profiler.profile(",
        "torch_npu.profiler.profile(",
        "with profile(",
        "prof.step(",
    )
    return any(signal in source for signal in signals)


def is_main_guard(node: ast.If) -> bool:
    test = node.test
    if not isinstance(test, ast.Compare):
        return False
    if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
        return False
    if len(test.comparators) != 1:
        return False
    left = test.left
    right = test.comparators[0]
    if not isinstance(left, ast.Name) or left.id != "__name__":
        return False
    if not isinstance(right, ast.Constant) or right.value != "__main__":
        return False
    return True


def extract_call_tokens(node: ast.AST) -> List[str]:
    tokens: List[str] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            func = child.func
            if isinstance(func, ast.Name):
                tokens.append(func.id.lower())
            elif isinstance(func, ast.Attribute):
                tokens.append(func.attr.lower())
    return tokens


class LoopCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.function_stack: List[str] = []
        self.main_guard_depth = 0
        self.loop_depth = 0
        self.candidates: List[LoopCandidate] = []
        self.main_guard_ids: List[int] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.function_stack.append(node.name.lower())
        self.generic_visit(node)
        self.function_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.function_stack.append(node.name.lower())
        self.generic_visit(node)
        self.function_stack.pop()

    def visit_If(self, node: ast.If) -> None:
        is_guard = is_main_guard(node)
        if is_guard:
            self.main_guard_ids.append(id(node))
            self.main_guard_depth += 1
        self.generic_visit(node)
        if is_guard:
            self.main_guard_depth -= 1

    def visit_For(self, node: ast.For) -> None:
        self._record_candidate(node)
        self.loop_depth += 1
        self.generic_visit(node)
        self.loop_depth -= 1

    def visit_While(self, node: ast.While) -> None:
        self._record_candidate(node)
        self.loop_depth += 1
        self.generic_visit(node)
        self.loop_depth -= 1

    def _record_candidate(self, node: ast.AST) -> None:
        score = 0
        reasons: List[str] = []
        function_name = self.function_stack[-1] if self.function_stack else None
        inside_main = self.main_guard_depth > 0

        if function_name in ENTRYPOINT_NAMES:
            score += 25
            reasons.append("entrypoint function")
        if inside_main:
            score += 20
            reasons.append("inside __main__")
        if self.loop_depth == 0:
            score += 10
            reasons.append("top-level loop")

        tokens = extract_call_tokens(node)
        matched_hints = sorted({token for token in tokens if token in LOOP_HINTS})
        if matched_hints:
            score += min(30, 10 * len(matched_hints))
            reasons.append("calls=" + ",".join(matched_hints))

        target_name = None
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name):
            target_name = node.target.id.lower()
        if target_name in {"step", "batch", "data", "sample", "idx", "i"}:
            score += 5
            reasons.append("iteration variable")

        self.candidates.append(
            LoopCandidate(
                node_id=id(node),
                score=score,
                reason="; ".join(reasons) if reasons else "generic loop",
                line=getattr(node, "lineno", -1),
                function_name=function_name,
                inside_main_guard=inside_main,
            )
        )


def build_profile_expr(stack: str, trace_dir: str, use_step_schedule: bool) -> ast.expr:
    keyword_parts = [
        "activities=[ProfilerActivity.CPU, ProfilerActivity.NPU]",
        "on_trace_ready=tensorboard_trace_handler({})".format(repr(trace_dir)),
    ]
    if use_step_schedule:
        keyword_parts.append(
            "schedule=schedule(wait=0, warmup=0, active=1, repeat=1, skip_first={})".format(
                STACK_SKIP_FIRST[stack]
            )
        )
    snippet = "profile({})".format(", ".join(keyword_parts))
    return ast.parse(snippet, mode="eval").body


def build_prof_step_stmt() -> ast.stmt:
    return ast.parse("prof.step()").body[0]


def build_with_stmt(stack: str, trace_dir: str, body: List[ast.stmt], use_step_schedule: bool) -> ast.With:
    with_node = ast.With(
        items=[
            ast.withitem(
                context_expr=build_profile_expr(stack, trace_dir, use_step_schedule),
                optional_vars=ast.Name(id="prof", ctx=ast.Store()),
            )
        ],
        body=body,
        type_comment=None,
    )
    return ast.fix_missing_locations(with_node)


class LoopInjector(ast.NodeTransformer):
    def __init__(self, stack: str, trace_dir: str, target_node_id: int) -> None:
        self.stack = stack
        self.trace_dir = trace_dir
        self.target_node_id = target_node_id
        self.applied = False

    def visit_For(self, node: ast.For):  # type: ignore[override]
        node = self.generic_visit(node)
        if id(node) != self.target_node_id:
            return node
        return self._wrap_loop(node)

    def visit_While(self, node: ast.While):  # type: ignore[override]
        node = self.generic_visit(node)
        if id(node) != self.target_node_id:
            return node
        return self._wrap_loop(node)

    def _wrap_loop(self, node):
        loop_node = copy.deepcopy(node)
        loop_node.body = list(loop_node.body) + [build_prof_step_stmt()]
        with_node = build_with_stmt(self.stack, self.trace_dir, [loop_node], use_step_schedule=True)
        self.applied = True
        return ast.copy_location(with_node, node)


class MainGuardWrapper(ast.NodeTransformer):
    def __init__(self, stack: str, trace_dir: str, target_if_id: int) -> None:
        self.stack = stack
        self.trace_dir = trace_dir
        self.target_if_id = target_if_id
        self.applied = False

    def visit_If(self, node: ast.If):  # type: ignore[override]
        node = self.generic_visit(node)
        if id(node) != self.target_if_id:
            return node
        node.body = [build_with_stmt(self.stack, self.trace_dir, list(node.body), use_step_schedule=False)]
        self.applied = True
        return node


def insert_imports(module: ast.Module, import_snippet: str) -> None:
    import_nodes = ast.parse(import_snippet).body
    insert_at = 0
    if module.body and isinstance(module.body[0], ast.Expr) and isinstance(module.body[0].value, ast.Constant):
        if isinstance(module.body[0].value.value, str):
            insert_at = 1
    while insert_at < len(module.body) and isinstance(module.body[insert_at], (ast.Import, ast.ImportFrom)):
        insert_at += 1
    module.body[insert_at:insert_at] = import_nodes


def wrap_module_executable_tail(module: ast.Module, stack: str, trace_dir: str) -> bool:
    first_exec = None
    for idx, node in enumerate(module.body):
        if idx == 0 and isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, str):
                continue
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        first_exec = idx
        break
    if first_exec is None:
        return False
    executable_tail = list(module.body[first_exec:])
    module.body = list(module.body[:first_exec]) + [
        build_with_stmt(stack, trace_dir, executable_tail, use_step_schedule=False)
    ]
    return True


def instrument_source(stack: str, source: str, trace_dir: str) -> dict:
    if has_existing_profiler_hooks(source):
        raise InjectionError("Profiler hooks already exist in the script. Refusing to inject duplicates.")

    tree = ast.parse(source)
    collector = LoopCollector()
    collector.visit(tree)

    insert_imports(tree, STACK_IMPORTS[stack])

    mode = None
    target = None
    selected_candidate = None
    if collector.candidates:
        selected_candidate = max(collector.candidates, key=lambda item: item.score)
        if selected_candidate.score >= 30:
            loop_injector = LoopInjector(stack, trace_dir, selected_candidate.node_id)
            tree = loop_injector.visit(tree)
            tree = ast.fix_missing_locations(tree)
            if loop_injector.applied:
                mode = "loop_step"
                target = {
                    "line": selected_candidate.line,
                    "reason": selected_candidate.reason,
                    "function_name": selected_candidate.function_name,
                }

    if mode is None and collector.main_guard_ids:
        wrapper = MainGuardWrapper(stack, trace_dir, collector.main_guard_ids[0])
        tree = wrapper.visit(tree)
        tree = ast.fix_missing_locations(tree)
        if wrapper.applied:
            mode = "entrypoint_context"
            target = {"main_guard": True}

    if mode is None:
        if wrap_module_executable_tail(tree, stack, trace_dir):
            tree = ast.fix_missing_locations(tree)
            mode = "module_context"
            target = {"main_guard": False}

    if mode is None:
        raise InjectionError("No safe profiler injection point was found.")

    instrumented = ast.unparse(tree) + "\n"
    return {
        "instrumented_source": instrumented,
        "mode": mode,
        "trace_dir": trace_dir,
        "target": target,
        "candidate_count": len(collector.candidates),
        "selected_candidate_score": selected_candidate.score if selected_candidate else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Inject a deterministic MindSpore or PTA profiler wrapper into a copied script")
    parser.add_argument("--stack", choices=["ms", "pta"], required=True, help="runtime stack")
    parser.add_argument("--input-script", required=True, help="source script path")
    parser.add_argument("--output-script", required=True, help="output copied script path")
    parser.add_argument("--trace-dir", required=True, help="profiler trace directory")
    parser.add_argument("--metadata-json", help="optional metadata JSON path")
    args = parser.parse_args()

    input_path = Path(args.input_script).resolve()
    output_path = Path(args.output_script).resolve()
    trace_dir = str(Path(args.trace_dir).resolve())

    source = input_path.read_text(encoding="utf-8")
    result = instrument_source(args.stack, source, trace_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result["instrumented_source"], encoding="utf-8")

    payload = {
        "stack": args.stack,
        "source_script": str(input_path),
        "output_script": str(output_path),
        "trace_dir": trace_dir,
        "mode": result["mode"],
        "target": result["target"],
        "candidate_count": result["candidate_count"],
        "selected_candidate_score": result["selected_candidate_score"],
    }
    if args.metadata_json:
        metadata_path = Path(args.metadata_json).resolve()
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
