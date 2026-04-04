import subprocess
import sys
import os
import re

CPPLINT_FILTER_FILE = os.path.join(os.getcwd(), '.jenkins/check/config/filter_cpplint.txt')
PYLINT_FILTER_FILE = os.path.join(os.getcwd(), '.jenkins/check/config/filter_pylint.txt')
PYLINT_RCFILE = os.path.join(os.getcwd(), '.jenkins/rules/pylint/pylintrc')
CPP_EXTENSIONS = {'h', 'cc', 'cpp', 'cxx', 'c++', 'hh', 'hpp', 'hxx', 'h++', 'c', 'cu', 'cuh', 'cl', 'tpp', 'txx'}
PY_EXTENSIONS = {'py'}

CPPLINT_ARGS = [
    '--filter=-build/header_guard,-build/c++11,-build/include_what_you_use,'
    '-whitespace/indent_namespace,-whitespace/newline,-readability/casting',
    '--linelength=120',
    # '--recursive',
    # 'mindspore'
]


def get_changed_files(ref, extensions):
    """Get list of changed files from git commit by extension."""
    try:
        output = subprocess.check_output(
            ['git', 'diff', '--diff-filter=ACMRTUXB', '--name-only', f'{ref}~', ref],
            encoding='utf-8'
        )
    except subprocess.CalledProcessError:
        output = subprocess.check_output(
            ['git', 'diff-tree', '--no-commit-id', '--name-only', '-r', ref],
            encoding='utf-8'
        )
    files = []
    for file in output.strip().split('\n'):
        if not file:
            continue
        extension = file.split('.')[-1]
        if extension in extensions:
            files.append(file)
    return [f for f in files if os.path.isfile(f)]


def load_filters(filter_file):
    """Load filter rules from file with format: "path/pattern" "error_type"."""
    filters = []
    if not os.path.isfile(filter_file):
        return filters
    with open(filter_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('"')
            if len(parts) >= 3:
                path_pattern = parts[1]
                error_type = parts[3] if len(parts) >= 4 else ''
                filters.append((path_pattern, error_type))
    return filters


def should_filter_error(error_line, filters):
    """Check if an error line should be filtered out."""
    for path_pattern, error_type in filters:
        try:
            if re.match(f'^{path_pattern}', error_line) and re.search(error_type, error_line):
                return True
        except re.error:
            if error_line.startswith(path_pattern) and error_type in error_line:
                return True
    return False


def run_clang_format(files):
    """Run clang-format on the given files."""
    if not files:
        return
    print("Running clang-format...")
    subprocess.run(f"clang-format --style=file -i {' '.join(files)}", shell=True)
    print("clang-format completed.")


def run_cpplint(files):
    """Run cpplint on the given files and return filtered errors."""
    if not files:
        print("No files to check with cpplint.")
        return 0

    print("Running cpplint...")
    filters = load_filters(CPPLINT_FILTER_FILE)
    mindspore_dir = os.path.basename(os.getcwd())
    files = [f'{mindspore_dir}/{file}' for file in files]

    cmd = ['cpplint'] + CPPLINT_ARGS + files
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(os.getcwd()))

    # cpplint outputs to stderr
    output = result.stderr
    if not output:
        print("cpplint: No issues found.")
        return 0

    # Filter out known exceptions
    filtered_lines = []
    for line in output.strip().split('\n'):
        if not line:
            continue
        if should_filter_error(line, filters):
            continue
        filtered_lines.append(line)

    if filtered_lines:
        print("cpplint issues found:")
        for line in filtered_lines:
            print(line)
        return 1

    print("cpplint: All issues filtered (known exceptions).")
    return 0


def run_pylint(files):
    """Run pylint on the given files and return filtered errors."""
    if not files:
        print("No files to check with pylint.")
        return 0

    print("Running pylint...")
    filters = load_filters(PYLINT_FILTER_FILE)
    repo_name = os.path.basename(os.getcwd())
    parent_dir = os.path.dirname(os.getcwd())
    targets = []
    for line in files:
        target = f'{repo_name}/{line}'
        if not os.path.isfile(os.path.join(parent_dir, target)):
            continue
        targets.append(target)

    cmd_base = [
        'pylint',
        f'--rcfile={PYLINT_RCFILE}',
        '-j',
        '4',
        '--output-format=parseable',
        '--max-line-length=120',
    ]
    output_lines = []
    if targets:
        try:
            result = subprocess.run(cmd_base + targets, capture_output=True, text=True, cwd=parent_dir)
            if result.stdout:
                output_lines.extend([s for s in result.stdout.splitlines() if s])
        except OSError:
            batch_size = 100
            for i in range(0, len(targets), batch_size):
                result = subprocess.run(
                    cmd_base + targets[i:i + batch_size],
                    capture_output=True,
                    text=True,
                    cwd=parent_dir,
                )
                if result.stdout:
                    output_lines.extend([s for s in result.stdout.splitlines() if s])

    if not output_lines:
        print("pylint: No issues found.")
        return 0

    filtered_lines = []
    for line in output_lines:
        if should_filter_error(line, filters):
            continue
        if line.startswith(f'{repo_name}/'):
            filtered_lines.append(line)

    if filtered_lines:
        print("pylint issues found:")
        for line in filtered_lines:
            print(line)
        return 1

    print("pylint: All issues filtered (known exceptions).")
    return 0


def main():
    if len(sys.argv) == 1:
        ref = 'HEAD'
    else:
        ref = sys.argv[1]

    target_cpp_files = get_changed_files(ref, CPP_EXTENSIONS)
    target_py_files = get_changed_files(ref, PY_EXTENSIONS)
    target_files = target_cpp_files + target_py_files

    print(f"Files affected ({len(target_files)}):")
    for f in sorted(target_files):
        print(f"  {f}")

    if not target_files:
        print("No C/C++ or Python files to check.")
        return 0

    cpplint_ret = 0
    pylint_ret = 0
    if target_cpp_files:
        run_clang_format(target_cpp_files)
        cpplint_ret = run_cpplint(target_cpp_files)
    if target_py_files:
        pylint_ret = run_pylint(target_py_files)
    return 1 if (cpplint_ret or pylint_ret) else 0


if __name__ == '__main__':
    sys.exit(main())
