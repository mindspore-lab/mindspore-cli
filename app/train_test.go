package main

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vigo999/ms-cli/configs"
	"github.com/vigo999/ms-cli/ui/model"
)

func TestParseTrainMetricStructuredLine(t *testing.T) {
	line := "step=120 total_steps=1000 loss=1.2345 throughput=512.7 grad_norm=0.82 model=qwen2.5-7b"
	parsed := parseTrainMetric(line)

	if !parsed.HasStep || parsed.Step != 120 {
		t.Fatalf("expected step=120, got %+v", parsed)
	}
	if !parsed.HasTotalStep || parsed.TotalStep != 1000 {
		t.Fatalf("expected total_steps=1000, got %+v", parsed)
	}
	if !parsed.HasLoss || parsed.Loss != 1.2345 {
		t.Fatalf("expected loss=1.2345, got %+v", parsed)
	}
	if !parsed.HasThroughput || parsed.Throughput != 512.7 {
		t.Fatalf("expected throughput=512.7, got %+v", parsed)
	}
	if !parsed.HasGradNorm || parsed.GradNorm != 0.82 {
		t.Fatalf("expected grad_norm=0.82, got %+v", parsed)
	}
	if !parsed.HasModel || parsed.Model != "qwen2.5-7b" {
		t.Fatalf("expected model=qwen2.5-7b, got %+v", parsed)
	}
}

func TestParseTrainMetricPythonDictLine(t *testing.T) {
	line := "{'step': 32, 'loss': 3.8123, 'grad_norm': 11.2, 'throughput': 487.1}"
	parsed := parseTrainMetric(line)

	if !parsed.HasStep || parsed.Step != 32 {
		t.Fatalf("expected step=32, got %+v", parsed)
	}
	if !parsed.HasLoss || parsed.Loss != 3.8123 {
		t.Fatalf("expected loss=3.8123, got %+v", parsed)
	}
	if !parsed.HasGradNorm || parsed.GradNorm != 11.2 {
		t.Fatalf("expected grad_norm=11.2, got %+v", parsed)
	}
	if !parsed.HasThroughput || parsed.Throughput != 487.1 {
		t.Fatalf("expected throughput=487.1, got %+v", parsed)
	}
}

func TestParseTrainMetricProgressBarLine(t *testing.T) {
	line := "  5/3708 [00:13<2:41:19,  2.61s/it]"
	parsed := parseTrainMetric(line)

	if !parsed.HasStep || parsed.Step != 5 {
		t.Fatalf("expected step=5 from progress bar, got %+v", parsed)
	}
	if !parsed.HasTotalStep || parsed.TotalStep != 3708 {
		t.Fatalf("expected total_step=3708 from progress bar, got %+v", parsed)
	}
}

func TestSanitizeTrainLogLineStripsCarriageReturnAndAnsi(t *testing.T) {
	line := "\r\x1b[32m  5/3708 [00:13<2:41:19,  2.61s/it]\x1b[0m"
	got := sanitizeTrainLogLine(line)
	if strings.Contains(got, "\r") || strings.Contains(got, "\x1b") {
		t.Fatalf("expected control sequences to be removed, got %q", got)
	}
	if got != "5/3708 [00:13<2:41:19,  2.61s/it]" {
		t.Fatalf("unexpected sanitized output: %q", got)
	}
}

func TestBuildLaunchCommandReplacesRunID(t *testing.T) {
	workflow := trainWorkflow{
		RunID: "run-001",
	}
	host := trainHost{
		Name:           "gpuA",
		User:           "user",
		Address:        "gpu-a.example.com",
		Target:         "user@gpuA",
		RemoteCodePath: "/remote/code",
		RunBaseDir:     "/remote/runs",
		TrainCommand:   "python -u examples/fake_log_generator.py --run-id {{RUN_ID}} --host {{HOST_NAME}} --total-steps 5",
	}

	cmd, logPath, pidPath := buildLaunchCommand(workflow, host)

	if !strings.Contains(cmd, "bash -s -- <<'__MSCLI_TRAIN__'") {
		t.Fatalf("expected launch command to stream script via heredoc, cmd=%s", cmd)
	}
	if strings.Contains(cmd, "ssh -tt ") {
		t.Fatalf("did not expect launch command to force a tty, cmd=%s", cmd)
	}
	if !strings.Contains(cmd, "nohup bash -lc") {
		t.Fatalf("expected launch command to run inside login bash, cmd=%s", cmd)
	}
	if !strings.Contains(cmd, `cd "/remote/code" && python -u examples/fake_log_generator.py --run-id run-001 --host gpuA --total-steps 5`) {
		t.Fatalf("template placeholders were not replaced, cmd=%s", cmd)
	}
	if !strings.Contains(cmd, `kill -0 "$pid"`) {
		t.Fatalf("expected launch command to verify process liveness, cmd=%s", cmd)
	}
	if logPath != "/remote/runs/run-001/log.txt" {
		t.Fatalf("unexpected log path: %s", logPath)
	}
	if pidPath != "/remote/runs/run-001/train.pid" {
		t.Fatalf("unexpected pid path: %s", pidPath)
	}
}

func TestBuildLaunchCommandExpandsTildePathsOnRemote(t *testing.T) {
	workflow := trainWorkflow{
		RunID: "run-001",
	}
	host := trainHost{
		Name:           "gpuA",
		User:           "user",
		Address:        "gpu-a.example.com",
		Target:         "user@gpuA",
		RemoteCodePath: "~/remote code",
		RunBaseDir:     "~/train runs",
		TrainCommand:   "python -u train.py --run-id {{RUN_ID}}",
	}

	cmd, logPath, pidPath := buildLaunchCommand(workflow, host)

	if strings.Contains(cmd, "'~/") {
		t.Fatalf("expected ~ paths to expand via $HOME, cmd=%s", cmd)
	}
	if strings.Contains(cmd, `'"'"'"'"'"'"'"'"'`) {
		t.Fatalf("expected launch command to avoid nested single-quote chains, cmd=%s", cmd)
	}
	if !strings.Contains(cmd, `cd "$HOME"/`) {
		t.Fatalf("expected remote_code_path to use $HOME expansion, cmd=%s", cmd)
	}
	if !strings.Contains(cmd, `nohup bash -lc 'cd "$HOME"/"remote code" && python -u train.py --run-id run-001'`) {
		t.Fatalf("expected login shell to cd into remote_code_path before launch, cmd=%s", cmd)
	}
	if !strings.Contains(cmd, `: > "$HOME"/`) {
		t.Fatalf("expected log file to be created before launch, cmd=%s", cmd)
	}
	if !strings.Contains(cmd, `> "$HOME"/`) {
		t.Fatalf("expected log redirection to use $HOME expansion, cmd=%s", cmd)
	}
	if !strings.Contains(cmd, `pid=$!`) {
		t.Fatalf("expected launch command to capture background pid, cmd=%s", cmd)
	}
	if !strings.Contains(cmd, `echo "$pid" > "$HOME"/`) {
		t.Fatalf("expected pid file write to use $HOME expansion, cmd=%s", cmd)
	}
	if !strings.Contains(cmd, `rm -f "$HOME"/`) {
		t.Fatalf("expected failed launch to clean pid file, cmd=%s", cmd)
	}
	if logPath != "~/train runs/run-001/log.txt" {
		t.Fatalf("unexpected log path: %s", logPath)
	}
	if pidPath != "~/train runs/run-001/train.pid" {
		t.Fatalf("unexpected pid path: %s", pidPath)
	}
}

func TestBuildLaunchCommandIncludesStartupCommandBeforeTrainCommand(t *testing.T) {
	workflow := trainWorkflow{RunID: "run-001"}
	host := trainHost{
		Name:           "gpuA",
		User:           "user",
		Address:        "gpu-a.example.com",
		Target:         "user@gpuA",
		RemoteCodePath: "/remote/code",
		RunBaseDir:     "/remote/runs",
		StartupCommand: "source ~/.bashrc && conda activate trainer",
		TrainCommand:   "python -u train.py --run-id {{RUN_ID}}",
	}

	cmd, _, _ := buildLaunchCommand(workflow, host)

	startupIdx := strings.Index(cmd, "source ~/.bashrc && conda activate trainer")
	trainIdx := strings.Index(cmd, "python -u train.py --run-id run-001")
	if startupIdx < 0 {
		t.Fatalf("expected startup command in launch script, got %s", cmd)
	}
	if trainIdx < 0 {
		t.Fatalf("expected rendered train command in launch script, got %s", cmd)
	}
	if startupIdx >= trainIdx {
		t.Fatalf("expected startup command before train command, got %s", cmd)
	}
}

func TestBuildLaunchDisplayCommandShowsRemoteLaunchInsteadOfSSHWrapper(t *testing.T) {
	workflow := trainWorkflow{RunID: "run-001"}
	host := trainHost{
		Name:           "gpuA",
		Target:         "user@gpuA",
		RemoteCodePath: "~/remote code",
		RunBaseDir:     "~/train runs",
		TrainCommand:   "python -u train.py --run-id {{RUN_ID}}",
	}

	display := buildLaunchDisplayCommand(workflow, host, "~/train runs/run-001/log.txt", "~/train runs/run-001/train.pid")

	if strings.Contains(display, "ssh ") {
		t.Fatalf("expected display command to omit ssh wrapper, got %s", display)
	}
	if !strings.Contains(display, `nohup bash -lc 'cd "$HOME"/"remote code" && python -u train.py --run-id run-001'`) {
		t.Fatalf("expected display command to show actual launch command, got %s", display)
	}
}

func TestBuildLaunchDisplayCommandShowsStartupCommandWhenConfigured(t *testing.T) {
	workflow := trainWorkflow{RunID: "run-001"}
	host := trainHost{
		Name:           "gpuA",
		Target:         "user@gpuA",
		RemoteCodePath: "/remote/code",
		RunBaseDir:     "/remote/runs",
		StartupCommand: "source ~/.bashrc && conda activate trainer",
		TrainCommand:   "python -u train.py --run-id {{RUN_ID}}",
	}

	display := buildLaunchDisplayCommand(workflow, host, "/remote/runs/run-001/log.txt", "/remote/runs/run-001/train.pid")
	if !strings.Contains(display, "source ~/.bashrc && conda activate trainer") {
		t.Fatalf("expected display command to include startup command, got %s", display)
	}
	if !strings.Contains(display, "python -u train.py --run-id run-001") {
		t.Fatalf("expected display command to include train command, got %s", display)
	}
}

func TestBuildTailCommandUsesPidAwareTail(t *testing.T) {
	host := trainHost{
		Target:      "user@gpuA",
		ControlPath: "~/.ssh/cm-gpua",
		LogPath:     "/remote/runs/run-001/log.txt",
		PIDPath:     "/remote/runs/run-001/train.pid",
	}

	cmd := buildTailCommand(host)

	if !strings.Contains(cmd, "bash -s -- <<'__MSCLI_TAIL__'") {
		t.Fatalf("expected tail command to stream script via heredoc, got %s", cmd)
	}
	if !strings.Contains(cmd, "tail --pid=") {
		t.Fatalf("expected pid-aware tail command, got %s", cmd)
	}
	if !strings.Contains(cmd, host.PIDPath) {
		t.Fatalf("expected pid path in tail command, got %s", cmd)
	}
}

func TestBuildTailCommandExpandsTildePathsOnRemote(t *testing.T) {
	host := trainHost{
		Target:      "user@gpuA",
		ControlPath: "~/.ssh/cm-gpua",
		LogPath:     "~/train runs/run-001/log.txt",
		PIDPath:     "~/train runs/run-001/train.pid",
	}

	cmd := buildTailCommand(host)

	if strings.Contains(cmd, "'~/train runs") {
		t.Fatalf("expected ~ paths to expand via $HOME, cmd=%s", cmd)
	}
	if strings.Contains(cmd, `'"'"'"'"'"'"'"'"'`) {
		t.Fatalf("expected tail command to avoid nested single-quote chains, cmd=%s", cmd)
	}
	if !strings.Contains(cmd, `cat "$HOME"/`) {
		t.Fatalf("expected pid path to use $HOME expansion, cmd=%s", cmd)
	}
	if !strings.Contains(cmd, `tail --pid="$pid" -n 0 -F "$HOME"/`) {
		t.Fatalf("expected log path to use $HOME expansion, cmd=%s", cmd)
	}
}

func TestBuildRsyncCommandUsesMuxGitIgnoreAndNoCompressionByDefault(t *testing.T) {
	workflow := trainWorkflow{
		LocalPath:             "/local/code",
		Exclude:               []string{".git", ".cache", "custom"},
		RsyncRespectGitIgnore: true,
	}
	host := trainHost{
		Target:         "user@gpuA",
		RemoteCodePath: "/remote/code",
		ControlPath:    "~/.ssh/cm-gpua",
		ControlPersist: "30m",
	}

	cmd := buildRsyncCommand(workflow, host)

	if !strings.Contains(cmd, "rsync -a --delete --omit-dir-times") {
		t.Fatalf("expected rsync archive command without implicit compression, got %s", cmd)
	}
	if strings.Contains(cmd, " -z ") {
		t.Fatalf("did not expect compression by default, got %s", cmd)
	}
	if !strings.Contains(cmd, "--filter ':- .gitignore'") {
		t.Fatalf("expected .gitignore filter, got %s", cmd)
	}
	if !strings.Contains(cmd, "ControlMaster=auto") {
		t.Fatalf("expected SSH multiplexing for rsync, got %s", cmd)
	}
	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatalf("resolve home dir: %v", err)
	}
	if !strings.Contains(cmd, "ControlPath="+filepath.Join(home, ".ssh", "cm-gpua")) {
		t.Fatalf("expected expanded control path, got %s", cmd)
	}
}

func TestBuildRsyncCommandCanEnableCompression(t *testing.T) {
	workflow := trainWorkflow{
		LocalPath:     "/local/code",
		RsyncCompress: true,
		Exclude:       []string{".git"},
	}
	host := trainHost{
		Target:         "user@gpuA",
		RemoteCodePath: "/remote/code",
	}

	cmd := buildRsyncCommand(workflow, host)

	if !strings.Contains(cmd, "rsync -a -z --delete") {
		t.Fatalf("expected compression flag when enabled, got %s", cmd)
	}
}

func TestBuildRsyncCommandUsesHostSpecificLocalDirectory(t *testing.T) {
	workflow := trainWorkflow{
		LocalPath: "/local/default",
	}
	host := trainHost{
		LocalPath:      "/local/host-a",
		LocalIsDir:     true,
		Target:         "user@gpuA",
		RemoteCodePath: "/remote/code",
	}

	cmd := buildRsyncCommand(workflow, host)

	if strings.Contains(cmd, "/local/default/") {
		t.Fatalf("did not expect global local_path when host override is set, got %s", cmd)
	}
	if !strings.Contains(cmd, "'/local/host-a/'") {
		t.Fatalf("expected host-specific directory source, got %s", cmd)
	}
}

func TestBuildRsyncCommandSupportsHostSpecificLocalFile(t *testing.T) {
	workflow := trainWorkflow{
		LocalPath: "/local/default",
	}
	host := trainHost{
		LocalPath:      "/local/train_host_b.py",
		LocalIsDir:     false,
		Target:         "user@gpuB",
		RemoteCodePath: "/remote/code",
	}

	cmd := buildRsyncCommand(workflow, host)

	if !strings.Contains(cmd, "'/local/train_host_b.py'") {
		t.Fatalf("expected file source path, got %s", cmd)
	}
	if strings.Contains(cmd, "/local/train_host_b.py/'") {
		t.Fatalf("did not expect trailing slash for file sync, got %s", cmd)
	}
}

func TestBuildMasterCommandChecksExistingSocketFirst(t *testing.T) {
	host := trainHost{
		Target:         "user@gpuA",
		ControlPath:    "~/.ssh/cm-gpua",
		ControlPersist: "30m",
	}

	cmd := buildMasterCommand(host)

	if !strings.Contains(cmd, "-O check") {
		t.Fatalf("expected socket check before opening master, got %s", cmd)
	}
	if !strings.Contains(cmd, "|| ssh -MNf") {
		t.Fatalf("expected idempotent master creation, got %s", cmd)
	}
}

func TestNormalizeTrainSyncParallelism(t *testing.T) {
	tests := []struct {
		name       string
		configured int
		hostCount  int
		want       int
	}{
		{name: "single host", configured: 0, hostCount: 1, want: 1},
		{name: "auto bounded", configured: 0, hostCount: 8, want: 4},
		{name: "auto small cluster", configured: 0, hostCount: 3, want: 3},
		{name: "explicit value", configured: 2, hostCount: 8, want: 2},
		{name: "explicit capped by hosts", configured: 8, hostCount: 3, want: 3},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := normalizeTrainSyncParallelism(tt.configured, tt.hostCount); got != tt.want {
				t.Fatalf("expected %d, got %d", tt.want, got)
			}
		})
	}
}

func TestBuildTrainErrorUpdateIncludesFailedCommand(t *testing.T) {
	err := &trainWorkflowError{
		Stage:   "launch",
		Host:    "gpuA",
		Command: "ssh user@gpuA 'cd /remote/code && nohup python -u train.py > log.txt 2>&1 &'",
		Message: "launch host gpuA: exit status 1",
		Output:  "log.txt: No such file or directory",
	}

	update := buildTrainErrorUpdate("run-001", err)

	if update.Kind != model.TrainUpdateError {
		t.Fatalf("expected train error update kind, got %+v", update)
	}
	if update.Stage != "launch" {
		t.Fatalf("expected stage=launch, got %q", update.Stage)
	}
	if update.Host != "gpuA" {
		t.Fatalf("expected host=gpuA, got %q", update.Host)
	}
	if update.Command != err.Command {
		t.Fatalf("expected failed command to be preserved, got %q", update.Command)
	}
	if !strings.Contains(update.Message, "exit status 1") {
		t.Fatalf("expected error summary in message, got %q", update.Message)
	}
	if !strings.Contains(update.Message, "log.txt: No such file or directory") {
		t.Fatalf("expected command output in message, got %q", update.Message)
	}
}

func TestStartTrainSessionStoresLastRunID(t *testing.T) {
	app := &Application{}

	if ok := app.startTrainSession("run-001", []string{"run-001"}, context.CancelFunc(func() {})); !ok {
		t.Fatalf("expected train session to start")
	}
	if app.trainRunID != "run-001" {
		t.Fatalf("expected current run_id to be stored, got %q", app.trainRunID)
	}
	if app.trainLastID != "run-001" {
		t.Fatalf("expected last run_id to be stored, got %q", app.trainLastID)
	}
	if len(app.trainLastArgs) != 1 || app.trainLastArgs[0] != "run-001" {
		t.Fatalf("expected last args to be stored, got %#v", app.trainLastArgs)
	}
}

func TestRetryTrainWorkflowWithoutPreviousRun(t *testing.T) {
	app := &Application{
		EventCh: make(chan model.Event, 1),
	}

	app.retryTrainWorkflow()

	select {
	case ev := <-app.EventCh:
		if ev.Type != model.AgentReply {
			t.Fatalf("expected agent reply event, got %+v", ev)
		}
		if !strings.Contains(ev.Message, "No previous /train workflow to retry.") {
			t.Fatalf("unexpected retry message: %q", ev.Message)
		}
	default:
		t.Fatalf("expected retry failure message")
	}
}

func TestBuildTrainWorkflowFromPromptResolvesScriptAndTarget(t *testing.T) {
	workDir := t.TempDir()
	scriptPath := filepath.Join(workDir, "scripts", "train_qwen.py")
	if err := os.MkdirAll(filepath.Dir(scriptPath), 0o755); err != nil {
		t.Fatalf("mkdir script dir: %v", err)
	}
	script := `import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--run-id")
parser.add_argument("--host")
parser.add_argument("--model")
if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
`
	if err := os.WriteFile(scriptPath, []byte(script), 0o644); err != nil {
		t.Fatalf("write script: %v", err)
	}

	cfg := configs.DefaultConfig()
	cfg.Training.Enabled = true
	cfg.Training.LocalPath = "."
	cfg.Training.RemoteCodePath = "/remote/code"
	cfg.Training.RunBaseDir = "/remote/runs"
	cfg.Training.HostsFile = ""
	cfg.Training.Hosts = []configs.TrainingHostConfig{
		{Name: "gpuA", User: "user", Address: "gpu-a.example.com"},
	}

	app := &Application{
		Config:  cfg,
		WorkDir: workDir,
	}

	workflow, err := app.buildTrainWorkflow([]string{"tuning", "qwen3", "with", "the", "current", "code"})
	if err != nil {
		t.Fatalf("buildTrainWorkflow: %v", err)
	}

	if workflow.Request != "tuning qwen3 with the current code" {
		t.Fatalf("unexpected request: %q", workflow.Request)
	}
	if workflow.Target != "qwen3" {
		t.Fatalf("unexpected target: %q", workflow.Target)
	}
	if workflow.ScriptPath != "scripts/train_qwen.py" {
		t.Fatalf("unexpected script path: %q", workflow.ScriptPath)
	}
	if len(workflow.Hosts) != 1 {
		t.Fatalf("expected one host, got %d", len(workflow.Hosts))
	}
	cmd := workflow.Hosts[0].TrainCommand
	if !strings.Contains(cmd, "python -u 'scripts/train_qwen.py'") {
		t.Fatalf("expected resolved script command, got %s", cmd)
	}
	if !strings.Contains(cmd, "--model 'qwen3'") {
		t.Fatalf("expected model flag from prompt target, got %s", cmd)
	}
	if !strings.Contains(cmd, "MSCLI_TRAIN_REQUEST='tuning qwen3 with the current code'") {
		t.Fatalf("expected train request env to be injected, got %s", cmd)
	}
}

func TestBuildTrainWorkflowSupportsPerHostLocalPathAndTrainScript(t *testing.T) {
	workDir := t.TempDir()
	hostADir := filepath.Join(workDir, "codeA")
	hostAScript := filepath.Join(hostADir, "scripts", "train_a.py")
	if err := os.MkdirAll(filepath.Dir(hostAScript), 0o755); err != nil {
		t.Fatalf("mkdir hostA script dir: %v", err)
	}
	if err := os.WriteFile(hostAScript, []byte("print('a')\n"), 0o644); err != nil {
		t.Fatalf("write hostA script: %v", err)
	}

	hostBDir := filepath.Join(workDir, "codeB")
	hostBScript := filepath.Join(hostBDir, "train_b.py")
	if err := os.MkdirAll(hostBDir, 0o755); err != nil {
		t.Fatalf("mkdir hostB dir: %v", err)
	}
	if err := os.WriteFile(hostBScript, []byte("print('b')\n"), 0o644); err != nil {
		t.Fatalf("write hostB script: %v", err)
	}

	cfg := configs.DefaultConfig()
	cfg.Training.Enabled = true
	cfg.Training.RemoteCodePath = "/remote/code"
	cfg.Training.RunBaseDir = "/remote/runs"
	cfg.Training.HostsFile = ""
	cfg.Training.Hosts = []configs.TrainingHostConfig{
		{
			Name:        "gpuA",
			User:        "user",
			Address:     "gpu-a.example.com",
			LocalPath:   hostADir,
			TrainScript: "scripts/train_a.py",
		},
		{
			Name:      "gpuB",
			User:      "user",
			Address:   "gpu-b.example.com",
			LocalPath: hostBScript,
		},
	}

	app := &Application{
		Config:  cfg,
		WorkDir: workDir,
	}

	workflow, err := app.buildTrainWorkflow([]string{"run-001"})
	if err != nil {
		t.Fatalf("buildTrainWorkflow: %v", err)
	}
	if len(workflow.Hosts) != 2 {
		t.Fatalf("expected two hosts, got %d", len(workflow.Hosts))
	}

	hostA := workflow.Hosts[0]
	hostB := workflow.Hosts[1]
	if hostA.LocalPath != hostADir || !hostA.LocalIsDir {
		t.Fatalf("expected hostA to use directory local_path, got %+v", hostA)
	}
	if hostA.TrainScript != "scripts/train_a.py" {
		t.Fatalf("expected hostA train script, got %q", hostA.TrainScript)
	}
	if hostB.LocalPath != hostBScript || hostB.LocalIsDir {
		t.Fatalf("expected hostB to use file local_path, got %+v", hostB)
	}
	if hostB.TrainScript != "train_b.py" {
		t.Fatalf("expected hostB file local_path to infer train script basename, got %q", hostB.TrainScript)
	}

	renderedA := renderTrainCommand(hostA.TrainCommand, workflow, hostA, "/remote/runs/run-001/log.txt", "/remote/runs/run-001/train.pid")
	renderedB := renderTrainCommand(hostB.TrainCommand, workflow, hostB, "/remote/runs/run-001/log.txt", "/remote/runs/run-001/train.pid")
	if !strings.Contains(renderedA, "python -u scripts/train_a.py") {
		t.Fatalf("expected hostA command to use host-specific train_script, got %s", renderedA)
	}
	if !strings.Contains(renderedB, "python -u train_b.py") {
		t.Fatalf("expected hostB command to use file-derived train_script, got %s", renderedB)
	}
}

func TestBuildTrainWorkflowAllowsFileLocalPathWithMatchingTrainScriptPath(t *testing.T) {
	workDir := t.TempDir()
	scriptPath := filepath.Join(workDir, "examples", "fake_log_generator.py")
	if err := os.MkdirAll(filepath.Dir(scriptPath), 0o755); err != nil {
		t.Fatalf("mkdir script dir: %v", err)
	}
	script := `
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--run-id")
parser.add_argument("--host")

if __name__ == "__main__":
    print("ok")
`
	if err := os.WriteFile(scriptPath, []byte(script), 0o644); err != nil {
		t.Fatalf("write script: %v", err)
	}

	cfg := configs.DefaultConfig()
	cfg.Training.Enabled = true
	cfg.Training.LocalPath = "."
	cfg.Training.RemoteCodePath = "/remote/code"
	cfg.Training.RunBaseDir = "/remote/runs"
	cfg.Training.HostsFile = ""
	cfg.Training.Hosts = []configs.TrainingHostConfig{
		{
			Name:        "gpuA",
			User:        "user",
			Address:     "gpu-a.example.com",
			LocalPath:   "./examples/fake_log_generator.py",
			TrainScript: "./examples/fake_log_generator.py",
		},
	}

	app := &Application{
		Config:  cfg,
		WorkDir: workDir,
	}

	workflow, err := app.buildTrainWorkflow([]string{"test", "with", "examples/fake_log_generator.py"})
	if err != nil {
		t.Fatalf("buildTrainWorkflow: %v", err)
	}
	if len(workflow.Hosts) != 1 {
		t.Fatalf("expected one host, got %d", len(workflow.Hosts))
	}
	host := workflow.Hosts[0]
	if host.LocalPath != scriptPath || host.LocalIsDir {
		t.Fatalf("expected file local_path to resolve to script file, got %+v", host)
	}
	if host.TrainScript != "fake_log_generator.py" {
		t.Fatalf("expected matching file train_script to collapse to basename, got %q", host.TrainScript)
	}
	if workflow.ScriptPath != "fake_log_generator.py" {
		t.Fatalf("expected workflow script path to use basename, got %q", workflow.ScriptPath)
	}
	rendered := renderTrainCommand(host.TrainCommand, workflow, host, "/remote/runs/run-001/log.txt", "/remote/runs/run-001/train.pid")
	if !strings.Contains(rendered, "python -u fake_log_generator.py") {
		t.Fatalf("expected rendered command to use synced basename, got %s", rendered)
	}
}

func TestBuildTrainWorkflowUsesHostSpecificStartupCommandOverride(t *testing.T) {
	workDir := t.TempDir()
	scriptPath := filepath.Join(workDir, "train.py")
	if err := os.WriteFile(scriptPath, []byte("print('ok')\n"), 0o644); err != nil {
		t.Fatalf("write script: %v", err)
	}

	cfg := configs.DefaultConfig()
	cfg.Training.Enabled = true
	cfg.Training.LocalPath = "."
	cfg.Training.StartupCommand = "source ~/.bashrc && conda activate global-trainer"
	cfg.Training.RemoteCodePath = "/remote/code"
	cfg.Training.RunBaseDir = "/remote/runs"
	cfg.Training.HostsFile = ""
	cfg.Training.Hosts = []configs.TrainingHostConfig{
		{
			Name:        "gpuA",
			User:        "user",
			Address:     "gpu-a.example.com",
			LocalPath:   "./train.py",
			TrainScript: "./train.py",
		},
		{
			Name:           "gpuB",
			User:           "user",
			Address:        "gpu-b.example.com",
			LocalPath:      "./train.py",
			TrainScript:    "./train.py",
			StartupCommand: "source /usr/local/Ascend/ascend-toolkit/set_env.sh",
		},
	}

	app := &Application{
		Config:  cfg,
		WorkDir: workDir,
	}

	workflow, err := app.buildTrainWorkflow([]string{"run-001"})
	if err != nil {
		t.Fatalf("buildTrainWorkflow: %v", err)
	}
	if len(workflow.Hosts) != 2 {
		t.Fatalf("expected two hosts, got %d", len(workflow.Hosts))
	}
	if workflow.Hosts[0].StartupCommand != "source ~/.bashrc && conda activate global-trainer" {
		t.Fatalf("expected hostA to inherit global startup command, got %q", workflow.Hosts[0].StartupCommand)
	}
	if workflow.Hosts[1].StartupCommand != "source /usr/local/Ascend/ascend-toolkit/set_env.sh" {
		t.Fatalf("expected hostB to override startup command, got %q", workflow.Hosts[1].StartupCommand)
	}
}
