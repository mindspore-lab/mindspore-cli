package main

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"
	"unicode"

	"github.com/vigo999/ms-cli/configs"
	"github.com/vigo999/ms-cli/ui/model"
)

const (
	defaultTrainCommandTemplate       = "python -u examples/fake_log_generator.py --run-id {{RUN_ID}} --host {{HOST_NAME}} --total-steps 120"
	defaultTrainScriptCommandTemplate = "python -u {{TRAIN_SCRIPT}} --run-id {{RUN_ID}} --host {{HOST_NAME}} --total-steps 120"
)

type trainWorkflow struct {
	RunID                 string
	Request               string
	Target                string
	ScriptPath            string
	SourceArgs            []string
	LocalPath             string
	Exclude               []string
	ControlPersist        string
	RsyncCompress         bool
	RsyncRespectGitIgnore bool
	SyncParallelism       int
	Hosts                 []trainHost
}

type trainHost struct {
	Name           string
	User           string
	Address        string
	Target         string
	LocalPath      string
	LocalIsDir     bool
	TrainScript    string
	StartupCommand string
	RemoteCodePath string
	RunBaseDir     string
	TrainCommand   string
	ControlPath    string
	ControlPersist string
	LogPath        string
	PIDPath        string
}

type trainMetricParse struct {
	Step       int
	TotalStep  int
	Loss       float64
	Throughput float64
	GradNorm   float64
	Model      string

	HasStep       bool
	HasTotalStep  bool
	HasLoss       bool
	HasThroughput bool
	HasGradNorm   bool
	HasModel      bool
}

type trainStreamResult struct {
	Host    string
	Command string
	Err     error
}

type trainCommandResult struct {
	Host    trainHost
	Command string
	Output  string
	Err     error
}

type trainRequestSpec struct {
	RunID         string
	Prompt        string
	Target        string
	ScriptHint    string
	ExplicitRunID bool
}

type trainScriptSpec struct {
	RelPath     string
	RunIDFlag   string
	HostFlag    string
	ModelFlag   string
	RequestFlag string
}

type trainWorkflowError struct {
	Stage   string
	Host    string
	Command string
	Message string
	Output  string
}

func (e *trainWorkflowError) Error() string {
	if e == nil {
		return ""
	}
	message := strings.TrimSpace(e.Message)
	output := strings.TrimSpace(e.Output)
	switch {
	case message == "":
		return output
	case output == "":
		return message
	default:
		return message + "\n" + output
	}
}

var (
	trainStepRE = regexp.MustCompile(`(?i)['"]?(?:step|iter(?:ation)?)['"]?\s*[:=]\s*(\d+)`)

	trainTotalStepRE = regexp.MustCompile(`(?i)['"]?total[_\s-]*steps?['"]?\s*[:=]\s*(\d+)`)

	trainProgressRE = regexp.MustCompile(`\b(\d+)\s*/\s*(\d+)\s*\[`)

	trainLossRE = regexp.MustCompile(`(?i)['"]?loss['"]?\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)`)

	trainThroughputRE = regexp.MustCompile(`(?i)(?:['"]?throughput['"]?|samples/s|sample/s|tok/s|tokens/s|it/s)\s*[:=]?\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)`)

	trainGradNormRE = regexp.MustCompile(`(?i)['"]?(?:grad(?:ient)?[_\s-]*norm)['"]?\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)`)

	trainModelRE = regexp.MustCompile(`(?i)['"]?model['"]?\s*[:=]\s*['"]?([A-Za-z0-9._/\-]+)['"]?`)

	trainANSIEscapeRE = regexp.MustCompile(`\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))`)
)

// cmdTrain handles "/train [run_id|retry|stop|task...]".
func (a *Application) cmdTrain(args []string) {
	if len(args) > 0 {
		switch strings.ToLower(strings.TrimSpace(args[0])) {
		case "stop":
			a.stopTrainWorkflow()
			return
		case "retry":
			a.retryTrainWorkflow()
			return
		case "help":
			a.EventCh <- model.Event{
				Type: model.AgentReply,
				Message: `Usage:
  /train                  Start workflow with timestamp run_id
  /train <run_id>         Start workflow with explicit run_id
  /train tuning qwen3 with the current code
  /train retry            Retry last run_id after failure
  /train stop             Stop current workflow and log streams`,
			}
			return
		}
	}

	workflow, err := a.buildTrainWorkflow(args)
	if err != nil {
		a.EventCh <- model.Event{
			Type:    model.AgentReply,
			Message: fmt.Sprintf("Cannot start /train: %v", err),
		}
		return
	}

	ctx, cancel := context.WithCancel(context.Background())
	if !a.startTrainSession(workflow.RunID, workflow.SourceArgs, cancel) {
		a.EventCh <- model.Event{
			Type:    model.AgentReply,
			Message: "A /train workflow is already running. Use `/train stop` first.",
		}
		cancel()
		return
	}

	hostNames := make([]string, 0, len(workflow.Hosts))
	for _, host := range workflow.Hosts {
		hostNames = append(hostNames, host.Name)
	}
	a.emitTrain(model.TrainUpdate{
		Kind:  model.TrainUpdateOpen,
		RunID: workflow.RunID,
		Hosts: hostNames,
	})
	a.emitTrain(model.TrainUpdate{
		Kind:    model.TrainUpdateNote,
		Message: fmt.Sprintf("run_id=%s", workflow.RunID),
	})
	if workflow.Request != "" {
		a.emitTrain(model.TrainUpdate{
			Kind:    model.TrainUpdateNote,
			Message: fmt.Sprintf("request=%s", workflow.Request),
		})
	}
	if workflow.Target != "" {
		a.emitTrain(model.TrainUpdate{
			Kind:    model.TrainUpdateNote,
			Message: fmt.Sprintf("target=%s", workflow.Target),
		})
	}
	if workflow.ScriptPath != "" {
		a.emitTrain(model.TrainUpdate{
			Kind:    model.TrainUpdateNote,
			Message: fmt.Sprintf("script=%s", workflow.ScriptPath),
		})
	}

	go a.runTrainWorkflow(ctx, workflow)
}

func (a *Application) retryTrainWorkflow() {
	a.trainMu.Lock()
	cancel := a.trainCancel
	lastRunID := a.trainLastID
	lastArgs := append([]string(nil), a.trainLastArgs...)
	a.trainMu.Unlock()

	if cancel != nil {
		a.EventCh <- model.Event{
			Type:    model.AgentReply,
			Message: "A /train workflow is already running. Use `/train stop` first.",
		}
		return
	}
	if strings.TrimSpace(lastRunID) == "" {
		a.EventCh <- model.Event{
			Type:    model.AgentReply,
			Message: "No previous /train workflow to retry.",
		}
		return
	}

	if len(lastArgs) == 0 {
		lastArgs = []string{lastRunID}
	}
	a.cmdTrain(lastArgs)
}

func (a *Application) stopTrainWorkflow() {
	a.trainMu.Lock()
	cancel := a.trainCancel
	runID := a.trainRunID
	a.trainMu.Unlock()

	if cancel == nil {
		a.EventCh <- model.Event{
			Type:    model.AgentReply,
			Message: "No active /train workflow.",
		}
		return
	}

	cancel()
	a.emitTrain(model.TrainUpdate{
		Kind:    model.TrainUpdateStopped,
		RunID:   runID,
		Message: fmt.Sprintf("stop requested for run_id=%s", runID),
	})
}

func (a *Application) runTrainWorkflow(ctx context.Context, workflow trainWorkflow) {
	defer a.finishTrainSession(workflow.RunID)

	if err := a.executeTrainWorkflow(ctx, workflow); err != nil {
		if errors.Is(err, context.Canceled) {
			a.emitTrain(model.TrainUpdate{
				Kind:    model.TrainUpdateStopped,
				RunID:   workflow.RunID,
				Message: "workflow stopped",
			})
			return
		}
		a.emitTrain(buildTrainErrorUpdate(workflow.RunID, err))
		return
	}
}

func (a *Application) executeTrainWorkflow(ctx context.Context, workflow trainWorkflow) error {
	if err := ctx.Err(); err != nil {
		return err
	}

	// 1) rsync sync
	a.emitTrain(model.TrainUpdate{
		Kind:    model.TrainUpdateStage,
		Stage:   "sync",
		Status:  string(model.TrainStageRunning),
		Message: "1/5 sync local code to remote hosts",
	})
	if err := a.syncTrainHosts(ctx, workflow); err != nil {
		return err
	}
	a.emitTrain(model.TrainUpdate{
		Kind:   model.TrainUpdateStage,
		Stage:  "sync",
		Status: string(model.TrainStageSuccess),
	})

	// 2) launch training
	a.emitTrain(model.TrainUpdate{
		Kind:    model.TrainUpdateStage,
		Stage:   "launch",
		Status:  string(model.TrainStageRunning),
		Message: "2/5 launch remote training via nohup",
	})
	for idx, host := range workflow.Hosts {
		cmd, logPath, pidPath := buildLaunchCommand(workflow, host)
		displayCmd := buildLaunchDisplayCommand(workflow, host, logPath, pidPath)
		workflow.Hosts[idx].LogPath = logPath
		workflow.Hosts[idx].PIDPath = pidPath

		a.emitTrain(model.TrainUpdate{
			Kind:    model.TrainUpdateHost,
			Host:    host.Name,
			Stage:   "launch",
			Status:  string(model.TrainHostRunning),
			Command: displayCmd,
			LogPath: logPath,
			Message: fmt.Sprintf("launching on %s", host.Name),
		})
		out, err := runShellCommand(ctx, cmd)
		if err != nil {
			a.emitTrain(model.TrainUpdate{
				Kind:    model.TrainUpdateHost,
				Host:    host.Name,
				Stage:   "launch",
				Status:  string(model.TrainHostFailed),
				LogPath: logPath,
				Message: fmt.Sprintf("launch failed on %s", host.Name),
			})
			return &trainWorkflowError{
				Stage:   "launch",
				Host:    host.Name,
				Command: displayCmd,
				Message: fmt.Sprintf("launch host %s: %v", host.Name, err),
				Output:  truncateOutput(out),
			}
		}
		a.emitTrain(model.TrainUpdate{
			Kind:    model.TrainUpdateHost,
			Host:    host.Name,
			Stage:   "launch",
			Status:  string(model.TrainHostSuccess),
			LogPath: logPath,
			Message: fmt.Sprintf("nohup started, pid file: %s", pidPath),
		})
	}
	a.emitTrain(model.TrainUpdate{
		Kind:   model.TrainUpdateStage,
		Stage:  "launch",
		Status: string(model.TrainStageSuccess),
	})

	// 3) create SSH masters
	a.emitTrain(model.TrainUpdate{
		Kind:    model.TrainUpdateStage,
		Stage:   "master",
		Status:  string(model.TrainStageRunning),
		Message: "3/5 create SSH control master",
	})
	for _, host := range workflow.Hosts {
		cmd := buildMasterCommand(host)
		a.emitTrain(model.TrainUpdate{
			Kind:    model.TrainUpdateHost,
			Host:    host.Name,
			Stage:   "master",
			Status:  string(model.TrainHostRunning),
			Command: cmd,
			Message: "creating ssh master",
		})
		out, err := runShellCommand(ctx, cmd)
		if err != nil {
			return &trainWorkflowError{
				Stage:   "master",
				Host:    host.Name,
				Command: cmd,
				Message: fmt.Sprintf("ssh master host %s: %v", host.Name, err),
				Output:  truncateOutput(out),
			}
		}
		a.emitTrain(model.TrainUpdate{
			Kind:    model.TrainUpdateHost,
			Host:    host.Name,
			Stage:   "master",
			Status:  string(model.TrainHostSuccess),
			Message: fmt.Sprintf("ssh master ready (%s)", host.ControlPath),
		})
	}
	a.emitTrain(model.TrainUpdate{
		Kind:   model.TrainUpdateStage,
		Stage:  "master",
		Status: string(model.TrainStageSuccess),
	})

	// 4) stream logs
	a.emitTrain(model.TrainUpdate{
		Kind:    model.TrainUpdateStage,
		Stage:   "stream",
		Status:  string(model.TrainStageRunning),
		Message: "4/5 attach remote tail -F streams",
	})

	results := make(chan trainStreamResult, len(workflow.Hosts))
	for _, host := range workflow.Hosts {
		cmd := buildTailCommand(host)
		a.emitTrain(model.TrainUpdate{
			Kind:    model.TrainUpdateHost,
			Host:    host.Name,
			Stage:   "stream",
			Status:  string(model.TrainHostRunning),
			Command: cmd,
			LogPath: host.LogPath,
			Message: "log stream connected",
		})
		go func(h trainHost, command string) {
			err := a.streamHostLogs(ctx, h, command)
			results <- trainStreamResult{Host: h.Name, Command: command, Err: err}
		}(host, cmd)
	}
	a.emitTrain(model.TrainUpdate{
		Kind:   model.TrainUpdateStage,
		Stage:  "stream",
		Status: string(model.TrainStageSuccess),
	})

	// 5) monitor + parse
	a.emitTrain(model.TrainUpdate{
		Kind:    model.TrainUpdateStage,
		Stage:   "dashboard",
		Status:  string(model.TrainStageRunning),
		Message: "5/5 parsing multi-host metrics into dashboard",
	})

	remainingStreams := len(workflow.Hosts)
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case result := <-results:
			if result.Err == nil || errors.Is(result.Err, context.Canceled) {
				if result.Host != "" {
					a.emitTrain(model.TrainUpdate{
						Kind:    model.TrainUpdateHost,
						Host:    result.Host,
						Stage:   "dashboard",
						Status:  string(model.TrainHostSuccess),
						Message: "log stream completed",
					})
				}
				remainingStreams--
				if remainingStreams == 0 {
					a.emitTrain(model.TrainUpdate{
						Kind:   model.TrainUpdateStage,
						Stage:  "dashboard",
						Status: string(model.TrainStageSuccess),
					})
					a.emitTrain(model.TrainUpdate{
						Kind:    model.TrainUpdateDone,
						RunID:   workflow.RunID,
						Message: "all host streams finished",
					})
					return nil
				}
				continue
			}
			return &trainWorkflowError{
				Stage:   "stream",
				Host:    result.Host,
				Command: result.Command,
				Message: fmt.Sprintf("log stream host %s: %v", result.Host, result.Err),
			}
		}
	}
}

func (a *Application) syncTrainHosts(ctx context.Context, workflow trainWorkflow) error {
	if len(workflow.Hosts) == 0 {
		return nil
	}

	syncCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	limit := workflow.SyncParallelism
	if limit < 1 {
		limit = 1
	}

	results := make(chan trainCommandResult, len(workflow.Hosts))
	sem := make(chan struct{}, limit)

	for _, host := range workflow.Hosts {
		host := host
		cmd := buildRsyncCommand(workflow, host)
		a.emitTrain(model.TrainUpdate{
			Kind:    model.TrainUpdateHost,
			Host:    host.Name,
			Stage:   "sync",
			Status:  string(model.TrainHostRunning),
			Command: cmd,
			Message: "rsync started",
		})

		go func() {
			select {
			case sem <- struct{}{}:
			case <-syncCtx.Done():
				results <- trainCommandResult{
					Host:    host,
					Command: cmd,
					Err:     syncCtx.Err(),
				}
				return
			}
			defer func() {
				<-sem
			}()

			out, err := runShellCommand(syncCtx, cmd)
			results <- trainCommandResult{
				Host:    host,
				Command: cmd,
				Output:  out,
				Err:     err,
			}
		}()
	}

	var firstErr error
	for i := 0; i < len(workflow.Hosts); i++ {
		result := <-results
		switch {
		case result.Err == nil:
			a.emitTrain(model.TrainUpdate{
				Kind:    model.TrainUpdateHost,
				Host:    result.Host.Name,
				Stage:   "sync",
				Status:  string(model.TrainHostSuccess),
				Message: "rsync finished",
			})
		case errors.Is(result.Err, context.Canceled):
			continue
		default:
			a.emitTrain(model.TrainUpdate{
				Kind:    model.TrainUpdateHost,
				Host:    result.Host.Name,
				Stage:   "sync",
				Status:  string(model.TrainHostFailed),
				Command: result.Command,
				Message: fmt.Sprintf("rsync failed: %v", result.Err),
			})
			if firstErr == nil {
				firstErr = &trainWorkflowError{
					Stage:   "sync",
					Host:    result.Host.Name,
					Command: result.Command,
					Message: fmt.Sprintf("sync host %s: %v", result.Host.Name, result.Err),
					Output:  truncateOutput(result.Output),
				}
				cancel()
			}
		}
	}

	if firstErr != nil {
		return firstErr
	}
	return ctx.Err()
}

func (a *Application) streamHostLogs(ctx context.Context, host trainHost, command string) error {
	cmd := exec.CommandContext(ctx, "bash", "-lc", command)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return err
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return err
	}
	if err := cmd.Start(); err != nil {
		return err
	}

	lines := make(chan string, 128)
	readPipe := func(scanner *bufio.Scanner) {
		for scanner.Scan() {
			select {
			case lines <- scanner.Text():
			case <-ctx.Done():
				return
			}
		}
	}

	stdoutScanner := bufio.NewScanner(stdout)
	stdoutScanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	stderrScanner := bufio.NewScanner(stderr)
	stderrScanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	go readPipe(stdoutScanner)
	go readPipe(stderrScanner)

	waitCh := make(chan error, 1)
	go func() {
		waitCh <- cmd.Wait()
	}()

	lastStep := 0
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case err := <-waitCh:
			if ctx.Err() != nil {
				return ctx.Err()
			}
			if err != nil {
				return err
			}
			return nil
		case line := <-lines:
			displayLine := sanitizeTrainLogLine(line)
			if strings.TrimSpace(displayLine) == "" {
				continue
			}
			a.emitTrainBestEffort(model.TrainUpdate{
				Kind:    model.TrainUpdateLog,
				Host:    host.Name,
				Stage:   "dashboard",
				Message: displayLine,
			})

			metric := parseTrainMetric(displayLine)
			if metric.HasLoss && !metric.HasStep {
				lastStep++
				metric.Step = lastStep
				metric.HasStep = true
			}
			if metric.HasStep && metric.Step > lastStep {
				lastStep = metric.Step
			}

			if !metric.HasLoss && !metric.HasStep && !metric.HasThroughput && !metric.HasGradNorm && !metric.HasModel && !metric.HasTotalStep {
				continue
			}

			a.emitTrainBestEffort(model.TrainUpdate{
				Kind:          model.TrainUpdateMetric,
				Host:          host.Name,
				Stage:         "dashboard",
				Step:          metric.Step,
				TotalStep:     metric.TotalStep,
				Loss:          metric.Loss,
				Throughput:    metric.Throughput,
				GradNorm:      metric.GradNorm,
				Model:         metric.Model,
				HasStep:       metric.HasStep,
				HasTotalStep:  metric.HasTotalStep,
				HasLoss:       metric.HasLoss,
				HasThroughput: metric.HasThroughput,
				HasGradNorm:   metric.HasGradNorm,
				HasModel:      metric.HasModel,
			})
		}
	}
}

func parseTrainMetric(line string) trainMetricParse {
	var parsed trainMetricParse
	line = sanitizeTrainLogLine(line)
	if line == "" {
		return parsed
	}

	if m := trainStepRE.FindStringSubmatch(line); len(m) == 2 {
		if v, err := strconv.Atoi(m[1]); err == nil {
			parsed.Step = v
			parsed.HasStep = true
		}
	}
	if m := trainTotalStepRE.FindStringSubmatch(line); len(m) == 2 {
		if v, err := strconv.Atoi(m[1]); err == nil {
			parsed.TotalStep = v
			parsed.HasTotalStep = true
		}
	}
	if m := trainProgressRE.FindStringSubmatch(line); len(m) == 3 {
		if !parsed.HasStep {
			if v, err := strconv.Atoi(m[1]); err == nil {
				parsed.Step = v
				parsed.HasStep = true
			}
		}
		if !parsed.HasTotalStep {
			if v, err := strconv.Atoi(m[2]); err == nil {
				parsed.TotalStep = v
				parsed.HasTotalStep = true
			}
		}
	}
	if m := trainLossRE.FindStringSubmatch(line); len(m) == 2 {
		if v, err := strconv.ParseFloat(m[1], 64); err == nil {
			parsed.Loss = v
			parsed.HasLoss = true
		}
	}
	if m := trainThroughputRE.FindStringSubmatch(line); len(m) == 2 {
		if v, err := strconv.ParseFloat(m[1], 64); err == nil {
			parsed.Throughput = v
			parsed.HasThroughput = true
		}
	}
	if m := trainGradNormRE.FindStringSubmatch(line); len(m) == 2 {
		if v, err := strconv.ParseFloat(m[1], 64); err == nil {
			parsed.GradNorm = v
			parsed.HasGradNorm = true
		}
	}
	if m := trainModelRE.FindStringSubmatch(line); len(m) == 2 {
		parsed.Model = strings.TrimSpace(m[1])
		parsed.HasModel = parsed.Model != ""
	}

	return parsed
}

func sanitizeTrainLogLine(line string) string {
	line = strings.TrimRight(strings.ReplaceAll(line, "\r\n", "\n"), "\n")
	if line == "" {
		return ""
	}

	if strings.Contains(line, "\r") {
		parts := strings.Split(line, "\r")
		for i := len(parts) - 1; i >= 0; i-- {
			part := strings.TrimSpace(parts[i])
			if part != "" {
				line = part
				break
			}
		}
	}

	line = trainANSIEscapeRE.ReplaceAllString(line, "")

	var b strings.Builder
	lastSpace := false
	for _, r := range line {
		switch {
		case r == '\n' || r == '\r' || r == '\t':
			if !lastSpace {
				b.WriteByte(' ')
				lastSpace = true
			}
		case unicode.IsControl(r):
			if !lastSpace {
				b.WriteByte(' ')
				lastSpace = true
			}
		default:
			b.WriteRune(r)
			lastSpace = unicode.IsSpace(r)
		}
	}

	return strings.TrimSpace(b.String())
}

func buildRsyncCommand(workflow trainWorkflow, host trainHost) string {
	args := []string{"rsync", "-a"}
	if workflow.RsyncCompress {
		args = append(args, "-z")
	}
	args = append(args, "--delete", "--omit-dir-times")
	if workflow.RsyncRespectGitIgnore {
		args = append(args, "--filter", shellQuote(":- .gitignore"))
	}
	for _, ex := range workflow.Exclude {
		ex = strings.TrimSpace(ex)
		if ex == "" {
			continue
		}
		args = append(args, "--exclude", shellQuote(ex))
	}
	args = append(args, "-e", shellQuote(buildRsyncShellCommand(host)))

	src := strings.TrimSpace(host.LocalPath)
	if src == "" {
		src = workflow.LocalPath
	}
	if host.LocalIsDir || src == "" {
		src = strings.TrimRight(src, "/") + "/"
	}
	dst := fmt.Sprintf("%s:%s/", host.Target, strings.TrimRight(host.RemoteCodePath, "/"))
	args = append(args, shellQuote(src), shellQuote(dst))
	return strings.Join(args, " ")
}

func buildLaunchCommand(workflow trainWorkflow, host trainHost) (cmd string, logPath string, pidPath string) {
	runDir := path.Join(host.RunBaseDir, workflow.RunID)
	logPath = path.Join(runDir, "log.txt")
	pidPath = path.Join(runDir, "train.pid")

	trainCmd := renderTrainCommand(host.TrainCommand, workflow, host, logPath, pidPath)
	startupCmd := renderTrainStartupCommand(host.StartupCommand, workflow, host, logPath, pidPath)
	launchCmd := buildTrainLaunchScript(host.RemoteCodePath, startupCmd, trainCmd)
	script := strings.Join([]string{
		"set -e",
		fmt.Sprintf("mkdir -p %s", remotePathExpr(runDir)),
		fmt.Sprintf("cd %s", remotePathExpr(host.RemoteCodePath)),
		fmt.Sprintf(": > %s", remotePathExpr(logPath)),
		fmt.Sprintf("nohup bash -lc %s > %s 2>&1 < /dev/null &", shellQuote(launchCmd), remotePathExpr(logPath)),
		`pid=$!`,
		fmt.Sprintf("echo \"$pid\" > %s", remotePathExpr(pidPath)),
		"sleep 1",
		`if ! kill -0 "$pid" 2>/dev/null; then`,
		`  echo "process exited immediately (pid=$pid)"`,
		fmt.Sprintf("  rm -f %s", remotePathExpr(pidPath)),
		fmt.Sprintf("  tail -n 20 %s 2>/dev/null || true", remotePathExpr(logPath)),
		"  exit 1",
		"fi",
	}, "\n")
	cmd = buildSSHScriptCommand(host, "auto", nil, "__MSCLI_TRAIN__", script)
	return cmd, logPath, pidPath
}

func buildLaunchDisplayCommand(workflow trainWorkflow, host trainHost, logPath string, pidPath string) string {
	runDir := path.Join(host.RunBaseDir, workflow.RunID)
	trainCmd := renderTrainCommand(host.TrainCommand, workflow, host, logPath, pidPath)
	startupCmd := renderTrainStartupCommand(host.StartupCommand, workflow, host, logPath, pidPath)
	launchCmd := buildTrainLaunchScript(host.RemoteCodePath, startupCmd, trainCmd)
	return strings.Join([]string{
		fmt.Sprintf("mkdir -p %s", remotePathExpr(runDir)),
		fmt.Sprintf(": > %s", remotePathExpr(logPath)),
		fmt.Sprintf("nohup bash -lc %s > %s 2>&1 < /dev/null &", shellQuote(launchCmd), remotePathExpr(logPath)),
		fmt.Sprintf("echo \"$!\" > %s", remotePathExpr(pidPath)),
	}, "\n")
}

func buildTrainLaunchScript(remoteCodePath, startupCmd, trainCmd string) string {
	if strings.TrimSpace(startupCmd) == "" {
		return fmt.Sprintf("cd %s && %s", remotePathExpr(remoteCodePath), trainCmd)
	}
	lines := []string{fmt.Sprintf("cd %s", remotePathExpr(remoteCodePath))}
	lines = append(lines, startupCmd)
	lines = append(lines, trainCmd)
	return strings.Join(lines, "\n")
}

func buildMasterCommand(host trainHost) string {
	checkCmd := buildSSHCommand(host, "auto", []string{"-O", "check"}, "")
	startCmd := buildSSHCommand(host, "yes", []string{"-MNf"}, "")
	return fmt.Sprintf("%s >/dev/null 2>&1 || %s", checkCmd, startCmd)
}

func buildTailCommand(host trainHost) string {
	remote := strings.Join([]string{
		fmt.Sprintf("pid=$(cat %s 2>/dev/null || true)", remotePathExpr(host.PIDPath)),
		`if [ -n "$pid" ]; then`,
		fmt.Sprintf("  tail --pid=\"$pid\" -n 0 -F %s", remotePathExpr(host.LogPath)),
		"else",
		fmt.Sprintf("  tail -n 0 -F %s", remotePathExpr(host.LogPath)),
		"fi",
	}, "\n")
	return buildSSHScriptCommand(host, "auto", nil, "__MSCLI_TAIL__", remote)
}

func buildRsyncShellCommand(host trainHost) string {
	args := []string{"ssh"}
	args = append(args, buildSSHControlOptions(host, "auto")...)
	return strings.Join(args, " ")
}

func buildSSHCommand(host trainHost, masterMode string, extraArgs []string, remote string) string {
	args := []string{"ssh"}
	args = append(args, extraArgs...)
	args = append(args, buildSSHControlOptions(host, masterMode)...)
	args = append(args, shellQuote(host.Target))
	if remote != "" {
		args = append(args, shellQuote(remote))
	}
	return strings.Join(args, " ")
}

func buildSSHScriptCommand(host trainHost, masterMode string, extraArgs []string, delimiter, script string) string {
	delimiter = strings.TrimSpace(delimiter)
	if delimiter == "" {
		delimiter = "__MSCLI_REMOTE__"
	}
	args := []string{"ssh"}
	args = append(args, extraArgs...)
	args = append(args, buildSSHControlOptions(host, masterMode)...)
	args = append(args, shellQuote(host.Target), "bash", "-s", "--")
	return strings.Join(args, " ") + fmt.Sprintf(" <<'%s'\n%s\n%s", delimiter, script, delimiter)
}

func buildSSHControlOptions(host trainHost, masterMode string) []string {
	args := []string{}
	if masterMode != "" {
		args = append(args, "-o", "ControlMaster="+masterMode)
	}
	controlPath := expandLocalPath(host.ControlPath)
	if controlPath != "" {
		args = append(args, "-o", "ControlPath="+controlPath)
	}
	if host.ControlPersist != "" {
		args = append(args, "-o", "ControlPersist="+host.ControlPersist)
	}
	return args
}

func runShellCommand(ctx context.Context, command string) (string, error) {
	cmd := exec.CommandContext(ctx, "bash", "-lc", command)
	output, err := cmd.CombinedOutput()
	return strings.TrimSpace(string(output)), err
}

func (a *Application) buildTrainWorkflow(args []string) (trainWorkflow, error) {
	cfg := a.Config.Training
	if !cfg.Enabled {
		return trainWorkflow{}, fmt.Errorf("training.enabled=false (set training.enabled=true in config)")
	}
	if len(cfg.Hosts) == 0 {
		return trainWorkflow{}, fmt.Errorf("training.hosts is empty in config (check training.hosts_file)")
	}

	req := parseTrainRequest(args)
	runID := buildTrainRunID(req)
	if !isValidRunID(runID) {
		return trainWorkflow{}, fmt.Errorf("run_id %q is invalid (allowed: a-z A-Z 0-9 . _ -)", runID)
	}

	absLocalPath, _, err := resolveTrainingLocalPath(a.WorkDir, cfg.LocalPath)
	if err != nil {
		return trainWorkflow{}, fmt.Errorf("resolve local_path: %w", err)
	}

	controlPersist := strings.TrimSpace(cfg.SSHControlPersist)
	if controlPersist == "" {
		controlPersist = "30m"
	}

	workflow := trainWorkflow{
		RunID:                 runID,
		Request:               req.Prompt,
		Target:                req.Target,
		SourceArgs:            append([]string(nil), args...),
		LocalPath:             absLocalPath,
		Exclude:               mergeExcludePatterns(configs.DefaultTrainingExcludes(), cfg.Exclude),
		ControlPersist:        controlPersist,
		RsyncCompress:         cfg.RsyncCompress,
		RsyncRespectGitIgnore: cfg.RsyncRespectGitIgnore,
		SyncParallelism:       normalizeTrainSyncParallelism(cfg.SyncParallelism, len(cfg.Hosts)),
		Hosts:                 make([]trainHost, 0, len(cfg.Hosts)),
	}

	for idx, hostCfg := range cfg.Hosts {
		host, err := resolveTrainHost(cfg, hostCfg, a.WorkDir)
		if err != nil {
			return trainWorkflow{}, fmt.Errorf("training.hosts[%d]: %w", idx, err)
		}
		host.ControlPersist = controlPersist
		if req.Prompt != "" {
			scriptSpec, err := resolveHostTrainScript(host, req)
			if err != nil {
				return trainWorkflow{}, fmt.Errorf("training.hosts[%d]: %w", idx, err)
			}
			host.TrainScript = scriptSpec.RelPath
			host.TrainCommand = buildRequestedTrainCommand(host.TrainCommand, req, scriptSpec)
			workflow.ScriptPath = mergeWorkflowScriptPath(workflow.ScriptPath, scriptSpec.RelPath)
		} else if host.TrainScript != "" {
			workflow.ScriptPath = mergeWorkflowScriptPath(workflow.ScriptPath, host.TrainScript)
		}
		workflow.Hosts = append(workflow.Hosts, host)
	}

	return workflow, nil
}

func parseTrainRequest(args []string) trainRequestSpec {
	if len(args) == 0 {
		return trainRequestSpec{}
	}

	if len(args) == 1 {
		token := strings.TrimSpace(args[0])
		if token != "" && isValidRunID(token) && !looksLikeScriptReference(token) {
			return trainRequestSpec{
				RunID:         token,
				ExplicitRunID: true,
			}
		}
	}

	spec := trainRequestSpec{
		Prompt: strings.TrimSpace(strings.Join(args, " ")),
	}
	for _, arg := range args {
		candidate := strings.Trim(strings.TrimSpace(arg), `"'`)
		if candidate == "" {
			continue
		}
		lower := strings.ToLower(candidate)
		switch {
		case spec.ScriptHint == "" && strings.HasPrefix(lower, "script="):
			spec.ScriptHint = strings.TrimSpace(candidate[len("script="):])
		case spec.Target == "" && strings.HasPrefix(lower, "model="):
			spec.Target = strings.TrimSpace(candidate[len("model="):])
		case spec.Target == "" && strings.HasPrefix(lower, "target="):
			spec.Target = strings.TrimSpace(candidate[len("target="):])
		case spec.ScriptHint == "" && looksLikeScriptReference(candidate):
			spec.ScriptHint = candidate
		}
	}
	if spec.Target == "" {
		spec.Target = extractTrainTarget(spec.Prompt)
	}
	return spec
}

func buildTrainRunID(req trainRequestSpec) string {
	if req.ExplicitRunID && req.RunID != "" {
		return req.RunID
	}
	runID := time.Now().Format("20060102-150405")
	suffix := sanitizeRunIDFragment(firstNonEmpty(req.Target, req.Prompt))
	if suffix == "" {
		return runID
	}
	if len(suffix) > 28 {
		suffix = suffix[:28]
	}
	return runID + "-" + suffix
}

func resolveTrainScript(localRoot string, req trainRequestSpec) (trainScriptSpec, error) {
	if strings.TrimSpace(req.ScriptHint) != "" {
		return resolveExplicitTrainScript(localRoot, req.ScriptHint)
	}
	return discoverTrainScript(localRoot, req)
}

func resolveExplicitTrainScript(localRoot, hint string) (trainScriptSpec, error) {
	hint = strings.Trim(strings.TrimSpace(hint), `"'`)
	if hint == "" {
		return trainScriptSpec{}, fmt.Errorf("train script hint is empty")
	}

	fullPath := hint
	if !filepath.IsAbs(fullPath) {
		fullPath = filepath.Join(localRoot, filepath.FromSlash(hint))
	}
	absPath, err := filepath.Abs(fullPath)
	if err != nil {
		return trainScriptSpec{}, fmt.Errorf("resolve train script %q: %w", hint, err)
	}
	relPath, err := filepath.Rel(localRoot, absPath)
	if err != nil {
		return trainScriptSpec{}, fmt.Errorf("resolve train script %q: %w", hint, err)
	}
	if strings.HasPrefix(relPath, "..") {
		return trainScriptSpec{}, fmt.Errorf("train script %q is outside training.local_path", hint)
	}
	info, err := os.Stat(absPath)
	if err != nil {
		return trainScriptSpec{}, fmt.Errorf("stat train script %q: %w", hint, err)
	}
	if info.IsDir() {
		return trainScriptSpec{}, fmt.Errorf("train script %q is a directory", hint)
	}
	if !strings.HasSuffix(strings.ToLower(absPath), ".py") {
		return trainScriptSpec{}, fmt.Errorf("train script %q must be a Python file", hint)
	}
	return inspectTrainScript(absPath, filepath.ToSlash(relPath))
}

func discoverTrainScript(localRoot string, req trainRequestSpec) (trainScriptSpec, error) {
	type candidate struct {
		spec  trainScriptSpec
		score int
	}

	tokens := trainPromptTokens(req.Prompt, req.Target)
	best := candidate{score: -1}
	walkErr := filepath.WalkDir(localRoot, func(fullPath string, d os.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if d.IsDir() {
			if shouldSkipTrainDiscoveryDir(d.Name()) && fullPath != localRoot {
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(strings.ToLower(d.Name()), ".py") {
			return nil
		}
		relPath, err := filepath.Rel(localRoot, fullPath)
		if err != nil {
			return nil
		}
		relPath = filepath.ToSlash(relPath)
		spec, err := inspectTrainScript(fullPath, relPath)
		if err != nil {
			return nil
		}
		score := scoreTrainScript(relPath, fullPath, tokens)
		if score > best.score || (score == best.score && best.spec.RelPath != "" && len(relPath) < len(best.spec.RelPath)) {
			best = candidate{spec: spec, score: score}
		}
		return nil
	})
	if walkErr != nil {
		return trainScriptSpec{}, fmt.Errorf("discover train script: %w", walkErr)
	}
	if best.spec.RelPath == "" {
		return trainScriptSpec{}, fmt.Errorf("cannot resolve a training Python script from request %q", req.Prompt)
	}
	return best.spec, nil
}

func inspectTrainScript(fullPath, relPath string) (trainScriptSpec, error) {
	data, err := os.ReadFile(fullPath)
	if err != nil {
		return trainScriptSpec{}, err
	}
	content := string(data)
	return trainScriptSpec{
		RelPath:     relPath,
		RunIDFlag:   findSupportedFlag(content, []string{"--run-id", "--run_id"}),
		HostFlag:    findSupportedFlag(content, []string{"--host", "--hostname", "--host-name"}),
		ModelFlag:   findSupportedFlag(content, []string{"--model", "--model-name", "--model_name", "--base-model", "--base_model", "--target-model", "--target_model"}),
		RequestFlag: findSupportedFlag(content, []string{"--task", "--prompt", "--request", "--goal", "--instruction"}),
	}, nil
}

func buildRequestedTrainCommand(template string, req trainRequestSpec, script trainScriptSpec) string {
	if hasTrainRequestPlaceholder(template) {
		return template
	}

	args := []string{
		"MSCLI_RUN_ID=" + shellQuote("{{RUN_ID}}"),
		"MSCLI_HOST_NAME=" + shellQuote("{{HOST_NAME}}"),
	}
	if req.Prompt != "" {
		args = append(args, "MSCLI_TRAIN_REQUEST="+shellQuote(req.Prompt))
	}
	if req.Target != "" {
		args = append(args, "MSCLI_TRAIN_TARGET="+shellQuote(req.Target))
	}

	args = append(args, "python", "-u", shellQuote(script.RelPath))
	if script.RunIDFlag != "" {
		args = append(args, script.RunIDFlag, shellQuote("{{RUN_ID}}"))
	}
	if script.HostFlag != "" {
		args = append(args, script.HostFlag, shellQuote("{{HOST_NAME}}"))
	}
	if req.Target != "" && script.ModelFlag != "" {
		args = append(args, script.ModelFlag, shellQuote(req.Target))
	}
	if req.Prompt != "" && script.RequestFlag != "" {
		args = append(args, script.RequestFlag, shellQuote(req.Prompt))
	}
	return strings.Join(args, " ")
}

func hasTrainRequestPlaceholder(template string) bool {
	for _, placeholder := range []string{"{{TRAIN_SCRIPT}}", "{{TRAIN_TARGET}}", "{{TRAIN_REQUEST}}"} {
		if strings.Contains(template, placeholder) {
			return true
		}
	}
	return false
}

func findSupportedFlag(content string, flags []string) string {
	lower := strings.ToLower(content)
	for _, flag := range flags {
		if strings.Contains(lower, strings.ToLower(flag)) {
			return flag
		}
	}
	return ""
}

func scoreTrainScript(relPath, fullPath string, tokens []string) int {
	score := 0
	lowerRel := strings.ToLower(filepath.ToSlash(relPath))
	lowerBase := strings.ToLower(filepath.Base(lowerRel))

	if strings.Contains(lowerBase, "train") {
		score += 24
	}
	for _, keyword := range []string{"tune", "finetune", "fine_tune", "sft", "lora", "pretrain"} {
		if strings.Contains(lowerBase, keyword) {
			score += 20
		}
		if strings.Contains(lowerRel, keyword) {
			score += 8
		}
	}
	if strings.HasSuffix(lowerBase, "train.py") || strings.HasSuffix(lowerBase, "tune.py") {
		score += 12
	}
	if strings.Contains(lowerRel, "/tests/") || strings.Contains(lowerRel, "/test/") || strings.HasSuffix(lowerBase, "_test.py") {
		score -= 24
	}

	data, err := os.ReadFile(fullPath)
	if err == nil {
		lower := strings.ToLower(string(data))
		if strings.Contains(lower, "argparse") {
			score += 4
		}
		if strings.Contains(lower, "__main__") {
			score += 4
		}
		for _, keyword := range []string{"trainer.train", ".fit(", "finetune", "lora", "pretrain", "trainingarguments"} {
			if strings.Contains(lower, keyword) {
				score += 6
			}
		}
		for _, token := range tokens {
			if strings.Contains(lower, token) {
				score += 3
			}
		}
	}

	for _, token := range tokens {
		if strings.Contains(lowerRel, token) {
			score += 8
		}
	}
	return score
}

func shouldSkipTrainDiscoveryDir(name string) bool {
	name = strings.TrimSpace(strings.ToLower(name))
	switch name {
	case ".git", ".hg", ".svn", ".venv", "venv", "env", "__pycache__", "node_modules", "dist", "build":
		return true
	}
	return strings.HasPrefix(name, ".")
}

func trainPromptTokens(prompt, target string) []string {
	text := strings.ToLower(strings.TrimSpace(prompt + " " + target))
	if text == "" {
		return nil
	}
	raw := regexp.MustCompile(`[a-z0-9._/-]+`).FindAllString(text, -1)
	stop := map[string]bool{
		"the": true, "with": true, "current": true, "code": true, "train": true, "training": true,
		"tuning": true, "tune": true, "model": true, "using": true, "use": true, "for": true,
		"and": true, "run": true, "start": true, "script": true,
	}
	seen := make(map[string]bool)
	tokens := make([]string, 0, len(raw))
	for _, token := range raw {
		token = strings.Trim(token, "._/-")
		if len(token) < 2 || stop[token] || seen[token] {
			continue
		}
		seen[token] = true
		tokens = append(tokens, token)
	}
	return tokens
}

func extractTrainTarget(prompt string) string {
	prompt = strings.TrimSpace(prompt)
	if prompt == "" {
		return ""
	}
	for _, re := range []*regexp.Regexp{
		regexp.MustCompile(`(?i)\b(?:tuning|tune|training|train|finetune|fine-tune|sft|pretrain|lora)\s+([A-Za-z0-9._/\-]+)\b`),
		regexp.MustCompile(`(?i)\bfor\s+([A-Za-z0-9._/\-]+)\b`),
	} {
		if m := re.FindStringSubmatch(prompt); len(m) == 2 {
			return strings.TrimSpace(m[1])
		}
	}
	return ""
}

func sanitizeRunIDFragment(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))
	if s == "" {
		return ""
	}
	var b strings.Builder
	lastDash := false
	for _, r := range s {
		switch {
		case r >= 'a' && r <= 'z', r >= '0' && r <= '9':
			b.WriteRune(r)
			lastDash = false
		case r == '.', r == '_', r == '-':
			if !lastDash {
				b.WriteRune('-')
				lastDash = true
			}
		case r == ' ' || r == '/':
			if !lastDash {
				b.WriteRune('-')
				lastDash = true
			}
		}
	}
	return strings.Trim(b.String(), "-")
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}

func normalizeTrainSyncParallelism(configured, hostCount int) int {
	if hostCount <= 1 {
		return 1
	}
	if configured > 0 {
		if configured > hostCount {
			return hostCount
		}
		return configured
	}
	if hostCount < 4 {
		return hostCount
	}
	return 4
}

func mergeExcludePatterns(base, extra []string) []string {
	merged := make([]string, 0, len(base)+len(extra))
	seen := make(map[string]bool, len(base)+len(extra))
	for _, values := range [][]string{base, extra} {
		for _, value := range values {
			value = strings.TrimSpace(value)
			if value == "" || seen[value] {
				continue
			}
			seen[value] = true
			merged = append(merged, value)
		}
	}
	return merged
}

func expandLocalPath(p string) string {
	p = strings.TrimSpace(p)
	switch {
	case p == "":
		return ""
	case p == "~":
		home, err := os.UserHomeDir()
		if err == nil {
			return home
		}
	case strings.HasPrefix(p, "~/"):
		home, err := os.UserHomeDir()
		if err == nil {
			return filepath.Join(home, p[2:])
		}
	}
	return p
}

func resolveTrainingLocalPath(workDir, value string) (string, bool, error) {
	pathValue := strings.TrimSpace(value)
	if pathValue == "" {
		pathValue = "."
	}
	pathValue = expandLocalPath(pathValue)
	if !filepath.IsAbs(pathValue) {
		pathValue = filepath.Join(workDir, pathValue)
	}
	absPath, err := filepath.Abs(pathValue)
	if err != nil {
		return "", false, err
	}
	info, err := os.Stat(absPath)
	if err != nil {
		return "", false, err
	}
	return absPath, info.IsDir(), nil
}

func resolveConfiguredTrainScript(localPath string, localIsDir bool, configured string) (string, error) {
	configured = strings.TrimSpace(configured)
	if configured == "" {
		if !localIsDir && strings.HasSuffix(strings.ToLower(localPath), ".py") {
			return filepath.ToSlash(filepath.Base(localPath)), nil
		}
		return "", nil
	}

	if !localIsDir {
		if !strings.HasSuffix(strings.ToLower(localPath), ".py") {
			return "", fmt.Errorf("local_path %q is a file, but not a Python file", localPath)
		}
		configuredBase := filepath.Base(filepath.Clean(filepath.FromSlash(configured)))
		localBase := filepath.Base(localPath)
		if configuredBase != localBase {
			return "", fmt.Errorf("train_script %q must match local_path file %q", configured, localBase)
		}
		spec, err := inspectTrainScript(localPath, filepath.ToSlash(localBase))
		if err != nil {
			return "", err
		}
		return spec.RelPath, nil
	}

	localRoot := localPath
	spec, err := resolveExplicitTrainScript(localRoot, configured)
	if err != nil {
		return "", err
	}
	return spec.RelPath, nil
}

func resolveHostTrainScript(host trainHost, req trainRequestSpec) (trainScriptSpec, error) {
	if host.LocalPath == "" {
		return trainScriptSpec{}, fmt.Errorf("local_path is required")
	}
	if host.LocalIsDir {
		if host.TrainScript != "" {
			return resolveExplicitTrainScript(host.LocalPath, host.TrainScript)
		}
		return resolveTrainScript(host.LocalPath, req)
	}
	if host.TrainScript != "" {
		return resolveExplicitTrainScript(filepath.Dir(host.LocalPath), host.TrainScript)
	}
	if strings.HasSuffix(strings.ToLower(host.LocalPath), ".py") {
		return inspectTrainScript(host.LocalPath, filepath.ToSlash(filepath.Base(host.LocalPath)))
	}
	return trainScriptSpec{}, fmt.Errorf("local_path %q is a file, but train_script is not set", host.LocalPath)
}

func mergeWorkflowScriptPath(current, next string) string {
	if strings.TrimSpace(next) == "" {
		return current
	}
	if current == "" || current == next {
		return next
	}
	return "per-host"
}

func looksLikeScriptReference(value string) bool {
	value = strings.Trim(strings.TrimSpace(value), `"'`)
	if value == "" {
		return false
	}
	lower := strings.ToLower(value)
	return strings.HasSuffix(lower, ".py") ||
		strings.HasPrefix(lower, "./") ||
		strings.HasPrefix(lower, "../") ||
		strings.HasPrefix(lower, "/") ||
		strings.Contains(lower, "/")
}

func resolveTrainHost(global configs.TrainingConfig, hostCfg configs.TrainingHostConfig, workDir string) (trainHost, error) {
	name := strings.TrimSpace(hostCfg.Name)
	if name == "" {
		return trainHost{}, fmt.Errorf("name is required")
	}
	user := strings.TrimSpace(hostCfg.User)
	if user == "" {
		return trainHost{}, fmt.Errorf("user is required")
	}
	address := strings.TrimSpace(hostCfg.Address)
	if address == "" {
		return trainHost{}, fmt.Errorf("address is required")
	}

	localPath := strings.TrimSpace(hostCfg.LocalPath)
	if localPath == "" {
		localPath = strings.TrimSpace(global.LocalPath)
	}
	absLocalPath, localIsDir, err := resolveTrainingLocalPath(workDir, localPath)
	if err != nil {
		return trainHost{}, fmt.Errorf("resolve local_path: %w", err)
	}

	remoteCodePath := strings.TrimSpace(hostCfg.RemoteCodePath)
	if remoteCodePath == "" {
		remoteCodePath = strings.TrimSpace(global.RemoteCodePath)
	}
	if remoteCodePath == "" {
		return trainHost{}, fmt.Errorf("remote_code_path is required")
	}

	runBaseDir := strings.TrimSpace(hostCfg.RunBaseDir)
	if runBaseDir == "" {
		runBaseDir = strings.TrimSpace(global.RunBaseDir)
	}
	if runBaseDir == "" {
		runBaseDir = path.Join(remoteCodePath, "runs")
	}

	trainScript := strings.TrimSpace(hostCfg.TrainScript)
	if trainScript == "" {
		trainScript = strings.TrimSpace(global.TrainScript)
	}
	resolvedTrainScript, err := resolveConfiguredTrainScript(absLocalPath, localIsDir, trainScript)
	if err != nil {
		return trainHost{}, fmt.Errorf("resolve train_script: %w", err)
	}

	hostTrainCommand := strings.TrimSpace(hostCfg.TrainCommand)
	trainCmd := strings.TrimSpace(hostCfg.TrainCommand)
	if trainCmd == "" {
		trainCmd = strings.TrimSpace(global.TrainCommand)
	}
	if resolvedTrainScript != "" && hostTrainCommand == "" {
		globalTrainCommand := strings.TrimSpace(global.TrainCommand)
		if globalTrainCommand == "" || globalTrainCommand == defaultTrainCommandTemplate {
			trainCmd = defaultTrainScriptCommandTemplate
		}
	}
	if trainCmd == "" {
		if resolvedTrainScript != "" {
			trainCmd = defaultTrainScriptCommandTemplate
		} else {
			trainCmd = defaultTrainCommandTemplate
		}
	}
	startupCmd := strings.TrimSpace(hostCfg.StartupCommand)
	if startupCmd == "" {
		startupCmd = strings.TrimSpace(global.StartupCommand)
	}

	target := fmt.Sprintf("%s@%s", user, address)
	controlPath := fmt.Sprintf("~/.ssh/cm-%s", sanitizeHostName(name))

	return trainHost{
		Name:           name,
		User:           user,
		Address:        address,
		Target:         target,
		LocalPath:      absLocalPath,
		LocalIsDir:     localIsDir,
		TrainScript:    resolvedTrainScript,
		StartupCommand: startupCmd,
		RemoteCodePath: remoteCodePath,
		RunBaseDir:     runBaseDir,
		TrainCommand:   trainCmd,
		ControlPath:    controlPath,
	}, nil
}

func (a *Application) startTrainSession(runID string, args []string, cancel context.CancelFunc) bool {
	a.trainMu.Lock()
	defer a.trainMu.Unlock()
	if a.trainCancel != nil {
		return false
	}
	a.trainRunID = runID
	a.trainLastID = runID
	a.trainLastArgs = append([]string(nil), args...)
	a.trainCancel = cancel
	return true
}

func (a *Application) finishTrainSession(runID string) {
	a.trainMu.Lock()
	defer a.trainMu.Unlock()
	if a.trainRunID == runID {
		a.trainRunID = ""
		a.trainCancel = nil
	}
}

func (a *Application) emitTrain(update model.TrainUpdate) {
	if update.At.IsZero() {
		update.At = time.Now()
	}
	a.recordTrainUpdate(update)
	a.EventCh <- model.Event{
		Type:  model.TrainUpdateEvent,
		Train: &update,
	}
}

func (a *Application) emitTrainBestEffort(update model.TrainUpdate) {
	if update.At.IsZero() {
		update.At = time.Now()
	}
	a.recordTrainUpdate(update)
	ev := model.Event{
		Type:  model.TrainUpdateEvent,
		Train: &update,
	}
	select {
	case a.EventCh <- ev:
	default:
	}
}

func (a *Application) recordTrainUpdate(update model.TrainUpdate) {
	a.trainMu.Lock()
	defer a.trainMu.Unlock()
	a.trainState.Apply(update)
}

func truncateOutput(out string) string {
	out = strings.TrimSpace(out)
	if out == "" {
		return "(no output)"
	}
	lines := strings.Split(out, "\n")
	if len(lines) > 12 {
		lines = lines[len(lines)-12:]
	}
	return strings.Join(lines, "\n")
}

func shellQuote(s string) string {
	return "'" + strings.ReplaceAll(s, "'", `'"'"'`) + "'"
}

func shellDoubleQuote(s string) string {
	replacer := strings.NewReplacer(
		`\`, `\\`,
		`"`, `\"`,
		`$`, `\$`,
		"`", "\\`",
	)
	return `"` + replacer.Replace(s) + `"`
}

func remotePathExpr(p string) string {
	p = strings.TrimSpace(p)
	switch {
	case p == "":
		return shellDoubleQuote("")
	case p == "~":
		return "\"$HOME\""
	case strings.HasPrefix(p, "~/"):
		return "\"$HOME\"/" + shellDoubleQuote(strings.TrimPrefix(p, "~/"))
	default:
		return shellDoubleQuote(p)
	}
}

func renderTrainCommand(template string, workflow trainWorkflow, host trainHost, logPath, pidPath string) string {
	trainCmd := strings.TrimSpace(template)
	if trainCmd == "" {
		if strings.TrimSpace(host.TrainScript) != "" {
			trainCmd = defaultTrainScriptCommandTemplate
		} else {
			trainCmd = defaultTrainCommandTemplate
		}
	}
	if !strings.Contains(trainCmd, "{{RUN_ID}}") && !hasTrainRequestPlaceholder(trainCmd) {
		trainCmd = strings.TrimSpace(trainCmd + " --run_id " + workflow.RunID)
	}
	trainScript := workflow.ScriptPath
	if strings.TrimSpace(host.TrainScript) != "" {
		trainScript = host.TrainScript
	}
	return renderTrainTemplate(trainCmd, workflow, host, logPath, pidPath, trainScript)
}

func renderTrainStartupCommand(template string, workflow trainWorkflow, host trainHost, logPath, pidPath string) string {
	startupCmd := strings.TrimSpace(template)
	if startupCmd == "" {
		return ""
	}
	trainScript := workflow.ScriptPath
	if strings.TrimSpace(host.TrainScript) != "" {
		trainScript = host.TrainScript
	}
	return renderTrainTemplate(startupCmd, workflow, host, logPath, pidPath, trainScript)
}

func renderTrainTemplate(template string, workflow trainWorkflow, host trainHost, logPath, pidPath, trainScript string) string {
	replacer := strings.NewReplacer(
		"{{RUN_ID}}", workflow.RunID,
		"{{HOST_NAME}}", host.Name,
		"{{HOST_USER}}", host.User,
		"{{HOST_ADDRESS}}", host.Address,
		"{{REMOTE_CODE_PATH}}", host.RemoteCodePath,
		"{{RUN_BASE_DIR}}", host.RunBaseDir,
		"{{TRAIN_REQUEST}}", workflow.Request,
		"{{TRAIN_TARGET}}", workflow.Target,
		"{{TRAIN_SCRIPT}}", trainScript,
		"{{LOG_PATH}}", logPath,
		"{{PID_PATH}}", pidPath,
	)
	return replacer.Replace(template)
}

func sanitizeHostName(name string) string {
	replacer := strings.NewReplacer(" ", "-", "/", "-", "\\", "-", ":", "-", "@", "-", ".", "-")
	name = replacer.Replace(strings.TrimSpace(strings.ToLower(name)))
	for strings.Contains(name, "--") {
		name = strings.ReplaceAll(name, "--", "-")
	}
	name = strings.Trim(name, "-")
	if name == "" {
		return "host"
	}
	return name
}

func isValidRunID(runID string) bool {
	matched, _ := regexp.MatchString(`^[A-Za-z0-9._-]+$`, runID)
	return matched
}

func inferStageFromError(err error) string {
	if err == nil {
		return ""
	}
	msg := strings.ToLower(err.Error())
	switch {
	case strings.Contains(msg, "sync host"):
		return "sync"
	case strings.Contains(msg, "launch host"):
		return "launch"
	case strings.Contains(msg, "ssh master"):
		return "master"
	case strings.Contains(msg, "log stream"):
		return "stream"
	default:
		return "dashboard"
	}
}

func buildTrainErrorUpdate(runID string, err error) model.TrainUpdate {
	update := model.TrainUpdate{
		Kind:  model.TrainUpdateError,
		RunID: runID,
	}
	if err == nil {
		return update
	}

	var workflowErr *trainWorkflowError
	if errors.As(err, &workflowErr) {
		update.Stage = workflowErr.Stage
		update.Host = workflowErr.Host
		update.Command = workflowErr.Command
		update.Message = workflowErr.Error()
		return update
	}

	update.Stage = inferStageFromError(err)
	update.Message = err.Error()
	return update
}
