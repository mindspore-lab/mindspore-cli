package shell

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

// Config holds the shell runner configuration.
type Config struct {
	WorkDir        string
	Timeout        time.Duration
	AllowedCmds    []string // Whitelist (empty = allow all)
	BlockedCmds    []string // Blacklist
	RequireConfirm []string // Commands requiring confirmation
	Env            map[string]string
}

// Result is the result of a command execution.
type Result struct {
	Stdout   string
	Stderr   string
	ExitCode int
	Error    error
}

// Runner executes shell commands.
type Runner struct {
	config Config
}

const (
	maxScannerTokenSize = 1024 * 1024
	maxOutputBytes      = 64 * 1024
)

// NewRunner creates a new shell runner.
func NewRunner(cfg Config) *Runner {
	if cfg.Timeout == 0 {
		cfg.Timeout = 60 * time.Second
	}
	return &Runner{config: cfg}
}

// Run executes a command and returns the result.
func (r *Runner) Run(ctx context.Context, command string) (*Result, error) {
	// Check if command is allowed
	if reason := r.checkAllowed(command); reason != "" {
		return &Result{
			ExitCode: -1,
			Error:    fmt.Errorf("command not allowed: %s", reason),
		}, nil
	}

	buildCmd := func(execCtx context.Context) *exec.Cmd {
		cmd := exec.CommandContext(execCtx, "sh", "-c", command)
		cmd.Dir = r.config.WorkDir
		cmd.Env = os.Environ()
		for k, v := range r.config.Env {
			cmd.Env = append(cmd.Env, fmt.Sprintf("%s=%s", k, v))
		}
		return cmd
	}

	// Create command
	cmd := buildCmd(ctx)

	// Run with timeout if context doesn't have one
	if _, hasDeadline := ctx.Deadline(); !hasDeadline && r.config.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, r.config.Timeout)
		defer cancel()
		cmd = buildCmd(ctx)
	}

	// Capture output
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("create stdout pipe: %w", err)
	}

	stderr, err := cmd.StderrPipe()
	if err != nil {
		return nil, fmt.Errorf("create stderr pipe: %w", err)
	}

	// Start command
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("start command: %w", err)
	}

	// Read output
	var stdoutOut, stderrOut string
	var stdoutErr, stderrErr error

	stdoutDone := make(chan struct{})
	go func() {
		stdoutOut, stdoutErr = readCapped(stdout, maxOutputBytes)
		close(stdoutDone)
	}()

	stderrDone := make(chan struct{})
	go func() {
		stderrOut, stderrErr = readCapped(stderr, maxOutputBytes)
		close(stderrDone)
	}()

	// Wait for output readers
	<-stdoutDone
	<-stderrDone
	if stdoutErr != nil {
		return nil, fmt.Errorf("read stdout: %w", stdoutErr)
	}
	if stderrErr != nil {
		return nil, fmt.Errorf("read stderr: %w", stderrErr)
	}

	// Wait for command
	err = cmd.Wait()

	result := &Result{
		Stdout:   stdoutOut,
		Stderr:   stderrOut,
		ExitCode: 0,
	}

	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			result.ExitCode = exitErr.ExitCode()
		} else {
			result.ExitCode = -1
			result.Error = err
		}
	}

	return result, nil
}

func readCapped(r io.Reader, maxBytes int) (string, error) {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 64*1024), maxScannerTokenSize)

	var b strings.Builder
	truncated := false
	for scanner.Scan() {
		line := scanner.Text()
		extra := len(line)
		if b.Len() > 0 {
			extra++
		}
		if b.Len()+extra <= maxBytes {
			if b.Len() > 0 {
				b.WriteByte('\n')
			}
			b.WriteString(line)
		} else {
			truncated = true
		}
	}
	if err := scanner.Err(); err != nil {
		return "", err
	}
	if truncated {
		if b.Len() > 0 {
			b.WriteString("\n")
		}
		b.WriteString("[output truncated]")
	}
	return b.String(), nil
}

// IsDangerous checks if a command might be dangerous.
func (r *Runner) IsDangerous(command string) bool {
	dangerous := []string{
		"rm -rf /", "rm -rf ~", "rm -rf /*",
		"> /dev/sda", "mkfs.", "dd if=",
		":(){ :|:& };:", // fork bomb
	}

	lower := strings.ToLower(command)
	for _, d := range dangerous {
		if strings.Contains(lower, d) {
			return true
		}
	}

	// Check destructive rm commands
	if strings.HasPrefix(lower, "rm ") && strings.Contains(lower, "-rf") {
		return true
	}

	return false
}

// checkAllowed checks if a command is allowed.
func (r *Runner) checkAllowed(command string) string {
	cmd := strings.TrimSpace(command)
	lower := strings.ToLower(cmd)

	// Check blacklist
	for _, blocked := range r.config.BlockedCmds {
		if strings.Contains(lower, strings.ToLower(blocked)) {
			return fmt.Sprintf("matches blocked pattern: %s", blocked)
		}
	}

	// Check whitelist (if defined)
	if len(r.config.AllowedCmds) > 0 {
		allowed := false
		for _, allowedCmd := range r.config.AllowedCmds {
			if strings.HasPrefix(lower, strings.ToLower(allowedCmd)) {
				allowed = true
				break
			}
		}
		if !allowed {
			return "not in allowed commands list"
		}
	}

	return ""
}

// RequiresConfirm checks if a command requires user confirmation.
func (r *Runner) RequiresConfirm(command string) bool {
	cmd := strings.TrimSpace(strings.ToLower(command))

	for _, prefix := range r.config.RequireConfirm {
		if strings.HasPrefix(cmd, strings.ToLower(prefix)) {
			return true
		}
	}

	// Check for destructive commands
	destructive := []string{"rm ", "mv ", "cp -r", "> ", ">> "}
	for _, d := range destructive {
		if strings.HasPrefix(cmd, d) {
			return true
		}
	}

	return false
}

// GetWorkDir returns the working directory.
func (r *Runner) GetWorkDir() string {
	return r.config.WorkDir
}

// SanitizePath sanitizes a path for use in commands.
func SanitizePath(path string) string {
	// Prevent command injection
	path = strings.ReplaceAll(path, ";", "")
	path = strings.ReplaceAll(path, "&", "")
	path = strings.ReplaceAll(path, "|", "")
	path = strings.ReplaceAll(path, "`", "")
	path = strings.ReplaceAll(path, "$", "")
	return filepath.Clean(path)
}
