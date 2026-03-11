package ssh

import (
	"bytes"
	"context"
	"fmt"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/pkg/sftp"
	"golang.org/x/crypto/ssh"
)

// Session represents a stateful SSH session with remote state caching.
type Session struct {
	client  *ssh.Client
	sftp    *sftp.Client // lazily initialized
	host    string
	user    string
	port    int
	workDir string            // cached remote working directory
	env     map[string]string // cached remote environment variables
	mu      sync.RWMutex
	sftpMu  sync.Once
}

// Run executes a command on the remote host with state preservation.
// Automatically handles workDir and env injection.
func (s *Session) Run(ctx context.Context, cmd string) (*Result, error) {
	s.mu.RLock()
	workDir := s.workDir
	env := make(map[string]string, len(s.env))
	for k, v := range s.env {
		env[k] = v
	}
	s.mu.RUnlock()

	// Build actual command with state
	var parts []string

	// Add environment variables
	for k, v := range env {
		parts = append(parts, fmt.Sprintf("export %s=%q", k, v))
	}

	// Add directory change
	if workDir != "" {
		parts = append(parts, fmt.Sprintf("cd %q", workDir))
	}

	// Add actual command
	parts = append(parts, cmd)

	actualCmd := strings.Join(parts, " && ")

	// Execute
	start := time.Now()
	stdout, stderr, exitCode, err := s.executeRaw(ctx, actualCmd)
	duration := time.Since(start)

	result := &Result{
		Stdout:   stdout,
		Stderr:   stderr,
		ExitCode: exitCode,
		Duration: duration,
	}

	if err != nil && exitCode == -1 {
		result.Error = err
	}

	// Update state if command succeeded
	if exitCode == 0 {
		s.updateStateFromCommand(cmd)
	}

	return result, nil
}

// SetWorkDir sets the cached remote working directory.
func (s *Session) SetWorkDir(dir string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.workDir = dir
}

// GetWorkDir returns the cached remote working directory.
func (s *Session) GetWorkDir() string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.workDir
}

// SetEnv sets a cached remote environment variable.
func (s *Session) SetEnv(key, value string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.env[key] = value
}

// GetEnv returns a cached remote environment variable.
func (s *Session) GetEnv(key string) (string, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	val, ok := s.env[key]
	return val, ok
}

// IsAlive checks if the SSH connection is still alive.
func (s *Session) IsAlive() bool {
	if s.client == nil {
		return false
	}
	// Try to create a session to test connection
	session, err := s.client.NewSession()
	if err != nil {
		return false
	}
	session.Close()
	return true
}

// Close closes the SSH session and SFTP client.
func (s *Session) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	var errs []error

	if s.sftp != nil {
		if err := s.sftp.Close(); err != nil {
			errs = append(errs, fmt.Errorf("failed to close SFTP: %w", err))
		}
		s.sftp = nil
	}

	if s.client != nil {
		if err := s.client.Close(); err != nil {
			errs = append(errs, fmt.Errorf("failed to close SSH client: %w", err))
		}
		s.client = nil
	}

	if len(errs) > 0 {
		return fmt.Errorf("errors closing session: %v", errs)
	}
	return nil
}

// executeRaw executes a raw command without state handling.
func (s *Session) executeRaw(ctx context.Context, cmd string) (stdout, stderr string, exitCode int, err error) {
	session, err := s.client.NewSession()
	if err != nil {
		return "", "", -1, fmt.Errorf("failed to create session: %w", err)
	}
	defer session.Close()

	// Set up context cancellation
	done := make(chan struct{})
	go func() {
		select {
		case <-ctx.Done():
			session.Signal(ssh.SIGTERM)
			time.Sleep(100 * time.Millisecond)
			session.Close()
		case <-done:
		}
	}()
	defer close(done)

	var stdoutBuf, stderrBuf bytes.Buffer
	session.Stdout = &stdoutBuf
	session.Stderr = &stderrBuf

	err = session.Run(cmd)
	exitCode = 0
	if err != nil {
		if exitErr, ok := err.(*ssh.ExitError); ok {
			exitCode = exitErr.ExitStatus()
		} else {
			exitCode = -1
		}
	}

	return stdoutBuf.String(), stderrBuf.String(), exitCode, err
}

// updateStateFromCommand updates cached state based on command execution.
func (s *Session) updateStateFromCommand(cmd string) {
	// Check for cd command
	if newDir, ok := extractCdTarget(cmd); ok {
		s.mu.Lock()
		defer s.mu.Unlock()
		s.workDir = resolvePath(s.workDir, newDir)
	}

	// Check for export command
	if key, value, ok := extractExport(cmd); ok {
		s.mu.Lock()
		defer s.mu.Unlock()
		s.env[key] = value
	}
}

// ensureSFTP initializes the SFTP client if needed.
func (s *Session) ensureSFTP() error {
	var initErr error
	s.sftpMu.Do(func() {
		if s.sftp == nil {
			s.sftp, initErr = sftp.NewClient(s.client)
		}
	})
	return initErr
}

// extractCdTarget extracts the target directory from a cd command.
func extractCdTarget(cmd string) (string, bool) {
	cmd = strings.TrimSpace(cmd)
	// Match "cd <path>" or "cd '<path>'" or "cd \"<path>\""
	re := regexp.MustCompile(`^cd\s+(?:["']?)([^"'\s].*?)(?:["']?)$`)
	matches := re.FindStringSubmatch(cmd)
	if len(matches) >= 2 {
		return strings.Trim(matches[1], `"'`), true
	}
	return "", false
}

// extractExport extracts key-value from export command.
func extractExport(cmd string) (key, value string, ok bool) {
	cmd = strings.TrimSpace(cmd)
	// Match "export KEY=value" or "export KEY='value'" or "export KEY=\"value\""
	re := regexp.MustCompile(`^export\s+([A-Za-z_][A-Za-z0-9_]*)=(?:["']?)(.*?)(?:["']?)$`)
	matches := re.FindStringSubmatch(cmd)
	if len(matches) >= 3 {
		return matches[1], matches[2], true
	}
	return "", "", false
}

// resolvePath resolves a relative path against a base directory.
func resolvePath(base, target string) string {
	if filepath.IsAbs(target) {
		return target
	}
	if base == "" {
		return target
	}
	return filepath.Join(base, target)
}
