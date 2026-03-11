// Package ssh provides SSH remote execution runtime with stateful session management.
package ssh

import (
	"fmt"
	"net"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/vigo999/ms-cli/configs"
	"golang.org/x/crypto/ssh"
	"golang.org/x/crypto/ssh/agent"
)

// ConnectOptions contains dynamic connection parameters from tool calls.
type ConnectOptions struct {
	Host     string        // 目标地址（或别名）
	User     string        // 用户名（可选，覆盖配置）
	Password string        // 密码（可选，覆盖配置）
	KeyPath  string        // 私钥路径（可选，覆盖配置）
	Port     int           // 端口（可选，默认22）
	Timeout  time.Duration // 超时（可选）
}

// Pool manages SSH sessions with connection reuse.
type Pool struct {
	mu       sync.RWMutex
	sessions map[string]*Session // key: normalized host+user
	config   configs.SSHConfig
	agentConn net.Conn            // SSH agent connection (kept open for auth)
}

// NewPool creates a new SSH connection pool.
func NewPool(cfg configs.SSHConfig) *Pool {
	return &Pool{
		sessions: make(map[string]*Session),
		config:   cfg,
	}
}

// Close closes the pool and all its sessions.
func (p *Pool) Close() error {
	if err := p.CloseAll(); err != nil {
		return err
	}
	if p.agentConn != nil {
		p.agentConn.Close()
		p.agentConn = nil
	}
	return nil
}

// Get gets or creates a session with the given options.
// Parameter merge priority: opts > pre-configured HostConfig > defaults
func (p *Pool) Get(opts ConnectOptions) (*Session, error) {
	// Resolve host alias to actual config
	resolved := p.resolveHost(opts.Host)

	// Merge options (dynamic opts override pre-configured values)
	merged := p.mergeOptions(opts, resolved)

	// Validate required fields
	if merged.Host == "" {
		return nil, fmt.Errorf("host is required")
	}
	if merged.User == "" {
		return nil, fmt.Errorf("user is required for host %s", merged.Host)
	}
	if merged.Port == 0 {
		merged.Port = 22
	}
	if merged.Timeout == 0 {
		merged.Timeout = time.Duration(p.config.DefaultTimeout) * time.Second
		if merged.Timeout == 0 {
			merged.Timeout = 60 * time.Second
		}
	}

	// Generate session key (normalized host + user)
	key := fmt.Sprintf("%s@%s:%d", merged.User, merged.Host, merged.Port)

	// Check for existing session
	p.mu.RLock()
	if sess, ok := p.sessions[key]; ok && sess.IsAlive() {
		p.mu.RUnlock()
		return sess, nil
	}
	p.mu.RUnlock()

	// Create new session
	p.mu.Lock()
	defer p.mu.Unlock()

	// Double-check after acquiring write lock
	if sess, ok := p.sessions[key]; ok && sess.IsAlive() {
		return sess, nil
	}

	sess, err := p.createSession(merged)
	if err != nil {
		return nil, fmt.Errorf("failed to create SSH session for %s: %w", key, err)
	}

	p.sessions[key] = sess
	return sess, nil
}

// CloseAll closes all sessions in the pool.
func (p *Pool) CloseAll() error {
	p.mu.Lock()
	defer p.mu.Unlock()

	var errs []error
	for key, sess := range p.sessions {
		if err := sess.Close(); err != nil {
			errs = append(errs, fmt.Errorf("failed to close session %s: %w", key, err))
		}
		delete(p.sessions, key)
	}

	if len(errs) > 0 {
		return fmt.Errorf("errors closing sessions: %v", errs)
	}
	return nil
}

// resolveHost resolves a host alias to its configuration.
// Also supports reverse lookup by IP address.
func (p *Pool) resolveHost(hostOrAlias string) configs.HostConfig {
	// Check if it's an alias in the config
	if cfg, ok := p.config.Hosts[hostOrAlias]; ok {
		return cfg
	}

	// Try reverse lookup: check if any configured host has this address
	for _, cfg := range p.config.Hosts {
		if cfg.Address == hostOrAlias {
			return cfg
		}
	}

	// Treat as direct host address
	return configs.HostConfig{
		Address: hostOrAlias,
	}
}

// mergeOptions merges dynamic options with pre-configured values.
// Priority: opts > cfg > defaults
func (p *Pool) mergeOptions(opts ConnectOptions, cfg configs.HostConfig) ConnectOptions {
	result := ConnectOptions{
		Host:     cfg.Address,
		User:     cfg.User,
		Password: cfg.Password,
		KeyPath:  cfg.KeyPath,
		Port:     cfg.Port,
	}

	// Override with dynamic options if provided
	if opts.Host != "" {
		result.Host = opts.Host // Keep original if it was an alias (already resolved)
	}
	if opts.User != "" {
		result.User = opts.User
	}
	if opts.Password != "" {
		result.Password = opts.Password
	}
	if opts.KeyPath != "" {
		result.KeyPath = opts.KeyPath
	}
	if opts.Port != 0 {
		result.Port = opts.Port
	}
	if opts.Timeout != 0 {
		result.Timeout = opts.Timeout
	}

	return result
}

// createSession creates a new SSH session with the given options.
func (p *Pool) createSession(opts ConnectOptions) (*Session, error) {
	sshConfig := &ssh.ClientConfig{
		User:            opts.User,
		Timeout:         opts.Timeout,
		HostKeyCallback: ssh.InsecureIgnoreHostKey(), // TODO: Support known_hosts
	}

	// Setup authentication methods
	authMethods := []ssh.AuthMethod{}

	// 1. Try key file if explicitly specified
	keyPath := expandHome(opts.KeyPath)
	if keyPath != "" {
		if keyAuth, err := p.getKeyAuth(keyPath); err == nil {
			authMethods = append(authMethods, keyAuth)
		} else {
			// Log key error for debugging (but don't fail yet, try other methods)
			fmt.Fprintf(os.Stderr, "ssh: failed to load key %s: %v\n", keyPath, err)
		}
	}

	// 2. Try SSH agent
	if agentAuth, ok := p.getAgentAuth(); ok {
		authMethods = append(authMethods, agentAuth)
	}

	// 3. Try default key paths (only if no explicit key and no agent)
	if keyPath == "" && len(authMethods) == 0 {
		for _, defaultPath := range []string{
			filepath.Join(os.Getenv("HOME"), ".ssh", "id_rsa"),
			filepath.Join(os.Getenv("HOME"), ".ssh", "id_ed25519"),
			filepath.Join(os.Getenv("HOME"), ".ssh", "id_ecdsa"),
		} {
			if _, err := os.Stat(defaultPath); err == nil {
				if keyAuth, err := p.getKeyAuth(defaultPath); err == nil {
					authMethods = append(authMethods, keyAuth)
					break
				}
			}
		}
	}

	// 5. Try password
	if opts.Password != "" {
		authMethods = append(authMethods, ssh.Password(opts.Password))
	}

	if len(authMethods) == 0 {
		return nil, fmt.Errorf("no authentication method available")
	}

	sshConfig.Auth = authMethods

	// Connect
	addr := fmt.Sprintf("%s:%d", opts.Host, opts.Port)
	client, err := ssh.Dial("tcp", addr, sshConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to dial SSH: %w", err)
	}

	return &Session{
		client:  client,
		host:    opts.Host,
		user:    opts.User,
		port:    opts.Port,
		env:     make(map[string]string),
		workDir: "",
	}, nil
}

// getAgentAuth returns SSH agent authentication if available.
// The agent connection is kept open in the pool and closed when the pool is closed.
func (p *Pool) getAgentAuth() (ssh.AuthMethod, bool) {
	// Reuse existing agent connection if available
	if p.agentConn != nil {
		return ssh.PublicKeysCallback(agent.NewClient(p.agentConn).Signers), true
	}

	socket := os.Getenv("SSH_AUTH_SOCK")
	if socket == "" {
		return nil, false
	}

	conn, err := net.Dial("unix", socket)
	if err != nil {
		return nil, false
	}

	// Store connection in pool for reuse (will be closed when pool is closed)
	p.agentConn = conn
	return ssh.PublicKeysCallback(agent.NewClient(p.agentConn).Signers), true
}

// getKeyAuth returns public key authentication from the given key file.
func (p *Pool) getKeyAuth(keyPath string) (ssh.AuthMethod, error) {
	key, err := os.ReadFile(keyPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read key file: %w", err)
	}

	signer, err := ssh.ParsePrivateKey(key)
	if err != nil {
		// Try with empty passphrase (for encrypted keys, user should use SSH agent)
		return nil, fmt.Errorf("failed to parse key (encrypted keys require SSH agent): %w", err)
	}

	return ssh.PublicKeys(signer), nil
}

// expandHome expands the ~ in the given path to the user's home directory.
func expandHome(path string) string {
	if strings.HasPrefix(path, "~/") {
		home, err := os.UserHomeDir()
		if err == nil {
			return filepath.Join(home, path[2:])
		}
	}
	return path
}
