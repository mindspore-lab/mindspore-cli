// Package ssh provides SSH remote execution runtime with stateful session management.
package ssh

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/vigo999/ms-cli/configs"
)

// TestCreateSession_WithConfigHosts 使用 .mscli/config.yaml 中配置的主机测试 createSession 功能
// 这是一个集成测试，需要实际的 SSH 服务器和密钥
func TestCreateSession_WithConfigHosts(t *testing.T) {
	// 诊断信息
	t.Logf("SSH_AUTH_SOCK: %s", os.Getenv("SSH_AUTH_SOCK"))
	t.Logf("HOME: %s", os.Getenv("HOME"))

	// 加载配置文件
	cfg, err := loadTestConfig()
	if err != nil {
		t.Skipf("无法加载配置文件: %v", err)
	}

	if len(cfg.SSH.Hosts) == 0 {
		t.Skip("配置文件中没有 SSH 主机配置")
	}

	pool := NewPool(cfg.SSH)
	defer pool.Close()

	// 测试每个配置的主机
	for alias, hostCfg := range cfg.SSH.Hosts {
		t.Run(alias, func(t *testing.T) {
			// 使用实际 IP 地址而非别名，因为 mergeOptions 会保留原始 Host 值
			opts := ConnectOptions{
				Host:    hostCfg.Address,
				User:    hostCfg.User,
				KeyPath: expandHome(hostCfg.KeyPath),
				Port:    hostCfg.Port,
				Timeout: 10 * time.Second,
			}

			session, err := pool.Get(opts)
			if err != nil {
				t.Fatalf("无法连接到主机 %s (%s): %v", alias, hostCfg.Address, err)
			}

			if session == nil {
				t.Fatal("session 不应为 nil")
			}

			// 验证会话状态
			if !session.IsAlive() {
				t.Error("新创建的会话应该是活跃的")
			}

			t.Logf("✓ 成功连接到 %s (%s@%s:%d)", alias, hostCfg.User, hostCfg.Address, hostCfg.Port)
		})
	}
}

// TestCreateSession_ConnectionReuse 测试连接复用功能
func TestCreateSession_ConnectionReuse(t *testing.T) {
	cfg, err := loadTestConfig()
	if err != nil {
		t.Skipf("无法加载配置文件: %v", err)
	}

	if len(cfg.SSH.Hosts) == 0 {
		t.Skip("配置文件中没有 SSH 主机配置")
	}

	pool := NewPool(cfg.SSH)
	defer pool.Close()

	// 获取第一个配置的主机
	var alias string
	var hostCfg configs.HostConfig
	for a, h := range cfg.SSH.Hosts {
		alias = a
		hostCfg = h
		break
	}

	t.Logf("测试主机 %s: %s@%s:%d", alias, hostCfg.User, hostCfg.Address, hostCfg.Port)

	opts := ConnectOptions{
		Host:    hostCfg.Address,
		User:    hostCfg.User,
		KeyPath: expandHome(hostCfg.KeyPath),
		Port:    hostCfg.Port,
		Timeout: 10 * time.Second,
	}

	// 第一次连接
	session1, err := pool.Get(opts)
	if err != nil {
		t.Fatalf("无法连接到主机 %s (%s): %v", alias, hostCfg.Address, err)
	}

	// 第二次连接（应该复用）
	session2, err := pool.Get(opts)
	if err != nil {
		t.Fatalf("第二次连接失败: %v", err)
	}

	// 验证是同一个会话
	if session1 != session2 {
		t.Error("相同主机的连接应该被复用")
	}

	t.Logf("✓ 连接复用成功: %s@%s:%d", hostCfg.User, hostCfg.Address, hostCfg.Port)
}

// TestCreateSession_InvalidHost 测试无效主机连接失败
func TestCreateSession_InvalidHost(t *testing.T) {
	cfg := configs.SSHConfig{
		DefaultTimeout: 5,
	}

	pool := NewPool(cfg)
	defer pool.Close()

	opts := ConnectOptions{
		Host:    "invalid.host.that.does.not.exist.example.com",
		User:    "test",
		Port:    22,
		Timeout: 2 * time.Second,
	}

	_, err := pool.Get(opts)
	if err == nil {
		t.Error("无效主机应该返回错误")
	} else {
		t.Logf("✓ 无效主机正确返回错误: %v", err)
	}
}

// TestCreateSession_MissingAuth 测试缺少认证信息时失败
func TestCreateSession_MissingAuth(t *testing.T) {
	// 创建一个临时目录，确保没有 SSH 密钥
	tmpDir := t.TempDir()

	// 设置临时的 HOME 目录，避免读取用户默认的 SSH 密钥
	origHome := os.Getenv("HOME")
	os.Setenv("HOME", tmpDir)
	defer os.Setenv("HOME", origHome)

	// 清除 SSH agent
	origAgent := os.Getenv("SSH_AUTH_SOCK")
	os.Unsetenv("SSH_AUTH_SOCK")
	defer os.Setenv("SSH_AUTH_SOCK", origAgent)

	cfg := configs.SSHConfig{
		DefaultTimeout: 5,
	}

	pool := NewPool(cfg)
	defer pool.Close()

	opts := ConnectOptions{
		Host:    "localhost",
		User:    "testuser",
		Port:    22,
		Timeout: 2 * time.Second,
	}

	_, err := pool.Get(opts)
	if err == nil {
		t.Error("缺少认证信息应该返回错误")
	} else {
		t.Logf("✓ 缺少认证正确返回错误: %v", err)
	}
}

// TestCreateSession_DirectAddress 测试直接使用 IP 地址而非别名
func TestCreateSession_DirectAddress(t *testing.T) {
	cfg, err := loadTestConfig()
	if err != nil {
		t.Skipf("无法加载配置文件: %v", err)
	}

	if len(cfg.SSH.Hosts) == 0 {
		t.Skip("配置文件中没有 SSH 主机配置")
	}

	// 获取第一个配置的主机，使用其地址直接连接
	var hostCfg configs.HostConfig
	for _, h := range cfg.SSH.Hosts {
		hostCfg = h
		break
	}

	pool := NewPool(cfg.SSH)
	defer pool.CloseAll()

	opts := ConnectOptions{
		Host:    hostCfg.Address,
		User:    hostCfg.User,
		KeyPath: expandHome(hostCfg.KeyPath),
		Port:    hostCfg.Port,
		Timeout: 10 * time.Second,
	}

	session, err := pool.Get(opts)
	if err != nil {
		t.Fatalf("无法连接到主机 %s: %v", hostCfg.Address, err)
	}

	if !session.IsAlive() {
		t.Error("会话应该是活跃的")
	}

	t.Logf("✓ 直接使用地址连接成功: %s@%s:%d", hostCfg.User, hostCfg.Address, hostCfg.Port)
}

// loadTestConfig 加载测试配置文件
func loadTestConfig() (*configs.Config, error) {
	// 尝试从多个位置加载配置
	configPaths := []string{
		".mscli/config.yaml",
		filepath.Join("..", "..", ".mscli/config.yaml"),
	}

	for _, path := range configPaths {
		if _, err := os.Stat(path); err == nil {
			return configs.LoadFromFile(path)
		}
	}

	return nil, os.ErrNotExist
}

// expandHome 展开路径中的 ~
func expandHome(path string) string {
	if strings.HasPrefix(path, "~/") {
		home, err := os.UserHomeDir()
		if err == nil {
			return filepath.Join(home, path[2:])
		}
	}
	return path
}
