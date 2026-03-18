// Package remote provides remote execution tools.
package remote

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/vigo999/ms-cli/configs"
	"github.com/vigo999/ms-cli/runtime/ssh"
)

// TestRemoteShellTool_DiskUsage 使用 .mscli/config.yaml 中配置的机器测试 remote_shell 工具
// 在目标机器上执行 df -h 命令查看磁盘占用情况
func TestRemoteShellTool_DiskUsage(t *testing.T) {
	// 加载配置文件
	cfg, err := loadTestConfig()
	if err != nil {
		t.Skipf("无法加载配置文件: %v", err)
	}

	if len(cfg.SSH.Hosts) == 0 {
		t.Skip("配置文件中没有 SSH 主机配置")
	}

	// 创建 SSH 连接池
	pool := ssh.NewPool(cfg.SSH)
	defer pool.Close()

	// 创建 remote_shell 工具
	tool := NewShellTool(pool)

	// 测试每个配置的主机
	for alias, hostCfg := range cfg.SSH.Hosts {
		t.Run(alias, func(t *testing.T) {
			t.Logf("测试主机: %s (%s@%s:%d)", alias, hostCfg.User, hostCfg.Address, hostCfg.Port)

			// 构建参数 - 使用 IP 地址 + 完整配置
			// 注意：使用别名会导致 mergeOptions 用别名覆盖 IP 地址
			params := map[string]interface{}{
				"host":     hostCfg.Address,
				"user":     hostCfg.User,
				"key_path": hostCfg.KeyPath,
				"command":  "df -h",
				"timeout":  30,
			}

			// 执行命令
			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			defer cancel()

			result, err := tool.Execute(ctx, mustMarshalJSON(params))
			if err != nil {
				t.Fatalf("执行远程命令失败: %v", err)
			}

			// 检查结果
			if result.Error != nil {
				t.Fatalf("远程命令返回错误: %v", result.Error)
			}

			t.Logf("命令执行结果: %s", result.Summary)
			t.Logf("磁盘占用情况:\n%s", result.Content)

			// 验证输出包含预期的磁盘信息
			content := strings.ToLower(result.Content)
			if !strings.Contains(content, "filesystem") && !strings.Contains(content, "文件系统") {
				t.Error("输出应该包含 'Filesystem' 或 '文件系统'")
			}
			if !strings.Contains(content, "size") && !strings.Contains(content, "容量") {
				t.Error("输出应该包含 'Size' 或 '容量'")
			}
			if !strings.Contains(content, "used") && !strings.Contains(content, "已用") {
				t.Error("输出应该包含 'Used' 或 '已用'")
			}
		})
	}
}

// TestRemoteShellTool_DiskUsageByDirectAddress 测试直接使用 IP 地址而非别名
func TestRemoteShellTool_DiskUsageByDirectAddress(t *testing.T) {
	cfg, err := loadTestConfig()
	if err != nil {
		t.Skipf("无法加载配置文件: %v", err)
	}

	if len(cfg.SSH.Hosts) == 0 {
		t.Skip("配置文件中没有 SSH 主机配置")
	}

	// 获取第一个配置的主机
	var alias string
	var hostCfg configs.HostConfig
	for a, h := range cfg.SSH.Hosts {
		alias = a
		hostCfg = h
		break
	}

	// 创建 SSH 连接池
	pool := ssh.NewPool(cfg.SSH)
	defer pool.Close()

	tool := NewShellTool(pool)

	// 使用直接地址和完整配置
	params := map[string]interface{}{
		"host":     hostCfg.Address,
		"user":     hostCfg.User,
		"key_path": expandHome(hostCfg.KeyPath),
		"command":  "df -h /",
		"timeout":  30,
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	result, err := tool.Execute(ctx, mustMarshalJSON(params))
	if err != nil {
		t.Fatalf("执行远程命令失败: %v", err)
	}

	if result.Error != nil {
		t.Fatalf("远程命令返回错误: %v", result.Error)
	}

	t.Logf("主机 %s (%s) 的根分区磁盘占用:\n%s", alias, hostCfg.Address, result.Content)
}

// TestRemoteShellTool_MultipleCommands 测试在会话中执行多个命令（验证状态保持）
func TestRemoteShellTool_MultipleCommands(t *testing.T) {
	cfg, err := loadTestConfig()
	if err != nil {
		t.Skipf("无法加载配置文件: %v", err)
	}

	if len(cfg.SSH.Hosts) == 0 {
		t.Skip("配置文件中没有 SSH 主机配置")
	}

	// 获取第一个配置的主机
	var alias string
	var hostCfg configs.HostConfig
	for a, h := range cfg.SSH.Hosts {
		alias = a
		hostCfg = h
		break
	}

	// 创建 SSH 连接池
	pool := ssh.NewPool(cfg.SSH)
	defer pool.Close()

	tool := NewShellTool(pool)

	// 执行多个命令
	commands := []struct {
		name    string
		command string
	}{
		{"磁盘总览", "df -h"},
		{"根分区详情", "df -h /"},
		{"当前目录", "pwd"},
		{"主机名", "hostname"},
	}

	for _, cmd := range commands {
		t.Run(cmd.name, func(t *testing.T) {
			params := map[string]interface{}{
				"host":     hostCfg.Address,
				"user":     hostCfg.User,
				"key_path": hostCfg.KeyPath,
				"command":  cmd.command,
				"timeout":  30,
			}

			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			result, err := tool.Execute(ctx, mustMarshalJSON(params))
			cancel()

			if err != nil {
				t.Fatalf("执行命令 '%s' 失败: %v", cmd.command, err)
			}

			if result.Error != nil {
				t.Fatalf("命令 '%s' 返回错误: %v", cmd.command, result.Error)
			}

			t.Logf("[%s] %s:\n%s", alias, cmd.name, result.Content)
		})
	}
}

// TestRemoteShellTool_InvalidCommand 测试无效命令返回错误
func TestRemoteShellTool_InvalidCommand(t *testing.T) {
	cfg, err := loadTestConfig()
	if err != nil {
		t.Skipf("无法加载配置文件: %v", err)
	}

	if len(cfg.SSH.Hosts) == 0 {
		t.Skip("配置文件中没有 SSH 主机配置")
	}

	// 获取第一个配置的主机
	var hostCfg configs.HostConfig
	for _, h := range cfg.SSH.Hosts {
		hostCfg = h
		break
	}

	// 创建 SSH 连接池
	pool := ssh.NewPool(cfg.SSH)
	defer pool.Close()

	tool := NewShellTool(pool)

	// 执行一个不存在的命令
	params := map[string]interface{}{
		"host":     hostCfg.Address,
		"user":     hostCfg.User,
		"key_path": hostCfg.KeyPath,
		"command":  "this_command_does_not_exist_12345",
		"timeout":  10,
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	result, err := tool.Execute(ctx, mustMarshalJSON(params))
	if err != nil {
		t.Fatalf("执行调用不应返回错误: %v", err)
	}

	// 命令不存在应该返回非零退出码，但 Execute 不会返回错误
	t.Logf("无效命令执行结果: %s", result.Summary)
	t.Logf("输出内容: %s", result.Content)

	// 检查是否返回了非零退出码
	if result.Summary == "exit=0" {
		t.Error("无效命令应该返回非零退出码")
	}
}

// TestRemoteShellTool_MissingRequiredParams 测试缺少必需参数
func TestRemoteShellTool_MissingRequiredParams(t *testing.T) {
	pool := ssh.NewPool(configs.SSHConfig{})
	defer pool.Close()

	tool := NewShellTool(pool)

	// 缺少 host
	params := map[string]interface{}{
		"command": "df -h",
	}

	ctx := context.Background()
	result, err := tool.Execute(ctx, mustMarshalJSON(params))
	if err != nil {
		t.Fatalf("执行调用不应返回错误: %v", err)
	}

	if result.Error == nil {
		t.Error("缺少 host 参数应该返回错误")
	} else {
		t.Logf("正确返回错误: %v", result.Error)
	}

	// 缺少 command
	params = map[string]interface{}{
		"host": "test-host",
	}

	result, err = tool.Execute(ctx, mustMarshalJSON(params))
	if err != nil {
		t.Fatalf("执行调用不应返回错误: %v", err)
	}

	if result.Error == nil {
		t.Error("缺少 command 参数应该返回错误")
	} else {
		t.Logf("正确返回错误: %v", result.Error)
	}
}

// loadTestConfig 加载测试配置文件
func loadTestConfig() (*configs.Config, error) {
	// 尝试从多个位置加载配置
	configPaths := []string{
		".mscli/config.yaml",
		filepath.Join("..", "..", ".mscli/config.yaml"),
		filepath.Join("..", "..", "..", ".mscli/config.yaml"),
	}

	for _, path := range configPaths {
		if _, err := os.Stat(path); err == nil {
			return configs.LoadFromFile(path)
		}
	}

	return nil, os.ErrNotExist
}

// mustMarshalJSON 将 map 序列化为 JSON，失败时 panic（仅用于测试）
func mustMarshalJSON(v map[string]interface{}) json.RawMessage {
	data, err := json.Marshal(v)
	if err != nil {
		panic(err)
	}
	return json.RawMessage(data)
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
