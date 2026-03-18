package ssh

import "time"

// Result holds the result of an SSH command execution.
type Result struct {
	Stdout   string        // 标准输出
	Stderr   string        // 标准错误
	ExitCode int           // 退出码（-1 表示执行错误）
	Error    error         // 执行错误
	Duration time.Duration // 执行耗时
}

// Success returns true if the command executed successfully (exit code 0).
func (r *Result) Success() bool {
	return r.ExitCode == 0 && r.Error == nil
}

// Combined returns stdout and stderr combined.
func (r *Result) Combined() string {
	if r.Stderr == "" {
		return r.Stdout
	}
	if r.Stdout == "" {
		return r.Stderr
	}
	return r.Stdout + "\n" + r.Stderr
}
