package loop

import "context"

type FSTool interface {
	Read(path string) (string, error)
	Glob(path, pattern string, maxMatches int) ([]string, error)
	Grep(path, pattern string, maxMatches int) ([]string, error)
	Edit(path, oldText, newText string) (string, error)
	Write(path, content string) (int, error)
}

type ShellTool interface {
	Run(ctx context.Context, command string) (string, int, error)
}

type TraceWriter interface {
	Write(eventType string, payload any) error
}
