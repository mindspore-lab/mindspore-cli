//go:build !unix

package shell

import "os/exec"

func configureCmdForCancel(cmd *exec.Cmd) {}

func terminateCmd(cmd *exec.Cmd) {
	if cmd == nil || cmd.Process == nil {
		return
	}
	_ = cmd.Process.Kill()
}
