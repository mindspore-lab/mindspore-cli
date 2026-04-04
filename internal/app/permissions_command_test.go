package app

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vigo999/mindspore-cli/configs"
	"github.com/vigo999/mindspore-cli/permission"
	"github.com/vigo999/mindspore-cli/ui/model"
)

func newPermAppForTest(t *testing.T) (*Application, *permission.DefaultPermissionService) {
	t.Helper()
	permSvc := permission.NewDefaultPermissionService(configs.PermissionsConfig{
		DefaultLevel: "ask",
	})
	app := &Application{
		EventCh:     make(chan model.Event, 8),
		permService: permSvc,
		Config:      configs.DefaultConfig(),
		WorkDir:     "/tmp/work",
	}
	return app, permSvc
}

func TestCmdPermissions_AlwaysEmitsPermissionsView(t *testing.T) {
	app, _ := newPermAppForTest(t)

	app.cmdPermissions(nil)
	ev := <-app.EventCh
	if ev.Type != model.PermissionsView {
		t.Fatalf("event type = %s, want %s", ev.Type, model.PermissionsView)
	}
	if ev.Permissions == nil {
		t.Fatal("permissions payload = nil, want payload")
	}
}

func TestHandleCommand_PermissionsIgnoresTrailingArgs(t *testing.T) {
	app, _ := newPermAppForTest(t)

	app.handleCommand("/permissions ignored args")
	ev := <-app.EventCh
	if ev.Type != model.PermissionsView {
		t.Fatalf("event type = %s, want %s", ev.Type, model.PermissionsView)
	}

	app.handleCommand("/permissions random text")
	ev = <-app.EventCh
	if ev.Type != model.PermissionsView {
		t.Fatalf("event type = %s, want %s", ev.Type, model.PermissionsView)
	}
}

func TestHandleCommand_PermissionShowsMigrationHint(t *testing.T) {
	app, _ := newPermAppForTest(t)

	app.handleCommand("/permission mode")
	ev := <-app.EventCh
	if ev.Type != model.AgentReply {
		t.Fatalf("event type = %s, want %s", ev.Type, model.AgentReply)
	}
	if !strings.Contains(ev.Message, "Use `/permissions`") {
		t.Fatalf("unexpected message: %q", ev.Message)
	}
}

func TestHandleCommand_InternalPermissionsTextCommandBlocked(t *testing.T) {
	app, _ := newPermAppForTest(t)

	app.handleCommand("/__permissions add tool shell allow_always")
	ev := <-app.EventCh
	if ev.Type != model.AgentReply || !strings.Contains(ev.Message, "Unknown command") {
		t.Fatalf("unexpected response: %#v", ev)
	}
}

func TestHandleCommand_InternalPermissionsAddAndRemoveToolRule(t *testing.T) {
	app, permSvc := newPermAppForTest(t)

	app.processInput(internalPermissionsActionPrefix + "add tool shell allow_always")
	ev := <-app.EventCh
	if ev.Type != model.AgentReply || !strings.Contains(ev.Message, "Added tool rule") {
		t.Fatalf("unexpected add response: %#v", ev)
	}
	if got := permSvc.Check("shell", ""); got != permission.PermissionAllowAlways {
		t.Fatalf("tool level = %s, want %s", got, permission.PermissionAllowAlways)
	}

	app.processInput(internalPermissionsActionPrefix + "remove tool shell")
	ev = <-app.EventCh
	if ev.Type != model.AgentReply || !strings.Contains(ev.Message, "Removed tool rule") {
		t.Fatalf("unexpected remove response: %#v", ev)
	}
	if got := permSvc.Check("shell", ""); got != permission.PermissionAsk {
		t.Fatalf("tool level after remove = %s, want %s", got, permission.PermissionAsk)
	}
}

func TestHandleCommand_InternalPermissionsAddAndRemoveDSLRule(t *testing.T) {
	app, permSvc := newPermAppForTest(t)

	app.processInput(internalPermissionsActionPrefix + "add allow Bash(npm test *)")
	ev := <-app.EventCh
	if ev.Type != model.AgentReply || !strings.Contains(ev.Message, "Added rule") {
		t.Fatalf("unexpected add response: %#v", ev)
	}
	if got := permSvc.Check("shell", "npm test ./..."); got != permission.PermissionAllowAlways {
		t.Fatalf("Check(shell npm test) = %s, want %s", got, permission.PermissionAllowAlways)
	}

	app.processInput(internalPermissionsActionPrefix + "remove Bash(npm test *)")
	ev = <-app.EventCh
	if ev.Type != model.AgentReply || !strings.Contains(ev.Message, "Removed rule") {
		t.Fatalf("unexpected remove response: %#v", ev)
	}
}

func TestHandleCommand_InternalPermissionsRemoveCommandRuleWithSpaces(t *testing.T) {
	app, permSvc := newPermAppForTest(t)

	app.processInput(internalPermissionsActionPrefix + "add allow Bash(npm test *)")
	ev := <-app.EventCh
	if ev.Type != model.AgentReply || !strings.Contains(ev.Message, "Added rule") {
		t.Fatalf("unexpected add response: %#v", ev)
	}

	app.processInput(internalPermissionsActionPrefix + "remove command npm test *")
	ev = <-app.EventCh
	if ev.Type != model.AgentReply || !strings.Contains(ev.Message, "Removed command rule") {
		t.Fatalf("unexpected remove response: %#v", ev)
	}
	if got := permSvc.Check("shell", "npm test ./..."); got == permission.PermissionAllowAlways {
		t.Fatalf("Check(shell npm test) = %s, want not %s", got, permission.PermissionAllowAlways)
	}
}

func TestHandleCommand_InternalPermissionsPathRuleWithSpaces(t *testing.T) {
	app, permSvc := newPermAppForTest(t)

	app.processInput(internalPermissionsActionPrefix + "add path /tmp/my dir/* deny")
	ev := <-app.EventCh
	if ev.Type != model.AgentReply || !strings.Contains(ev.Message, "Added path rule") {
		t.Fatalf("unexpected add response: %#v", ev)
	}
	if got := permSvc.CheckPath("/tmp/my dir/file.txt"); got != permission.PermissionDeny {
		t.Fatalf("CheckPath(/tmp/my dir/file.txt) = %s, want %s", got, permission.PermissionDeny)
	}

	app.processInput(internalPermissionsActionPrefix + "remove path /tmp/my dir/*")
	ev = <-app.EventCh
	if ev.Type != model.AgentReply || !strings.Contains(ev.Message, "Removed path rule") {
		t.Fatalf("unexpected remove response: %#v", ev)
	}
	if got := permSvc.CheckPath("/tmp/my dir/file.txt"); got != permission.PermissionAllowAlways {
		t.Fatalf("CheckPath(/tmp/my dir/file.txt) = %s, want %s", got, permission.PermissionAllowAlways)
	}
}

func TestHandleCommand_InternalPermissionsRemoveManagedRuleRejected(t *testing.T) {
	permSvc := permission.NewDefaultPermissionService(configs.PermissionsConfig{
		DefaultLevel: "ask",
		Deny:         []string{"Bash(git push origin *)"},
		RuleSources: map[string]string{
			"Bash(git push origin *)": "managed",
		},
	})
	app := &Application{
		EventCh:     make(chan model.Event, 8),
		permService: permSvc,
	}

	app.processInput(internalPermissionsActionPrefix + "remove Bash(git push origin *)")
	ev := <-app.EventCh
	if ev.Type != model.AgentReply || !strings.Contains(strings.ToLower(ev.Message), "managed rule is immutable") {
		t.Fatalf("unexpected response: %#v", ev)
	}
}

func TestHandleCommand_InternalPermissionsAddRuleWithScopePersistsToLocalSettings(t *testing.T) {
	app, permSvc := newPermAppForTest(t)
	workDir := t.TempDir()
	app.WorkDir = workDir

	app.processInput(internalPermissionsActionPrefix + "add allow_always Bash(ls -la) --scope project")
	ev := <-app.EventCh
	if ev.Type != model.AgentReply || !strings.Contains(ev.Message, "saved to") {
		t.Fatalf("unexpected add response: %#v", ev)
	}
	if got := permSvc.Check("shell", "ls -la"); got != permission.PermissionAllowAlways {
		t.Fatalf("Check(shell ls -la) = %s, want %s", got, permission.PermissionAllowAlways)
	}

	path := filepath.Join(workDir, ".mscli", "permissions.json")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile(%s) err = %v", path, err)
	}
	text := string(raw)
	if !strings.Contains(text, `"permissions"`) || !strings.Contains(text, `"Bash(ls -la)"`) {
		t.Fatalf("permissions.json content = %s", text)
	}
}

func TestCmdYoloTogglesLoadSkillPermission(t *testing.T) {
	app, permSvc := newPermAppForTest(t)

	app.cmdYolo()
	ev := <-app.EventCh
	if ev.Type != model.AgentReply || !strings.Contains(ev.Message, "enabled") {
		t.Fatalf("unexpected enable response: %#v", ev)
	}
	if got := permSvc.Check("load_skill", ""); got != permission.PermissionAllowAlways {
		t.Fatalf("Check(load_skill) after enable = %s, want %s", got, permission.PermissionAllowAlways)
	}

	app.cmdYolo()
	ev = <-app.EventCh
	if ev.Type != model.AgentReply || !strings.Contains(ev.Message, "disabled") {
		t.Fatalf("unexpected disable response: %#v", ev)
	}
	if got := permSvc.Check("load_skill", ""); got != permission.PermissionAsk {
		t.Fatalf("Check(load_skill) after disable = %s, want %s", got, permission.PermissionAsk)
	}
}
