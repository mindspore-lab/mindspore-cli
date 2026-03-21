package app

import (
	"strings"
	"testing"

	"github.com/vigo999/ms-cli/configs"
	"github.com/vigo999/ms-cli/permission"
	"github.com/vigo999/ms-cli/ui/model"
)

func TestCmdPermissions_ModeSet(t *testing.T) {
	permSvc := permission.NewDefaultPermissionService(configs.PermissionsConfig{
		DefaultLevel: "ask",
		DefaultMode:  "default",
	})
	app := &Application{
		EventCh:     make(chan model.Event, 8),
		permService: permSvc,
	}

	app.cmdPermissions([]string{"mode", "plan"})
	ev := <-app.EventCh
	if ev.Type != model.AgentReply {
		t.Fatalf("event type = %s, want %s", ev.Type, model.AgentReply)
	}
	if !strings.Contains(ev.Message, "Permission mode set to: plan") {
		t.Fatalf("unexpected message: %q", ev.Message)
	}
	if got := permSvc.Mode(); got != permission.ModePlan {
		t.Fatalf("mode = %s, want %s", got, permission.ModePlan)
	}
}

func TestCmdPermissions_NoArgsEmitsPermissionsView(t *testing.T) {
	permSvc := permission.NewDefaultPermissionService(configs.PermissionsConfig{
		DefaultLevel: "ask",
		DefaultMode:  "default",
	})
	permSvc.Grant("shell", permission.PermissionAllowAlways)
	permSvc.Grant("write", permission.PermissionAsk)
	permSvc.Grant("delete", permission.PermissionDeny)
	app := &Application{
		EventCh:     make(chan model.Event, 8),
		permService: permSvc,
		Config:      configs.DefaultConfig(),
		WorkDir:     "/tmp/work",
	}

	app.cmdPermissions(nil)
	ev := <-app.EventCh
	if ev.Type != model.PermissionsView {
		t.Fatalf("event type = %s, want %s", ev.Type, model.PermissionsView)
	}
	if ev.Permissions == nil {
		t.Fatal("permissions payload = nil, want payload")
	}
	if ev.Permissions.Mode != "default" {
		t.Fatalf("mode = %q, want %q", ev.Permissions.Mode, "default")
	}
	if len(ev.Permissions.Allow) == 0 || len(ev.Permissions.Ask) == 0 || len(ev.Permissions.Deny) == 0 {
		t.Fatalf("unexpected buckets allow=%v ask=%v deny=%v", ev.Permissions.Allow, ev.Permissions.Ask, ev.Permissions.Deny)
	}
	if len(ev.Permissions.RuleSources) == 0 {
		t.Fatal("rule sources should be populated")
	}
}

func TestHandleCommand_PermissionShowsMigrationHint(t *testing.T) {
	permSvc := permission.NewDefaultPermissionService(configs.PermissionsConfig{
		DefaultLevel: "ask",
		DefaultMode:  "default",
	})
	app := &Application{
		EventCh:     make(chan model.Event, 8),
		permService: permSvc,
	}

	app.handleCommand("/permission mode")
	ev := <-app.EventCh
	if ev.Type != model.AgentReply {
		t.Fatalf("event type = %s, want %s", ev.Type, model.AgentReply)
	}
	if !strings.Contains(ev.Message, "Use `/permissions`") {
		t.Fatalf("unexpected message: %q", ev.Message)
	}
}

func TestCmdPermissions_AddAndRemoveToolRule(t *testing.T) {
	permSvc := permission.NewDefaultPermissionService(configs.PermissionsConfig{
		DefaultLevel: "ask",
		DefaultMode:  "default",
	})
	app := &Application{
		EventCh:     make(chan model.Event, 8),
		permService: permSvc,
	}

	app.cmdPermissions([]string{"add", "tool", "shell", "allow_always"})
	ev := <-app.EventCh
	if ev.Type != model.AgentReply || !strings.Contains(ev.Message, "Added tool rule") {
		t.Fatalf("unexpected add response: %#v", ev)
	}
	if got := permSvc.Check("shell", ""); got != permission.PermissionAllowAlways {
		t.Fatalf("tool level = %s, want %s", got, permission.PermissionAllowAlways)
	}

	app.cmdPermissions([]string{"remove", "tool", "shell"})
	ev = <-app.EventCh
	if ev.Type != model.AgentReply || !strings.Contains(ev.Message, "Removed tool rule") {
		t.Fatalf("unexpected remove response: %#v", ev)
	}
	if got := permSvc.Check("shell", ""); got != permission.PermissionAsk {
		t.Fatalf("tool level after remove = %s, want %s", got, permission.PermissionAsk)
	}
}

func TestCmdPermissions_AddCommandAndPathRule(t *testing.T) {
	permSvc := permission.NewDefaultPermissionService(configs.PermissionsConfig{
		DefaultLevel: "ask",
		DefaultMode:  "default",
	})
	app := &Application{
		EventCh:     make(chan model.Event, 8),
		permService: permSvc,
	}

	app.cmdPermissions([]string{"add", "command", "git", "allow_session"})
	ev := <-app.EventCh
	if ev.Type != model.AgentReply || !strings.Contains(ev.Message, "Added command rule") {
		t.Fatalf("unexpected command add response: %#v", ev)
	}
	if got := permSvc.CheckCommand("git status"); got != permission.PermissionAllowSession {
		t.Fatalf("command level = %s, want %s", got, permission.PermissionAllowSession)
	}

	app.cmdPermissions([]string{"add", "path", "*.md", "deny"})
	ev = <-app.EventCh
	if ev.Type != model.AgentReply || !strings.Contains(ev.Message, "Added path rule") {
		t.Fatalf("unexpected path add response: %#v", ev)
	}
	if got := permSvc.CheckPath("readme.md"); got != permission.PermissionDeny {
		t.Fatalf("path level = %s, want %s", got, permission.PermissionDeny)
	}
}

func TestCmdPermissions_WorkspaceAddAndRemove(t *testing.T) {
	permSvc := permission.NewDefaultPermissionService(configs.PermissionsConfig{
		DefaultLevel: "ask",
		DefaultMode:  "default",
	})
	app := &Application{
		EventCh:     make(chan model.Event, 8),
		permService: permSvc,
	}

	app.cmdPermissions([]string{"workspace", "add", "/tmp/project"})
	ev := <-app.EventCh
	if ev.Type != model.AgentReply || !strings.Contains(ev.Message, "Workspace directory added") {
		t.Fatalf("unexpected workspace add response: %#v", ev)
	}
	if got := permSvc.CheckPath("/tmp/project/file.txt"); got != permission.PermissionAllowAlways {
		t.Fatalf("workspace path level = %s, want %s", got, permission.PermissionAllowAlways)
	}

	app.cmdPermissions([]string{"workspace", "remove", "/tmp/project"})
	ev = <-app.EventCh
	if ev.Type != model.AgentReply || !strings.Contains(ev.Message, "Workspace directory removed") {
		t.Fatalf("unexpected workspace remove response: %#v", ev)
	}
	if got := permSvc.CheckPath("/tmp/project/file.txt"); got != permission.PermissionAllowAlways {
		t.Fatalf("workspace path level after remove = %s, want default allow_always", got)
	}
}

func TestCmdPermissions_AddAndRemoveDSLRule(t *testing.T) {
	permSvc := permission.NewDefaultPermissionService(configs.PermissionsConfig{
		DefaultLevel: "ask",
		DefaultMode:  "default",
	})
	app := &Application{
		EventCh:     make(chan model.Event, 8),
		permService: permSvc,
	}

	app.cmdPermissions([]string{"add", "allow", "Bash(npm test *)"})
	ev := <-app.EventCh
	if ev.Type != model.AgentReply || !strings.Contains(ev.Message, "Added rule") {
		t.Fatalf("unexpected add response: %#v", ev)
	}
	if got := permSvc.Check("shell", "npm test ./..."); got != permission.PermissionAllowAlways {
		t.Fatalf("Check(shell npm test) = %s, want %s", got, permission.PermissionAllowAlways)
	}
	app.cmdPermissions(nil)
	ev = <-app.EventCh
	if ev.Type != model.PermissionsView || ev.Permissions == nil {
		t.Fatalf("permissions view event = %#v", ev)
	}
	if got := ev.Permissions.RuleSources["Bash(npm test *)"]; got != "project" {
		t.Fatalf("rule source = %q, want %q", got, "project")
	}

	app.cmdPermissions([]string{"remove", "Bash(npm test *)"})
	ev = <-app.EventCh
	if ev.Type != model.AgentReply || !strings.Contains(ev.Message, "Removed rule") {
		t.Fatalf("unexpected remove response: %#v", ev)
	}
	if got := permSvc.Check("shell", "npm test ./..."); got == permission.PermissionAllowAlways {
		t.Fatalf("Check(shell npm test) after remove = %s, want not %s", got, permission.PermissionAllowAlways)
	}
}

func TestCmdPermissions_RemoveManagedRuleRejected(t *testing.T) {
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

	app.cmdPermissions([]string{"remove", "Bash(git push origin *)"})
	ev := <-app.EventCh
	if ev.Type != model.AgentReply || !strings.Contains(strings.ToLower(ev.Message), "managed rule is immutable") {
		t.Fatalf("unexpected response: %#v", ev)
	}
}
