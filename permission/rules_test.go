package permission

import (
	"testing"

	"github.com/mindspore-lab/mindspore-cli/configs"
)

func TestParseRule_BasicAndSpecifier(t *testing.T) {
	t.Parallel()

	r, err := ParsePermissionRule("Bash(npm run test *)")
	if err != nil {
		t.Fatalf("ParsePermissionRule() err = %v", err)
	}
	if r.Tool != "bash" {
		t.Fatalf("tool = %q, want bash", r.Tool)
	}
	if r.Specifier != "npm run test *" {
		t.Fatalf("specifier = %q", r.Specifier)
	}

	r, err = ParsePermissionRule("Read")
	if err != nil {
		t.Fatalf("ParsePermissionRule() err = %v", err)
	}
	if r.Tool != "read" || r.Specifier != "" {
		t.Fatalf("parsed read mismatch: %+v", r)
	}
}

func TestParseRule_Invalid(t *testing.T) {
	t.Parallel()

	if _, err := ParsePermissionRule("Bash("); err == nil {
		t.Fatal("ParsePermissionRule(Bash() err = nil, want error")
	}
	if _, err := ParsePermissionRule("("); err == nil {
		t.Fatal("ParsePermissionRule(() err = nil, want error")
	}
}

func TestRuleEvaluation_PriorityDenyAskAllow(t *testing.T) {
	t.Parallel()

	svc := NewDefaultPermissionService(configs.PermissionsConfig{
		DefaultLevel: "ask",
		Deny:         []string{"Bash(git push *)"},
		Ask:          []string{"Bash(git *)"},
		Allow:        []string{"Bash(*)"},
	})

	if got := svc.Check("shell", "git push origin main"); got != PermissionDeny {
		t.Fatalf("Check(shell git push) = %s, want deny", got)
	}
	if got := svc.Check("shell", "git status"); got != PermissionAsk {
		t.Fatalf("Check(shell git status) = %s, want ask", got)
	}
	if got := svc.Check("shell", "echo hello"); got != PermissionAllowAlways {
		t.Fatalf("Check(shell echo) = %s, want allow_always", got)
	}
}

func TestRuleEvaluation_BashWildcardSpacing(t *testing.T) {
	t.Parallel()

	svc := NewDefaultPermissionService(configs.PermissionsConfig{
		DefaultLevel: "ask",
		Allow:        []string{"Bash(ls *)"},
	})

	if got := svc.Check("shell", "ls -la"); got != PermissionAllowAlways {
		t.Fatalf("Check(shell ls -la) = %s, want allow_always", got)
	}
	if got := svc.Check("shell", "lsof -i"); got == PermissionAllowAlways {
		t.Fatalf("Check(shell lsof) = %s, want not allow_always", got)
	}
}

func TestRuleEvaluation_ReadPathSyntaxKinds(t *testing.T) {
	t.Parallel()

	svc := NewDefaultPermissionService(configs.PermissionsConfig{
		DefaultLevel: "ask",
		Deny: []string{
			"Read(./.env)",
			"Read(/secrets/*)",
		},
	})

	if got := svc.Check("read", ""); got != PermissionAllowAlways {
		t.Fatalf("Check(read) = %s, want allow_always", got)
	}
	if got := svc.CheckPath(".env"); got != PermissionDeny {
		t.Fatalf("CheckPath(.env) = %s, want deny", got)
	}
	if got := svc.CheckPath("secrets/token.txt"); got != PermissionDeny {
		t.Fatalf("CheckPath(secrets/token.txt) = %s, want deny", got)
	}
}

func TestRuleEvaluation_ReadPathAbsoluteAndHome(t *testing.T) {
	t.Parallel()

	svc := NewDefaultPermissionService(configs.PermissionsConfig{
		DefaultLevel: "ask",
		Deny: []string{
			"Read(//tmp/mscli-secret.txt)",
			"Read(~/private/*)",
		},
	})

	if got := svc.CheckPath("/tmp/mscli-secret.txt"); got != PermissionDeny {
		t.Fatalf("CheckPath(/tmp/mscli-secret.txt) = %s, want deny", got)
	}
}

func TestRuleEvaluation_ReadPathGitignoreStyle(t *testing.T) {
	t.Parallel()

	svc := NewDefaultPermissionService(configs.PermissionsConfig{
		DefaultLevel: "ask",
		Deny: []string{
			"Read(secrets/)",
			"Read(/root-only/*.md)",
			"Read(*.env)",
			"Read(**/private/*.txt)",
		},
	})

	if got := svc.CheckPath("secrets/token.txt"); got != PermissionDeny {
		t.Fatalf("CheckPath(secrets/token.txt) = %s, want deny", got)
	}
	if got := svc.CheckPath("nested/secrets/token.txt"); got != PermissionDeny {
		t.Fatalf("CheckPath(nested/secrets/token.txt) = %s, want deny", got)
	}
	if got := svc.CheckPath("root-only/a.md"); got != PermissionDeny {
		t.Fatalf("CheckPath(root-only/a.md) = %s, want deny", got)
	}
	if got := svc.CheckPath("nested/root-only/a.md"); got == PermissionDeny {
		t.Fatalf("CheckPath(nested/root-only/a.md) = %s, want not deny", got)
	}
	if got := svc.CheckPath(".env"); got != PermissionDeny {
		t.Fatalf("CheckPath(.env) = %s, want deny", got)
	}
	if got := svc.CheckPath("app/.env"); got != PermissionDeny {
		t.Fatalf("CheckPath(app/.env) = %s, want deny", got)
	}
	if got := svc.CheckPath("a/private/secret.txt"); got != PermissionDeny {
		t.Fatalf("CheckPath(a/private/secret.txt) = %s, want deny", got)
	}
}

func TestRuleEvaluation_MCPAndAgent(t *testing.T) {
	t.Parallel()

	svc := NewDefaultPermissionService(configs.PermissionsConfig{
		DefaultLevel: "ask",
		Allow:        []string{"mcp__puppeteer__*"},
		Deny:         []string{"Agent(Plan)"},
	})

	if got := svc.Check("mcp__puppeteer__puppeteer_navigate", ""); got != PermissionAllowAlways {
		t.Fatalf("Check(mcp tool) = %s, want allow_always", got)
	}
	if got := svc.Check("agent", "Plan"); got != PermissionDeny {
		t.Fatalf("Check(agent Plan) = %s, want deny", got)
	}
}

func TestRuleEvaluation_WebFetchDomainHostMatching(t *testing.T) {
	t.Parallel()

	svc := NewDefaultPermissionService(configs.PermissionsConfig{
		DefaultLevel: "ask",
		Allow:        []string{"WebFetch(domain:example.com)"},
	})

	if got := svc.Check("webfetch", "https://example.com/docs"); got != PermissionAllowAlways {
		t.Fatalf("Check(webfetch example.com) = %s, want allow_always", got)
	}
	if got := svc.Check("webfetch", "https://api.example.com/v1"); got != PermissionAllowAlways {
		t.Fatalf("Check(webfetch api.example.com) = %s, want allow_always", got)
	}
	if got := svc.Check("webfetch", "https://evil-example.com/phish"); got != PermissionAsk {
		t.Fatalf("Check(webfetch evil-example.com) = %s, want ask", got)
	}
}
