package permission

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/vigo999/mindspore-code/configs"
)

type stubPermissionUI struct {
	granted  bool
	remember bool
	err      error
}

func (s stubPermissionUI) RequestPermission(tool, action, path string) (bool, bool, error) {
	return s.granted, s.remember, s.err
}

type captureStore struct {
	decisions []PermissionDecision
}

func (s *captureStore) SaveDecision(decision PermissionDecision) error {
	s.decisions = append(s.decisions, decision)
	return nil
}

func (s *captureStore) LoadDecisions() ([]PermissionDecision, error) {
	return nil, nil
}

func (s *captureStore) ClearDecisions() error {
	s.decisions = nil
	return nil
}

type preloadStore struct {
	decisions []PermissionDecision
}

func (s *preloadStore) SaveDecision(decision PermissionDecision) error { return nil }
func (s *preloadStore) LoadDecisions() ([]PermissionDecision, error)   { return s.decisions, nil }
func (s *preloadStore) ClearDecisions() error                          { return nil }

func TestRequestRemember_EditScopesToSessionAndNotPersisted(t *testing.T) {
	svc := NewDefaultPermissionService(configs.PermissionsConfig{
		DefaultLevel: "ask",
	})
	store := &captureStore{}
	svc.SetStore(store)
	svc.SetUI(stubPermissionUI{granted: true, remember: true})

	granted, err := svc.Request(context.Background(), "write", "", "blank.md")
	if err != nil {
		t.Fatalf("Request() err = %v", err)
	}
	if !granted {
		t.Fatal("Request() granted = false, want true")
	}

	if got := svc.Check("write", ""); got != PermissionAllowSession {
		t.Fatalf("Check(write) = %v, want %v", got, PermissionAllowSession)
	}
	if got := svc.Check("edit", ""); got != PermissionAllowSession {
		t.Fatalf("Check(edit) = %v, want %v", got, PermissionAllowSession)
	}
	if got := len(store.decisions); got != 0 {
		t.Fatalf("saved decisions = %d, want 0 for edit session allow", got)
	}
}

func TestRequestRemember_ShellPersistsCommandDecision(t *testing.T) {
	svc := NewDefaultPermissionService(configs.PermissionsConfig{
		DefaultLevel: "ask",
	})
	store := &captureStore{}
	svc.SetStore(store)
	svc.SetUI(stubPermissionUI{granted: true, remember: true})

	granted, err := svc.Request(context.Background(), "shell", "npm test ./...", "")
	if err != nil {
		t.Fatalf("Request() err = %v", err)
	}
	if !granted {
		t.Fatal("Request() granted = false, want true")
	}

	if got := svc.CheckCommand("npm run build"); got != PermissionAllowSession {
		t.Fatalf("CheckCommand(npm run build) = %v, want %v", got, PermissionAllowSession)
	}
	if got := len(store.decisions); got != 1 {
		t.Fatalf("saved decisions = %d, want 1", got)
	}
	if got := store.decisions[0].Tool; got != "shell" {
		t.Fatalf("saved decision tool = %q, want shell", got)
	}
}

func TestRuleConfig_ClaudeStyleToolSpecificMatching(t *testing.T) {
	svc := NewDefaultPermissionService(configs.PermissionsConfig{
		DefaultLevel: "ask",
		Allow: []string{
			"Bash(npm test *)",
			"WebFetch(domain:example.com)",
			"mcp__puppeteer__*",
		},
		Deny: []string{
			"Agent(Plan)",
			"Bash(rm -rf /)",
		},
		Ask: []string{
			"Bash(git push *)",
		},
	})

	if got := svc.Check("shell", "npm test ./..."); got != PermissionAllowAlways {
		t.Fatalf("Check(shell npm test) = %v, want %v", got, PermissionAllowAlways)
	}
	if got := svc.Check("shell", "git push origin main"); got != PermissionAsk {
		t.Fatalf("Check(shell git push) = %v, want %v", got, PermissionAsk)
	}
	if got := svc.Check("shell", "rm -rf /"); got != PermissionDeny {
		t.Fatalf("Check(shell rm -rf /) = %v, want %v", got, PermissionDeny)
	}
	if got := svc.Check("webfetch", "https://example.com/docs"); got != PermissionAllowAlways {
		t.Fatalf("Check(webfetch) = %v, want %v", got, PermissionAllowAlways)
	}
	if got := svc.Check("agent", "Plan"); got != PermissionDeny {
		t.Fatalf("Check(agent Plan) = %v, want %v", got, PermissionDeny)
	}
	if got := svc.Check("mcp__puppeteer__puppeteer_navigate", ""); got != PermissionAllowAlways {
		t.Fatalf("Check(mcp puppeteer tool) = %v, want %v", got, PermissionAllowAlways)
	}
}

func TestRuleConfig_BashOperatorAwareMatching(t *testing.T) {
	svc := NewDefaultPermissionService(configs.PermissionsConfig{
		DefaultLevel: "ask",
		Allow:        []string{"Bash(npm test *)"},
	})

	if got := svc.CheckCommand("npm test ./..."); got != PermissionAllowAlways {
		t.Fatalf("CheckCommand(npm test ./...) = %v, want %v", got, PermissionAllowAlways)
	}
	if got := svc.CheckCommand("npm test ./... && rm -rf /tmp/x"); got != PermissionAsk {
		t.Fatalf("CheckCommand(compound) = %v, want %v", got, PermissionAsk)
	}
}

func TestRuleConfig_BashFindPatternMatchesQuoteAndGroupingVariants(t *testing.T) {
	svc := NewDefaultPermissionService(configs.PermissionsConfig{
		DefaultLevel: "ask",
		Allow: []string{
			`Bash(find . -type f -name "*.py" -o -name "*.yaml" -o -name "*.yml" -o -name "*.json" -o -name "*.md" -o -name "*.txt" -o -name "*.sh")`,
		},
	})

	variants := []string{
		`find . -type f -name "*.py" -o -name "*.yaml" -o -name "*.yml" -o -name "*.json" -o -name "*.md" -o -name "*.txt" -o -name "*.sh"`,
		`find . -type f -name '*.py' -o -name '*.yaml' -o -name '*.yml' -o -name '*.json' -o -name '*.md' -o -name '*.txt' -o -name '*.sh'`,
		`find . -type f \( -name "*.py" -o -name "*.yaml" -o -name "*.yml" -o -name "*.json" -o -name "*.md" -o -name "*.txt" -o -name "*.sh" \)`,
	}
	for _, cmd := range variants {
		if got := svc.Check("shell", cmd); got != PermissionAllowAlways {
			t.Fatalf("Check(shell %q) = %v, want %v", cmd, got, PermissionAllowAlways)
		}
		if got := svc.CheckCommand(cmd); got != PermissionAllowAlways {
			t.Fatalf("CheckCommand(%q) = %v, want %v", cmd, got, PermissionAllowAlways)
		}
	}

	nonMatch := `find . -type f -name "*.go" -o -name "*.sum"`
	if got := svc.Check("shell", nonMatch); got != PermissionAsk {
		t.Fatalf("Check(shell %q) = %v, want %v", nonMatch, got, PermissionAsk)
	}
}

func TestRequestRemember_ShellCompositeCommandSplitsAndCapsAtFive(t *testing.T) {
	svc := NewDefaultPermissionService(configs.PermissionsConfig{
		DefaultLevel: "ask",
	})
	store := &captureStore{}
	svc.SetStore(store)
	svc.SetUI(stubPermissionUI{granted: true, remember: true})

	action := "npm test ./... && go test ./... && make build && docker run x && kubectl apply -f y && terraform plan"
	granted, err := svc.Request(context.Background(), "shell", action, "")
	if err != nil {
		t.Fatalf("Request() err = %v", err)
	}
	if !granted {
		t.Fatal("Request() granted = false, want true")
	}

	for _, cmd := range []string{"npm test ./...", "go test ./...", "make build", "docker run x", "kubectl apply -f y"} {
		if got := svc.CheckCommand(cmd); got != PermissionAllowSession {
			t.Fatalf("CheckCommand(%q) = %v, want %v", cmd, got, PermissionAllowSession)
		}
	}
	// 6th command should not be persisted (cap at 5)
	if got := svc.CheckCommand("terraform plan"); got != PermissionAsk {
		t.Fatalf("CheckCommand(terraform plan) = %v, want %v", got, PermissionAsk)
	}
	if got := len(store.decisions); got != 5 {
		t.Fatalf("saved decisions = %d, want 5", got)
	}
}

func TestSetStore_LoadedDecisionsUseStateSource(t *testing.T) {
	svc := NewDefaultPermissionService(configs.PermissionsConfig{
		DefaultLevel: "ask",
	})
	svc.SetStore(&preloadStore{
		decisions: []PermissionDecision{
			{Tool: "shell", Action: "npm test ./...", Level: PermissionAllowSession, Timestamp: time.Now()},
			{Tool: "edit", Path: "*.md", Level: PermissionDeny, Timestamp: time.Now()},
		},
	})

	views := svc.GetRuleViews()
	if len(views) == 0 {
		t.Fatal("GetRuleViews() empty, want loaded state rules")
	}
	for _, v := range views {
		if v.Rule == "Bash(npm *)" || v.Rule == "Edit(*.md)" {
			if v.Source != "state" {
				t.Fatalf("rule %q source = %q, want state", v.Rule, v.Source)
			}
		}
	}
}

func TestConfigRuleSourcesPropagateToRuleViews(t *testing.T) {
	svc := NewDefaultPermissionService(configs.PermissionsConfig{
		DefaultLevel: "ask",
		Allow:        []string{"Read"},
		RuleSources: map[string]string{
			"Read": "project",
		},
	})

	views := svc.GetRuleViews()
	if len(views) == 0 {
		t.Fatal("GetRuleViews() empty")
	}
	found := false
	for _, v := range views {
		if v.Rule == "Read" {
			found = true
			if v.Source != "project" {
				t.Fatalf("Read source = %q, want project", v.Source)
			}
		}
	}
	if !found {
		t.Fatal("Read rule not found in views")
	}
}

func TestManagedRuleCannotBeOverriddenByProjectAddRule(t *testing.T) {
	svc := NewDefaultPermissionService(configs.PermissionsConfig{
		DefaultLevel: "ask",
		Deny:         []string{"Bash(git push *)"},
		RuleSources: map[string]string{
			"Bash(git push *)": "managed",
		},
	})

	if err := svc.AddRule("Bash(git push *)", PermissionAllowAlways); !errors.Is(err, ErrManagedRuleLocked) {
		t.Fatalf("AddRule() err = %v, want ErrManagedRuleLocked", err)
	}
	if got := svc.Check("shell", "git push origin main"); got != PermissionDeny {
		t.Fatalf("Check(shell git push) = %v, want %v", got, PermissionDeny)
	}

	views := svc.GetRuleViews()
	for _, v := range views {
		if v.Rule == "Bash(git push *)" && v.Source != "managed" {
			t.Fatalf("managed rule source changed to %q", v.Source)
		}
	}
}

func TestAllowBucketFirstMatchLockedByOrder(t *testing.T) {
	svc := NewDefaultPermissionService(configs.PermissionsConfig{
		DefaultLevel: "ask",
	})
	if err := svc.AddRule("Bash(git *)", PermissionAllowOnce); err != nil {
		t.Fatalf("AddRule(git *) err = %v", err)
	}
	if err := svc.AddRule("Bash(git push *)", PermissionAllowAlways); err != nil {
		t.Fatalf("AddRule(git push *) err = %v", err)
	}

	if got := svc.Check("shell", "git push origin main"); got != PermissionAllowOnce {
		t.Fatalf("Check(shell git push) = %v, want %v", got, PermissionAllowOnce)
	}
}

func TestUpdateSameSourceRuleKeepsOriginalOrder(t *testing.T) {
	svc := NewDefaultPermissionService(configs.PermissionsConfig{
		DefaultLevel: "ask",
	})
	if err := svc.AddRule("Bash(git *)", PermissionAllowOnce); err != nil {
		t.Fatalf("AddRule(git *) err = %v", err)
	}
	if err := svc.AddRule("Bash(git push *)", PermissionAllowAlways); err != nil {
		t.Fatalf("AddRule(git push *) err = %v", err)
	}
	if err := svc.AddRule("Bash(git *)", PermissionAllowSession); err != nil {
		t.Fatalf("AddRule(update git *) err = %v", err)
	}

	if got := svc.Check("shell", "git push origin main"); got != PermissionAllowSession {
		t.Fatalf("Check(shell git push) = %v, want %v", got, PermissionAllowSession)
	}
}

func TestSourcePrecedenceAcrossBuckets_ManagedAllowBeatsProjectDeny(t *testing.T) {
	svc := NewDefaultPermissionService(configs.PermissionsConfig{
		DefaultLevel: "ask",
		Allow:        []string{"Bash(git push origin *)"},
		RuleSources: map[string]string{
			"Bash(git push origin *)": "managed",
		},
	})
	_ = svc.AddRule("Bash(git push *)", PermissionDeny)

	if got := svc.Check("shell", "git push origin main"); got != PermissionAllowAlways {
		t.Fatalf("Check(shell git push) = %v, want %v", got, PermissionAllowAlways)
	}
}

func TestSourcePrecedenceAcrossBuckets_HigherSourceAskBeatsLowerDeny(t *testing.T) {
	svc := NewDefaultPermissionService(configs.PermissionsConfig{
		DefaultLevel: "ask",
		Ask:          []string{"Bash(git push origin *)"},
		RuleSources: map[string]string{
			"Bash(git push origin *)": "managed",
		},
	})
	_ = svc.AddRule("Bash(git push *)", PermissionDeny)

	if got := svc.Check("shell", "git push origin main"); got != PermissionAsk {
		t.Fatalf("Check(shell git push) = %v, want %v", got, PermissionAsk)
	}
}

func TestManagedRuleHardLock_AddAndRemoveRejected(t *testing.T) {
	svc := NewDefaultPermissionService(configs.PermissionsConfig{
		DefaultLevel: "ask",
		Deny:         []string{"Bash(git push origin *)"},
		RuleSources: map[string]string{
			"Bash(git push origin *)": "managed",
		},
	})

	if err := svc.AddRule("Bash(git push origin *)", PermissionAllowAlways); !errors.Is(err, ErrManagedRuleLocked) {
		t.Fatalf("AddRule err = %v, want ErrManagedRuleLocked", err)
	}
	ok, err := svc.RemoveRule("Bash(git push origin *)")
	if !errors.Is(err, ErrManagedRuleLocked) {
		t.Fatalf("RemoveRule err = %v, want ErrManagedRuleLocked", err)
	}
	if ok {
		t.Fatal("RemoveRule ok = true, want false")
	}
}

func TestManagedRuleHardLock_GrantAndRevokeCannotBypass(t *testing.T) {
	svc := NewDefaultPermissionService(configs.PermissionsConfig{
		DefaultLevel: "ask",
		Deny:         []string{"Bash"},
		RuleSources: map[string]string{
			"Bash": "managed",
		},
	})

	svc.Grant("shell", PermissionAllowAlways)
	if got := svc.Check("shell", "echo hello"); got != PermissionDeny {
		t.Fatalf("Check(shell echo) after Grant = %v, want %v", got, PermissionDeny)
	}

	svc.Revoke("shell")
	if got := svc.Check("shell", "echo hello"); got != PermissionDeny {
		t.Fatalf("Check(shell echo) after Revoke = %v, want %v", got, PermissionDeny)
	}
}

func TestManagedRuleHardLock_GrantCommandAndRevokeCommandCannotBypass(t *testing.T) {
	svc := NewDefaultPermissionService(configs.PermissionsConfig{
		DefaultLevel: "ask",
		Deny:         []string{"Bash(git *)"},
		RuleSources: map[string]string{
			"Bash(git *)": "managed",
		},
	})

	svc.GrantCommand("git push", PermissionAllowAlways)
	if got := svc.Check("shell", "git push origin main"); got != PermissionDeny {
		t.Fatalf("Check(shell git push) after GrantCommand = %v, want %v", got, PermissionDeny)
	}

	svc.RevokeCommand("git push")
	if got := svc.Check("shell", "git push origin main"); got != PermissionDeny {
		t.Fatalf("Check(shell git push) after RevokeCommand = %v, want %v", got, PermissionDeny)
	}
}

func TestManagedRuleHardLock_GrantPathAndRevokePathCannotBypass(t *testing.T) {
	svc := NewDefaultPermissionService(configs.PermissionsConfig{
		DefaultLevel: "ask",
		Deny:         []string{"Edit(secrets/*)"},
		RuleSources: map[string]string{
			"Edit(secrets/*)": "managed",
		},
	})

	svc.GrantPath("secrets/*", PermissionAllowAlways)
	if got := svc.CheckPath("secrets/a.txt"); got != PermissionDeny {
		t.Fatalf("CheckPath(secrets/a.txt) after GrantPath = %v, want %v", got, PermissionDeny)
	}

	svc.RevokePath("secrets/*")
	if got := svc.CheckPath("secrets/a.txt"); got != PermissionDeny {
		t.Fatalf("CheckPath(secrets/a.txt) after RevokePath = %v, want %v", got, PermissionDeny)
	}
}
