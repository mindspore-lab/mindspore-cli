package loop

import "testing"

func TestPermissionManager_ApproveOnceFlow(t *testing.T) {
	pm := NewPermissionManager(false, nil)

	decision, req, err := pm.Request("shell", "rm -rf /tmp/mscli-test", "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if decision != DecisionPending || req == nil {
		t.Fatal("expected dangerous shell to require approval")
	}

	if _, err := pm.ApproveOncePending(); err != nil {
		t.Fatalf("ApproveOncePending failed: %v", err)
	}

	decision, _, err = pm.Request("shell", "rm -rf /tmp/mscli-test", "")
	if err != nil || decision != DecisionAllow {
		t.Fatalf("expected once-approved request to pass, decision=%v err=%v", decision, err)
	}

	decision, req, err = pm.Request("shell", "rm -rf /tmp/mscli-test", "")
	if err != nil || decision != DecisionPending || req == nil {
		t.Fatal("expected approval requirement after once approval consumed")
	}
}

func TestPermissionManager_WhitelistBlacklistAndYolo(t *testing.T) {
	pm := NewPermissionManager(false, []string{"read", "grep"})

	decision, _, err := pm.Request("read", "a.txt", "a.txt")
	if err != nil || decision != DecisionAllow {
		t.Fatalf("expected read to be allowed, decision=%v err=%v", decision, err)
	}

	decision, _, err = pm.Request("edit", "a.txt", "a.txt")
	if err != nil || decision != DecisionDeny {
		t.Fatalf("expected edit denied by whitelist, decision=%v err=%v", decision, err)
	}

	pm.AddBlacklist("read")
	decision, _, err = pm.Request("read", "a.txt", "a.txt")
	if err != nil || decision != DecisionDeny {
		t.Fatalf("expected read denied by blacklist, decision=%v err=%v", decision, err)
	}

	pm.SetYolo(true)
	decision, _, err = pm.Request("shell", "rm -rf /tmp/x", "")
	if err != nil || decision != DecisionAllow {
		t.Fatalf("expected yolo to allow, decision=%v err=%v", decision, err)
	}
}

func TestPermissionManager_SafeShellNoApproval(t *testing.T) {
	pm := NewPermissionManager(false, nil)
	decision, _, err := pm.Request("shell", "ls -la", "")
	if err != nil || decision != DecisionAllow {
		t.Fatalf("safe shell command should be allowed without approval, decision=%v err=%v", decision, err)
	}
}

func TestPermissionManager_RejectThenDenyOnce(t *testing.T) {
	pm := NewPermissionManager(false, nil)

	decision, req, err := pm.Request("edit", "a.txt", "a.txt")
	if err != nil || decision != DecisionPending || req == nil {
		t.Fatalf("expected pending request, decision=%v err=%v req=%v", decision, err, req)
	}
	if _, err := pm.RejectPending(); err != nil {
		t.Fatalf("RejectPending failed: %v", err)
	}

	decision, deniedReq, err := pm.Request("edit", "a.txt", "a.txt")
	if err != nil || decision != DecisionDeny || deniedReq == nil {
		t.Fatalf("expected deny after reject, decision=%v err=%v req=%v", decision, err, deniedReq)
	}

	decision, req, err = pm.Request("edit", "a.txt", "a.txt")
	if err != nil || decision != DecisionPending || req == nil {
		t.Fatalf("expected pending again after deny consumed, decision=%v err=%v req=%v", decision, err, req)
	}
}

func TestPermissionManager_PendingQueueFIFO(t *testing.T) {
	pm := NewPermissionManager(false, nil)

	d1, r1, err := pm.Request("edit", "a.txt", "a.txt")
	if err != nil || d1 != DecisionPending || r1 == nil {
		t.Fatalf("expected first pending request")
	}
	d2, r2, err := pm.Request("write", "b.txt", "b.txt")
	if err != nil || d2 != DecisionPending || r2 == nil {
		t.Fatalf("expected second pending request")
	}

	approved, err := pm.ApproveOncePending()
	if err != nil {
		t.Fatalf("ApproveOncePending failed: %v", err)
	}
	if approved.ID != r1.ID {
		t.Fatalf("expected FIFO approval id=%d got=%d", r1.ID, approved.ID)
	}
}
