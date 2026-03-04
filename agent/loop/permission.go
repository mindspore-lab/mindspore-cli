package loop

import (
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"
)

type Decision string

const (
	DecisionAllow   Decision = "allow"
	DecisionDeny    Decision = "deny"
	DecisionPending Decision = "pending"
)

// PermissionService controls tool-call permissions.
type PermissionService interface {
	Request(tool, action, path string) (Decision, *ApprovalRequest, error)
}

type ApprovalRequest struct {
	ID        int64
	Tool      string
	Action    string
	Path      string
	Key       string
	CreatedAt time.Time
}

type ApprovalRequiredError struct {
	Request ApprovalRequest
}

func (e *ApprovalRequiredError) Error() string {
	return fmt.Sprintf(
		"approval required for %s: %s (id=%d). use /approve once | /approve session | /reject",
		e.Request.Tool,
		strings.TrimSpace(e.Request.Action),
		e.Request.ID,
	)
}

type PermissionStatus struct {
	Yolo            bool
	Whitelist       []string
	Blacklist       []string
	SessionApproved int
	Pending         *ApprovalRequest
	PendingCount    int
}

type PermissionManager struct {
	mu               sync.Mutex
	yoloMode         bool
	whitelist        map[string]struct{}
	blacklist        map[string]struct{}
	sessionApprovals map[string]struct{}
	onceApprovalKey  string
	deniedByKey      map[string]ApprovalRequest
	pendingQueue     []ApprovalRequest
	pendingByKey     map[string]ApprovalRequest
	nextID           int64
}

func NewPermissionManager(skipRequests bool, allowedTools []string) *PermissionManager {
	m := &PermissionManager{
		yoloMode:         skipRequests,
		whitelist:        make(map[string]struct{}, len(allowedTools)),
		blacklist:        map[string]struct{}{},
		sessionApprovals: map[string]struct{}{},
		deniedByKey:      map[string]ApprovalRequest{},
		pendingQueue:     make([]ApprovalRequest, 0, 8),
		pendingByKey:     map[string]ApprovalRequest{},
		nextID:           0,
	}
	for _, t := range allowedTools {
		name := normalizeTool(t)
		if name == "" {
			continue
		}
		m.whitelist[name] = struct{}{}
	}
	return m
}

// Backward-compatible constructor.
func NewStaticPermissionService(skipRequests bool, allowedTools []string) *PermissionManager {
	return NewPermissionManager(skipRequests, allowedTools)
}

func (m *PermissionManager) Request(tool, action, path string) (Decision, *ApprovalRequest, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	t := normalizeTool(tool)
	a := strings.TrimSpace(action)
	p := strings.TrimSpace(path)

	if m.yoloMode {
		return DecisionAllow, nil, nil
	}
	if _, blocked := m.blacklist[t]; blocked {
		return DecisionDeny, nil, nil
	}
	if len(m.whitelist) > 0 {
		if _, ok := m.whitelist[t]; !ok {
			return DecisionDeny, nil, nil
		}
	}

	if !requiresApproval(t, a, p) {
		return DecisionAllow, nil, nil
	}

	key := requestKey(t, a, p)
	if _, ok := m.sessionApprovals[key]; ok {
		return DecisionAllow, nil, nil
	}
	if m.onceApprovalKey == key {
		m.onceApprovalKey = ""
		return DecisionAllow, nil, nil
	}

	if deniedReq, denied := m.deniedByKey[key]; denied {
		delete(m.deniedByKey, key)
		req := deniedReq
		return DecisionDeny, &req, nil
	}

	if req, ok := m.pendingByKey[key]; ok {
		c := req
		return DecisionPending, &c, nil
	}

	m.nextID++
	req := ApprovalRequest{
		ID:        m.nextID,
		Tool:      t,
		Action:    a,
		Path:      p,
		Key:       key,
		CreatedAt: time.Now().UTC(),
	}
	m.pendingQueue = append(m.pendingQueue, req)
	m.pendingByKey[key] = req
	c := req
	return DecisionPending, &c, nil
}

func (m *PermissionManager) ApproveOncePending() (ApprovalRequest, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	req, ok := m.popNextPending()
	if !ok {
		return ApprovalRequest{}, fmt.Errorf("no pending approval request")
	}
	m.onceApprovalKey = req.Key
	return req, nil
}

func (m *PermissionManager) ApproveSessionPending() (ApprovalRequest, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	req, ok := m.popNextPending()
	if !ok {
		return ApprovalRequest{}, fmt.Errorf("no pending approval request")
	}
	m.sessionApprovals[req.Key] = struct{}{}
	return req, nil
}

func (m *PermissionManager) RejectPending() (ApprovalRequest, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	req, ok := m.popNextPending()
	if !ok {
		return ApprovalRequest{}, fmt.Errorf("no pending approval request")
	}
	m.deniedByKey[req.Key] = req
	return req, nil
}

func (m *PermissionManager) SetYolo(enabled bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.yoloMode = enabled
}

func (m *PermissionManager) AddWhitelist(tool string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	t := normalizeTool(tool)
	if t != "" {
		m.whitelist[t] = struct{}{}
	}
}

func (m *PermissionManager) RemoveWhitelist(tool string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	delete(m.whitelist, normalizeTool(tool))
}

func (m *PermissionManager) AddBlacklist(tool string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	t := normalizeTool(tool)
	if t != "" {
		m.blacklist[t] = struct{}{}
	}
}

func (m *PermissionManager) RemoveBlacklist(tool string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	delete(m.blacklist, normalizeTool(tool))
}

func (m *PermissionManager) Status() PermissionStatus {
	m.mu.Lock()
	defer m.mu.Unlock()
	s := PermissionStatus{
		Yolo:            m.yoloMode,
		Whitelist:       sortedKeys(m.whitelist),
		Blacklist:       sortedKeys(m.blacklist),
		SessionApproved: len(m.sessionApprovals),
		PendingCount:    len(m.pendingQueue),
	}
	if len(m.pendingQueue) > 0 {
		c := m.pendingQueue[0]
		s.Pending = &c
	}
	return s
}

func (m *PermissionManager) popNextPending() (ApprovalRequest, bool) {
	if len(m.pendingQueue) == 0 {
		return ApprovalRequest{}, false
	}
	req := m.pendingQueue[0]
	m.pendingQueue = append([]ApprovalRequest{}, m.pendingQueue[1:]...)
	delete(m.pendingByKey, req.Key)
	return req, true
}

func normalizeTool(tool string) string {
	return strings.ToLower(strings.TrimSpace(tool))
}

func requestKey(tool, action, path string) string {
	return tool + "|" + action + "|" + path
}

func requiresApproval(tool, action, path string) bool {
	switch tool {
	case "shell":
		return shellNeedsApproval(action)
	case "edit", "write":
		return true
	case "read", "grep", "glob":
		return false
	default:
		return true
	}
}

func shellNeedsApproval(command string) bool {
	cmd := strings.ToLower(strings.TrimSpace(command))
	if cmd == "" {
		return true
	}

	// Dangerous patterns require explicit approval.
	dangerous := []string{
		"rm -rf",
		"sudo ",
		"shutdown",
		"reboot",
		"mkfs",
		"dd if=",
		"chmod ",
		"chown ",
		"git reset --hard",
		"git clean -fd",
		"sed -i",
		">",
		">>",
	}
	for _, p := range dangerous {
		if strings.Contains(cmd, p) {
			return true
		}
	}
	return false
}

func sortedKeys(m map[string]struct{}) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}
