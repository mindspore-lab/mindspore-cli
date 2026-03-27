package loop

import (
	"fmt"
	"sort"
	"strings"

	"github.com/vigo999/ms-cli/integrations/llm"
)

type toolPairCleanupReport struct {
	removedToolCalls   int
	removedToolResults int
	ids                []string
}

func (r toolPairCleanupReport) changed() bool {
	return r.removedToolCalls > 0 || r.removedToolResults > 0
}

func (r toolPairCleanupReport) warningMessage() string {
	if !r.changed() {
		return ""
	}

	parts := make([]string, 0, 2)
	if r.removedToolCalls > 0 {
		parts = append(parts, describeRemovedCount(r.removedToolCalls, "unpaired tool call"))
	}
	if r.removedToolResults > 0 {
		parts = append(parts, describeRemovedCount(r.removedToolResults, "unpaired tool result"))
	}

	msg := fmt.Sprintf("warning: removed %s before llm request", joinWithAnd(parts))
	if len(r.ids) > 0 {
		msg += fmt.Sprintf(" (tool ids: %s)", strings.Join(r.ids, ", "))
	}
	return msg
}

func describeRemovedCount(count int, singular string) string {
	if count == 1 {
		return fmt.Sprintf("1 %s", singular)
	}
	return fmt.Sprintf("%d %ss", count, singular)
}

func joinWithAnd(parts []string) string {
	switch len(parts) {
	case 0:
		return ""
	case 1:
		return parts[0]
	case 2:
		return parts[0] + " and " + parts[1]
	default:
		return strings.Join(parts[:len(parts)-1], ", ") + ", and " + parts[len(parts)-1]
	}
}

func validToolCallIDs(messages []llm.Message) map[string]struct{} {
	callCount := make(map[string]int)
	callPos := make(map[string]int)
	resultCount := make(map[string]int)
	resultPos := make(map[string]int)

	for i, msg := range messages {
		switch msg.Role {
		case "assistant":
			for _, tc := range msg.ToolCalls {
				id := strings.TrimSpace(tc.ID)
				if id == "" {
					continue
				}
				if callCount[id] == 0 {
					callPos[id] = i
				}
				callCount[id]++
			}
		case "tool":
			id := strings.TrimSpace(msg.ToolCallID)
			if id == "" {
				continue
			}
			if resultCount[id] == 0 {
				resultPos[id] = i
			}
			resultCount[id]++
		}
	}

	valid := make(map[string]struct{})
	for id := range callCount {
		if callCount[id] == 1 && resultCount[id] == 1 && callPos[id] < resultPos[id] {
			valid[id] = struct{}{}
		}
	}
	return valid
}

func sanitizeMessagesForValidToolCallIDs(messages []llm.Message, valid map[string]struct{}) ([]llm.Message, toolPairCleanupReport) {
	if len(messages) == 0 {
		return nil, toolPairCleanupReport{}
	}

	sanitized := make([]llm.Message, 0, len(messages))
	removedIDs := make(map[string]struct{})
	report := toolPairCleanupReport{}

	for _, msg := range messages {
		switch msg.Role {
		case "assistant":
			if len(msg.ToolCalls) == 0 {
				sanitized = append(sanitized, msg)
				continue
			}

			filtered := make([]llm.ToolCall, 0, len(msg.ToolCalls))
			for _, tc := range msg.ToolCalls {
				id := strings.TrimSpace(tc.ID)
				if id == "" {
					report.removedToolCalls++
					continue
				}
				if _, ok := valid[id]; !ok {
					report.removedToolCalls++
					removedIDs[id] = struct{}{}
					continue
				}
				filtered = append(filtered, tc)
			}

			if len(filtered) == 0 {
				if strings.TrimSpace(msg.Content) == "" {
					continue
				}
				msg.ToolCalls = nil
				sanitized = append(sanitized, msg)
				continue
			}

			msg.ToolCalls = filtered
			sanitized = append(sanitized, msg)

		case "tool":
			id := strings.TrimSpace(msg.ToolCallID)
			if id == "" {
				report.removedToolResults++
				continue
			}
			if _, ok := valid[id]; !ok {
				report.removedToolResults++
				removedIDs[id] = struct{}{}
				continue
			}
			sanitized = append(sanitized, msg)

		default:
			sanitized = append(sanitized, msg)
		}
	}

	if len(removedIDs) > 0 {
		report.ids = make([]string, 0, len(removedIDs))
		for id := range removedIDs {
			report.ids = append(report.ids, id)
		}
		sort.Strings(report.ids)
	}

	return sanitized, report
}
