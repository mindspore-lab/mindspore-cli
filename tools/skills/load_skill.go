package skills

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/mindspore-lab/mindspore-cli/integrations/llm"
	skillslib "github.com/mindspore-lab/mindspore-cli/integrations/skills"
	"github.com/mindspore-lab/mindspore-cli/tools"
)

// LoadSkillTool implements tools.Tool and loads skill instructions on demand.
type LoadSkillTool struct {
	loader *skillslib.Loader
}

// NewLoadSkillTool creates a new load_skill tool backed by the given Loader.
func NewLoadSkillTool(loader *skillslib.Loader) *LoadSkillTool {
	return &LoadSkillTool{loader: loader}
}

func (t *LoadSkillTool) Name() string { return "load_skill" }

func (t *LoadSkillTool) Description() string {
	return "Load a skill's detailed instructions into the conversation. " +
		"Call this when the user's task matches an available skill. " +
		"The skill content will be returned as instructions to follow."
}

func (t *LoadSkillTool) Schema() llm.ToolSchema {
	return llm.ToolSchema{
		Type: "object",
		Properties: map[string]llm.Property{
			"name": {
				Type:        "string",
				Description: "Name of the skill to load",
				Enum:        t.loader.Names(),
			},
		},
		Required: []string{"name"},
	}
}

type loadSkillParams struct {
	Name string `json:"name"`
}

func (t *LoadSkillTool) Execute(_ context.Context, params json.RawMessage) (*tools.Result, error) {
	var p loadSkillParams
	if err := tools.ParseParams(params, &p); err != nil {
		return tools.ErrorResult(err), nil
	}
	if p.Name == "" {
		return tools.ErrorResultf("skill name is required"), nil
	}
	content, err := t.loader.Load(p.Name)
	if err != nil {
		return tools.ErrorResultf("load skill: %v", err), nil
	}
	return tools.StringResultWithSummary(content, fmt.Sprintf("loaded skill: %s", p.Name)), nil
}
