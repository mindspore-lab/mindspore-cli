package panels

import (
	"strings"
	"testing"

	"github.com/mindspore-lab/mindspore-cli/ui/model"
)

func TestRenderHintBarIncludesModelAndProvider(t *testing.T) {
	state := model.NewState("test", "/tmp/project", "", "Kimi K2.5", 262144, "MindSpore CLI Free")

	result := RenderHintBar(state, 120)
	for _, want := range []string{"Kimi K2.5", "MindSpore CLI Free", "/tmp/project"} {
		if !strings.Contains(result, want) {
			t.Fatalf("expected hint bar to include %q, got:\n%s", want, result)
		}
	}
}
