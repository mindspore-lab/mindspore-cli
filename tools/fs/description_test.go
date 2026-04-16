package fs

import (
	"strings"
	"testing"
)

func TestToolDescriptionsIncludeDedicatedToolGuidance(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name  string
		got   string
		wants []string
	}{
		{
			name:  "read",
			got:   NewReadTool(".").Description(),
			wants: []string{"To read files use read instead of cat, head, tail, or sed"},
		},
		{
			name:  "edit",
			got:   NewEditTool(".").Description(),
			wants: []string{"To edit files use edit instead of sed or awk"},
		},
		{
			name:  "write",
			got:   NewWriteTool(".").Description(),
			wants: []string{"To create files use write instead of cat with heredoc or echo redirection"},
		},
		{
			name:  "glob",
			got:   NewGlobTool(".").Description(),
			wants: []string{"To search for files use glob instead of find or ls"},
		},
		{
			name:  "grep",
			got:   NewGrepTool(".").Description(),
			wants: []string{"To search the content of files, use grep instead of grep or rg"},
		},
		{
			name:  "list_dir",
			got:   NewListDirTool(".").Description(),
			wants: []string{"Prefer this over glob or shell ls/find/tree when exploring directory layout"},
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			for _, want := range tc.wants {
				if !strings.Contains(tc.got, want) {
					t.Fatalf("%s description missing %q: %q", tc.name, want, tc.got)
				}
			}
		})
	}
}
