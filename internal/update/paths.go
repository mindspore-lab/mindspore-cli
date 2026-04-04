package update

import (
	"os"
	"path/filepath"
	"runtime"
)

const (
	defaultMirrorManifestURL = "http://47.115.175.134/mscli/releases/latest/manifest.json"
	defaultGitHubManifestURL = "https://github.com/vigo999/mindspore-cli/releases/latest/download/manifest.json"
)

// InstallDir returns ~/.mscli/bin.
func InstallDir() string {
	return filepath.Join(ConfigDir(), "bin")
}

// BinaryPath returns the expected binary path.
func BinaryPath() string {
	name := "mscli"
	if runtime.GOOS == "windows" {
		name += ".exe"
	}
	return filepath.Join(InstallDir(), name)
}

// ConfigDir returns ~/.mscli.
func ConfigDir() string {
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".mscli")
}

// ManifestURL returns the first manifest URL candidate.
func ManifestURL() string {
	urls := ManifestURLs()
	if len(urls) == 0 {
		return ""
	}
	return urls[0]
}

// ManifestURLs returns manifest URL candidates in lookup order.
// If MSCLI_MANIFEST_URL is set, it is used exclusively.
func ManifestURLs() []string {
	if u := os.Getenv("MSCLI_MANIFEST_URL"); u != "" {
		return []string{u}
	}
	return []string{defaultMirrorManifestURL, defaultGitHubManifestURL}
}
