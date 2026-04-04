package update

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"runtime"
	"strconv"
	"strings"
	"time"
)

// Check fetches the manifest and compares the current version.
// Returns nil, nil for dev builds.
func Check(ctx context.Context, currentVersion string) (*CheckResult, error) {
	if currentVersion == "dev" || currentVersion == "" {
		return nil, nil
	}

	manifest, err := fetchManifest(ctx)
	if err != nil {
		return nil, err
	}

	result := &CheckResult{
		CurrentVersion: currentVersion,
		LatestVersion:  manifest.Latest,
	}

	if compareSemver(currentVersion, manifest.Latest) < 0 {
		result.UpdateAvailable = true
		result.DownloadURL = buildDownloadURL(manifest.DownloadBase, manifest.Latest)
	}

	if manifest.MinAllowed != "" && compareSemver(currentVersion, manifest.MinAllowed) < 0 {
		result.ForceUpdate = true
		result.UpdateAvailable = true
		result.DownloadURL = buildDownloadURL(manifest.DownloadBase, manifest.Latest)
	}

	return result, nil
}

func fetchManifest(ctx context.Context) (*Manifest, error) {
	return fetchManifestFromURLs(ctx, ManifestURLs())
}

func fetchManifestFromURLs(ctx context.Context, urls []string) (*Manifest, error) {
	if len(urls) == 0 {
		return nil, fmt.Errorf("no manifest URLs configured")
	}

	var errs []string
	for _, url := range urls {
		m, err := fetchManifestFromURL(ctx, url)
		if err == nil {
			return m, nil
		}
		errs = append(errs, fmt.Sprintf("%s: %v", url, err))
	}

	return nil, fmt.Errorf("fetch manifest: %s", strings.Join(errs, "; "))
}

func fetchManifestFromURL(ctx context.Context, url string) (*Manifest, error) {
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("manifest fetch returned %d", resp.StatusCode)
	}

	var m Manifest
	if err := json.NewDecoder(resp.Body).Decode(&m); err != nil {
		return nil, fmt.Errorf("decode manifest: %w", err)
	}
	return &m, nil
}

type semverIdentifier struct {
	raw      string
	numeric  bool
	numericV int
}

type semverVersion struct {
	core       [3]int
	prerelease []semverIdentifier
}

// compareSemver compares two semver strings, including prerelease ordering.
// Returns -1 if a < b, 0 if equal, 1 if a > b.
func compareSemver(a, b string) int {
	pa := parseSemver(a)
	pb := parseSemver(b)
	for i := 0; i < 3; i++ {
		if pa.core[i] < pb.core[i] {
			return -1
		}
		if pa.core[i] > pb.core[i] {
			return 1
		}
	}

	if len(pa.prerelease) == 0 && len(pb.prerelease) == 0 {
		return 0
	}
	if len(pa.prerelease) == 0 {
		return 1
	}
	if len(pb.prerelease) == 0 {
		return -1
	}

	for i := 0; i < len(pa.prerelease) && i < len(pb.prerelease); i++ {
		cmp := compareSemverIdentifier(pa.prerelease[i], pb.prerelease[i])
		if cmp != 0 {
			return cmp
		}
	}

	switch {
	case len(pa.prerelease) < len(pb.prerelease):
		return -1
	case len(pa.prerelease) > len(pb.prerelease):
		return 1
	default:
		return 0
	}
}

func compareSemverIdentifier(a, b semverIdentifier) int {
	switch {
	case a.numeric && b.numeric:
		switch {
		case a.numericV < b.numericV:
			return -1
		case a.numericV > b.numericV:
			return 1
		default:
			return 0
		}
	case a.numeric && !b.numeric:
		return -1
	case !a.numeric && b.numeric:
		return 1
	default:
		switch {
		case a.raw < b.raw:
			return -1
		case a.raw > b.raw:
			return 1
		default:
			return 0
		}
	}
}

func parseSemver(v string) semverVersion {
	v = strings.TrimPrefix(v, "v")
	corePart := v
	prereleasePart := ""
	if idx := strings.IndexByte(v, '-'); idx >= 0 {
		corePart = v[:idx]
		prereleasePart = v[idx+1:]
	}

	parts := strings.SplitN(corePart, ".", 3)
	var result semverVersion
	for i := 0; i < 3 && i < len(parts); i++ {
		n, _ := strconv.Atoi(parts[i])
		result.core[i] = n
	}

	if prereleasePart == "" {
		return result
	}

	for _, ident := range strings.Split(prereleasePart, ".") {
		token := semverIdentifier{raw: ident}
		if n, err := strconv.Atoi(ident); err == nil {
			token.numeric = true
			token.numericV = n
		}
		result.prerelease = append(result.prerelease, token)
	}
	return result
}

// FetchReleaseNotes fetches the body of a GitHub release by tag.
// Returns empty string on any failure (non-fatal).
func FetchReleaseNotes(ctx context.Context, version string) string {
	version = strings.TrimPrefix(version, "v")
	url := fmt.Sprintf("https://api.github.com/repos/mindspore-lab/mindspore-cli/releases/tags/v%s", version)

	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return ""
	}
	req.Header.Set("Accept", "application/vnd.github+json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return ""
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return ""
	}

	var release struct {
		Body string `json:"body"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&release); err != nil {
		return ""
	}
	return release.Body
}

func buildDownloadURL(base, version string) string {
	version = strings.TrimPrefix(version, "v")
	name := fmt.Sprintf("mscli-%s-%s", runtime.GOOS, runtime.GOARCH)
	if runtime.GOOS == "windows" {
		name += ".exe"
	}
	return fmt.Sprintf("%s/v%s/%s", strings.TrimRight(base, "/"), version, name)
}
