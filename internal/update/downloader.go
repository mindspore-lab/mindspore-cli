package update

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"time"
)

const maxBinarySize = 500 << 20 // 500 MiB

// ProgressFunc is called periodically during download with bytes downloaded and total size.
// total is -1 when Content-Length is unknown.
type ProgressFunc func(downloaded, total int64)

// progressReader wraps an io.Reader and calls fn after each Read.
type progressReader struct {
	r          io.Reader
	fn         ProgressFunc
	downloaded int64
	total      int64
}

func (pr *progressReader) Read(p []byte) (int, error) {
	n, err := pr.r.Read(p)
	pr.downloaded += int64(n)
	if pr.fn != nil {
		pr.fn(pr.downloaded, pr.total)
	}
	return n, err
}

// DownloadWithProgress is like Download but calls progressFn periodically with progress.
func DownloadWithProgress(ctx context.Context, url string, progressFn ProgressFunc) (string, error) {
	tmpDir := filepath.Join(ConfigDir(), "tmp")
	if err := os.MkdirAll(tmpDir, 0755); err != nil {
		return "", fmt.Errorf("create tmp dir: %w", err)
	}

	ctx, cancel := context.WithTimeout(ctx, 5*time.Minute)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return "", fmt.Errorf("create request: %w", err)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("download binary: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("download returned %d", resp.StatusCode)
	}

	total := resp.ContentLength // -1 if unknown

	tmpFile, err := os.CreateTemp(tmpDir, "mscli-update-*")
	if err != nil {
		return "", fmt.Errorf("create temp file: %w", err)
	}
	tmpPath := tmpFile.Name()

	limited := &io.LimitedReader{R: resp.Body, N: maxBinarySize}
	pr := &progressReader{r: limited, fn: progressFn, total: total}
	if _, err := io.Copy(tmpFile, pr); err != nil {
		tmpFile.Close()
		os.Remove(tmpPath)
		return "", fmt.Errorf("write binary: %w", err)
	}

	if limited.N == 0 {
		tmpFile.Close()
		os.Remove(tmpPath)
		return "", fmt.Errorf("download exceeds maximum size (%d MiB)", maxBinarySize>>20)
	}

	if err := tmpFile.Close(); err != nil {
		os.Remove(tmpPath)
		return "", fmt.Errorf("close temp file: %w", err)
	}

	return tmpPath, nil
}

// Download fetches the binary at url into ~/.mscli/tmp/ and returns the temp file path.
func Download(ctx context.Context, url string) (string, error) {
	return DownloadWithProgress(ctx, url, nil)
}
