package app

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

const (
	inputHistoryLoadMax               = 100
	inputHistoryMaxBytes        int64 = 512 * 1024
	inputHistoryTrimTargetBytes       = inputHistoryMaxBytes * 4 / 5
)

type inputHistoryEntry struct {
	Display   string `json:"display"`
	Timestamp int64  `json:"timestamp"`
	WorkDir   string `json:"workdir"`
}

func inputHistoryFilePath() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("resolve home dir: %w", err)
	}
	return filepath.Join(home, ".mscli", "history.jsonl"), nil
}

func loadInputHistoryForWorkdir(workDir string) ([]string, error) {
	path, err := inputHistoryFilePath()
	if err != nil {
		return nil, err
	}
	file, err := os.Open(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, nil
		}
		return nil, fmt.Errorf("open history: %w", err)
	}
	defer file.Close()

	var matches []string
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 0, 64*1024), int(inputHistoryMaxBytes))
	for scanner.Scan() {
		var row inputHistoryEntry
		if err := json.Unmarshal(scanner.Bytes(), &row); err != nil {
			continue
		}
		if strings.TrimSpace(row.Display) == "" || row.WorkDir != workDir {
			continue
		}
		matches = append(matches, row.Display)
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("scan history: %w", err)
	}
	if len(matches) > inputHistoryLoadMax {
		matches = matches[len(matches)-inputHistoryLoadMax:]
	}
	return matches, nil
}

func appendInputHistory(workDir, text string) error {
	text = strings.TrimSpace(text)
	if text == "" {
		return nil
	}
	path, err := inputHistoryFilePath()
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return fmt.Errorf("create history dir: %w", err)
	}

	row := inputHistoryEntry{
		Display:   text,
		Timestamp: time.Now().UnixMilli(),
		WorkDir:   workDir,
	}
	data, err := json.Marshal(row)
	if err != nil {
		return fmt.Errorf("marshal history entry: %w", err)
	}
	data = append(data, '\n')

	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o600)
	if err != nil {
		return fmt.Errorf("open history for append: %w", err)
	}
	_, writeErr := f.Write(data)
	closeErr := f.Close()
	if writeErr != nil {
		return fmt.Errorf("append history entry: %w", writeErr)
	}
	if closeErr != nil {
		return fmt.Errorf("close history file: %w", closeErr)
	}
	return trimInputHistoryIfNeeded(path)
}

func trimInputHistoryIfNeeded(path string) error {
	info, err := os.Stat(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil
		}
		return fmt.Errorf("stat history: %w", err)
	}
	if info.Size() <= inputHistoryMaxBytes {
		return nil
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("read history for trim: %w", err)
	}
	lines := strings.Split(string(data), "\n")
	if len(lines) == 0 {
		return nil
	}

	var kept []string
	var size int64
	for i := len(lines) - 1; i >= 0; i-- {
		line := strings.TrimSpace(lines[i])
		if line == "" {
			continue
		}
		lineSize := int64(len(line) + 1)
		if size+lineSize > inputHistoryTrimTargetBytes && len(kept) > 0 {
			break
		}
		kept = append(kept, line)
		size += lineSize
	}
	for left, right := 0, len(kept)-1; left < right; left, right = left+1, right-1 {
		kept[left], kept[right] = kept[right], kept[left]
	}

	content := ""
	if len(kept) > 0 {
		content = strings.Join(kept, "\n") + "\n"
	}
	tmpPath := path + ".tmp"
	if err := os.WriteFile(tmpPath, []byte(content), 0o600); err != nil {
		return fmt.Errorf("write trimmed history: %w", err)
	}
	if err := os.Rename(tmpPath, path); err != nil {
		_ = os.Remove(tmpPath)
		return fmt.Errorf("replace trimmed history: %w", err)
	}
	return nil
}
