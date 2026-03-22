package configs

import (
	"fmt"
	"os"

	"gopkg.in/yaml.v3"
)

type ServerConfig struct {
	Server  ServerListenConfig `yaml:"server"`
	Storage StorageConfig      `yaml:"storage"`
	Auth    AuthConfig         `yaml:"auth"`
}

type ServerListenConfig struct {
	Addr string `yaml:"addr"`
}

type StorageConfig struct {
	Driver string `yaml:"driver"`
	DSN    string `yaml:"dsn"`
}

type AuthConfig struct {
	Tokens []TokenEntry `yaml:"tokens"`
}

type TokenEntry struct {
	Token string `yaml:"token"`
	User  string `yaml:"user"`
	Role  string `yaml:"role"`
}

func LoadServerConfig(path string) (*ServerConfig, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read server config %q: %w", path, err)
	}
	var cfg ServerConfig
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("parse server config %q: %w", path, err)
	}
	if cfg.Server.Addr == "" {
		cfg.Server.Addr = ":8080"
	}
	if cfg.Storage.Driver == "" {
		cfg.Storage.Driver = "sqlite"
	}
	if cfg.Storage.DSN == "" {
		cfg.Storage.DSN = "issues.db"
	}
	return &cfg, nil
}
