package main

import (
	"database/sql"
	"flag"
	"log"
	"net/http"

	_ "github.com/mattn/go-sqlite3"
	"github.com/vigo999/ms-cli/configs"
	"github.com/vigo999/ms-cli/internal/server"
)

func main() {
	cfgPath := flag.String("config", "configs/server.yaml", "path to server config file")
	flag.Parse()

	cfg, err := configs.LoadServerConfig(*cfgPath)
	if err != nil {
		log.Fatalf("load config: %v", err)
	}

	db, err := sql.Open("sqlite3", cfg.Storage.DSN)
	if err != nil {
		log.Fatalf("open database: %v", err)
	}
	defer db.Close()

	store, err := server.NewStore(db)
	if err != nil {
		log.Fatalf("init store: %v", err)
	}

	mux := server.NewMux(store, cfg.Auth.Tokens)

	log.Printf("ms-cli-server listening on %s", cfg.Server.Addr)
	if err := http.ListenAndServe(cfg.Server.Addr, mux); err != nil {
		log.Fatalf("listen: %v", err)
	}
}
