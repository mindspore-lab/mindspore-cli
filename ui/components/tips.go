package components

import "math/rand"

var tips = []string{
	"Use /report to submit feedback",
	"Press ctrl+o to view full history",
	"Use /model to switch models",
	"Use ↑/↓ to recall input history",
	"Use /compact to free up context space",
	"Use /clear to start a fresh conversation",
	"Use /permissions to manage tool access",
	"Use /train to launch training sessions",
	"Use /diagnose to investigate issues",
	"Use /help to see all commands",
}

// RandomTip returns a random tip string.
func RandomTip() string {
	return tips[rand.Intn(len(tips))]
}
