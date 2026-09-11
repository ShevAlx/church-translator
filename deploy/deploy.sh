#!/usr/bin/env bash
# Push this working tree to the booth Pi and (re)start the Telegram bot there.
#
#   deploy/deploy.sh                      # default host: shevchuk@shevchuk.local
#   deploy/deploy.sh shevchuk@10.0.0.42   # by IP, e.g. on the church network
#
# Code only. The Pi keeps its own .env (API keys, bot token), config.yaml
# (its device names, its voice ids) and logs/ — rsync never touches excluded
# files on the receiving side, --delete included. Private audio (*.wav) and
# start_on_new_machine.md (real keys) never leave the Mac.
set -euo pipefail

HOST="${1:-shevchuk@shevchuk.local}"
cd "$(dirname "$0")/.."

rsync -az --delete \
  --exclude .git/ --exclude .venv/ --exclude logs/ --exclude __pycache__/ \
  --exclude .DS_Store --exclude .env --exclude config.yaml \
  --exclude start_on_new_machine.md --exclude '*.wav' --exclude replay-out/ \
  ./ "$HOST:church-translator/"

ssh "$HOST" bash -s <<'REMOTE'
set -euo pipefail
cd ~/church-translator
# --extra providers, always: a bare sync would uninstall them (see README).
~/.local/bin/uv sync --extra providers --frozen --quiet

mkdir -p ~/.config/systemd/user ~/.config/wireplumber/wireplumber.conf.d
WP=~/.config/wireplumber/wireplumber.conf.d/51-church-translator-scarlett.conf
if ! cmp -s deploy/wireplumber-scarlett.conf "$WP"; then
  cp deploy/wireplumber-scarlett.conf "$WP"
  systemctl --user restart wireplumber 2>/dev/null || true
  echo "WirePlumber rule updated: PipeWire leaves the Scarlett alone"
fi
cp deploy/church-translator-bot.service ~/.config/systemd/user/
systemctl --user daemon-reload

python3 -c "import ctypes.util, sys; sys.exit(0 if ctypes.util.find_library('portaudio') else 1)" \
  || echo "!! libportaudio2 is missing — run once: sudo apt install -y libportaudio2"
missing=""
[ -f .env ] || missing="$missing .env"
[ -f config.yaml ] || missing="$missing config.yaml"
if [ -n "$missing" ]; then
  echo "!! Not starting the bot, missing:$missing (see PI_SETUP.md)"
  exit 0
fi
systemctl --user enable --quiet church-translator-bot
systemctl --user restart church-translator-bot
sleep 3
systemctl --user --no-pager --lines=8 status church-translator-bot || true
REMOTE
