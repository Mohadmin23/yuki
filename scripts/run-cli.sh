#!/usr/bin/env bash
# Shortcut: launch the terminal chatbot
exec "$(dirname "$0")/../llama" cli "$@"
