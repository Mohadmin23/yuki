#!/usr/bin/env bash
# Shortcut: run the test suite
exec "$(dirname "$0")/../llama" test "$@"
