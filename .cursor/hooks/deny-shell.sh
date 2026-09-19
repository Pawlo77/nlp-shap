#!/usr/bin/env bash
# Deny pip install and pre-commit CLI; force uv + prek.
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
python3 "$ROOT/.cursor/hooks/deny_shell.py"
exit 0
