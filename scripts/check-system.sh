#!/usr/bin/env bash
set -euo pipefail
sejong_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$sejong_root"
source scripts/env.sh
python scripts/fetch-minilm.py
cargo test -p sejong-runtime native::checks::input_sized_exact_composition
cargo run -p sejong-train --example parser_resume
cargo run -p sejong-train --example memory_contract -- "${1:-.cache/scalable-memory-$$}"
