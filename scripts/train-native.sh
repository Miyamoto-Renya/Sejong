#!/usr/bin/env bash
set -euo pipefail
sejong_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$sejong_root"
source scripts/env.sh
sejong_output="${1:-runs/native-v6}"
sejong_data="${2:-data/native-v6}"
if [[ -e "$sejong_output" ]]; then
  printf 'Output already exists: %s\n' "$sejong_output" >&2
  exit 1
fi
python scripts/fetch-minilm.py
if [[ ! -d "$sejong_data" ]]; then
  python scripts/create-native-data.py --out "$sejong_data"
fi
cargo build --release -p sejong-cli
mkdir -p "$sejong_output"
for sejong_role in 0 1 2 3 4 5; do
  target/release/sejong train-role --spec "$sejong_data/role-$sejong_role.json" --out "$sejong_output/role-$sejong_role"
done
for sejong_fact in a b; do
  target/release/sejong train-memory --spec "$sejong_data/$sejong_fact.json" --out "$sejong_output/$sejong_fact"
done
target/release/sejong train-value --spec "$sejong_data/native-b.json" --out "$sejong_output/native-b"
cp "$sejong_data/assembly.json" "$sejong_output/assembly.json"
target/release/sejong connect --spec "$sejong_output/assembly.json" --out "$sejong_output/model"
