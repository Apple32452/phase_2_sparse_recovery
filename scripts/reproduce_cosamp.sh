#!/usr/bin/env bash
set -e

mkdir -p results/cosamp
mkdir -p figures/cosamp

python experiments/cosamp_stress_test.py \
  --out-dir results/cosamp \
  --out-prefix cosamp_stress_test

mv -f results/cosamp/cosamp_stress_test.png figures/cosamp/

echo "Wrote results/cosamp/cosamp_stress_test.json"
echo "Wrote figures/cosamp/cosamp_stress_test.png"
