#!/usr/bin/env bash
set -e

# 1) Create directories
dirs=(
  calibration
  noise
  simulator
  fitting
  benchmarks
  tests
  examples
  docs
  .github/workflows
)
for d in "${dirs[@]}"; do
  mkdir -p "$d"
done

# 2) Create __init__.py in noise so it's a package
touch noise/__init__.py

# 3) Touch all the placeholder files
files=(
  calibration/calibration.py
  noise/reset_error.py
  noise/decay_error.py
  noise/measurement_error.py
  noise/gate_error.py
  noise/crosstalk_error.py
  simulator/simulate.py
  fitting/fitting.py
  benchmarks/ghz.py
  benchmarks/w_state.py
  tests/test_calibration.py
  tests/test_noise_models.py
  tests/test_simulation.py
  examples/tutorial.ipynb
  docs/index.md
  .github/workflows/ci.yml
  .gitignore
  setup.py
  README.md
  CONTRIBUTING.md
  CODE_OF_CONDUCT.md
  requirements.txt
)

for f in "${files[@]}"; do
  # ensure parent directory exists (just in case)
  mkdir -p "$(dirname "$f")"
  # create the empty file if it doesn't exist
  [ -f "$f" ] || touch "$f"
done

echo "✅ Repository scaffold created!"
