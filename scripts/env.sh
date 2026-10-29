# Source from Bash: source /home/renya/Sejong/scripts/env.sh
export SEJONG_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$SEJONG_ROOT/.venv/bin/activate"
export ROCM_PATH="$("$VIRTUAL_ENV/bin/python" -c 'import importlib.metadata as m; print(m.distribution("rocm-sdk-core").locate_file("_rocm_sdk_core"))')"
export HIP_PATH="$ROCM_PATH"
export PATH="$ROCM_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TRITON_CACHE_DIR="$SEJONG_ROOT/.cache/triton"
