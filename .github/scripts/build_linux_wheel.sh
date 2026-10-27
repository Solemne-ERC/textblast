#!/usr/bin/env bash
# Build the Linux wheel inside a manylinux container (the repository mounted at /io, the engine cache at
# $TEXTBLAST_CACHE). Builds the engine unless the cache has it, bundles it, and lets auditwheel copy in the
# libraries that manylinux does not allow and set the platform tag.
#
#   docker run --rm -v "$PWD:/io" -v "$CACHE:/cache" -e TEXTBLAST_CACHE=/cache -e PLAT=manylinux_2_28_x86_64 \
#       quay.io/pypa/manylinux_2_28_x86_64 bash /io/.github/scripts/build_linux_wheel.sh
set -euo pipefail

PLAT="${PLAT:-manylinux_2_28_x86_64}"
OPENMP="${WHEEL_OPENMP:-auto}"
PYTHON=/opt/python/cp312-cp312/bin/python

echo "== $(gcc --version | head -1) · $(ldd --version | head -1) · auditwheel $(auditwheel --version)"
"$PYTHON" -m venv /tmp/venv
/tmp/venv/bin/pip install --quiet /io
/tmp/venv/bin/textblast engine build --openmp "$OPENMP"

/tmp/venv/bin/python /io/.github/scripts/build_wheel.py --plat-name linux_x86_64 --openmp "$OPENMP" --out /tmp/dist
for binary in "$TEXTBLAST_CACHE"/engine/*/bin/*; do
  echo "== ldd $binary"
  ldd "$binary"
done
auditwheel show /tmp/dist/*.whl
auditwheel repair --plat "$PLAT" --wheel-dir /io/wheelhouse /tmp/dist/*.whl
auditwheel show /io/wheelhouse/*.whl
ls -l /io/wheelhouse
