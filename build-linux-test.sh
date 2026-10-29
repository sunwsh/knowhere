#!/usr/bin/env bash
# 在 Apple Silicon 的 Mac 上，用 linux/amd64 容器编出可拷到 x86_64 服务器的
# benchmark_float_qps。脚本本身不改 Knowhere 源码。
# PAG 用例在支持 AVX-512（含 VNNI）的机器上运行：
#   ./dist-amd64/bin/benchmark_float_qps --gtest_filter=Benchmark_float_qps.TEST_PAG
#
# 用法（在 knowhere 目录或任意目录）：
#   ./build-linux-test.sh
# 清空本次 CMake 构建目录后重编：
#   CLEAN=1 ./build-linux-test.sh
# 并行度（容器约 8GB 内存时 8 能编过）：
#   JOBS=4 ./build-linux-test.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
IMAGE="${IMAGE:-vipdocker-f9nub.vclound.com/milvusdb/milvus-cpu-devel:ubuntu22.04_v25_19}"
CONTAINER="${CONTAINER:-knowhere-amd64-build}"
CONAN_VOLUME="${CONAN_VOLUME:-knowhere-conan-cache}"
JOBS="${JOBS:-8}"
CLEAN="${CLEAN:-0}"

if ! docker info >/dev/null 2>&1; then
  echo "docker 不可用" >&2
  exit 1
fi

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  docker pull --platform linux/amd64 "$IMAGE"
fi

docker volume create "$CONAN_VOLUME" >/dev/null

if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  running="$(docker inspect -f '{{.State.Running}}' "$CONTAINER")"
  if [[ "$running" != "true" ]]; then
    docker start "$CONTAINER" >/dev/null
  fi
else
  docker run -d --name "$CONTAINER" --platform linux/amd64 \
    -v "$ROOT":/work/knowhere \
    -v "$CONAN_VOLUME":/root/.conan \
    -w /work/knowhere \
    "$IMAGE" \
    sleep infinity >/dev/null
fi

docker exec -i -e JOBS="$JOBS" -e CLEAN="$CLEAN" "$CONTAINER" bash -s <<'EOS'
set -euo pipefail

# conanfile 把 CMAKE_*_COMPILER_LAUNCHER 设成 ccache，CMake 又把
# RULE_LAUNCH_COMPILE 设成 ccache，两层套在一起会报 Recursive invocation。
# 容器里换成直接 exec 编译器，源码保持不动。
if [[ ! -f /usr/bin/ccache.real ]]; then
  mv /usr/bin/ccache /usr/bin/ccache.real
fi
printf '%s\n' '#!/bin/sh' 'exec "$@"' > /usr/bin/ccache
chmod +x /usr/bin/ccache

if ! command -v patchelf >/dev/null 2>&1; then
  apt-get update -qq
  apt-get install -y -qq patchelf
fi

export CONAN_SYSREQUIRES_MODE=enabled
export CONAN_CPU_COUNT="$JOBS"
mkdir -p /root/.conan
if [[ -f /root/.conan/global.conf ]] && grep -q '^tools.build:jobs=' /root/.conan/global.conf; then
  sed -i "s/^tools.build:jobs=.*/tools.build:jobs=${JOBS}/" /root/.conan/global.conf
else
  echo "tools.build:jobs=${JOBS}" >> /root/.conan/global.conf
fi

cd /work/knowhere
if [[ "$CLEAN" == "1" ]]; then
  rm -rf build dist-amd64
fi
mkdir -p build
cd build

conan profile new default --detect --force
conan profile update settings.compiler.libcxx=libstdc++11 default
if ! conan remote list | grep -q '^default-conan-local:'; then
  conan remote add default-conan-local https://milvus01.jfrog.io/artifactory/api/conan/default-conan-local
fi

conan install .. --build=missing \
  -o with_benchmark=True \
  -s compiler.libcxx=libstdc++11 \
  -s build_type=Release
conan build ..

src=/work/knowhere/build/Release
dest=/work/knowhere/dist-amd64
rm -rf "$dest"
mkdir -p "$dest/bin" "$dest/lib"
bins=(
  benchmark_float_qps
  benchmark_float
  benchmark_float_bitset
  benchmark_float_range
  benchmark_float_range_bitset
  benchmark_binary
  benchmark_binary_range
  benchmark_simd_qps
  benchmark_sparse_simd
  gen_hdf5_file
  gen_fbin_file
)
for b in "${bins[@]}"; do
  cp -a "$src/benchmark/$b" "$dest/bin/"
done
cp -a "$src/libknowhere.so" "$dest/lib/"
cp -a "$src/milvus-common-build/libmilvus-common.so" "$dest/lib/"

python3 - <<'PY'
import os, shutil, subprocess
dest = "/work/knowhere/dist-amd64/lib"
out = subprocess.check_output(["ldd", "/work/knowhere/dist-amd64/bin/benchmark_float_qps"], text=True)
seen = set()
for line in out.splitlines():
    if "=>" not in line:
        continue
    path = line.split("=>", 1)[1].strip().split()[0]
    if not path.startswith("/root/.conan/") or not os.path.exists(path):
        continue
    real = os.path.realpath(path)
    parent = os.path.dirname(real)
    base = os.path.basename(real).split(".so")[0]
    for name in os.listdir(parent):
        if not name.startswith(base + ".so"):
            continue
        src = os.path.join(parent, name)
        dst = os.path.join(dest, name)
        if src in seen:
            continue
        seen.add(src)
        if os.path.lexists(dst):
            os.remove(dst)
        if os.path.islink(src):
            os.symlink(os.readlink(src), dst)
        else:
            shutil.copy2(src, dst)
PY

for b in "$dest"/bin/*; do
  patchelf --set-rpath '$ORIGIN/../lib' "$b"
done
for so in "$dest"/lib/*; do
  [[ -L "$so" ]] && continue
  patchelf --set-rpath '$ORIGIN' "$so"
done

echo "---- $(file "$dest/bin/benchmark_float_qps")"
"$dest/bin/benchmark_float_qps" --gtest_list_tests
EOS

echo "完成: $ROOT/dist-amd64/bin/benchmark_float_qps"
echo "PAG 用例: $ROOT/dist-amd64/bin/benchmark_float_qps --gtest_filter=Benchmark_float_qps.TEST_PAG"
