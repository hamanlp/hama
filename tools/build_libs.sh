#!/usr/bin/env bash
# Build the native hama engine libraries for the shipped Python wheel platforms
# and stage them under python/src/hama/_libs/<platform>/. Zig cross-compiles all
# targets from one host. The platform dir name matches _engine.py's lookup:
#   f"{platform.system().lower()}-{platform.machine().lower()}"
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ZIG_DIR="$ROOT/zig"
LIBS="$ROOT/python/src/hama/_libs"
OPT="${HAMA_OPT:-ReleaseFast}"

build() {
  local target="$1" platdir="$2" libname="$3"
  echo "==> $target -> _libs/$platdir/$libname"
  ( cd "$ZIG_DIR" && zig build -Dtarget="$target" -Doptimize="$OPT" -p "zig-out/$platdir" )
  mkdir -p "$LIBS/$platdir"
  cp "$ZIG_DIR/zig-out/$platdir/lib/"*"${libname##*.}" "$LIBS/$platdir/$libname" 2>/dev/null \
    || cp "$ZIG_DIR/zig-out/$platdir/lib/$libname" "$LIBS/$platdir/$libname"
}

# Linux targets pin glibc 2.17 (manylinux2014 baseline) so the .so loads on
# older distros as well as new ones.
build aarch64-macos             darwin-arm64    libhama.dylib
build x86_64-macos              darwin-x86_64   libhama.dylib
build x86_64-linux-gnu.2.17     linux-x86_64    libhama.so
build aarch64-linux-gnu.2.17    linux-aarch64   libhama.so

echo "done. staged libs:"
find "$LIBS" -type f -name 'libhama.*' | sed "s#$ROOT/##"
