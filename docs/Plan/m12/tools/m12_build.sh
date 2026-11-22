#!/bin/sh
# M12: build gemma_native from the exact v0.2.1 source snapshot (provenance native-20261014-r4, manifest sha 92f6d81f…)
# + the 3-hunk vocab patch (docs/Plan/m12/raw/gemma_native_m12.diff), same flags/deps as ALife build.mjs. Output: out/m12/bin/<platform>/
# usage: m12_build.sh mac|linux
set -eu
P=$1; Q=/Users/lexchien/Documents/AI/gemma4-qat; A=/Users/lexchien/Documents/AI/ALife-Platform/Google-Gemma4
SRC=$Q/out/m12/work/src/gemma-native; B=$Q/out/m12/work/build-$P; T=$A/originals/tools/native-20261014-r1
L=$A/build/legacy-20261013/_deps
SPM=/Users/lexchien/Documents/AI/ALife-Platform/runs/google_gemma4_private_upload/20261014/export-repo/Google-Gemma4/work/source/native-dependencies/native-20261014-r1/sentencepiece
ARGS="-S $SRC -B $B -DCMAKE_BUILD_TYPE=Release -DCMAKE_DISABLE_FIND_PACKAGE_OpenSSL=ON -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DHWY_ENABLE_TESTS=OFF -DHWY_ENABLE_EXAMPLES=OFF -DNATIVE_SQLITE_SOURCE=$T/sqlite-amalgamation-3500400 -DNATIVE_ABSL_SOURCE=$T/abseil-source -DFETCHCONTENT_SOURCE_DIR_HIGHWAY=$L/highway-src -DFETCHCONTENT_SOURCE_DIR_JSON=$L/json-src -DFETCHCONTENT_SOURCE_DIR_HTTPLIB=$L/httplib-src -DFETCHCONTENT_SOURCE_DIR_BENCHMARK=$L/benchmark-src -DFETCHCONTENT_SOURCE_DIR_SENTENCEPIECE=$SPM"
mkdir -p $Q/out/m12/bin/$P
if [ $P = mac ]; then
  cmake $ARGS "-DCMAKE_CXX_FLAGS=-mcpu=apple-m1 -DHWY_HAVE_SCALAR_BF16_TYPE=0"
  cmake --build $B --target gemma_native --parallel 6
  cp $B/gemma_native $Q/out/m12/bin/mac/gemma_native; /usr/bin/strip -S $Q/out/m12/bin/mac/gemma_native
else
  export NATIVE_ZIG=$T/zig-aarch64-macos-0.15.2/zig ZIG_GLOBAL_CACHE_DIR=$B/zig-cache ZIG_LOCAL_CACHE_DIR=$B/zig-local
  cmake $ARGS -DCMAKE_TOOLCHAIN_FILE=$SRC/linux-musl.cmake "-DCMAKE_EXE_LINKER_FLAGS=-static -Wl,--gc-sections -Wl,--strip-all"
  cmake --build $B --target gemma_native --parallel 4
  cp $B/gemma_native $Q/out/m12/bin/linux-x86_64-gemma_native
fi
echo BUILD_DONE $P
