#!/usr/bin/env bash
# Package a built gfpgan-ncnn-vulkan binary into a nihui-style portable zip:
#   gfpgan-ncnn-vulkan-<version>-<platform>/
#     gfpgan-ncnn-vulkan[.exe]   models/   LICENSE   README.md
#
# Runs on Linux/macOS CI and inside Git Bash on Windows runners. Expects the
# working directory to be the repo root with ./build present; platform-specific
# extras (OpenCV DLLs) are pulled from the well-known cache paths the release
# workflow sets up.
set -euo pipefail

PLATFORM="${1:?usage: package.sh <ubuntu|windows|macos>}"
APP="gfpgan-ncnn-vulkan"
VERSION="${GITHUB_REF_NAME:-dev}"
VERSION="${VERSION//\//_}"
PKG="${APP}-${VERSION}-${PLATFORM}"

unzip_any() {
    # unzip_any <archive> <dest>
    if command -v unzip >/dev/null 2>&1; then
        unzip -q -o "$1" -d "$2"
    elif command -v 7z >/dev/null 2>&1; then
        7z x -aoa -o"$2" "$1" >/dev/null
    else
        python -c "import sys,zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$1" "$2"
    fi
}

zip_any() {
    # zip_any <zip> <dir>
    if command -v zip >/dev/null 2>&1; then
        zip -9 -q -r "$1" "$2"
    elif command -v 7z >/dev/null 2>&1; then
        7z a -tzip -mx9 "$1" "$2" >/dev/null
    else
        powershell.exe -NoProfile -Command "Compress-Archive -Path '${2}/*' -DestinationPath '${1}' -Force"
    fi
}

rm -rf "$PKG"
mkdir -p "$PKG/models"

# ── binary ───────────────────────────────────────────────────────────────────
case "$PLATFORM" in
    windows)
        BIN="build/Release/${APP}.exe"
        [ -f "$BIN" ] || BIN="build/${APP}.exe"
        ;;
    *)
        BIN="build/${APP}"
        ;;
esac
[ -f "$BIN" ] || { echo "binary not found (looked for build/**/${APP})" >&2; exit 1; }
cp "$BIN" "$PKG/"
[ "$PLATFORM" = "windows" ] && chmod +x "$PKG/${APP}.exe"

# Windows: the exe links opencv_world dynamically — ship the DLL next to it.
if [ "$PLATFORM" = "windows" ]; then
    WORLD_DLL="$(find OpenCV -name 'opencv_world*.dll' -path '*vc16*bin*' 2>/dev/null | head -1 || true)"
    [ -z "$WORLD_DLL" ] && WORLD_DLL="$(find OpenCV -name 'opencv_world*.dll' 2>/dev/null | head -1 || true)"
    if [ -n "$WORLD_DLL" ]; then
        cp "$WORLD_DLL" "$PKG/"
    else
        echo "WARNING: opencv_world DLL not found; the exe may not start on machines without OpenCV" >&2
    fi
fi

# ── models ───────────────────────────────────────────────────────────────────
# Normalized from the upstream model bundle: whatever folder carries
# encoder.param becomes models/.
MODELS_URL="${MODELS_URL:-https://github.com/onuralpszr/GFPGAN-ncnn-vulkan/releases/download/v0.0.1-models/GFPGAN-ncnn-models.zip}"
rm -rf .models-tmp && mkdir .models-tmp
echo "downloading models from $MODELS_URL"
curl -sL "$MODELS_URL" -o .models-tmp/models.zip
unzip_any .models-tmp/models.zip .models-tmp
ENCODER="$(find .models-tmp -name 'encoder*.param' | head -1)"
[ -n "$ENCODER" ] || { echo "model bundle did not contain encoder*.param" >&2; exit 1; }
MODEL_ROOT="$(dirname "$ENCODER")"
cp "$MODEL_ROOT"/* "$PKG/models/"
rm -rf .models-tmp

# ── docs + metadata ──────────────────────────────────────────────────────────
cp README.md LICENSE "$PKG/" 2>/dev/null || cp LICENSE "$PKG/"

# Linux: strip for size (symbol tables only; keeping dynamic symbols).
if [ "$PLATFORM" = "ubuntu" ] && command -v strip >/dev/null 2>&1; then
    strip -g "$PKG/${APP}" 2>/dev/null || true
fi

zip_any "${PKG}.zip" "$PKG"
echo "packaged ${PKG}.zip"
