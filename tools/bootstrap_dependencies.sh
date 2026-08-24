#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
third_party_dir="$repo_root/third_party"
download_dir="$third_party_dir/downloads"
vikit_dir="$repo_root/src/rpg_vikit"
vikit_commit="6c886c8e5d83997806e00294826d528cea3581dd"
libtorch_url="https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcu118.zip"
libtorch_zip="$download_dir/libtorch-cxx11-abi-shared-with-deps-2.0.1+cu118.zip"

if [[ ${1:-} == "--dry-run" ]]; then
  printf 'rpg_vikit=%s@%s\n' "https://github.com/xuankuzcr/rpg_vikit.git" "$vikit_commit"
  printf 'libtorch=%s\n' "$libtorch_url"
  exit 0
fi

mkdir -p "$download_dir"

if [[ ! -d "$vikit_dir/.git" ]]; then
  git clone https://github.com/xuankuzcr/rpg_vikit.git "$vikit_dir"
fi
git -C "$vikit_dir" fetch --depth=1 origin "$vikit_commit"
git -C "$vikit_dir" checkout --detach "$vikit_commit"

if [[ ! -d "$third_party_dir/libtorch" ]]; then
  if [[ ! -f "$libtorch_zip" ]]; then
    curl --fail --location --retry 3 --output "$libtorch_zip" "$libtorch_url"
  fi
  unzip -q "$libtorch_zip" -d "$third_party_dir"
fi

sha256sum "$libtorch_zip" > "$libtorch_zip.sha256"
git -C "$vikit_dir" rev-parse HEAD
