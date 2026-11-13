#!/usr/bin/env bash
# Clone the backbone code at the commits used in the paper into third_party/.
#   bash scripts/setup_third_party.sh            # FastVGGT (VGGT implementation) and pi3
#   bash scripts/setup_third_party.sh all        # also Depth Anything 3, VGGT-Omega, and official VGGT (training losses)
#   bash scripts/setup_third_party.sh da3 vggt   # selected repositories
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p third_party

clone() {  # folder, url, commit
  if [ ! -d "third_party/$1/.git" ]; then
    git clone --quiet "$2" "third_party/$1"
  fi
  git -C "third_party/$1" fetch --quiet origin "$3" 2>/dev/null || true
  git -C "third_party/$1" checkout --quiet "$3"
  echo "third_party/$1 at $3"
}

want=("$@")
[ ${#want[@]} -eq 0 ] && want=(fastvggt pi3)
[ "${want[0]}" = "all" ] && want=(fastvggt pi3 da3 vggt-omega vggt)
for w in "${want[@]}"; do
  case "$w" in
    fastvggt) clone FastVGGT https://github.com/mystorm16/FastVGGT.git 6526e275a29572653a034762bb3c6c9ce280ff55 ;;
    pi3) clone Pi3 https://github.com/yyfz/Pi3.git 9fa3ddb3f8d53041f8b2738df404f62223bbaa7b ;;
    da3) clone Depth-Anything-3 https://github.com/ByteDance-Seed/Depth-Anything-3.git 3d835ec1a5802d64a8b8b15f817a1ab54809bfe4
         echo "Install its dependencies with: pip install -e third_party/Depth-Anything-3" ;;
    vggt-omega) clone vggt-omega https://github.com/facebookresearch/vggt-omega.git 48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6 ;;
    vggt) clone vggt https://github.com/facebookresearch/vggt.git a288dd0f14786c93483e45524328726ab7b1b4ce ;;
    *) echo "unknown repository: $w (choose from fastvggt, pi3, da3, vggt-omega, vggt, all)"; exit 1 ;;
  esac
done
