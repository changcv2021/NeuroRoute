#!/usr/bin/env bash
# Run only after authenticating GitHub CLI as the requested repository owner.
set -euo pipefail
release_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$release_root"
repository="changcv2021/NeuroRoute"
remote_url="https://github.com/${repository}.git"

if [[ "${1:-}" != "--publish" ]]; then
  echo "Prepared target: https://github.com/$repository (public)"
  echo "After gh auth login, run: bash tools/publish_github.sh --publish"
  exit 0
fi
command -v gh >/dev/null || { echo 'GitHub CLI (gh) is required.' >&2; exit 1; }
actual_user="$(gh api user --jq .login)"
[[ "$actual_user" == "changcv2021" ]] || {
  echo 'Authenticate GitHub CLI as changcv2021 before publishing.' >&2
  exit 1
}
"${NR_CHECK_PYTHON:-python3}" tools/check_release.py
[[ -z "$(git status --porcelain)" ]] || {
  echo 'Review and commit local changes before publishing.' >&2
  exit 1
}
[[ "$(git branch --show-current)" == main ]] || {
  echo 'Expected the prepared main branch.' >&2
  exit 1
}

if gh repo view "$repository" --json nameWithOwner,isPrivate >/dev/null 2>&1; then
  [[ "$(gh repo view "$repository" --json isPrivate --jq .isPrivate)" == false ]] || {
    echo 'An existing private repository will not be made public automatically.' >&2
    exit 1
  }
else
  gh repo create "$repository" --public \
    --description 'NeuroRoute / FlyCNS-v2: source code for communication-constrained Go1 locomotion'
fi

if git remote get-url origin >/dev/null 2>&1; then
  [[ "$(git remote get-url origin)" == "$remote_url" ]] || {
    echo 'Existing origin differs from the prepared repository; review it first.' >&2
    exit 1
  }
else
  git remote add origin "$remote_url"
fi

# Use the authenticated CLI without changing global Git configuration or storing
# a token in a URL. Normal push rejects divergent history; never force push.
git -c credential.helper= -c 'credential.helper=!gh auth git-credential' \
  push --set-upstream origin main
local_head="$(git rev-parse HEAD)"
remote_head="$(gh api "repos/$repository/commits/main" --jq .sha)"
[[ "$local_head" == "$remote_head" ]] || {
  echo 'Remote main does not match the local commit.' >&2
  exit 1
}
echo "Published and verified: https://github.com/$repository"
