#!/usr/bin/env bash
set -euo pipefail

die() {
    echo "error: $*" >&2
    exit 1
}

usage() {
    cat <<'EOF'
Usage:
  ./bump_and_publish.sh <crate> <version> [--publish] [--dry-run]

Arguments:
  <crate>    Crate to release: af-anndata or convert-af
  <version>  New version (X.Y.Z format)

Options:
  --publish  Publish to crates.io after bumping and committing
  --dry-run  Show what would be done without modifying anything
  -h, --help Show this help message

Examples:
  ./bump_and_publish.sh af-anndata 0.3.4 --publish
  ./bump_and_publish.sh convert-af 0.2.4 --publish
  ./bump_and_publish.sh af-anndata 0.3.4 --dry-run
EOF
}

print_cmd() {
    printf '+'
    printf ' %q' "$@"
    printf '\n'
}

run() {
    print_cmd "$@"
    if [[ "$DRY_RUN" == true ]]; then
        return 0
    fi
    "$@"
}

replace_first_version() {
    local file="$1"
    local version="$2"
    if [[ "$DRY_RUN" == false ]]; then
        sed -i.bak "1,/^version = /s/^version = \".*\"/version = \"${version}\"/" "$file"
        rm -f "${file}.bak"
    fi
}

replace_dep_version() {
    local file="$1"
    local dep="$2"
    local version="$3"
    if [[ "$DRY_RUN" == false ]]; then
        python3 - "$file" "$dep" "$version" <<'PY'
import pathlib
import re
import sys

file_path = pathlib.Path(sys.argv[1])
dep = sys.argv[2]
version = sys.argv[3]
text = file_path.read_text()
pattern = rf'^({re.escape(dep)}\s*=\s*\{{[^\n]*\bversion\s*=\s*")[^"]+(")'
updated, count = re.subn(pattern, rf'\g<1>{version}\2', text, count=1, flags=re.MULTILINE)
if count != 1:
    raise SystemExit(f"error: failed to update dependency version for {dep} in {file_path}")
file_path.write_text(updated)
PY
    fi
}

CRATE=""
VERSION=""
PUBLISH=false
DRY_RUN=false
declare -a EXTRA_FILES=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --publish)
            PUBLISH=true
            ;;
        --dry-run)
            DRY_RUN=true
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            die "unknown option: $1"
            ;;
        *)
            if [[ -z "$CRATE" ]]; then
                CRATE="$1"
            elif [[ -z "$VERSION" ]]; then
                VERSION="$1"
            else
                die "too many positional arguments"
            fi
            ;;
    esac
    shift
done

[[ -n "$CRATE" ]] || { usage; exit 1; }
[[ -n "$VERSION" ]] || { usage; exit 1; }

if ! [[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+([+-][0-9A-Za-z.-]+)*$ ]]; then
    die "version must look like X.Y.Z, optionally with prerelease/build suffixes"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

case "$CRATE" in
    af-anndata)
        CARGO_TOML="af-anndata/Cargo.toml"
        TAG="${CRATE}-v${VERSION}"
        PUBLISH_ARGS="-p af-anndata"
        EXTRA_FILES=("convert-af/Cargo.toml")
        ;;
    convert-af)
        CARGO_TOML="convert-af/Cargo.toml"
        TAG="${CRATE}-v${VERSION}"
        PUBLISH_ARGS="-p convert-af"
        EXTRA_FILES=()
        ;;
    *)
        die "unknown crate: $CRATE (expected: af-anndata or convert-af)"
        ;;
esac

LOCKFILE="Cargo.lock"

[[ -f "$CARGO_TOML" ]] || die "not found: $CARGO_TOML"
[[ -f "$LOCKFILE" ]] || die "not found: $LOCKFILE"

CURRENT_VERSION="$(sed -n 's/^version = "\(.*\)"/\1/p' "$CARGO_TOML" | head -1)"
[[ -n "$CURRENT_VERSION" ]] || die "could not determine current crate version from $CARGO_TOML"

if [[ "$CURRENT_VERSION" == "$VERSION" ]]; then
    die "crate version is already $VERSION"
fi

if git rev-parse "$TAG" >/dev/null 2>&1; then
    die "tag $TAG already exists"
fi

if [[ -n "$(git status --porcelain)" ]]; then
    die "working tree is not clean; commit or stash existing changes first"
fi

echo "Crate                 : $CRATE"
echo "Cargo.toml            : $CARGO_TOML"
echo "Current version       : $CURRENT_VERSION"
echo "New version           : $VERSION"
echo "Tag                   : $TAG"
if [[ "$PUBLISH" == true ]]; then
    echo "Publish crate         : yes"
else
    echo "Publish crate         : no"
fi
if [[ "$DRY_RUN" == true ]]; then
    echo "Dry-run               : yes"
else
    echo "Dry-run               : no"
fi
echo

echo "Updating $CARGO_TOML"
echo "  version: $CURRENT_VERSION -> $VERSION"
replace_first_version "$CARGO_TOML" "$VERSION"

if [[ "$CRATE" == "af-anndata" ]]; then
    echo "Updating convert-af/Cargo.toml"
    echo "  af-anndata dependency version: $CURRENT_VERSION -> $VERSION"
    replace_dep_version "convert-af/Cargo.toml" "af-anndata" "$VERSION"
fi

UPDATED_VERSION="$(sed -n 's/^version = "\(.*\)"/\1/p' "$CARGO_TOML" | head -1)"
if [[ "$DRY_RUN" == false ]]; then
    [[ "$UPDATED_VERSION" == "$VERSION" ]] || die "crate version update failed"
else
    echo "Dry-run: would rewrite manifests and refresh $LOCKFILE"
fi

run cargo check -q
if [[ ${#EXTRA_FILES[@]} -gt 0 ]]; then
    run git add "$CARGO_TOML" "${EXTRA_FILES[@]}"
else
    run git add "$CARGO_TOML"
fi
run git add -f "$LOCKFILE"
run git commit -m "chore(release): bump ${CRATE} to v${VERSION}"

if [[ "$PUBLISH" == true ]]; then
    run cargo publish $PUBLISH_ARGS --allow-dirty
fi

run git tag -a "$TAG" -m "${CRATE} v${VERSION}"
run git push origin HEAD
run git push origin "$TAG"

if [[ "$DRY_RUN" == true ]]; then
    echo
    echo "Dry-run complete"
else
    echo
    echo "Release bump complete for ${CRATE} v${VERSION}"
fi
