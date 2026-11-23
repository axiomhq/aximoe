#!/usr/bin/env bash
# Publish one trained model to the Hugging Face repository axiomhq/aximoe as a single commit on
# main, tagged with the model's name: its weights as model.safetensors, its model.json pointing at
# them, the model card with its license and figures from hf/, and what runs the model: the package
# python/aximoe without its tests, pyproject.toml, uv.lock and the example series. Afterwards main
# holds exactly those files. A name the Hub already has as a tag is refused, so a tag always means
# the same weights; -r publishes such a name again when the weights at its tag equal these byte
# for byte, and moves the tag to the new commit.
#
#   scripts/hf-publish.sh aximoe-261115          publish data/models/aximoe-261115
#   scripts/hf-publish.sh path/to/run            publish a model directory, tagged with its name
#   scripts/hf-publish.sh -n aximoe-261115       prepare the files and run every check, upload nothing
#   scripts/hf-publish.sh -r aximoe-261115       publish again under a tag the Hub has, same weights
#
# Needs `hf auth login` with write access to the repository.
set -euo pipefail
HUB_REPO=${AXIMOE_HF_REPO:-axiomhq/aximoe}
REPO=$(cd "$(dirname "$0")/.." && pwd)
CARD=$REPO/hf
PY=${AXIMOE_PYTHON:-$REPO/.venv/bin/python}
die() { echo "hf-publish: $*" >&2; exit 1; }

DRY=0; AGAIN=0
while [ $# -gt 0 ]; do
  case $1 in
    -n) DRY=1; shift ;;
    -r) AGAIN=1; shift ;;
    *) break ;;
  esac
done
[ $# -eq 1 ] || { sed -n '2,15p' "$0" >&2; exit 2; }
if [ -d "$1" ]; then
  dir=$(cd "$1" && pwd); name=$(basename "$dir")
else
  name=$1; dir=$REPO/data/models/$name
fi
[ -f "$dir/student.pt" ] && [ -f "$dir/model.json" ] || die "$dir holds no student.pt and model.json"
case $name in (*[!A-Za-z0-9._-]* | '') die "$name is not usable as a tag name" ;; esac
command -v hf >/dev/null || die "the hf CLI is not installed"
[ -x "$PY" ] || die "no Python at $PY; set AXIMOE_PYTHON"
for f in README.md LICENSE; do [ -f "$CARD/$f" ] || die "$CARD/$f is missing"; done

# the tracked files that go beside the weights: the code with its project files and example, the card
CODE=(); CARDFILES=()
while IFS= read -r f; do CODE+=("$f"); done < <(git -C "$REPO" ls-files -- 'python/aximoe/*.py' pyproject.toml uv.lock examples \
  | grep -v -E '^python/aximoe/test_[^/]*\.py$')
while IFS= read -r f; do CARDFILES+=("$f"); done < <(git -C "$REPO" ls-files -- hf/README.md hf/LICENSE 'hf/images/*.png')
[ ${#CODE[@]} -gt 0 ] && [ ${#CARDFILES[@]} -gt 0 ] || die "git lists no files to publish in $REPO"
# the Hub holds committed files, so the repository's history tells what each release shipped
changed=$(git -C "$REPO" diff --name-only HEAD -- "${CODE[@]}" "${CARDFILES[@]}")
[ -z "$changed" ] || die "files to publish have uncommitted changes: $(echo "$changed" | tr '\n' ' ')"

hf auth whoami >/dev/null || die "not logged in; run hf auth login"
tags=$(hf repos tag list "$HUB_REPO" --format json)
# the commit the Hub's tag of this name points at, empty without such a tag
old=$("$PY" -c 'import json, sys
for t in json.loads(sys.argv[1]):
    if isinstance(t, dict) and t.get("name") == sys.argv[2]:
        print(t.get("target_commit") or "unknown")
    elif t == sys.argv[2]:
        print("unknown")' "$tags" "$name")
if [ -n "$old" ]; then
  [ "$AGAIN" = 1 ] || die "$HUB_REPO already has the tag $name; -r publishes it again if the weights are the same"
else
  [ "$AGAIN" = 0 ] || die "$HUB_REPO has no tag $name to publish again"
fi

stage=$(mktemp -d); check=$(mktemp -d); trap 'rm -rf "${stage:?}" "${check:?}"' EXIT
for f in "${CODE[@]}"; do mkdir -p "$stage/$(dirname "$f")"; cp "$REPO/$f" "$stage/$f"; done
for f in "${CARDFILES[@]}"; do t=${f#hf/}; mkdir -p "$stage/$(dirname "$t")"; cp "$REPO/$f" "$stage/$t"; done

# weights and description beside the code, converted and read back by the code that is published
# the conversion runs at the model's patch width
width=$("$PY" -c 'import json, sys; print(json.load(open(sys.argv[1])).get("patch", 32))' "$dir/model.json")
AXIMOE_PATCH=$width "$PY" - "$dir" "$stage" <<'PY'
import json, os, sys
import torch
from safetensors.torch import save_file
src, out = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.join(out, "python"))
from aximoe import student
if not os.path.realpath(student.__file__).startswith(os.path.realpath(out) + os.sep):
    sys.exit(f"the conversion imported {student.__file__}, not the files to publish")
description, model = student.load(src)
state = {k: v.contiguous() for k, v in model.state_dict().items()}
save_file(state, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
description["weights"] = "model.safetensors"
with open(os.path.join(out, "model.json"), "w") as f:
    json.dump(description, f, indent=2)
    f.write("\n")
_, back = student.load(out)
got = back.state_dict()
bad = [k for k in state if k not in got or not torch.equal(state[k], got[k])]
if got.keys() != state.keys() or bad:
    sys.exit(f"the converted weights differ from {src}: {bad[:5]}")
print(f"{len(state)} tensors converted, equal to {src}")
PY

# the card's command on the files as they are published
if ! (cd "$stage" && "$PY" python/aximoe/forecast.py --model . --in examples/series.f32 --out "$check/bands.f32" \
    --resolution 60 --tag auto --length 2016 >"$check/forecast.log" 2>&1); then
  tail -5 "$check/forecast.log" >&2
  die "the forecaster to publish fails on the example to publish"
fi
[ -s "$check/bands.f32" ] || die "the forecaster to publish wrote no bands"
echo "the card's forecast runs on these files: $(wc -c <"$check/bands.f32") bytes of bands"
find "$stage" -name __pycache__ -prune -exec rm -rf {} +
expected=$(printf '%s\n' "${CODE[@]}" "${CARDFILES[@]#hf/}" model.safetensors model.json | sort)
staged=$(cd "$stage" && find . -type f | sed 's|^\./||' | sort)
[ "$expected" = "$staged" ] || die "the files to publish are not the listed ones: $(diff <(echo "$expected") <(echo "$staged") | grep '^[<>]' | tr '\n' ' ')"

if [ "$AGAIN" = 1 ]; then
  hf download "$HUB_REPO" model.safetensors --revision "$name" --local-dir "$check/tagged" --quiet >/dev/null
  cmp -s "$check/tagged/model.safetensors" "$stage/model.safetensors" \
    || die "the weights at tag $name differ from those of $dir; a tag keeps its weights"
  echo "the weights at tag $name equal these byte for byte"
fi

echo "to $HUB_REPO main, tagged $name:"
(cd "$stage" && find . -type f | sed 's|^\./||' | sort | xargs ls -l | awk '{printf "  %10d  %s\n", $5, $NF}')
if [ "$DRY" = 1 ]; then echo "dry run: nothing uploaded"; exit 0; fi

before=$(hf models info "$HUB_REPO" --format json | "$PY" -c 'import json, sys; print(json.load(sys.stdin)["sha"])')
hf upload "$HUB_REPO" "$stage" . --delete '*' \
  --commit-message "Publish $name" \
  --commit-description "From $(git -C "$REPO" rev-parse --short HEAD) of github.com/axiomhq/aximoe, model directory $(basename "$dir")."
after=$(hf models info "$HUB_REPO" --format json | "$PY" -c 'import json, sys; print(json.load(sys.stdin)["sha"])')
[ "$after" != "$before" ] || die "the upload made no commit; main still is $before"
if [ "$AGAIN" = 1 ]; then
  hf repos tag delete "$HUB_REPO" "$name" --yes
  hf repos tag create "$HUB_REPO" "$name" --revision "$after" -m "Aximö $name" \
    || die "the tag $name is deleted and could not be set to $after; it was at $old: hf repos tag create $HUB_REPO $name --revision $old"
else
  hf repos tag create "$HUB_REPO" "$name" --revision "$after" -m "Aximö $name"
fi

# the tag holds these weights: fetch them back at the tag and compare byte for byte
hf download "$HUB_REPO" model.safetensors --revision "$name" --local-dir "$check/published" --quiet >/dev/null
cmp -s "$check/published/model.safetensors" "$stage/model.safetensors" || die "the weights at tag $name differ from the converted ones"
echo "published $name to https://huggingface.co/$HUB_REPO/tree/$name (commit $after)"
