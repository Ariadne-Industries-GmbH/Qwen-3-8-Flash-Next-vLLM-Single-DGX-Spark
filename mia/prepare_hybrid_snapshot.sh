#!/usr/bin/env bash
# Container-side driver for `./scripts/qwen38f-vllm.sh prepare-hybrid`.
# Builds an FP8-hybrid snapshot from the local RadixArk checkpoint:
#
#   1) hardlink clone of the checkpoint  (cp -al: instant, 0 bytes of extra
#      space -- the 419 file names point at the same inodes)
#   2) pull model.safetensors.index.json OF THE CLONE onto its own inode
#   3) run fp8_convert.py over the clone (shutil.move of the 4 affected
#      shards -> .bf16.bak, the new fp8 shards as real files)
#   4) prove that the ORIGINAL is untouched
#
# WHY HARDLINKS INSTEAD OF SYMLINKS (deviation from blazux prepare-hybrid.sh):
# The template builds the snapshot as a symlink clone of an HF snapshot whose
# entries are themselves already relative symlinks onto ../../blobs/<sha> -- so
# that works within an HF cache hierarchy. Our checkpoint
# (Qwen3.8-Flash-Next-NVFP4-RadixArk) is real files in a flat
# directory. Relative symlinks would only work if the serve
# container mounted the same parent directory AND addressed the model via
# the sub-path; the hybrid service mounts only the snapshot
# directory. Hardlinks make the snapshot autonomous: a real
# directory, no resolution, no dependence on the original
# staying where it is -- and only ~7.7 GiB of new bytes (4 fp8 shards + index).
#
# THE ONE DANGER (hence step 2 + 4, and hence this script is
# anchor-pinned instead of inline in the docker run): fp8_convert.py writes the index IN
# PLACE (`json.dump(idx, open(idx_path, "w"))`). As long as the snapshot entry
# is a hardlink onto the original index, that is the same inode -> the
# converter would bend the index OF THE ORIGINAL CHECKPOINT to fp8 and make
# the running k3/mia setup unusable. The shards themselves are harmless
# (shutil.move + save_file = new inode). Hence: decouple the index, and after
# the run actively check that the original contains no weight_scale_inv.
#
# NO chmod on *.safetensors (which the template does): with hardlinks a
# chmod would also change the permissions of the original files. The container
# runs with --user of the host user, so new files inherit 644/owner anyway.
#
# Invocation (container view, from prepare_hybrid in scripts/qwen38f-vllm.sh):
#   bash /tools/prepare_hybrid_snapshot.sh <SRC> <DST>
# Environment overrides (for the local self-test without torch/safetensors):
#   CONV=<path>   converter (default /tools/fp8_convert.py)
#   PY=<path>     Python  (default python3)
set -euo pipefail

SRC="${1:?Usage: prepare_hybrid_snapshot.sh <SRC> <DST>}"
DST="${2:?Usage: prepare_hybrid_snapshot.sh <SRC> <DST>}"
CONV="${CONV:-/tools/fp8_convert.py}"
PY="${PY:-python3}"
INDEX="model.safetensors.index.json"

die() { echo "!! $*" >&2; exit 1; }

[[ -f "${SRC}/${INDEX}" ]] || die "SRC has no ${INDEX}: ${SRC}"
[[ -d "${SRC}" ]] || die "SRC is not a directory: ${SRC}"
[[ -f "${CONV}" ]] || die "converter missing: ${CONV}"

# The target must not contain anything we did not create ourselves. No rm -rf
# on foreign data: better abort and show the user the command.
if [[ -e "${DST}" ]] && [[ -n "$(ls -A "${DST}" 2>/dev/null)" ]]; then
  die "target exists and is not empty: ${DST}
   If an earlier attempt aborted:  rm -rf '${DST}'   then retry.
   If a checkpoint already lives here: check the target path (QFN_HYBRID_MODEL_DIR)."
fi

# Hardlinks need the same filesystem as the source.
mkdir -p "${DST}"
src_dev="$(stat -c %d "${SRC}")"
dst_dev="$(stat -c %d "${DST}")"
[[ "${src_dev}" == "${dst_dev}" ]] ||
  die "SRC (dev ${src_dev}) and DST (dev ${dst_dev}) are on different filesystems -- hardlinks work, but the snapshot would hang off SRC. Put the target next to the checkpoint (QFN_HYBRID_MODEL_DIR)."

# Record the before/after state of the original for the integrity check:
# exactly the shards the converter touches, plus original ownership.
SRC_SHARDS="$(
  PY="${PY}" SRC="${SRC}" INDEX="${INDEX}" "${PY}" - <<'PYEOF'
import json, os, re
src, idx = os.environ["SRC"], os.environ["INDEX"]
T = re.compile(
    r"model\.language_model\.layers\.\d+\.("
    r"linear_attn\.(in_proj_qkv|in_proj_z|out_proj)"
    r"|self_attn\.(q_proj|k_proj|v_proj|o_proj)"
    r"|mlp\.shared_expert\.(gate_proj|up_proj|down_proj)"
    r")\.weight$")
wm = json.load(open(f"{src}/{idx}"))["weight_map"]
for f in sorted({v for k, v in wm.items() if T.search(k)}):
    print(f)
PYEOF
)"
[[ -n "${SRC_SHARDS}" ]] || die "no fp8 target tensors in the index of ${SRC} -- wrong checkpoint?"
echo "Affected shards (become fp8 in the snapshot, unchanged in the original):" >&2
while IFS= read -r f; do
  [[ -n "${f}" ]] || continue
  sz="$(stat -c %s "${SRC}/${f}" 2>/dev/null || echo 0)"
  echo "  ${f}  ($(( sz / 1073741824 )) GiB)" >&2
done <<< "${SRC_SHARDS}"

echo "Hardlink clone: ${SRC} -> ${DST}" >&2
cp -al "${SRC}/." "${DST}/"

# --- Step 2: decouple the index (see header) --------------------------------
cp --remove-destination "${SRC}/${INDEX}" "${DST}/${INDEX}"
[[ "$(stat -c %i "${SRC}/${INDEX}")" != "$(stat -c %i "${DST}/${INDEX}")" ]] ||
  die "the index could not be decoupled from the original (same inode) -- aborting before anything is written."

# --- Step 3: conversion (CPU only, keeps one shard fully in RAM) -----------
echo "Converting side layers to blockwise FP8 (${CONV}) ..." >&2
"${PY}" "${CONV}" "${DST}"

# --- Step 4: prove original integrity --------------------------------------
SRC="${SRC}" DST="${DST}" INDEX="${INDEX}" SHARDS="${SRC_SHARDS}" "${PY}" - <<'PYEOF'
import json, os, sys
src, dst, idx = (os.environ[x] for x in ("SRC", "DST", "INDEX"))
shards = [s for s in os.environ["SHARDS"].split("\n") if s]
d = json.load(open(f"{dst}/{idx}"))["weight_map"]
s = json.load(open(f"{src}/{idx}"))["weight_map"]
fp8_new = sum(1 for k in d if k.endswith(".weight_scale_inv"))
fp8_src = sum(1 for k in s if k.endswith(".weight_scale_inv"))
missing = [f for f in shards if not os.path.isfile(f"{src}/{f}")]
print(f"snapshot: {fp8_new} weight_scale_inv tensors, {len(shards)} shards converted")
if fp8_new == 0:
    sys.exit("!! conversion produced no fp8 tensors")
if fp8_src:
    sys.exit(f"!! ORIGINAL index contains {fp8_src} weight_scale_inv tensors -- "
             f"the converter wrote into the original. Restore {src}/{idx}!")
if missing:
    sys.exit(f"!! original shards missing: {missing}")
for f in shards:
    a, b = os.path.getsize(f"{src}/{f}"), os.path.getsize(f"{dst}/{f}")
    print(f"  {f}: original {a/2**30:.2f} GiB -> fp8 {b/2**30:.2f} GiB ({b/a:.0%})")
PYEOF

touch "${DST}/.prepared"
echo "Hybrid snapshot prepared: ${DST}" >&2
