#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"

if [[ -f "${PROJECT_DIR}/.env" ]]; then
  # .env provides defaults. Variables already set in the shell environment
  # keep precedence, so launch-time overrides continue to work
  # (e.g. QFN_KV_CACHE_MEMORY_BYTES=18253611008 ./scripts/qwen38f-vllm.sh).
  while IFS= read -r env_line || [[ -n "${env_line}" ]]; do
    [[ -z "${env_line}" || "${env_line}" == \#* ]] && continue
    env_key="${env_line%%=*}"
    [[ -z "${!env_key+x}" ]] && export "${env_line}"
  done < "${PROJECT_DIR}/.env"
fi

resolve_project_path() {
  local path="$1"
  if [[ "${path}" = /* ]]; then
    printf '%s\n' "${path}"
  else
    printf '%s/%s\n' "${PROJECT_DIR}" "${path}"
  fi
}

export MODEL_DIR="$(resolve_project_path "${MODEL_DIR:-${WORKSPACE_DIR}/Qwen3.8-Flash-Next-NVFP4-RadixArk}")"
export HF_CACHE_DIR="$(resolve_project_path "${HF_CACHE_DIR:-${HOME}/.cache/huggingface}")"
export VLLM_CACHE_DIR="$(resolve_project_path "${VLLM_CACHE_DIR:-${HOME}/.cache/vllm}")"
export TRITON_CACHE_DIR="$(resolve_project_path "${TRITON_CACHE_DIR:-${HOME}/.triton}")"
# Patched Flash-Next image (build chain: official Qwen3.8-Flash-Next image
# @ digest + blazux/qwen3.8-Flash-DGX @ BUILD_COMMIT; see .env/README).
export VLLM_IMAGE="${VLLM_IMAGE:-qwen38-flash-dgx:656cdb0}"
# Mia profile image = VLLM_IMAGE + reduced-vocab MTP patch layer (mia/Dockerfile),
# built via ./scripts/qwen38f-vllm.sh build-image-mia.
export MIA_IMAGE="${MIA_IMAGE:-qwen38-flash-mia:2}"
# Hardlink snapshot for the hybrid profile (created by `prepare-hybrid`;
# default = sibling directory of the local checkpoint, same parent folder --
# prepare_hybrid enforces this because the container mounts the parent directory).
export QFN_HYBRID_MODEL_DIR="$(resolve_project_path "${QFN_HYBRID_MODEL_DIR:-${MODEL_DIR}-fp8hybrid}")"
# Draft vocab for the Mia/hybrid profiles. The default artifact is the seed vocab
# shipped in this repo, built exclusively from public sources
# (mia/draft_vocab_public.txt; provenance record: <name>.provenance.json).
# Alternative vocabularies require an explicit DRAFT_VOCAB_FILE override.
# DRAFT_VOCAB_PROJECT points at the tool set used for the build (here in the repo).
export DRAFT_VOCAB_PROJECT="${DRAFT_VOCAB_PROJECT:-${PROJECT_DIR}/tools/draft-vocab}"
export DRAFT_VOCAB_OUT="${DRAFT_VOCAB_OUT:-${PROJECT_DIR}/mia/draft_vocab.txt}"
export DRAFT_VOCAB_PUBLIC="${DRAFT_VOCAB_PUBLIC:-${PROJECT_DIR}/mia/draft_vocab_public.txt}"
export VLLM_PORT="${VLLM_PORT:-44415}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.8-flash-next}"
# Build repo + commit for `build-image` (pins verified 07.09.2026).
export BUILD_REPO="${BUILD_REPO:-blazux/qwen3.8-Flash-DGX}"
export BUILD_COMMIT="${BUILD_COMMIT:-656cdb075b57671c9bd6807b57f630fb3c1641fe}"
export CURL_MAX_TIME="${CURL_MAX_TIME:-300}"

# Profiles: k3 (default; MTP3, 4x180k, FP8-KV) and mia (k3 + reduced MTP draft
# vocab + explicit CUDA graph widths + V2 runner; never parallel with k3).
export QFN_PROFILE="${QFN_PROFILE:-k3}"
RESTART_OVERRIDE=""
POSITIONAL_ARGS=()

while (($#)); do
  case "$1" in
    --no-restart)
      RESTART_OVERRIDE="no"
      shift
      ;;
    --restart-policy)
      if (($# < 2)); then
        echo "--restart-policy requires a value." >&2
        exit 1
      fi
      RESTART_OVERRIDE="$2"
      shift 2
      ;;
    --mtp)
      if (($# < 2)); then
        echo "--mtp requires a value (e.g. 2 or 3)." >&2
        exit 1
      fi
      QFN_MTP_NUM_SPECULATIVE_TOKENS="$2"
      shift 2
      ;;
    *)
      POSITIONAL_ARGS+=("$1")
      shift
      ;;
  esac
done

set -- "${POSITIONAL_ARGS[@]}"

if [[ -n "${RESTART_OVERRIDE}" ]]; then
  export RESTART_POLICY="${RESTART_OVERRIDE}"
fi

compose() {
  docker compose --project-directory "${PROJECT_DIR}" -f "${PROJECT_DIR}/docker-compose.yml" "$@"
}

usage() {
  cat <<USAGE
Usage:
  $0 [--no-restart] [--mtp <k>] [profile]     Launch (profiles: k3 | mia | hybrid)
  $0 logs [profile]
  $0 kv [profile]
  $0 test
  $0 build-image
  $0 build-image-mia
  $0 vocab-public     public sources only (seed vocab for this repo)
  $0 draft-vocab      + own model outputs (stays local, gitignored)
  $0 prepare-hybrid
  $0 dropcaches
  $0 disable-restart
  $0 down
  $0 ps [profile]
  scripts/host-tuning.sh status|apply|revert   (optional, host RAM tuning, sudo)

Profiles:
  k3     Default: native MTP3 (no external drafter), 4 parallel requests x
         max. 180k context (KV budget 4 x 190k), FP8-KV, BF16 GDN/Mamba state.
  mia    k3 + MiaAI-Lab optimisation: reduced MTP draft vocab (65.5k IDs, drafter
         reads 0.34 instead of 1.27 GB per draft step), explicit CUDA graph capture
         widths [4,8,12,16], V2 model runner. Requires MIA_IMAGE (build-image-mia)
         and mia/draft_vocab_public.txt. Never parallel with k3.
         Details: mia/README.md.
  hybrid mia + blazux FP8 hybrid checkpoint: ~15 GiB BF16 side layers converted
         locally to block-wise FP8 (NVFP4 experts unchanged) ->
         ~7 GiB fewer resident weights, ~+20 % decode (blazux reference).
         Requires prepare-hybrid (one-time) + MIA_IMAGE + draft vocab like mia.
         VLLM_FP8_HYBRID=1 + VLLM_USE_DEEP_GEMM=0 are set in the profile.
         Never parallel with k3/mia. Details: docs/hybrid-profile.md.
Options:
  --no-restart              override restart policy for this launch with restart=no
  --restart-policy <value>  explicit override, e.g. unless-stopped
  --mtp <k>                 override the speculative token count (default 3; K sweep:
                            K=3 optimal at every concurrency)

Environment overrides:
  MODEL_DIR=${MODEL_DIR}
  VLLM_IMAGE=${VLLM_IMAGE}
  MIA_IMAGE=${MIA_IMAGE}
  QFN_HYBRID_MODEL_DIR=${QFN_HYBRID_MODEL_DIR}
  QFN_PLE_MMAP_MADV_RANDOM=${QFN_PLE_MMAP_MADV_RANDOM:-1}
  DRAFT_VOCAB_FILE=${DRAFT_VOCAB_FILE:-./mia/draft_vocab_public.txt}
  DRAFT_VOCAB_PUBLIC=${DRAFT_VOCAB_PUBLIC}
  DRAFT_VOCAB_OUT=${DRAFT_VOCAB_OUT}
  DRAFT_VOCAB_PROJECT=${DRAFT_VOCAB_PROJECT}
  DRAFT_VOCAB_CAPTURE_IN=<export.jsonl|folder>   (draft-vocab only, optional)
  DRAFT_VOCAB_ENGLISH_WEIGHT=${DRAFT_VOCAB_ENGLISH_WEIGHT:-1}
  DRAFT_VOCAB_GERMAN_WEIGHT=${DRAFT_VOCAB_GERMAN_WEIGHT:-3}
  DRAFT_VOCAB_CODE_WEIGHT=${DRAFT_VOCAB_CODE_WEIGHT:-3}
  DRAFT_VOCAB_MODEL_OUT_WEIGHT=${DRAFT_VOCAB_MODEL_OUT_WEIGHT:-20}
  DRAFT_VOCAB_SIZE=${DRAFT_VOCAB_SIZE:-65536}
  VLLM_PORT=${VLLM_PORT}
  SERVED_MODEL_NAME=${SERVED_MODEL_NAME}
  BUILD_REPO=${BUILD_REPO}
  BUILD_COMMIT=${BUILD_COMMIT}
  QFN_MAX_MODEL_LEN=${QFN_MAX_MODEL_LEN:-180000}
  QFN_MAX_NUM_SEQS=${QFN_MAX_NUM_SEQS:-4}
  QFN_MTP_NUM_SPECULATIVE_TOKENS=${QFN_MTP_NUM_SPECULATIVE_TOKENS:-3}
  QFN_KV_CACHE_DTYPE=${QFN_KV_CACHE_DTYPE:-fp8_e4m3}
  QFN_MAMBA_SSM_CACHE_DTYPE=${QFN_MAMBA_SSM_CACHE_DTYPE:-bfloat16}
  QFN_KV_CACHE_MEMORY_BYTES=${QFN_KV_CACHE_MEMORY_BYTES:-12459440000}
  QFN_GPU_MEMORY_UTILIZATION=${QFN_GPU_MEMORY_UTILIZATION:-0.80}
  QFN_MAX_NUM_BATCHED_TOKENS=${QFN_MAX_NUM_BATCHED_TOKENS:-8192}
  QFN_PLE_MMAP_WORKERS=${QFN_PLE_MMAP_WORKERS:-32}
  QFN_PLE_MMAP_PREWARM=${QFN_PLE_MMAP_PREWARM:-0}
  QFN_QSA_DET_TOPK=${QFN_QSA_DET_TOPK:-1}
  QFN_QSA_EXACT_TOPK=${QFN_QSA_EXACT_TOPK:-0}
  HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
  CURL_MAX_TIME=${CURL_MAX_TIME}
USAGE
}

ensure_model() {
  if [[ ! -f "${MODEL_DIR}/config.json" || ! -f "${MODEL_DIR}/model.safetensors.index.json" ]]; then
    echo "Model directory is incomplete or missing: ${MODEL_DIR}" >&2
    echo "Expected local Hugging Face checkpoint RadixArk/Qwen3.8-Flash-Next-NVFP4." >&2
    exit 1
  fi
  local ple_shards
  ple_shards="$(find "${MODEL_DIR}" -maxdepth 1 -name 'model-plefp8-*.safetensors' 2>/dev/null | wc -l | tr -d ' ')"
  if [[ "${ple_shards}" -lt 1 ]]; then
    echo "PLE shards (model-plefp8-*.safetensors) missing in ${MODEL_DIR} -- without them" >&2
    echo "the PLE-mmap path cannot start." >&2
    exit 1
  fi
  echo "Model check: OK (${ple_shards} PLE shards)."
}

ensure_image() {
  local image="${1:-${VLLM_IMAGE}}"
  if docker image inspect "${image}" >/dev/null 2>&1; then
    return
  fi
  case "${image}" in
    "${MIA_IMAGE}")
      echo "Mia image ${image} missing. Build it with (needs ${VLLM_IMAGE} as base):" >&2
      echo "  ./scripts/qwen38f-vllm.sh build-image-mia" >&2
      exit 1
      ;;
    *)
      echo "Docker image ${image} not found locally. Build it with:" >&2
      echo "  ./scripts/qwen38f-vllm.sh build-image" >&2
      exit 1
      ;;
  esac
}

# Mia launch prerequisites: the draft vocab file must exist and hold the
# expected line count. The shipped public vocabulary is the default.
ensure_draft_vocab() {
  local file="$(resolve_project_path "${DRAFT_VOCAB_FILE:-./mia/draft_vocab_public.txt}")"
  local public="$(resolve_project_path "${DRAFT_VOCAB_PUBLIC:-./mia/draft_vocab_public.txt}")"
  if [[ ! -s "${file}" ]]; then
    if [[ -s "${public}" ]]; then
      echo "Configured draft vocab missing (${file}) -- using the public seed vocab:" >&2
      echo "  ${public}" >&2
      file="${public}"
    else
      echo "No draft vocab found." >&2
      echo "  checked: ${file}" >&2
      echo "  checked: ${public}" >&2
      echo "Build it:" >&2
      echo "  ./scripts/qwen38f-vllm.sh vocab-public   public sources only (into the repo)" >&2
      echo "  ./scripts/qwen38f-vllm.sh draft-vocab    public sources + own outputs" >&2
      exit 1
    fi
  fi
  # Compose binds the mount to this variable -- without the export the
  # container would mount the compose default instead of the vocab actually chosen.
  export DRAFT_VOCAB_FILE="${file}"
  local lines
  lines="$(wc -l < "${file}")"
  echo "Draft vocab: ${lines} lines (expected ${DRAFT_VOCAB_SIZE:-65536}) -> ${file}" >&2
  if [[ "${lines}" -ne "${DRAFT_VOCAB_SIZE:-65536}" ]]; then
    echo "WARNING: line count != DRAFT_VOCAB_SIZE -- rebuild the vocab or" >&2
    echo "         adjust DRAFT_VOCAB_SIZE, otherwise the patch crashes at launch." >&2
  fi
}

meminfo_gib() {
  awk -v k="$1" '$1 == k {printf "%.1f", $2/1048576}' /proc/meminfo
}

drop_caches() {
  echo "Before:  MemAvailable=$(meminfo_gib "MemAvailable:") GiB | Inactive(file)=$(meminfo_gib "Inactive(file):") GiB"
  local ok=0
  if sync && (echo 3 > /proc/sys/vm/drop_caches) 2>/dev/null; then
    ok=1
  elif command -v sudo >/dev/null 2>&1; then
    if sudo -n true >/dev/null 2>&1; then
      if sudo -n sync && sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null; then
        ok=1
      fi
    fi
  fi
  if [[ "${ok}" -eq 1 ]]; then
    echo "After:   MemAvailable=$(meminfo_gib "MemAvailable:") GiB | Inactive(file)=$(meminfo_gib "Inactive(file):") GiB"
    echo "Page cache (weight-file residue) evicted."
  else
    echo "WARNING: /proc/sys/vm/drop_caches is not writable without root." >&2
    echo "Manual command on the host (needs sudo):" >&2
    echo "  sudo sh -c 'sync && echo 3 > /proc/sys/vm/drop_caches'" >&2
    echo "Note: on Flash-Next, dropcaches clears ONLY the 126 GB weight residue"
    echo "before the pool check. IMPORTANT: never evict the PLE page cache (48 GB,"
    echo "needed at runtime) once the server has started."
  fi
}

build_image() {
  local repo_dir="${WORKSPACE_DIR}/qwen3.8-Flash-DGX"
  if [[ -d "${repo_dir}/.git" ]]; then
    echo "Build repo present: ${repo_dir}" >&2
  else
    echo "Cloning ${BUILD_REPO} into ${repo_dir} ..." >&2
    git clone "https://github.com/${BUILD_REPO}.git" "${repo_dir}" >&2
  fi
  echo "Checkout ${BUILD_COMMIT:0:12} ..." >&2
  git -C "${repo_dir}" checkout "${BUILD_COMMIT}" >&2
  local current
  current="$(git -C "${repo_dir}" rev-parse HEAD)"
  if [[ "${current}" != "${BUILD_COMMIT}" ]]; then
    echo "HEAD(${current}) != BUILD_COMMIT(${BUILD_COMMIT})" >&2
    exit 1
  fi
  echo "Building ${VLLM_IMAGE} (base image pulled from the Dockerfile; ~10.5 GB +" >&2
  echo "patches incl. nvcc build of the det top-k kernel; takes several minutes depending on cache) ..." >&2
  docker build -t "${VLLM_IMAGE}" "${repo_dir}"
  echo "Image built: ${VLLM_IMAGE}" >&2
}

build_image_mia() {
  if ! docker image inspect "${VLLM_IMAGE}" >/dev/null 2>&1; then
    echo "Base image ${VLLM_IMAGE} missing. Build it first: ./scripts/qwen38f-vllm.sh build-image" >&2
    exit 1
  fi
  local work="${PROJECT_DIR}/mia"
  # INSTALLED locations inside the image (dist-packages) -- not the GitHub repo
  # layout: vLLM imports exactly these files at runtime, and the patch must be
  # validated against them. Paths inside the image (qwen38-flash-dgx:656cdb0):
  #   /usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/mtp.py
  #   /usr/local/lib/python3.12/dist-packages/vllm_ple_mmap.py
  local mtp_path="usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/mtp.py"
  local ple_path="usr/local/lib/python3.12/dist-packages/vllm_ple_mmap.py"
  # MiaAI reference flow: extract the unpatched files FROM the image, then
  # patch them (both patches verify each anchor exactly once, abort on drift).
  # Build intermediates stay gitignored (mia/.gitignore).
  echo "Extracting UNPATCHED mtp.py + vllm_ple_mmap.py from ${VLLM_IMAGE} ..." >&2
  # docker create starts NO server (no CUDA, no weights -- layer extraction
  # only); the throwaway container is removed immediately.
  local cid rc=0
  cid="$(docker create "${VLLM_IMAGE}" /bin/true)"
  docker cp "${cid}:/${mtp_path}" "${work}/mtp_patched.py.orig" || rc=$?
  docker cp "${cid}:/${ple_path}" "${work}/vllm_ple_mmap_patched.py.orig" || rc=$?
  docker rm "${cid}" >/dev/null 2>&1
  if [[ ${rc} -ne 0 ]]; then
    echo "!! Extraction failed -- image layout shifted?" >&2
    echo "   Expected: /${mtp_path}" >&2
    echo "             /${ple_path}" >&2
    exit 1
  fi
  echo "Applying the MiaAI vocab patch (anchor verification, aborts on drift) ..." >&2
  ( cd "${work}" && python3 patch_mtp_draft_vocab.py )
  [[ -f "${work}/mtp_patched.py" ]] || { echo "Patch failed (no mtp_patched.py)." >&2; exit 1; }
  echo "Applying the MADV_RANDOM patch (anchor verification, aborts on drift) ..." >&2
  ( cd "${work}" && python3 patch_ple_mmap_madvise.py )
  [[ -f "${work}/vllm_ple_mmap_patched.py" ]] || { echo "Patch failed (no vllm_ple_mmap_patched.py)." >&2; exit 1; }
  echo "Building ${MIA_IMAGE} (mia/Dockerfile: one layer on top of ${VLLM_IMAGE}," >&2
  echo "incl. grep sanity against the files baked into the image) ..." >&2
  docker build --build-arg "VLLM_IMAGE=${VLLM_IMAGE}" -t "${MIA_IMAGE}" "${work}"
  echo "Image built: ${MIA_IMAGE}" >&2
}

# Hybrid snapshot: hardlink clone of the checkpoint with fp8 side layers (built by
# prepare_hybrid). Without this check the launch aborts with a cryptic loader
# message, because the hybrid dispatcher looks for scale tensors that a plain
# clone does not have.
ensure_hybrid_snapshot() {
  local dir="${QFN_HYBRID_MODEL_DIR}"
  local index="${dir}/model.safetensors.index.json"
  if [[ ! -f "${dir}/.prepared" ]]; then
    echo "Hybrid snapshot not prepared: ${dir}" >&2
    echo "Build it once (Flash-Next must be down, ~10 min, CPU):" >&2
    echo "  ./scripts/qwen38f-vllm.sh prepare-hybrid" >&2
    exit 1
  fi
  if [[ ! -f "${index}" || ! -f "${dir}/config.json" ]]; then
    echo "Hybrid snapshot incomplete: ${dir}" >&2
    echo "(index.json and/or config.json missing -- delete the directory and" >&2
    echo " run prepare-hybrid again)." >&2
    exit 1
  fi
  # The index must really contain the fp8 scales: catches an interrupted
  # converter run that had already written .prepared.
  if ! grep -q 'weight_scale_inv' "${index}"; then
    echo "Hybrid snapshot without fp8 side layer tensors: ${index}" >&2
    echo "(.prepared present, but no weight_scale_inv -> interrupted run?)" >&2
    echo "Delete the directory and run prepare-hybrid again." >&2
    exit 1
  fi
  echo "Hybrid snapshot: ${dir} (prepared, fp8 side layers confirmed)." >&2
}

# Historical PLE prefetch experiment (no observed performance gain): hybrid mounts a patched
# vllm_ple_mmap.py over the image module, so the patch survives a
# container recreate. Both files (patched module + .bak of the original) are
# versioned in the repo, so this module needs no extraction from Docker. The block below
# only (re)creates them when they are missing, in two steps:
# (1) extract the module from ${MIA_IMAGE} (docker create, no CUDA/weights),
# (2) apply the anchor-verified patch from ple_patches/ple_prefetch_patch.py to
# it. The patch aborts on anchor drift instead of staying silently ineffective.
# The source is ${MIA_IMAGE}, NOT ${VLLM_IMAGE}: hybrid runs the mia image, whose
# module already carries the MADV_RANDOM layer. Extracting from the base image
# instead would push that layer out through the bind mount -- a ~2 GiB page-cache
# lever gone without any error message, hence the MADV_RANDOM gate below.
ensure_ple_patch() {
  local dir="$(resolve_project_path "${QFN_PLE_PATCH_DIR:-ple_patches}")"
  local orig="${dir}/vllm_ple_mmap_orig_backup.py"
  local patched="${dir}/vllm_ple_mmap_patched.py"
  local patcher="${PROJECT_DIR}/ple_patches/ple_prefetch_patch.py"
  local ple_path="usr/local/lib/python3.12/dist-packages/vllm_ple_mmap.py"
  local base_image="${MIA_IMAGE}"
  if [[ -s "${patched}" && -s "${orig}" ]]; then
    echo "PLE prefetch patch present: ${dir}/" >&2
    return 0
  fi
  if [[ ! -f "${patcher}" ]]; then
    echo "Patch script missing: ${patcher}" >&2
    exit 1
  fi
  echo "Creating the PLE prefetch patch in ${dir}/ (one-time, from ${base_image}) ..." >&2
  mkdir -p "${dir}"
  local cid rc=0
  # docker create starts NO server -- layer extraction only, container removed.
  cid="$(docker create "${base_image}" /bin/true)"
  docker cp "${cid}:/${ple_path}" "${orig}" || rc=$?
  docker rm "${cid}" >/dev/null 2>&1
  if [[ ${rc} -ne 0 || ! -s "${orig}" ]]; then
    echo "!! Cannot extract the module: /${ple_path} from ${base_image}" >&2
    echo "   Image layout changed? Rebuild it: ./scripts/qwen38f-vllm.sh build-image-mia" >&2
    rm -f "${orig}"
    exit 1
  fi
  # Gate: the running profile is the mia image, so the extracted module must still
  # carry its MADV_RANDOM layer. Without it the bind mount would silently undo the
  # page-cache fix (see mia/Dockerfile patch B) -- regenerate only from a correct image.
  if ! grep -q 'MADV_RANDOM' "${orig}"; then
    echo "!! ${base_image} module has no MADV_RANDOM layer: the mia image is stale or" >&2
    echo "   built from another base. Refusing to regenerate -- the bind mount would" >&2
    echo "   drop that fix silently. Rebuild it:" >&2
    echo "   ./scripts/qwen38f-vllm.sh build-image-mia" >&2
    rm -f "${orig}"
    exit 1
  fi
  cp "${orig}" "${patched}"
  if ! python3 "${patcher}" "${patched}"; then
    rm -f "${patched}"
    echo "!! PLE patch failed (anchor drift inside the image module?)." >&2
    exit 1
  fi
  rm -f "${patched}.bak"
  echo "PLE prefetch patch created: ${patched}" >&2
}

# One-time conversion of the local RadixArk checkpoint into the FP8 hybrid snapshot
# (~15 GiB BF16 side layers -> block-wise FP8; NVFP4 experts stay as they
# are). Layout driver + rationale for deviating from the template:
# mia/prepare_hybrid_snapshot.sh. In short:
#  - blazux @ 656cdb0 clones the HF snapshot via symlinks to ../../blobs/...;
#    our checkpoint is REAL files in a flat folder. Hence a
#    hardlink clone: instant, 0 bytes extra space, the serve container needs only
#    the snapshot directory (no symlink resolution, no parent mount).
#  - Conversion runs INSIDE the image (torch + safetensors present, CPU only).
#    The converter holds a whole shard in RAM (largest: 9.99 GiB), hence a
#    server-down check + MemAvailable lower bound.
#  - Real new bytes: only the 4 converted shards (~7.7 GiB) + index copy.
#  - Container runs with the host user --user, so the snapshot can be deleted
#    again without sudo.
prepare_hybrid() {
  local dst="${QFN_HYBRID_MODEL_DIR}"
  local driver="${PROJECT_DIR}/mia/prepare_hybrid_snapshot.sh"
  local log="${PROJECT_DIR}/logs/prepare-hybrid.log"
  # Lower bounds (overridable for dry runs/tests: QFN_HYBRID_MIN_*).
  local min_free_gib="${QFN_HYBRID_MIN_FREE_GIB:-12}"
  local min_avail_gib="${QFN_HYBRID_MIN_AVAIL_GIB:-20}"
  local src_parent dst_parent src_name dst_name

  if [[ -f "${dst}/.prepared" ]]; then
    echo "Hybrid snapshot already prepared: ${dst}" >&2
    echo "(To rebuild: delete the directory and run prepare-hybrid.)" >&2
    return 0
  fi

  # 0) Layout guard: the container mounts the shared parent directory
  #    exactly once -> the snapshot must be a sibling of the source checkpoint.
  #    A target on another filesystem would be formally possible,
  #    but would have no hardlinks to the original inodes; we rule that out.
  src_parent="$(dirname "${MODEL_DIR}")"
  dst_parent="$(dirname "${dst}")"
  src_name="$(basename "${MODEL_DIR}")"
  dst_name="$(basename "${dst}")"
  if [[ "${src_parent}" != "${dst_parent}" ]]; then
    echo "!! QFN_HYBRID_MODEL_DIR must sit beside MODEL_DIR, same folder:" >&2
    echo "   MODEL_DIR             = ${MODEL_DIR}" >&2
    echo "   QFN_HYBRID_MODEL_DIR  = ${dst}" >&2
    exit 1
  fi
  if [[ "${src_name}" == "${dst_name}" ]]; then
    echo "!! Snapshot and source directories are identical: ${dst}" >&2
    exit 1
  fi
  [[ -d "${src_parent}" ]] || { echo "!! parent folder missing: ${src_parent}" >&2; exit 1; }
  [[ -f "${MODEL_DIR}/model.safetensors.index.json" ]] || { echo "!! index missing in ${MODEL_DIR}" >&2; exit 1; }
  [[ -f "${driver}" ]] || { echo "!! layout driver missing: ${driver}" >&2; exit 1; }
  ensure_image "${VLLM_IMAGE}"

  echo "FP8 hybrid snapshot: ${dst}" >&2
  echo "Source:              ${MODEL_DIR}" >&2

  # 1) No Flash-Next container may run (RAM + I/O); the conversion needs the
  #    memory field for itself.
  local code
  code="$(curl -s -o /dev/null -w '%{http_code}' --connect-timeout 2 --max-time 5 \
    "http://127.0.0.1:${VLLM_PORT}/v1/models" 2>/dev/null || true)"
  if [[ "${code}" == "200" ]]; then
    echo "!! Flash-Next is running on ${VLLM_PORT} -- bring it down before converting:" >&2
    echo "   ./scripts/qwen38f-vllm.sh down" >&2
    exit 1
  fi
  if [[ -n "$(docker ps -q --filter "name=q38f-" 2>/dev/null)" ]]; then
    echo "!! A q38f container is still running -- run ./scripts/qwen38f-vllm.sh down first" >&2
    exit 1
  fi

  # 2) Disk: hardlinks cost nothing, but the 4 fp8 shards are real files.
  local avail_gib
  avail_gib="$(df -Pk "${dst_parent}" | awk 'NR==2{printf "%.0f", $4/1048576}')"
  echo "Free space at the target: ${avail_gib} GiB (minimum ${min_free_gib} GiB; realistically ~7.7 GiB new)." >&2
  if (( avail_gib < min_free_gib )); then
    echo "!! Too little free space at the target directory." >&2
    exit 1
  fi

  # 3) RAM: the converter loads a complete shard (model-bf16-00011 = 9.99
  #    GiB) and puts the fp8 copy + float intermediate right beside it.
  local avail_mem_gib
  avail_mem_gib="$(meminfo_gib "MemAvailable:")"
  echo "MemAvailable: ${avail_mem_gib} GiB (minimum ${min_avail_gib} GiB)." >&2
  if (( $(awk -v a="${avail_mem_gib}" -v m="${min_avail_gib}" 'BEGIN{print (a<m)?1:0}') )); then
    echo "!! Not enough free RAM to run the conversion." >&2
    echo "   Page-cache residue: ./scripts/qwen38f-vllm.sh dropcaches (needs sudo)." >&2
    echo "   Other large processes on this host compete for the same pool." >&2
    exit 1
  fi

  mkdir -p "${PROJECT_DIR}/logs"
  echo "Starting the conversion inside the container (${VLLM_IMAGE}, CPU only, ~10 min) ..." >&2
  echo "Log: ${log}" >&2
  local rc=0
  docker run --rm --name q38f-fp8convert \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    --entrypoint bash \
    -v "${src_parent}:/models-root" \
    -v "${PROJECT_DIR}/mia:/tools:ro" \
    "${VLLM_IMAGE}" \
    /tools/prepare_hybrid_snapshot.sh \
      "/models-root/${src_name}" "/models-root/${dst_name}" \
    2>&1 | tee "${log}" || rc=$?

  if [[ ! -f "${dst}/.prepared" ]]; then
    echo "!! Conversion aborted (rc=${rc}, no .prepared). The target may" >&2
    echo "   half-finished:  rm -rf '${dst}'   run it again." >&2
    echo "   Log: ${log}" >&2
    exit 1
  fi

  echo "Hybrid snapshot ready: ${dst}" >&2
  echo "Next steps:" >&2
  echo "  ./scripts/qwen38f-vllm.sh build-image-mia   (unless already built)" >&2
  echo "  ./scripts/qwen38f-vllm.sh hybrid" >&2
  echo "The snapshot carries its own data (hardlinks); it also survives" >&2
  echo "moving the original -- k3/mia still need ${src_name} though." >&2
}

# Build a draft vocab. $1 = public | local
#   public: only the public sources pinned in sources.json ->
#           mia/draft_vocab_public.txt, including the provenance record. That is
#           the artifact for this repo.
#   local:  the same sources plus own model outputs -> mia/draft_vocab.txt;
#           stays on the local machine (gitignored).
vocab_build() {
  local mode="$1"
  local tool="${DRAFT_VOCAB_PROJECT}"
  if [[ ! -d "${tool}/scripts" ]]; then
    echo "Vocab tool missing: ${tool}/scripts" >&2
    echo "  DRAFT_VOCAB_PROJECT must point at a toolkit containing scripts/" >&2
    echo "  (default: tools/draft-vocab in this repo)." >&2
    exit 1
  fi
  local py="${tool}/.venv/bin/python"
  [[ -x "${py}" ]] || py="python3"
  local corpus="${tool}/corpus"
  local out prov_file
  if [[ "${mode}" == "public" ]]; then
    out="$(resolve_project_path "${DRAFT_VOCAB_PUBLIC}")"
    prov_file="${out}.provenance.json"
  else
    out="$(resolve_project_path "${DRAFT_VOCAB_OUT}")"
    # The provenance record of a local build lists the own sources --
    # so it stays local, just like the vocab itself (*.provenance.local.json is
    # gitignored). Only the public artifact is published together with its record.
    prov_file="${out}.provenance.local.json"
  fi

  local wt_en wt_de wt_code wt_out
  wt_en="${DRAFT_VOCAB_ENGLISH_WEIGHT:-1}"
  wt_de="${DRAFT_VOCAB_GERMAN_WEIGHT:-3}"
  wt_code="${DRAFT_VOCAB_CODE_WEIGHT:-3}"
  wt_out="${DRAFT_VOCAB_MODEL_OUT_WEIGHT:-20}"

  echo "== Building draft vocab (${mode}) -> ${out} =="

  if [[ -s "${corpus}/english.txt" || -s "${corpus}/german.txt" || -s "${corpus}/code.txt" ]]; then
    echo "1/3 Public sources present -- skip. (To refetch: empty ${corpus})" >&2
  else
    echo "1/3 Fetching public sources (pins in ${tool}/sources.json; several GiB download) ..." >&2
    ( cd "${tool}" && "${py}" scripts/build_public_corpus.py )
  fi

  local -a specs=()
  [[ -s "${corpus}/english.txt" ]] && specs+=("corpus/english.txt:${wt_en}")
  [[ -s "${corpus}/german.txt" ]]  && specs+=("corpus/german.txt:${wt_de}")
  [[ -s "${corpus}/code.txt" ]]    && specs+=("corpus/code.txt:${wt_code}")

  if [[ "${mode}" == "local" ]]; then
    if [[ -s "${corpus}/model_out.jsonl" ]]; then
      echo "2/3 model_out.jsonl present ($(wc -l < "${corpus}/model_out.jsonl") lines) -- skip." >&2
    elif [[ -n "${DRAFT_VOCAB_CAPTURE_IN:-}" && -e "${DRAFT_VOCAB_CAPTURE_IN}" ]]; then
      echo "2/3 Capturing own model outputs (they pass the redaction) ..." >&2
      ( cd "${tool}" && "${py}" scripts/capture_model_outputs.py \
          --in "${DRAFT_VOCAB_CAPTURE_IN}" --out corpus/model_out.jsonl \
          --since "${DRAFT_VOCAB_SINCE:-}" --strict )
    else
      echo "2/3 No own model outputs given -> public-only ranking." >&2
      echo "    Also capture: DRAFT_VOCAB_CAPTURE_IN=<export.jsonl|folder> ./scripts/qwen38f-vllm.sh draft-vocab" >&2
    fi
    [[ -s "${corpus}/model_out.jsonl" ]] && specs+=("corpus/model_out.jsonl:${wt_out}")
  elif [[ -s "${corpus}/model_out.jsonl" ]]; then
    echo "2/3 corpus/model_out.jsonl NOT used: vocab-public keeps own material out." >&2
  else
    echo "2/3 No own outputs (as intended with vocab-public)." >&2
  fi

  if ((${#specs[@]} == 0)); then
    echo "No corpus: no usable file in ${corpus}." >&2
    exit 1
  fi

  # The mode belongs to the file, not to the run. Pinned public sources
  # structurally contain paths and mails AS CONTENT (documentation names
  # home paths in the form /home/<user>/, weblink lists carry a contact address,
  # CPython file headers author addresses -- the corpus built here holds 194 blocks of
  # home_path noise). A hit there is
  # provenance noise. Own material (model_out.jsonl) has no such relief:
  # there every hit stays a hard error. One shared --strict run over
  # both kinds would therefore not be a gate, but a build abort.
  echo "3/3 Corpus check (trigger patterns in the source files) ..." >&2
  local -a pub_files=() own_files=()
  local spec
  for spec in "${specs[@]}"; do
    case "${spec%%:*}" in
      corpus/model_out.jsonl) own_files+=("${spec%%:*}") ;;
      *)                      pub_files+=("${spec%%:*}") ;;
    esac
  done
  if ((${#pub_files[@]})); then
    echo "    public sources (--public): ${pub_files[*]}" >&2
    ( cd "${tool}" && "${py}" scripts/verify_corpus.py --public "${pub_files[@]}" ) || exit 1
  fi
  if ((${#own_files[@]})); then
    echo "    own material (--strict, hard): ${own_files[*]}" >&2
    ( cd "${tool}" && "${py}" scripts/verify_corpus.py --strict "${own_files[@]}" ) || exit 1
  fi

  echo "Tokenize + rank (${specs[*]##*/}, size ${DRAFT_VOCAB_SIZE:-65536})." >&2
  echo "Duration varies in step with the corpus size; progress shows in stderr." >&2
  ( cd "${tool}" && "${py}" scripts/build_draft_vocab.py "${specs[@]}" \
      --model "${MODEL_DIR}" \
      --size "${DRAFT_VOCAB_SIZE:-65536}" \
      --out "${out}" \
      --provenance "${prov_file}" \
      --manifest corpus/manifest.json ) || exit 1

  # Gate against the artifact itself: every ID is decoded back and checked
  # against the finding patterns. That is the test that counts -- the IDs ship,
  # not the texts.
  echo "Gate: decoded artifact check ..." >&2
  ( cd "${tool}" && "${py}" scripts/verify_vocab.py --model "${MODEL_DIR}" "${out}" ) || exit 1

  echo "Draft vocab written: ${out}" >&2
  echo "Provenance record: ${prov_file}" >&2
  if [[ "${mode}" == "public" ]]; then
    echo "  (public artifact -> ship it with the repo)" >&2
    echo "Profile test: ./scripts/qwen38f-vllm.sh mia   (the public vocabulary is the default)" >&2
  fi
}

draft_vocab()  { vocab_build local; }
vocab_public() { vocab_build public; }

start_profile() {
  local profile="$1"
  local service="q38f-${profile}"

  ensure_model
  mkdir -p "${HF_CACHE_DIR}" "${VLLM_CACHE_DIR}" "${TRITON_CACHE_DIR}"
  if [[ "${profile}" == "mia" ]]; then
    ensure_image "${MIA_IMAGE}"
    ensure_draft_vocab
  elif [[ "${profile}" == "hybrid" ]]; then
    # hybrid needs both: the fp8 snapshot (prepare-hybrid) AND the
    # MIA_IMAGE (MTP vocab patch + MADV_RANDOM) incl. draft vocab.
    ensure_hybrid_snapshot
    ensure_image "${MIA_IMAGE}"
    ensure_draft_vocab
    ensure_ple_patch
  else
    ensure_image
  fi
  compose --profile "${profile}" up -d --force-recreate "${service}"
  compose logs -f "${service}"
}

logs() {
  local profile="${1:-${QFN_PROFILE}}"
  compose logs -f "q38f-${profile}"
}

kv() {
  local profile="${1:-${QFN_PROFILE}}"
  compose logs --no-color "q38f-${profile}" 2>&1 \
    | grep -Ei "available kv|gpu kv|maximum concurrency|max model len|kv cache|speculat|mtp|ple mmap|draft vocab" || true
}

test_text() {
  echo "Checking server readiness on http://127.0.0.1:${VLLM_PORT}/v1/models" >&2
  curl --fail-with-body --connect-timeout 5 --max-time 15 -sS \
    "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null
  echo "POST http://127.0.0.1:${VLLM_PORT}/v1/chat/completions" >&2
  curl --fail-with-body --connect-timeout 5 --max-time "${CURL_MAX_TIME}" -sS \
    "http://127.0.0.1:${VLLM_PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"${SERVED_MODEL_NAME}\",
      \"messages\": [
        {\"role\": \"system\", \"content\": \"Answer directly and concisely.\"},
        {\"role\": \"user\", \"content\": \"Say only: Server running.\"}
      ],
      \"max_tokens\": 32
    }"
  echo
}

disable_restart() {
  docker update --restart=no q38f-k3-vllm >/dev/null 2>&1 || true
  docker update --restart=no q38f-mia-vllm >/dev/null 2>&1 || true
  docker update --restart=no q38f-hybrid-vllm >/dev/null 2>&1 || true
}

down_all() {
  disable_restart
  compose --profile k3 down --remove-orphans || true
  compose --profile mia down --remove-orphans || true
  compose --profile hybrid down --remove-orphans || true
  docker rm -f q38f-k3-vllm >/dev/null 2>&1 || true
  docker rm -f q38f-mia-vllm >/dev/null 2>&1 || true
  docker rm -f q38f-hybrid-vllm >/dev/null 2>&1 || true
}

if [[ $# -eq 0 ]]; then
  set -- "${QFN_PROFILE}"
fi

case "$1" in
  k3)
    start_profile "k3"
    ;;
  mia)
    start_profile "mia"
    ;;
  hybrid)
    start_profile "hybrid"
    ;;
  logs)
    shift
    logs "${1:-${QFN_PROFILE}}"
    ;;
  kv)
    shift
    kv "${1:-${QFN_PROFILE}}"
    ;;
  test)
    test_text
    ;;
  build-image)
    build_image
    ;;
  build-image-mia)
    build_image_mia
    ;;
  vocab-public)
    vocab_public
    ;;
  draft-vocab)
    draft_vocab
    ;;
  prepare-hybrid)
    prepare_hybrid
    ;;
  dropcaches)
    drop_caches
    ;;
  disable-restart)
    disable_restart
    ;;
  down)
    down_all
    ;;
  ps)
    shift
    compose --profile "${1:-${QFN_PROFILE}}" ps
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    usage >&2
    exit 1
    ;;
esac
