# Mia layer image: VLLM_IMAGE (qwen38-flash-dgx:656cdb0, the base of k3) plus
# two build-time patches, both anchor-verified (drift = build abort):
#
#   A) reduced-vocabulary MTP drafting (MiaAI-Lab technique, AGPL-3.0-or-later,
#      MiaAI-Lab -- licence in the header of patch_mtp_draft_vocab.py).
#      Inert as long as VLLM_MTP_DRAFT_VOCAB is not set inside the container.
#   B) MADV_RANDOM for the PLE mmap shard views (patch_ple_mmap_madvise.py;
#      reason: blazux@656cdb0 sets no madvise -- PLE lookups are random
#      accesses, the read-ahead window pads the page cache with up to ~2 GiB
#      nobody reads; MiaAI measurement 1366 vs. 57 KiB SSD read per decoded
#      token).
#      Switch: VLLM_PLE_MMAP_MADV_RANDOM (default on, "0" = off).
#
# Build: ./scripts/qwen38f-vllm.sh build-image-mia
#   extracts the UNPATCHED mtp.py + vllm_ple_mmap.py from VLLM_IMAGE into
#   mia/*.orig (gitignored), applies both patches DURING THE BUILD (base image
#   has python3; each anchor verified exactly once) and builds ONE layer.
#
# k3 keeps running on VLLM_IMAGE and stays untouched by both pins
# (smallest possible intrusion into the running setup). Profiles mia + hybrid
# use this image.
ARG VLLM_IMAGE=qwen38-flash-dgx:656cdb0
FROM ${VLLM_IMAGE}

COPY mtp_patched.py.orig /work/mtp_patched.py.orig
COPY patch_mtp_draft_vocab.py /work/patch_mtp_draft_vocab.py
RUN python3 /work/patch_mtp_draft_vocab.py \
 && grep -q '_attach_draft_vocab' /work/mtp_patched.py \
 && grep -q 'VLLM_MTP_DRAFT_VOCAB' /work/mtp_patched.py \
 && grep -q '_draft_id_to_target_id' /work/mtp_patched.py \
 && cp /work/mtp_patched.py /usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/mtp.py

# --- B) PLE mmap: MADV_RANDOM (no read-ahead for random n-gram lookups) ---------
# The block hooks MmapPleTable.__init__ and madvises all shard memmaps;
# it runs only when VLLM_PLE_MMAP=1 is active (mia/hybrid set it).
# Default on; VLLM_PLE_MMAP_MADV_RANDOM=0 switches it off (base behaviour).
COPY vllm_ple_mmap_patched.py.orig /work/vllm_ple_mmap_patched.py.orig
COPY patch_ple_mmap_madvise.py /work/patch_ple_mmap_madvise.py
RUN python3 /work/patch_ple_mmap_madvise.py \
 && grep -q 'MADV_RANDOM' /work/vllm_ple_mmap_patched.py \
 && grep -q 'VLLM_PLE_MMAP_MADV_RANDOM' /work/vllm_ple_mmap_patched.py \
 && grep -q 'MmapPleTable.__init__ = _ple_madvise_random_after_init' /work/vllm_ple_mmap_patched.py \
 && cp /work/vllm_ple_mmap_patched.py /usr/local/lib/python3.12/dist-packages/vllm_ple_mmap.py
