#!/usr/bin/env bash
# host-tuning.sh -- optional kernel VM tuning for the DGX Spark (GB10)
# under Flash-Next. NOT part of a server launch; deliberately a separate
# script, because it (a) needs root and (b) shifts how MemAvailable reads
# (see the caveat below) -- that call has to be made explicitly.
#
# Values taken from MiaAI-Lab files/sysctl-spark3.conf (AGPL-3.0-or-later,
# MiaAI-Lab; fetched 10.09.2026, main branch). Context there: on spark1 six
# crash-free bring-up runs with exactly these three values, after three
# hanging hangs without them; not measured on spark3 yet. The kernel
# default on the Spark is vm.min_free_kbytes~45 MB and vm.watermark_scale_
# factor=10 (kswapd kicks in at 0.1 % free) -- against a GPU driver that
# allocates in GiB/s bursts from the SAME unified pool, that is neither a
# floor nor an early warning (MiaAI documents an NV_ERR_NO_MEMORY death at
# ~3 GiB MemFree).
#
# WHAT THE VALUES COST (caveat, close to the MiaAI config wording):
#   The reserve is CUDA-visible memory that vLLM can no longer take.
#   The vLLM pool check (gpu-memory-utilization x MemTotal) demands the ~4 GiB
#   as "free" anyway -- so the reserve only legalises what the launch
#   already requires. BUT: MemAvailable then reads roughly
#   ~11 GiB LOWER (watermark formula: low~7.7 GiB, high~11.3 GiB at 4 GiB
#   min_free + 3 % scale), because the kernel stops counting anything below
#   high as available. Confusion is built in; that is not real
#   availability. Recalibrate watchdog thresholds (if present)
#   after the first idle run.
#
# Usage:
#   sudo ./scripts/host-tuning.sh apply     (sets the three values; reset on
#                                            reboot -- persist only after
#                                            it has proven itself, see show_persist)
#   ./scripts/host-tuning.sh status         (runs without root; shows values +
#                                            swap in/out activity)
#   sudo ./scripts/host-tuning.sh revert    (back to Ubuntu default 60/10;
#                                            min_free back to the boot value
#                                            of the kernel -- note it beforehand!)
#   ./scripts/host-tuning.sh show-persist   (prints the persist command)
#
# Swap note (user research 10.09.2026, NVIDIA Known Issues): do NOT disable
# swap -- UMA frees DRAM through reclaim/swap; without a safety
# reserve "something lags" quickly turns into "CUDA allocation failed". 16/16 GiB
# swap OCCUPIED is not swap thrash: only si/so activity is relevant
# (see status).
#
# ORDER (measured live 10.09.2026, Flash-Next running on 44415):
#   MemFree 0.6 GiB | MemAvailable 4.1 GiB | Swap 16/16 GiB full
#   si/so 0 in the last sample (no acute thrash), but bi 50-113
#   MB/s -- PLE pages are reloaded from NVMe continuously, the page cache is
#   evicted. In EXACTLY such a state do not set min_free_kbytes=4 GiB.
#   The reserve is a before-the-fact tool, not a during-the-fact tool:
#     apply  ->  dropcaches  ->  qwen38f-vllm.sh k3
#   apply() therefore refuses below 8 GiB MemAvailable (sudo env FORCE=1
#   overrides the lock).
set -euo pipefail

# Target values -- movable via env overrides without touching this file:
#   sudo MIN_FREE_KBYTES=2097152 ./scripts/host-tuning.sh apply
MIN_FREE_KBYTES="${MIN_FREE_KBYTES:-4194304}"            # 4 GiB
WATERMARK_SCALE_FACTOR="${WATERMARK_SCALE_FACTOR:-300}"  # 3 % (kernel default 10 = 0.1 %)
SWAPPINESS="${SWAPPINESS:-30}"                           # kernel default 60

# Integers only: a typo in an override would otherwise land as junk in
# sysctl (and in the persist snippet of show-persist). The kernel rejects
# an absurd watermark_scale_factor on its own as well.
for _spec in "min_free_kbytes=${MIN_FREE_KBYTES}" \
             "watermark_scale_factor=${WATERMARK_SCALE_FACTOR}" \
             "swappiness=${SWAPPINESS}"; do
  if [[ ! "${_spec#*=}" =~ ^[0-9]+$ ]]; then
    echo "!! ${_spec%%=*}='${_spec#*=}' is not an integer -- remove or fix the override." >&2
    exit 1
  fi
done

read_sys() { cat "/proc/sys/vm/$1" 2>/dev/null || echo "?"; }

status() {
  echo "== Kernel VM values (current) =="
  printf '  vm.min_free_kbytes      = %s\n' "$(read_sys min_free_kbytes)"
  printf '  vm.watermark_scale_factor = %s\n' "$(read_sys watermark_scale_factor)"
  printf '  vm.swappiness           = %s\n' "$(read_sys swappiness)"
  echo "== Target values (MiaAI spark3 template) =="
  printf '  min_free_kbytes=%s  watermark_scale_factor=%s  swappiness=%s\n' \
    "$MIN_FREE_KBYTES" "$WATERMARK_SCALE_FACTOR" "$SWAPPINESS"
  echo "== RAM/Swap =="
  awk '/MemTotal|MemAvailable|MemFree|Inactive\(file\)|SwapTotal|SwapFree/ \
       {printf "  %-16s %8.1f GiB\n", $1, $2/1048576}' /proc/meminfo
  echo "== Swap activity (decisive, not the swap fill level) =="
  if command -v vmstat >/dev/null 2>&1; then
    echo "  vmstat 1 x 3 (si/so must stay near 0; permanently high = thrash):"
    vmstat 1 3 | sed 's/^/    /'
  else
    echo "  vmstat not installed -- sampling via /proc/vmstat:"
    local a b c d
    a=$(awk '/^pswpin /{print $2}' /proc/vmstat); b=$(awk '/^pswpout /{print $2}' /proc/vmstat)
    sleep 2
    c=$(awk '/^pswpin /{print $2}' /proc/vmstat); d=$(awk '/^pswpout /{print $2}' /proc/vmstat)
    printf '  pswpin delta/2s: %s   pswpout delta/2s: %s  (1 unit = 4 KiB)\n' "$((c-a))" "$((d-b))"
  fi
  echo "  Reading: si/so ~ 0 -> swap is just old eviction, no problem."
  echo "           si/so permanently high -> swap thrash explains the system lag."
  echo "  bi high with si/so ~ 0: the server loads PLE pages from NVMe"
  echo "  afterwards because its page cache is being evicted. That is NOT a swap"
  echo "  problem (disabling swap would be harmful, see the header), but a"
  echo "  room shortage inside the same pool -> biggest lever is the FP8 hybrid"
  echo "  conversion (~7 GiB fewer resident weights; README hybrid)."
}

apply() {
  # Under memory pressure the reserve is the most harmful thing that can
  # happen right now: the kernel takes it back at once via reclaim (page
  # cache + PLE pages gone, while serving). The lock does not bother idle.
  local avail
  avail="$(awk '/MemAvailable/{printf "%.0f", $2/1048576}' /proc/meminfo)"
  if [[ "${avail}" -lt 8 && "${FORCE:-0}" != "1" ]]; then
    echo "!! MemAvailable = ${avail} GiB (< 8): something large is already running." >&2
    echo "   Setting min_free_kbytes=${MIN_FREE_KBYTES} now forces reclaim right" >&2
    echo "   through serving. Correct order:" >&2
    echo "     Server down  ->  sudo $0 apply  ->  dropcaches  ->  Server up" >&2
    echo "   Deliberately anyway:  sudo env FORCE=1 $0 apply" >&2
    exit 1
  fi
  if ! sysctl -w "vm.min_free_kbytes=${MIN_FREE_KBYTES}" \
             "vm.watermark_scale_factor=${WATERMARK_SCALE_FACTOR}" \
             "vm.swappiness=${SWAPPINESS}"; then
    echo "!! sysctl needs root: sudo $0 apply" >&2
    exit 1
  fi
  echo "Applied (reset on reboot). Caveat on the MemAvailable reading: see header."
  status
}

revert() {
  # Defaults of a vanilla Ubuntu/Spark install. min_free_kbytes depends on
  # the kernel/boot (~45155 per MiaAI); so note it BEFORE apply.
  sysctl -w vm.swappiness=60 vm.watermark_scale_factor=10 || {
    echo "!! sysctl needs root: sudo $0 revert" >&2; exit 1; }
  echo "swappiness/watermark back to default; min_free_kbytes NOT touched"
  echo "(boot target value -- reset it yourself when needed, see the status output)."
}

show_persist() {
  cat <<EOF
# Persist only after it has proven itself (at least a few days in production):
#   sudo tee /etc/sysctl.d/99-spark-vllm.conf <<'CONF'
vm.min_free_kbytes = ${MIN_FREE_KBYTES}
vm.watermark_scale_factor = ${WATERMARK_SCALE_FACTOR}
vm.swappiness = ${SWAPPINESS}
#   CONF
#   sudo sysctl --system
EOF
}

case "${1:-status}" in
  status) status ;;
  apply) apply ;;
  revert) revert ;;
  show-persist) show_persist ;;
  help|-h|--help) sed -n '2,53p' "$0" ;;   # full header incl. ORDER
  *) echo "Usage: $0 [status|apply|revert|show-persist]" >&2; exit 1 ;;
esac
