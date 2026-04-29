#!/usr/bin/env bash
set -euo pipefail

PARTITION="${1:-main}"
REQ_CPUS="${REQ_CPUS:-}"
REQ_MEM_GB="${REQ_MEM_GB:-}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/check_main_resources.sh [partition]

Optional environment variables:
  REQ_CPUS=15      Mark whether each node can satisfy this CPU request.
  REQ_MEM_GB=60    Mark whether each node can satisfy this memory request in GiB.

Examples:
  bash scripts/check_main_resources.sh
  REQ_CPUS=15 REQ_MEM_GB=60 bash scripts/check_main_resources.sh
  REQ_CPUS=7 REQ_MEM_GB=60 bash scripts/check_main_resources.sh main
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

scontrol show node | awk \
  -v partition="$PARTITION" \
  -v req_cpus="$REQ_CPUS" \
  -v req_mem_gb="$REQ_MEM_GB" '
function reset_record() {
  node = ""
  partitions = ""
  state = ""
  cpu_alloc = ""
  cpu_total = ""
  real_mem_mb = ""
  alloc_mem_mb = ""
}

function partition_matches(list, want,    n, i, parts) {
  n = split(list, parts, ",")
  for (i = 1; i <= n; i++) {
    if (parts[i] == want) {
      return 1
    }
  }
  return 0
}

function is_clean_state(value) {
  return value !~ /DRAIN|DOWN|FAIL|NOT_RESPONDING|MAINT/
}

function emit_record(    free_cpus, free_mem_gb, clean, fits_cpu, fits_mem, fit_status) {
  if (node == "" || !partition_matches(partitions, partition)) {
    return
  }

  free_cpus = cpu_total - cpu_alloc
  free_mem_gb = (real_mem_mb - alloc_mem_mb) / 1024.0
  clean = is_clean_state(state) ? "yes" : "no"

  fits_cpu = (req_cpus == "" || free_cpus >= req_cpus) ? "yes" : "no"
  fits_mem = (req_mem_gb == "" || free_mem_gb >= req_mem_gb) ? "yes" : "no"
  fit_status = (clean == "yes" && fits_cpu == "yes" && fits_mem == "yes") ? "yes" : "no"

  printf "%-10s %-16s %7d/%-7d %7d %-11.1f/%-11.1f %-11.1f %-5s %-8s %-8s %-8s\n",
    node,
    state,
    cpu_alloc,
    cpu_total,
    free_cpus,
    alloc_mem_mb / 1024.0,
    real_mem_mb / 1024.0,
    free_mem_gb,
    clean,
    fits_cpu,
    fits_mem,
    fit_status
}

BEGIN {
  reset_record()
  printf "Partition: %s\n", partition
  if (req_cpus != "" || req_mem_gb != "") {
    printf "Requested fit: cpus=%s mem_gb=%s\n",
      (req_cpus == "" ? "-" : req_cpus),
      (req_mem_gb == "" ? "-" : req_mem_gb)
  }
  printf "%-10s %-16s %-15s %-7s %-24s %-11s %-5s %-8s %-8s %-8s\n",
    "NODE",
    "STATE",
    "CPU_ALLOC/TOT",
    "FREECPU",
    "MEM_ALLOC/TOT_GB",
    "FREEMEM_GB",
    "CLEAN",
    "FIT_CPU",
    "FIT_MEM",
    "FIT_ALL"
}

{
  for (i = 1; i <= NF; i++) {
    split($i, kv, "=")
    key = kv[1]
    value = kv[2]

    if (key == "NodeName") {
      emit_record()
      reset_record()
      node = value
    } else if (key == "Partitions") {
      partitions = value
    } else if (key == "State") {
      state = value
    } else if (key == "CPUAlloc") {
      cpu_alloc = value + 0
    } else if (key == "CPUTot") {
      cpu_total = value + 0
    } else if (key == "RealMemory") {
      real_mem_mb = value + 0
    } else if (key == "AllocMem") {
      alloc_mem_mb = value + 0
    }
  }
}

END {
  emit_record()
}
'
