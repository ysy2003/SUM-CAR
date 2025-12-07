usage() {
  cat <<'EOF'
Auto mode (recommended):
  run_metrics.sh --auto [--patches task=path,...] [--merged-memory PATH] [--hits PATH] [--remap PATH]
                  [--tasks t1,t2,t3] [--base-model MODEL] [--max-samples N]

Manual mode (legacy):
  run_metrics.sh \
    --single SINGLE_JSON --merged MERGED_JSON --restored RESTORED_JSON \
    --home-hits HOME_JSON --cross-hits CROSS_JSON [--util UTIL_JSON --conflict CONFLICT_JSON] \
    [--metric-map METRIC_MAP_JSON] [--tasks TASKS] [--out OUTPUT_JSON]

Default output: out/metrics_v2.json
EOF
}

OUT="out/metrics_v2.json"
TASKS=""
METRIC_MAP=""
UTIL=""
CONFLICT=""
AUTO_MODE=false
AUTO_PATCHES=""
AUTO_MERGED="out/merged/memory.pt"
AUTO_HITS="out/logs/memory_hits.csv"
AUTO_REMAP="out/logs/remap_events.csv"
AUTO_BASE="meta-llama/Meta-Llama-3-8B-Instruct"
AUTO_MAX_SAMPLES=100
AUTO_K_TOP=8
AUTO_ALPHA=1.0
AUTO_USE_COT=false
AUTO_USE_FP16=true

bool_to_flag() {
  if [[ "$1" == true ]]; then
    echo "True"
  else
    echo "False"
  fi
}
METRIC_MAP=""
UTIL=""
CONFLICT=""
    --auto)
      AUTO_MODE=true
      shift ;;
    --patches)
      AUTO_PATCHES="$2"; shift 2 ;;
    --merged-memory)
      AUTO_MERGED="$2"; shift 2 ;;
    --hits)
      AUTO_HITS="$2"; shift 2 ;;
    --remap)
      AUTO_REMAP="$2"; shift 2 ;;
    --base-model)
      AUTO_BASE="$2"; shift 2 ;;
    --max-samples)
      AUTO_MAX_SAMPLES="$2"; shift 2 ;;
    --auto-k-top)
      AUTO_K_TOP="$2"; shift 2 ;;
    --auto-alpha)
      AUTO_ALPHA="$2"; shift 2 ;;
    --auto-use-cot)
      AUTO_USE_COT=true; shift ;;
    --auto-use-fp32)
      AUTO_USE_FP16=false; shift ;;

while [[ $# -gt 0 ]]; do
  case "$1" in
    --single)
      SINGLE="$2"; shift 2 ;;
    --merged)
      MERGED="$2"; shift 2 ;;
    --restored)
      RESTORED="$2"; shift 2 ;;
    --home-hits)
      HOME_HITS="$2"; shift 2 ;;
    --cross-hits)
      CROSS_HITS="$2"; shift 2 ;;
    --util)
      UTIL="$2"; shift 2 ;;
    --conflict)
      CONFLICT="$2"; shift 2 ;;
    --metric-map)
      METRIC_MAP="$2"; shift 2 ;;
    --tasks)
      TASKS="$2"; shift 2 ;;
    --out)
      OUT="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "[run_metrics] Unknown argument: $1" >&2
      usage
      exit 1 ;;
  esac
done

if $AUTO_MODE; then
  export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
  args=(
    python -m sumcar.cli.auto_metrics
    --out "$OUT"
    --tasks "${TASKS:-gsm8k,humaneval,finqa}"
    --merged_memory "$AUTO_MERGED"
    --hits_csv "$AUTO_HITS"
    --remap_csv "$AUTO_REMAP"
    --base_model "$AUTO_BASE"
    --k_top "$AUTO_K_TOP"
    --alpha "$AUTO_ALPHA"
    --max_samples "$AUTO_MAX_SAMPLES"
  )
  if [[ -n "$AUTO_PATCHES" ]]; then
    args+=(--patches "$AUTO_PATCHES")
  fi
  if [[ "$AUTO_USE_COT" == true ]]; then
    args+=(--use_cot)
  fi
  if [[ "$AUTO_USE_FP16" == false ]]; then
    args+=(--use_fp16 False)
  fi
  if [[ -n "$METRIC_MAP" ]]; then
    args+=(--metric_map "$METRIC_MAP")
  fi
  "${args[@]}"
  exit $?
fi

for var in SINGLE MERGED RESTORED HOME_HITS CROSS_HITS; do
  if [[ -z "${!var:-}" ]]; then
    echo "[run_metrics] Missing required --${var,,} argument" >&2
    usage
    exit 1
  fi
  if [[ ! -f "${!var}" ]]; then
    echo "[run_metrics] File not found: ${!var}" >&2
    exit 1
  fi
done

if [[ -n "$UTIL" && -z "$CONFLICT" ]] || [[ -z "$UTIL" && -n "$CONFLICT" ]]; then
  echo "[run_metrics] --util and --conflict must be provided together" >&2
  exit 1
fi

if [[ -n "$UTIL" && ! -f "$UTIL" ]]; then
  echo "[run_metrics] File not found: $UTIL" >&2
  exit 1
fi
if [[ -n "$CONFLICT" && ! -f "$CONFLICT" ]]; then
  echo "[run_metrics] File not found: $CONFLICT" >&2
  exit 1
fi
if [[ -n "$METRIC_MAP" && ! -f "$METRIC_MAP" ]]; then
  echo "[run_metrics] File not found: $METRIC_MAP" >&2
  exit 1
fi

export SINGLE MERGED RESTORED HOME_HITS CROSS_HITS UTIL CONFLICT OUT TASKS METRIC_MAP
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"

python - <<'PY'
import json
import os
import pathlib

from sumcar.metrics import (
    compute_per_task_scores,
    compute_retention_metrics,
    compute_reversible_metrics,
    compute_routing_diagnostics,
)


def load_json(path: str | None):
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_tasks(raw: str):
    raw = (raw or "").strip()
    if not raw:
        return None
    return [part.strip() for part in raw.split(",") if part.strip()]


single_scores = load_json(os.environ["SINGLE"])
merged_scores = load_json(os.environ["MERGED"])
restored_scores = load_json(os.environ["RESTORED"])
home_hits = load_json(os.environ["HOME_HITS"])
cross_hits = load_json(os.environ["CROSS_HITS"])
util_counts = load_json(os.environ.get("UTIL", ""))
conflict_counts = load_json(os.environ.get("CONFLICT", ""))
metric_map = load_json(os.environ.get("METRIC_MAP", ""))
tasks = parse_tasks(os.environ.get("TASKS", ""))

per_task = compute_per_task_scores(
    single_scores,
    merged_scores,
    metric_keys=metric_map,
    tasks=tasks,
)

retention = compute_retention_metrics(
    single_scores,
    merged_scores,
    metric_keys=metric_map,
    tasks=tasks,
)

reversible = compute_reversible_metrics(
    merged_scores,
    restored_scores,
    metric_keys=metric_map,
    tasks=tasks,
)

routing = compute_routing_diagnostics(
    home_hits,
    cross_hits,
    util=util_counts,
    conflict=conflict_counts,
    tasks=tasks,
)

result = {
    "per_task": per_task,
    "retention": retention,
    "reversible": reversible,
    "routing": routing,
}

out_path = pathlib.Path(os.environ["OUT"])
out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w", encoding="utf-8") as handle:
    json.dump(result, handle, indent=2)

print(f"[run_metrics] Metrics written to {out_path}")
PY
