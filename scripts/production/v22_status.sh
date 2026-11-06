#!/bin/bash
# Status of a v22 block job: job state, driver output, per-node log tails, failed runs.
#   v22_status.sh <jobid> <version_dir> [wait]   ("wait": first wait until the job runs + 2 min)
JOB=$1; V=$2
if [ "${3:-}" = wait ]; then
  for i in $(seq 1 50); do s=$(squeue -h -j "$JOB" -o %T); [ "$s" = RUNNING ] && break; [ -z "$s" ] && break; sleep 30; done
  sleep 120
fi
squeue -j "$JOB" -o "%.10i %.20j %.8T %.10M %.10L %.5D" 2>/dev/null
sacct -j "$JOB" -X -o JobID,State,Elapsed,NNodes -n
cat "$V"/logs/blocks/*_"$JOB".out 2>/dev/null | tail -5
n=$(ls "$V"/logs/blocks/*_"$JOB"_*.log 2>/dev/null | wc -l)
echo "node logs: $n; nodes done: $(grep -l 'done failed_runs' "$V"/logs/blocks/*_"$JOB"_*.log 2>/dev/null | wc -l)"
echo "runs ok: $(cat "$V"/logs/blocks/*_"$JOB"_*.log 2>/dev/null | grep -c 'rc=0$') failed: $(cat "$V"/logs/blocks/*_"$JOB"_*.log 2>/dev/null | grep -cE 'run [0-9]+ rc=[1-9]')"
ls "$V"/logs/blocks/*_"$JOB"_*.log 2>/dev/null | head -1 | xargs -r tail -n 3
