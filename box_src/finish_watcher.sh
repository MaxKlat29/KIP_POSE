#!/usr/bin/env bash
# Box-side autonomous finish trigger. Runs ON the box (nohup), independent of any
# external session or the operator's connectivity. Polls the phase2 chain log; when
# PHASE2_TRAIN_DONE appears it launches e2e_finish.sh exactly once (guard file).
# On PHASE2_FAILED it exits without launching. This closes the gap where the GDRNPP
# training finishes autonomously but the final eval/A-B/viewer/report (e2e_finish)
# previously depended on the operator's 2h cron.
set -u
REPO=/mnt/data/kip_pose
L=/mnt/data/bop/logs/phase2_chain.log
FIN=/mnt/data/bop/logs/e2e_finish.log
GUARD=/mnt/data/bop/logs/.e2e_finish_started   # shared with the operator cron to avoid double-launch
STAMP() { date -u +%FT%TZ; }

echo "$(STAMP) finish_watcher armed (waiting for PHASE2_TRAIN_DONE in $L)" >> "$FIN"
while true; do
    if grep -aq PHASE2_TRAIN_DONE "$L" 2>/dev/null; then
        if [ ! -e "$GUARD" ] && ! pgrep -f "[e]2e_finish.sh" >/dev/null 2>&1; then
            : > "$GUARD"
            echo "$(STAMP) PHASE2_TRAIN_DONE seen -> launching e2e_finish.sh" >> "$FIN"
            cd "$REPO" && setsid nohup bash box_src/e2e_finish.sh >> "$FIN" 2>&1 &
            echo "$(STAMP) e2e_finish.sh launched (pid $!)" >> "$FIN"
        else
            echo "$(STAMP) PHASE2_TRAIN_DONE seen but e2e_finish already started (guard/proc present) -> noop" >> "$FIN"
        fi
        exit 0
    fi
    if grep -aq PHASE2_FAILED "$L" 2>/dev/null; then
        echo "$(STAMP) PHASE2_FAILED seen -> NOT launching e2e_finish (needs inspection)" >> "$FIN"
        exit 1
    fi
    sleep 120
done
