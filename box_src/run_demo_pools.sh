#!/usr/bin/env bash
# Demo-Puffer fuer alle drei Pipelines bauen (T-193, Max 06.08.).
#
# Die Box hat 15 GB. Isaac braucht beim Rendern rund 5 GB, der Pose-Worker
# belegt dauerhaft 6 GB. Wenn zusaetzlich das Mesh laeuft, landet die Kiste im
# Swap und Isaac kommt nicht mehr hoch. Deshalb: Mesh waehrend des Renderns
# runter, fuer die Auswertung wieder hoch.
set -u
cd /mnt/data/kip_pose
LOG=/tmp/demo_pools.log
: > "$LOG"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

mesh_down() { docker stop $(docker ps -q --filter name=kip_mesh) >/dev/null 2>&1; \
              sudo systemctl stop kip-gdrnpp-svc >/dev/null 2>&1; sleep 3; }
mesh_up()   { sudo systemctl start kip-gdrnpp-svc >/dev/null 2>&1; \
              docker start $(docker ps -aq --filter name=kip_mesh) >/dev/null 2>&1; sleep 20; }

for KIND in rgb rgbd moe; do
  say "=== $KIND : rendern ==="
  mesh_down
  RAW=/mnt/data/kip_pose/sim_raw_$KIND
  rm -rf "$RAW"; mkdir -p "$RAW"
  /mnt/data/isaacsim-venv/bin/python box_src/gen_sdg_arm_visible.py \
    --scene /mnt/data/kip_pose/data/SDG/IsaacSim/USD-Files/GST_Scene.usd \
    --usd-dir /mnt/data/kip_pose/data/SDG/IsaacSim/USD-Files \
    --output "$RAW" --num-scenes 100 --force-counts 8 \
    --focus-frac 1.0 --force-each-focus \
    --spawn-x 0.117,0.723 --spawn-y 0.102,0.477 \
    --arm-clear 0.25 --table-margin 0.08 >> /tmp/render_$KIND.log 2>&1
  say "$KIND : $(ls "$RAW"/rgb_*.png 2>/dev/null | wc -l) Szenen gerendert"

  say "=== $KIND : auswerten ==="
  mesh_up
  python3 box_src/build_demo_pools.py --kind "$KIND" --n 100 --keep 5 \
    --min-parts 4 --skip-render >> "$LOG" 2>&1
  say "$KIND : fertig (exit $?)"
done
say "=== ALLE DREI DURCH ==="
curl -s --max-time 6 localhost:8077/api/sim/pool | tee -a "$LOG"
