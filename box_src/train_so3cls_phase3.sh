#!/usr/bin/env bash
# PHASE-3 TRAIN-KOMMANDO — Zahnrad SO(3)-Klassifikations-Kopf (T-058 / S-049).
#
#   STATUS: SCAFFOLD. NICHT JETZT AUSFÜHREN — die GPU trainiert/rendert den
#   laufenden Retrain (Crop 320/80 + 8k DR + 160ep). Dieses Skript ist die
#   ready-to-run Phase-3-Aktion fürs GPU-frei-werden, NACH dem aktuellen Lauf.
#
#   Es startet KEINEN GPU-Job, solange die GPU belegt ist: es prüft die Belegung
#   und bricht ab, wenn etwas läuft (kein blindes Davor-Schieben eines Retrains).
#
# ABFOLGE BEIM GPU-FREI-WERDEN:
#   0) git pull im gdrnpp-repo + SO3_INTEGRATION.md abarbeiten (3 Einhänge-Punkte)
#      und so3_rotation_head.py nach core/gdrn_modeling/models/heads/ kopieren.
#   1) bash box_src/train_so3cls_phase3.sh --self-check   # kein GPU-Job
#   2) bash box_src/train_so3cls_phase3.sh --smoke        # 1 Iter, OOM-Gate
#   3) bash box_src/train_so3cls_phase3.sh --train        # voller 160ep-Lauf
#   4) eval + A/B gegen die Regressions-Baseline (zahnrad.py):
#        box_src/eval_bop.sh ... ODER box_src/e2e_finish.sh
#      -> Zahnrad-AR(so3_cls) vs Zahnrad-AR(allo_rot6d, =0.36) dokumentieren.
set -euo pipefail

GDRN_REPO="${GDRN_REPO:-/mnt/data/bop/repos/gdrnpp}"
CFG_REL="configs/gdrn/poseIsaacPbrSO/zahnrad_so3cls.py"
VENV="${GDRN_VENV:-/mnt/data/bop/bop-venv/bin/python}"
MODE="${1:---help}"

gpu_busy() {
  # >5% util ODER >1GB belegt -> als belegt werten (konservativ).
  local util mem
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0)
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0)
  [ "${util:-0}" -gt 5 ] || [ "${mem:-0}" -gt 1024 ]
}

guard_gpu() {
  if gpu_busy; then
    echo "[phase3] GPU BELEGT (laufender Retrain) — KEIN GPU-Job gestartet." >&2
    echo "[phase3] Erst nach dem aktuellen Lauf erneut aufrufen." >&2
    exit 3
  fi
}

case "$MODE" in
  --self-check)
    # KEIN GPU-Job: nur der Kopf-Scaffold (Anker + Forward + Loss).
    echo "[phase3] Self-Check des SO(3)-Kopf-Scaffolds (kein GPU-Job) ..."
    "$VENV" "$(dirname "$0")/so3_rotation_head.py"
    echo "[phase3] Config-Stub: $GDRN_REPO/$CFG_REL"
    [ -f "$GDRN_REPO/$CFG_REL" ] && echo "[phase3] Config deployed: ja" \
      || echo "[phase3] Config noch NICHT deployed (Schritt 0)."
    ;;
  --smoke)
    guard_gpu
    echo "[phase3] SMOKE (1 Iter, OOM-Gate) — Zahnrad SO(3)-cls ..."
    cd "$GDRN_REPO"
    "$VENV" core/gdrn_modeling/main_gdrn.py \
      --config-file "$CFG_REL" --num-gpus 1 \
      SOLVER.TOTAL_EPOCHS 1 TRAIN.PRINT_FREQ 1 DEBUG True
    ;;
  --train)
    guard_gpu
    echo "[phase3] VOLLER TRAIN (160ep) — Zahnrad SO(3)-cls ..."
    cd "$GDRN_REPO"
    "$VENV" core/gdrn_modeling/main_gdrn.py \
      --config-file "$CFG_REL" --num-gpus 1
    echo "[phase3] fertig -> output/gdrn/poseIsaacPbrSO/zahnrad_so3cls/"
    echo "[phase3] JETZT A/B: eval so3_cls vs Baseline 0.36 (siehe Kopf §4)."
    ;;
  *)
    sed -n '1,30p' "$0"
    ;;
esac
