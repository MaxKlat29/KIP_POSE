"""PHASE-3 CONFIG STUB — Zahnrad mit SO(3)-Rotations-KLASSIFIKATIONS-Kopf.

> Ticket T-058 · S-049 · STATUS: SCAFFOLD, NICHT TRAINIERT.
> Deploy-Ziel auf der Box: configs/gdrn/poseIsaacPbrSO/zahnrad_so3cls.py
> AKTIVIERUNG: erst NACH dem laufenden Retrain (GPU frei). NICHT JETZT starten.

WARUM NUR DAS ZAHNRAD ZUERST
----------------------------
Der Anker-Median ist bereits gut (5–8°, cont-Y); sein Restfehler ist der 180°-
Flip-Tail, den M2 (MegaPose-Multi-Hyp) adressiert. Das ZAHNRAD ist der Teil im
falschen Becken (AR 0.36) — genau die Krankheit, die ein SO(3)-Klassifikations-
Kopf heilt (SC6D ITODD 30.3 AR symmetrie-agnostisch). Also: erst das Zahnrad als
A/B gegen die Regressions-Baseline (zahnrad.py). Bei Erfolg auf Anker ausrollen.

WAS DIESER STUB ÄNDERT GEGENÜBER zahnrad.py
-------------------------------------------
Erbt den GESAMTEN per-Objekt-Setup von base_so.py (Daten, Solver, Backbone,
Geo-Head, RGB-only, Symmetrie via models_info). Überschreibt NUR den Rotations-
Pfad des PNP_NET: statt `ROT_TYPE="allo_rot6d"` (Regression EINER Rotation) ->
SO(3)-Klassifikation über ein HEALPix-Anker-Grid (box_src/so3_rotation_head.py).

>>> INTEGRATION (3 Punkte in der GDRNPP-Modell-Klasse — siehe SO3_INTEGRATION.md):
    (a) GDRN_double_mask.from_config: baue SO3RotationClassificationHead statt
        get_pnp_net-Rot-Zweig, wenn PNP_NET.ROT_TYPE == "so3_cls".
    (b) GDRN_double_mask.forward: pred_rot_ = head(pnp_feat) -> Logits; pred_rot_m
        = head.decode(logits, k=1) statt get_rot_mat(...).
    (c) Loss: ersetze den Rot-Term durch head.loss(logits, R_gt, sym_anchor_sets)
        wobei sym_anchor_sets aus get_symmetry_transformations(models_info) +
        nearest_anchor_index berechnet wird (one-to-many für C_7).
    Translation/Centroid/Z bleiben UNVERÄNDERT (centroid_z, der gelöste Teil).
"""
_base_ = ["./base_so.py"]

OUTPUT_DIR = "output/gdrn/poseIsaacPbrSO/zahnrad_so3cls"

DATASETS = dict(
    TRAIN=("pose_isaac_zahnrad_train_pbr",),
    TEST=("pose_isaac_zahnrad_val",),
)

# --- der EINE Architektur-Wechsel: Rotations-Kopf von Regression -> SO(3)-cls ---
MODEL = dict(
    POSE_NET=dict(
        PNP_NET=dict(
            # NEU: signalisiert der Modell-Klasse, den SO(3)-cls-Kopf zu bauen.
            # (Erfordert die 3 Integrations-Punkte oben; ohne sie ignoriert GDRNPP
            #  ROT_TYPE-Unbekanntes nicht -> deshalb ist der Stub bis zur
            #  Integration NICHT lauffähig. Das ist gewollt: SCAFFOLD.)
            ROT_TYPE="so3_cls",
            SO3_CLS=dict(
                # HEALPix-SO(3)-Anker (Implicit-PDF-Stil). Training klein halten,
                # Inferenz hochskalieren (SC6D: 5k train / 480k infer).
                N_VIEW_TRAIN=600, N_INPLANE_TRAIN=24,     # ~14.4k Anker (train)
                N_VIEW_INFER=2000, N_INPLANE_INFER=60,    # ~120k Anker (infer)
                EMB_DIM=64,            # SC6D-Default
                TAU=0.1,              # Cosinus-Softmax-Temperatur (SC6D)
                # one-to-many CE über die C_7-Repräsentanten (symmetrie-nativ):
                MULTI_POSITIVE_SYM=True,
                N_FOLD=7,            # Zahnrad C_7 (aus models_info)
            ),
        ),
        LOSS_CFG=dict(
            # Rot-Term wird vom SO3-Kopf (CE) gestellt; PM_LOSS bleibt als
            # geometrischer Zusatz-Term optional an (symmetrie-aware).
            PM_LOSS_SYM=True, PM_R_ONLY=True, PM_LW=1.0,
            # Gewicht des neuen Klassifikations-Rot-Terms:
            ROT_CLS_LW=1.0,
        ),
    ),
)

# Phase-3: identischer Solver wie base_so (kein Hyperparam-Drift im A/B). Bei OOM
# durch das große Anker-Grid: N_VIEW_TRAIN runter ODER IMS_PER_BATCH 12.
SOLVER = dict(TOTAL_EPOCHS=160)      # an den laufenden Retrain angeglichen
