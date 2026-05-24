# SO(3)-Klassifikations-Kopf — GDRNPP-Integrations-Anleitung (Phase-3, T-058)

> SCAFFOLD-Doku. Beschreibt die 3 Einhänge-Punkte, um `so3_rotation_head.py` in
> die GDRNPP-Modell-Klasse `GDRN_DoubleMask` zu integrieren. NICHT JETZT machen
> (GPU belegt). Abarbeiten beim GPU-frei-werden, vor `train_so3cls_phase3.sh`.

## Voraussetzung
```bash
cp box_src/so3_rotation_head.py \
   /mnt/data/bop/repos/gdrnpp/core/gdrn_modeling/models/heads/so3_rotation_head.py
cp box_src/configs_phase3/zahnrad_so3cls.py \
   /mnt/data/bop/repos/gdrnpp/configs/gdrn/poseIsaacPbrSO/zahnrad_so3cls.py
```

## Datei: `core/gdrn_modeling/models/GDRN_double_mask.py`

GDRNPP-Ist (gelesen 2026-05-24): in `forward()` (~Zeile 158–164)
```python
pred_rot_, pred_t_ = self.pnp_net(coor_feat, region=region, extents=extents, mask_attention=mask_atten)
rot_type = pnp_net_cfg.ROT_TYPE
pred_rot_m = get_rot_mat(pred_rot_, rot_type)          # <- Regression EINER Rotation
```
und in `from_config()` (~Zeile 570): `pnp_net, _ = get_pnp_net(cfg)`.

### (a) Kopf bauen — `from_config()`
```python
from .heads.so3_rotation_head import SO3RotationClassificationHead, healpix_so3_grid
...
pnp_net, pnp_net_params = get_pnp_net(cfg)             # bleibt: liefert die Features
so3_head = None
if net_cfg.PNP_NET.ROT_TYPE == "so3_cls":
    so3 = net_cfg.PNP_NET.SO3_CLS
    anchors = healpix_so3_grid(n_view=so3.N_VIEW_TRAIN, n_inplane=so3.N_INPLANE_TRAIN)
    so3_head = SO3RotationClassificationHead(
        in_dim=<pnp_feat_dim>, anchors=anchors, emb_dim=so3.EMB_DIM, tau=so3.TAU)
    params_lr_list.append({"params": so3_head.parameters(), "lr": cfg.SOLVER.BASE_LR})
# so3_head an den Konstruktor durchreichen (Attribut self.so3_head).
```
`<pnp_feat_dim>` = die geflachte Feature-Dim, die `pnp_net` VOR seiner Rot-Schicht
hat (ConvPnPNet: `featdim * final_spatial_size[0] * final_spatial_size[1]`, mit den
config-Werten). Den Rot-Output-Layer des PnP-Net abgreifen ODER `pnp_net` so
abändern, dass es das Feature ZUSÄTZLICH zurückgibt (kleinster Eingriff: ein
`return_feat=True`-Pfad).

### (b) Forward — Logits statt Regression
```python
if rot_type == "so3_cls":
    rot_logits = self.so3_head(pnp_feat)               # (B, N) Verteilung über SO(3)
    pred_rot_m = self.so3_head.decode(rot_logits, k=1) # (B,3,3) Top-1 (allo)
    # Top-K für den planaren Prior / M2-Multi-Hyp:
    # rot_topk = self.so3_head.decode(rot_logits, k=K)
else:
    pred_rot_m = get_rot_mat(pred_rot_, rot_type)
```
`decode` liefert ALLO-Rotationen (Anker sind im Allo-Frame definiert) — exakt der
Frame, den GDRNPP nach dem PnP-Net erwartet (`allo_rot6d` ist auch allo). Der
nachgelagerte `allo->ego`-Schritt bleibt unverändert.

### (c) Loss — CE statt Rot-Regression
In der Loss-Funktion (`gdrn_loss` / wo `PM_LOSS`/`rot_loss` berechnet wird):
```python
if rot_type == "so3_cls":
    # GT-Allo-Rotation R_gt_allo (wie für den PM-Loss schon berechnet).
    sym_sets = None
    if so3_cfg.MULTI_POSITIVE_SYM:
        sym_sets = build_sym_anchor_sets(R_gt_allo, self.so3_head.anchors,
                                         n_fold=so3_cfg.N_FOLD, sym_axis=(0,1,0))
    loss_rot = self.so3_head.loss(rot_logits, R_gt_allo, sym_anchor_sets=sym_sets)
    loss_dict["loss_rot_cls"] = loss_rot * loss_cfg.ROT_CLS_LW
    # PM_LOSS optional als geometrischer Zusatz-Term behalten.
```

`build_sym_anchor_sets` (einmal pro Batch, klein) — die C_N-Repräsentanten jeder
GT-Rotation auf Anker-Indizes mappen:
```python
def build_sym_anchor_sets(R_gt, anchors, n_fold, sym_axis=(0,1,0)):
    from .heads.so3_rotation_head import nearest_anchor_index
    import numpy as np
    ax = np.asarray(sym_axis, float)
    sets = []
    for R in R_gt.detach().cpu().numpy():
        idxs = set()
        for k in range(n_fold):                        # k * 2π/N um die Sym-Achse
            th = 2*np.pi*k/n_fold
            c, s = np.cos(th), np.sin(th)
            # Rodrigues um ax:
            K = np.array([[0,-ax[2],ax[1]],[ax[2],0,-ax[0]],[-ax[1],ax[0],0]])
            Rk = np.eye(3) + s*K + (1-c)*(K@K)
            idxs.add(nearest_anchor_index(R @ Rk, anchors.cpu().numpy()))
        sets.append(sorted(idxs))
    return sets
```
Für cont-Y-Teile (Anker): `n_fold` groß wählen (z.B. 36) -> approximiert das
Kontinuum als viele Positive, derselbe Mechanismus.

## Inferenz-Skalierung
Beim Test die Anker hochskalieren (SC6D 480k): in `from_config` für den
TEST-Pfad `healpix_so3_grid(N_VIEW_INFER, N_INPLANE_INFER)` als zweites Anker-
Buffer registrieren und in `forward` bei `self.training is False` nutzen. Das
Bild-Embedding bleibt gleich; nur gegen mehr Anker scoren (reine matmul-Kosten).

## A/B (der eigentliche Beweis)
`zahnrad_so3cls.py` (so3_cls) vs `zahnrad.py` (allo_rot6d, Baseline AR 0.36) auf
demselben gefilterten val-GT (n=1077), symmetrie-bewusst (`eval_bop.py`). Erst
wenn so3_cls das Zahnrad messbar hebt: auf Anker ausrollen.

## Quellen
- SC6D (SO(3)-Einbettung + Cosinus-Klassifikation, sym-agnostisch): https://arxiv.org/pdf/2208.02129 · https://github.com/dingdingcai/SC6D-pose
- Implicit-PDF (HEALPix-SO(3)-Grid, Dichte): https://arxiv.org/pdf/2106.05965 · https://implicit-pdf.github.io/
- SymCode/SymNet (one-to-many CE für diskrete Symmetrie): https://arxiv.org/html/2405.10557v1
- GDRNPP (Architektur, allo_rot6d, get_rot_mat): https://arxiv.org/html/2102.12145v5
