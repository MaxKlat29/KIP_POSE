# S-009 / T-144 — Modell-Weights auf der Box (Bruno / Security)

> Box: `max@100.85.216.95` (`maxgpuserverobk`). Weights-Root: `/mnt/data/kip_pose_weights/`
> Download-venv (reuse): `/mnt/data/kip_pose_weights/.dl-venv` (isoliert, kein System-Python-Pollute).
> Regel: Weights auf `/mnt/data` (3.0 T frei), NIE auf `/` (226 G, ~88 % voll — Sams Build-/Docker-Disk).

## S-009 (T-135) — FoundationPose + GigaPose Weights (Kontext, done 2026-06-07)

- FoundationPose- + GigaPose-Weights liegen unter `/mnt/data/kip_pose_weights/{foundationpose,gigapose}/`.
- Integritäts-Methodik: Magic-Bytes statt Größe (`head -c4 | xxd`) — torch-Checkpoints sind `PK\x03\x04`
  (`504b 0304`, ZIP, weil `torch.save` ZIP schreibt); eine getarnte HTML-Quota-Seite startet `3c68` (`<htm`).
- `.gitignore` gehärtet (Defense-in-Depth): `*.ckpt`/`*.pth`/`weights/`/`pretrained/`/`hf_token*` ergänzt.
- FP-Weights sind **NON-COMMERCIAL** (NVIDIA-Lizenz §3.3 — nur Research/Eval, ADR-021).

---

## sam3 (T-144) — `facebook/sam3` (gated) → CACHED, OFFLINE-MOUNTBAR, TOKEN NICHT PERSISTIERT

**Status: DONE.** Max hat Lizenz akzeptiert + Read-Token geliefert. Snapshot transient gezogen, Token nirgends persistiert.

### Cache-Pfad (HF-Hub-Layout, offline auflösbar)

```
HF_HOME      = /mnt/data/kip_pose_weights/hf_cache
Hub-Cache    = /mnt/data/kip_pose_weights/hf_cache/hub
Snapshot     = /mnt/data/kip_pose_weights/hf_cache/hub/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7
Repo sha     = 3c879f39826c281e95690f02c7821c4de09afae7   (gated=manual, user=maxkl, role=read)
Größe        = 3.3 G (model.safetensors 3,439,938,512 B + configs/tokenizer)
```

Heruntergeladen via `huggingface_hub.snapshot_download(..., allow_patterns=[...])` ins **echte HF-Hub-Cache-Layout**
(`models--facebook--sam3/{blobs,snapshots,refs}`), damit `sam3-svc` es mit `HF_HUB_OFFLINE=1` rein lokal auflöst —
**ohne Token, ohne Netzwerk** (live bewiesen, siehe unten).

### Files im Snapshot (10) — bewusst OHNE `sam3.pt`

| File | Größe | Hinweis |
|---|---|---|
| `model.safetensors` | 3.44 GB | **gezogen** — 1797 Tensoren, dtype F32, arch `Sam3VideoModel` |
| `config.json` | 25 KB | `model_type=sam3_video` |
| `tokenizer.json` / `tokenizer_config.json` / `vocab.json` / `merges.txt` / `special_tokens_map.json` / `processor_config.json` | klein | gezogen |
| `README.md` / `LICENSE` | klein | gezogen |
| ~~`sam3.pt`~~ | ~~3.45 GB~~ | **NICHT gezogen** — Pickle → arbitrary-code-execution bei `torch.load` |

**Security-Entscheidung (Bruno):** `model.safetensors` enthält dieselben Weights wie `sam3.pt`, aber als reines
Tensor-Format **ohne Code-Ausführung**. Den Pickle-`.pt` per `allow_patterns` ausgeschlossen → **−3.45 GB Disk +
RCE-Deserialisierungs-Vektor eliminiert**. Falls `sam3-svc` zwingend `sam3.pt` lädt (unwahrscheinlich für HF-`AutoModel`),
ist Nachziehen trivial (`allow_patterns=["sam3.pt"]`) — sollte aber bevorzugt auf safetensors umgestellt werden.

### Integrität (verifiziert)

- **safetensors-Header valide:** erste Bytes `e8cc 0300 0000 0000 7b22 5f5f 6d65 7461` = u64-Header-Length (0x3cce8
  = 249064 B) + JSON `{"__metadata__...`. **Kein getarntes HTML** (`<htm`/`3c68` = Quota-Seite). JSON parst:
  1797 Tensoren, dtype F32, sample `detector_model.detr_decoder.box_head.layer1.bias` shape `[256]`, `__metadata__={'format':'pt'}`.
- **config.json valide JSON:** `model_type=sam3_video`, `architectures=['Sam3VideoModel']`.
- **Keine `.incomplete`-Files**, Download vollständig, sha matcht model_info-sha.

### Mount für `sam3-svc` (`:ro`, offline, kein Token)

`sam3-svc` mountet den HF-Cache **read-only** und läuft mit `HF_HUB_OFFLINE=1` + `HF_HOME` auf den Mountpunkt.
Kein `HF_TOKEN` im Container — der Snapshot ist bereits lokal vollständig.

docker-compose (sam3-svc-Service):
```yaml
  sam3-svc:
    # ...
    environment:
      HF_HOME: /hf_cache
      HF_HUB_OFFLINE: "1"          # rein lokal auflösen, kein Netz/Token
      HF_HUB_DISABLE_TELEMETRY: "1"
      SAM3_CONF: "0.2"             # BEKANNTE FALLE: 0.2, NICHT 0.5 (yannic-Messung)
    volumes:
      - /mnt/data/kip_pose_weights/hf_cache:/hf_cache:ro   # read-only
```

Äquiv. `docker run`:
```
-e HF_HOME=/hf_cache -e HF_HUB_OFFLINE=1 -e HF_HUB_DISABLE_TELEMETRY=1 \
-v /mnt/data/kip_pose_weights/hf_cache:/hf_cache:ro
```

**Offline-Resolve live bewiesen** (ohne Token, ohne Netz, `local_files_only=True`):
`snapshot_download("facebook/sam3")` → o.g. Snapshot-Pfad; `model.safetensors` 3,439,938,512 B + `config.json` exists.

### Token-Hygiene — Befund: CLEAN (Token NICHT persistiert)

Token war **ausschließlich transient**: per **stdin** an einen Remote-Python-Prozess gepiped → dort als
`snapshot_download(token=...)`-Argument. **Nie** als CLI-arg (kein `ps aux`-Leak), **nie** in einer Datei,
**nie** via `hf auth login` (das `~/.cache/huggingface/token` schreiben würde), **nie** als persistente env-Var.

Negativ-Beweis post-download (alle CLEAN):

| Prüfung | Befund |
|---|---|
| `~/.cache/huggingface/token` | absent |
| `~/.huggingface/token` (legacy) | absent |
| `~/.cache/huggingface/stored_tokens` | absent |
| `grep -aroE 'hf_[A-Za-z0-9]{34,}'` in `~/.cache/huggingface` + `hf_cache` | **0 Treffer** |
| `~/.bash_history` (+ zsh/fish) auf Token-Muster | **0 Treffer** |
| Literal-Prefix `hf_FXmZ` in history/`/tmp` | nirgends |
| `env | grep HF_TOKEN/HUGGING_FACE_HUB_TOKEN` | leer |
| transiente Artefakte `/tmp/sam3_dl.py` + `/tmp/sam3_dl.log` | **entfernt** |

> **Methodik-Hinweis (wichtig):** `grep hf_` allein ist **zu breit** — es matcht harmlos die HF-Library-
> User-Agent-Strings `hf_xet/1.x` und `hf_hub/1.x` in `~/.cache/huggingface/xet/logs/*.log` sowie zufällige
> `hf_`-Substrings in binären Blob-Daten. Das **präzise** Muster für ein echtes HF-Read-Token ist
> `hf_[A-Za-z0-9]{34,}` (Token = `hf_` + 34 base62). Damit: **null** Treffer in jedem HF-Cache.

**→ Max kann den Token jetzt revoken** (huggingface.co → Settings → Access Tokens). Der Box-Cache bleibt
funktionsfähig (offline, kein Token nötig). Es bleibt **kein** Token-Rest auf der Box.

### Disk-Befund (Mount-Trennung gewahrt)

- `/mnt/data`: 494 G used / 3.0 T frei — sam3 (+3.3 G) hier abgelegt.
- `/`: 28 G frei (88 %) — **unberührt** (Sams Docker-/Build-Disk, T-132 läuft parallel).

### Notiz für Sam (sam3-svc:8004 freischaltbar)

`sam3-svc:8004` war in S-006 (T-132) deferred wegen fehlender gated-Weights — **kann jetzt hochgefahren werden**:
- Mount wie oben (`:ro`, `HF_HUB_OFFLINE=1`, kein `HF_TOKEN`).
- **Falle:** `SAM3_CONF=0.2` (nicht 0.5).
- **Bekanntes Limit (yannic-Messung):** sam3 **trennt `anker_kurz` / `anker_lang` nicht** sauber →
  **YOLO bleibt die klassifizierende Quelle**; sam3 liefert Maske/Detection, nicht die Klasse.
