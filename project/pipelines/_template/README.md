# Pipeline-Template

Kopiervorlage für die Anbindung **einer** komplett anderen Pose-Pipeline.

## Schritte

1. **Kopieren:** dieses Verzeichnis → `pipelines/<deine_id>/` (z. B. `pipelines/pipeline_x/`).
2. **Vendor:** das rohe externe Python-Projekt (von Max, unverändert) nach `<id>/vendor/`.
3. **Deps isolieren:** Pins in `<id>/requirements.txt`, eigene venv:
   ```bash
   python3 -m venv pipelines/<id>/.venv
   pipelines/<id>/.venv/bin/pip install -r pipelines/<id>/requirements.txt
   ```
4. **Adapter:** in `adapter.py`
   - `id` / `name` / `description` setzen,
   - `available` auf einen echten Check umstellen (z. B. `vendor/`-Entrypoint vorhanden),
   - `infer()` implementieren: in `vendor/` reinrufen, rohen Output über
     `pipelines.contract.assemble_doc(...)` auf den **eingefrorenen** `pose_result`-Contract
     mappen (face/upright werden dort analytisch abgeleitet).
5. **Registrieren:** die `register(...)`-Zeile am Ende von `adapter.py` entkommentieren.

Danach taucht die Pipeline automatisch in `/api/pipelines`, im Modell-Dropdown und im
Vergleichs-Harness (`compare_pipelines.py`) auf.

Details + Contract-Felder + Eval-Rückrechnung: [`../../docs/PIPELINE_INTEGRATION.md`](../../docs/PIPELINE_INTEGRATION.md)
