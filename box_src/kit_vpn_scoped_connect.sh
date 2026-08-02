#!/usr/bin/env bash
# =============================================================================
# kit_vpn_scoped_connect.sh — On-demand SCOPED KIT-VPN auf der GPU-Workstation.
# -----------------------------------------------------------------------------
# Verbindet die Workstation mit dem KIT-OpenVPN, routet aber NUR den Jetson-
# Zellenrechner ($JETSON_IP) durch den Tunnel. KEIN Full-Tunnel, KEIN
# DNS-Hijack -> der restliche Workstation-Traffic (Civion-GPU-Daemon, Civion-API,
# kip-server, Tailscale) bleibt voellig unangetastet.
#
# Mechanik: openvpn --route-nopull (ignoriert ALLE gepushten Routen/redirect-
# gateway/DNS) + ein --route-up-Hook, der GENAU EINE Host-Route auf das frische
# tun-Device setzt. So pullt die Workstation den Jetson-live_server on-demand.
#
# Voraussetzung (einmalig auf der Box ablegen, KITVPN_DIR, default /mnt/data/kitvpn):
#   kit.ovpn   -> https://www.scc.kit.edu/scc/net/openvpn/conf/kit.ovpn (direkt ladbar)
#   auth.txt   -> Zeile1 = KIT-U-KUERZEL (z.B. 'ulumu', NICHT die E-Mail!), Zeile2 = Passwort
#   sudo apt-get install -y openvpn
#
# Nutzung:   sudo box_src/kit_vpn_scoped_connect.sh
# Trennen:   sudo box_src/kit_vpn_scoped_disconnect.sh
# =============================================================================
set -euo pipefail

D="${KITVPN_DIR:-/mnt/data/kitvpn}"
JETSON="${JETSON_IP:?JETSON_IP not set — see project/.env.example}"
CFG="$D/kit.ovpn"
AUTH="$D/auth.txt"

[ -f "$CFG" ] && [ -f "$AUTH" ] || {
  echo "FEHLER: $CFG und/oder $AUTH fehlen."
  echo "  -> kit.ovpn von https://www.scc.kit.edu/scc/net/openvpn/conf/kit.ovpn nach $D"
  echo "  -> auth.txt: Zeile1 = U-Kuerzel (z.B. ulumu), Zeile2 = Passwort"
  exit 1; }
command -v openvpn >/dev/null || { echo "FEHLER: openvpn fehlt (sudo apt-get install -y openvpn)"; exit 1; }
mkdir -p "$D"

# route-up: NUR die eine Jetson-Host-Route ueber das frische tun-Device.
cat > "$D/route-up.sh" <<RU
#!/usr/bin/env bash
DEV="\${dev:-\$1}"
ip route replace ${JETSON}/32 dev "\$DEV" || true
echo "\$(date '+%F %T') scoped route ${JETSON}/32 dev \$DEV" >> "$D/openvpn.log"
RU
chmod +x "$D/route-up.sh"

echo "[kit-vpn] verbinde (scoped: nur ${JETSON}, kein Full-Tunnel) ..."
openvpn --config "$CFG" --auth-user-pass "$AUTH" --auth-nocache \
  --route-nopull --script-security 2 --route-up "$D/route-up.sh" \
  --connect-timeout 8 --connect-retry-max 6 \
  --daemon --log "$D/openvpn.log" --writepid "$D/openvpn.pid"

sleep 6
if ping -c1 -W3 "$JETSON" >/dev/null 2>&1; then
  echo "[kit-vpn] OK — scoped VPN steht, Jetson ${JETSON} erreichbar. Restliches Netz unberuehrt."
  ip route get "$JETSON" 2>/dev/null | head -1
else
  echo "[kit-vpn] VPN-Daemon gestartet, aber ${JETSON} noch nicht pingbar."
  echo "  (KIT-Auth = U-Kuerzel, nicht E-Mail? Log:)"
  tail -10 "$D/openvpn.log" 2>/dev/null || true
fi
