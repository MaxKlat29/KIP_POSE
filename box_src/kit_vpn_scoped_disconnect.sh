#!/usr/bin/env bash
# Trennt das scoped KIT-VPN auf der Workstation (Gegenstueck zu kit_vpn_scoped_connect.sh).
D="${KITVPN_DIR:-/mnt/data/kitvpn}"
if [ -f "$D/openvpn.pid" ] && kill "$(cat "$D/openvpn.pid")" 2>/dev/null; then
  rm -f "$D/openvpn.pid"; echo "[kit-vpn] scoped VPN getrennt."
elif pkill -f "openvpn --config $D/kit.ovpn"; then
  echo "[kit-vpn] getrennt (pkill)."
else
  echo "[kit-vpn] kein laufender VPN-Prozess."
fi
