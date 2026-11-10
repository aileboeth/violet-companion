#!/usr/bin/env bash
# Start of Violet's SnowLuma container (deploy/snowluma/Dockerfile), written for Violet 2026-11-10.
# Violet's installer starts it with VNC_PASSWD, SNOWLUMA_UID / SNOWLUMA_GID (the account QQ runs as on the host) and
# writes SnowLuma's config/onebot.json itself before; this script only prepares the container and starts:
#   a virtual screen (Xvfb) with a window manager (fluxbox), a VNC server for it (QQ's window in Violet's panel),
#   then QQ and SnowLuma under supervisord.
set -euo pipefail

# a huge inherited open-files limit makes some desktop helpers hang at start: lower it (never raise it)
soft="$(ulimit -Sn)"; hard="$(ulimit -Hn)"
if [ "${soft}" = "unlimited" ] || [ "${soft}" -gt 65536 ] 2>/dev/null; then ulimit -Sn 65536 || true; fi
if [ "${hard}" = "unlimited" ] || [ "${hard}" -gt 1048576 ] 2>/dev/null; then ulimit -Hn 1048576 || true; fi

: "${VNC_PASSWD:=}"
: "${SNOWLUMA_UID:=1000}"
: "${SNOWLUMA_GID:=1000}"
: "${SNOWLUMA_DATA:=/app/data}"
: "${SNOWLUMA_SCREEN:=1920x1080x24}"
: "${SNOWLUMA_QQ_FLAGS:=--disable-gpu --disable-software-rasterizer --disable-gpu-compositing}"
export DISPLAY="${DISPLAY:-:1}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/runtime-${SNOWLUMA_UID}}"
export SNOWLUMA_HOOK_RUNTIME_DIR="${SNOWLUMA_HOOK_RUNTIME_DIR:-${XDG_RUNTIME_DIR}}"
export SNOWLUMA_QQ_FLAGS SNOWLUMA_LOG_LEVEL="${SNOWLUMA_LOG_LEVEL:-info}" SNOWLUMA_HOOK_AUTOLOAD="${SNOWLUMA_HOOK_AUTOLOAD:-1}"
screen_no="${DISPLAY#:}"; screen_no="${screen_no%%.*}"

# what a previous run of this container left (a restart keeps the writable layer)
chmod 1777 /tmp || true
mkdir -p /tmp/.X11-unix && chmod 1777 /tmp/.X11-unix || true
rm -f /run/dbus/pid /run/dbus/system_bus_socket "/tmp/.X${screen_no}-lock" "/tmp/.X11-unix/X${screen_no}" /tmp/dbus-*
mkdir -p /var/run/dbus /root/.vnc "${XDG_RUNTIME_DIR}" "${SNOWLUMA_DATA}/config" /app/.cache /app/.config /app/.local/share

# a machine id that survives a new container (QQ ties its login to it)
if [ ! -s "${SNOWLUMA_DATA}/config/machine-id" ]; then
  dbus-uuidgen > "${SNOWLUMA_DATA}/config/machine-id"
fi
ln -sf "${SNOWLUMA_DATA}/config/machine-id" /etc/machine-id

# QQ and SnowLuma run as the host's account (its files in the mounted folders are that account's)
groupmod -o -g "${SNOWLUMA_GID}" snowluma
usermod -o -u "${SNOWLUMA_UID}" -g "${SNOWLUMA_GID}" snowluma
chown "${SNOWLUMA_UID}:${SNOWLUMA_GID}" /app
chown -R "${SNOWLUMA_UID}:${SNOWLUMA_GID}" "${SNOWLUMA_DATA}" /app/.cache /app/.config /app/.local/share "${XDG_RUNTIME_DIR}"
chmod 700 "${XDG_RUNTIME_DIR}"
# stale hook sockets of a dead QQ must not look like a live one
find "${XDG_RUNTIME_DIR}" -maxdepth 1 -type s -name 'mojo.*.*.sock' -delete 2>/dev/null || true

# QQ's silent hot update would replace QQ's files under SnowLuma's version-bound hooks: its host goes nowhere
grep -q 'qqpatch\.gtimg\.cn' /etc/hosts 2>/dev/null || printf '0.0.0.0 qqpatch.gtimg.cn\n' >> /etc/hosts

# the VNC password (Violet always gives one; else a random one, kept in the data folder for root only)
pass_file="${SNOWLUMA_DATA}/config/vnc-password"
if [ -n "${VNC_PASSWD}" ]; then
  printf '%s\n' "${VNC_PASSWD}" > "${pass_file}"
elif [ ! -s "${pass_file}" ]; then
  head -c 12 /dev/urandom | base64 | tr -d '/+=' | head -c 8 > "${pass_file}"
fi
chown root:root "${pass_file}"; chmod 600 "${pass_file}"
x11vnc -storepasswd "$(head -n 1 "${pass_file}")" /root/.vnc/passwd >/dev/null

# the virtual screen, its window manager and the VNC server
export DBUS_SESSION_BUS_ADDRESS=""
if address="$(su -s /bin/bash -c 'dbus-daemon --session --fork --print-address' snowluma 2>/dev/null)"; then
  export DBUS_SESSION_BUS_ADDRESS="${address}"
fi
dbus-daemon --config-file=/usr/share/dbus-1/system.conf --print-address >/dev/null &
Xvfb "${DISPLAY}" -screen 0 "${SNOWLUMA_SCREEN}" &
xvfb_pid=$!
for _ in $(seq 1 200); do
  [ -S "/tmp/.X11-unix/X${screen_no}" ] && break
  kill -0 "${xvfb_pid}" 2>/dev/null || { echo "Xvfb stopped before the screen was ready" >&2; exit 1; }
  sleep 0.1
done
[ -S "/tmp/.X11-unix/X${screen_no}" ] || { echo "the virtual screen did not start" >&2; exit 1; }
# no wallpaper: fluxbox calls fbsetbg at start, which finds no wallpaper program and opens an "fbsetbg: I can't find
# an app to set the wallpaper" box next to QQ's window (seen on the server 2026-11-10) -- a do-nothing fbsetbg first
# in PATH
ln -sf /bin/true /usr/local/bin/fbsetbg
fluxbox >/dev/null 2>&1 &
x11vnc -display "${DISPLAY}" -noxrecord -noxfixes -noxdamage -forever -threads -rfbauth /root/.vnc/passwd -quiet &
vnc_pid=$!
sleep 0.5
kill -0 "${vnc_pid}" 2>/dev/null || { echo "the VNC server did not start" >&2; exit 1; }

exec supervisord -c /etc/supervisord.conf
