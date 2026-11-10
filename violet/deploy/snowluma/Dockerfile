# Violet's own build of the container for the channel "QQ" (2026-11-10): SnowLuma's Docker image and the repository
# it was built from went offline, so `violet qq install` builds the same kind of container on the server itself
# (tools/violet_snowluma.py). Written for Violet after what the old image did (its `docker history`):
#
#   base     node:24.22.1-bookworm-slim (Node.js, which SnowLuma runs on)
#   desktop  Xvfb + fluxbox + x11vnc: a virtual screen for QQ's window (its QR code is scanned in Violet's panel)
#   QQ       the official Linux QQ 3.2.32, downloaded during the build -- Tencent's own address first, else the
#            Rodert/qq-versions archive (the old image's mirror); the file must match the SHA-256 below either way.
#            QQ is never stored in Violet's repository or releases.
#   SnowLuma snowluma-runtime-2.2.1-linux-x64.tar.gz next to this file (MIT, see LICENSE.SnowLuma.txt): the exact
#            build that ran with QQ 3.2.32 in the old image
#
# QQ's silent hot update is blocked (start.sh) so QQ stays the version SnowLuma's hooks were made for.
ARG BASE_IMAGE=node:24.22.1-bookworm-slim
FROM ${BASE_IMAGE}

ARG APT_MIRROR=
ARG QQ_VERSION=3.2.32_260812
ARG QQ_SOURCES="https://qqdl.gtimg.cn/qqfile/QQNT/9.9.33/release/3f89efc5 https://github.com/Rodert/qq-versions/releases/download/qq-packages-20260813-1d08f1d4"
ARG QQ_AMD64_SHA256=d085dd89397225061eb9f194308f688129818ed445777e97a4a0a16e13d7b0e8
ARG QQ_ARM64_SHA256=8796ccfd66acc025ef18db37185532d40bc8c58921e17da6d28c295acbcf8f92

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai \
    DISPLAY=:1 \
    SNOWLUMA_HOME=/app/runtime \
    SNOWLUMA_DATA=/app/data \
    SNOWLUMA_UID=1000 \
    SNOWLUMA_GID=1000 \
    SNOWLUMA_SCREEN=1920x1080x24 \
    SNOWLUMA_HOOK_AUTOLOAD=1 \
    SNOWLUMA_LOG_LEVEL=info \
    SNOWLUMA_QQ_FLAGS="--disable-gpu --disable-software-rasterizer --disable-gpu-compositing"

# Debian packages (APT_MIRROR: a mirror instead of deb.debian.org, chosen by the installer when that is slow)
RUN set -eux; \
    if [ -n "${APT_MIRROR}" ]; then \
      sed -i "s#http://deb.debian.org#${APT_MIRROR}#g" /etc/apt/sources.list.d/debian.sources; \
    fi; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      ca-certificates curl dbus dbus-user-session gnutls-bin iproute2 libcap2-bin procps supervisor tzdata xdg-utils \
      ffmpeg fonts-wqy-zenhei \
      libasound2 libatspi2.0-0 libgbm1 libgtk-3-0 libnotify4 libnss3 libsecret-1-0 \
      fluxbox x11vnc x11-utils xorg xvfb; \
    echo "${TZ}" > /etc/timezone; \
    ln -sf "/usr/share/zoneinfo/${TZ}" /etc/localtime; \
    rm -rf /var/lib/apt/lists/*

# QQ: the pinned version, from the first source that has it with the right SHA-256
RUN set -eux; \
    arch="$(dpkg --print-architecture)"; \
    case "${arch}" in \
      amd64) sha="${QQ_AMD64_SHA256}" ;; \
      arm64) sha="${QQ_ARM64_SHA256}" ;; \
      *) echo "QQ for Linux has no ${arch} build here" >&2; exit 1 ;; \
    esac; \
    deb="QQ_${QQ_VERSION}_${arch}_01.deb"; \
    ok=0; \
    for source in ${QQ_SOURCES}; do \
      rm -f /tmp/linuxqq.deb; \
      echo "QQ: trying ${source}"; \
      curl -fL --retry 3 --retry-delay 2 --connect-timeout 20 -A "Violet" -o /tmp/linuxqq.deb "${source}/${deb}" || continue; \
      if echo "${sha}  /tmp/linuxqq.deb" | sha256sum -c -; then ok=1; break; fi; \
      echo "QQ: ${source} gave a different file (SHA-256), not used" >&2; \
    done; \
    [ "${ok}" = 1 ] || { echo "QQ ${QQ_VERSION} could not be downloaded from any source" >&2; exit 1; }; \
    apt-get update; \
    apt-get install -y --no-install-recommends /tmp/linuxqq.deb; \
    rm -f /tmp/linuxqq.deb; \
    rm -rf /var/lib/apt/lists/*; \
    test -x /opt/QQ/qq; \
    chmod 777 /opt/QQ

# SnowLuma (it attaches to QQ: node needs ptrace) and the account it runs as
COPY snowluma-runtime-2.2.1-linux-x64.tar.gz start.sh supervisord.conf /tmp/violet/
RUN set -eux; \
    setcap cap_sys_ptrace+ep /usr/local/bin/node; \
    groupadd --gid 1001 snowluma; \
    useradd --no-log-init --uid 1001 --gid 1001 --home-dir /app --shell /bin/bash snowluma; \
    install -d -o snowluma -g snowluma "${SNOWLUMA_HOME}" "${SNOWLUMA_DATA}" /app/.cache /app/.config /app/.local/share; \
    tar -xzf /tmp/violet/snowluma-runtime-2.2.1-linux-x64.tar.gz -C "${SNOWLUMA_HOME}"; \
    chown -R snowluma:snowluma "${SNOWLUMA_HOME}"; \
    native="$(case "$(dpkg --print-architecture)" in amd64) echo x64 ;; *) echo none ;; esac)"; \
    test -f "${SNOWLUMA_HOME}/index.mjs"; \
    test -f "${SNOWLUMA_HOME}/native/snowluma-linux-${native}.node" || { echo "SnowLuma here is built for x64 only" >&2; exit 1; }; \
    install -m 755 /tmp/violet/start.sh /root/start.sh; \
    install -m 644 /tmp/violet/supervisord.conf /etc/supervisord.conf; \
    mkdir -p /etc/supervisor/conf.d; \
    rm -rf /tmp/violet

LABEL violet.built="snowluma" violet.snowluma="2.2.1" violet.qq="3.2.32_260812"
WORKDIR /app/data
VOLUME ["/app/data", "/app/.config", "/app/.local/share"]
EXPOSE 3001 5900
CMD ["/root/start.sh"]
