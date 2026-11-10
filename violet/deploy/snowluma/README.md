# QQ container (SnowLuma) built by Violet

SnowLuma's own Docker image (`motricseven7/snowluma`) and the repositories it came from went offline in November
2026. `violet qq install` therefore builds the container for the channel "QQ" on your server from this folder:

| File | What it is |
| --- | --- |
| `Dockerfile` | the build: Node.js base image, a virtual screen (Xvfb, fluxbox, x11vnc), QQ, SnowLuma |
| `start.sh`, `supervisord.conf` | start the screen and the VNC server, then QQ and SnowLuma (written for Violet) |
| `snowluma-runtime-2.2.1-linux-x64.tar.gz` | SnowLuma 2.2.1 for Linux x64, the build that ran with QQ 3.2.32 |
| `LICENSE.SnowLuma.txt` | SnowLuma's license (MIT) |

**QQ is not part of Violet.** It is Tencent's software and is downloaded during the build: Linux QQ 3.2.32
(`QQ_3.2.32_260812`) from Tencent's own address first, else from the
[Rodert/qq-versions](https://github.com/Rodert/qq-versions) archive. Either file must match the SHA-256 in the
`Dockerfile`, or the build stops. QQ's silent hot update is blocked inside the container, so QQ stays the version
SnowLuma was made for.

Third-party parts inside the SnowLuma package keep their own licenses: SQLite (`native/sqlite/LICENSE.sqlite3`) and
the FFmpeg addon (`native/ffmpeg/`, FFmpeg is LGPL/GPL software). SnowLuma is an independent project, not affiliated
with Tencent; using a personal QQ account through it is at your own risk (QQ's terms of service).

Only x64 (amd64) servers can build it: the SnowLuma package here is the x64 build. Other servers can still load a
saved copy of an image (`violet qq install --image-file …`).
