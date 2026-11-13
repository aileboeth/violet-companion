# tools/

**English** | [简体中文](README.zh-CN.md)

This directory holds two kinds of things: install and management scripts used on the server, and test tools used during development. The full record of development testing (Phase 7–9 acceptance, real-model evaluations) is in [TESTING-HISTORY.md](TESTING-HISTORY.md).

## Used on the server

Users normally only run `/opt/violet/violet <command>`, which calls the scripts below. The one-click install buttons in the web control panel call them too.

| File | Command | What it does |
|---|---|---|
| `violet_installer.py` | `install.sh`, `violet update / backup / restore / uninstall / info / reset-password / set-port …` | Install, repair, upgrade (automatic rollback on failure), backup, uninstall, access mode and firewall. Python standard library only |
| `violet_setup.py` | (runs automatically during install) | Creates the configuration and data directories; safe to run again |
| `violet_service.py` | (when there is no systemd) | Violet's own supervisor, for Windows, macOS, containers and minimal Linux |
| `violet_admin.py` | `python -m tools.violet_admin set-password` | Control panel admin password |
| `violet_searxng.py` | `violet searxng install / uninstall / status` | Private SearXNG web search, listening only on the server itself, no Docker needed |
| `violet_snowluma.py` | `violet qq install / uninstall / status / restart` | Personal QQ account (SnowLuma, runs in Docker). Switches to a mirror automatically when pulling the image |
| `violet_voice.py`, `voice_server.py` | `violet voice install / uninstall / status` | Her voice: CosyVoice running on the server CPU, cloning a voice from one recording |
| `violet_relay.py` | `violet mcp-relay install / status / uninstall / set-port` | Relay for the Violet MCP phone app |
| `make_release.py` | (maintainers) | Builds the release files (`install.sh`, `violet.tar.gz`, `SHA256SUMS`). GitHub Actions uses it when publishing |

## Used during development

The evaluation tools run the real Violet flow in an isolated sandbox: their own throw-away database, no connection to WeChat or other chat platforms, and no access to your chats or memories. `--mock` uses a simulated model, which costs nothing; otherwise they use the model of the settings file you point them at.

| File | What it does |
|---|---|
| `scenario_eval.py` | Does she use her abilities when she should? About 40 everyday situations (voice, stickers, the weather, the phone, reminders, her mood, small talk, groups) run once each through the real code with your model; every one is scored and its tokens are counted, to compare before and after a change |
| `virtual_test.py` | Long-run test in virtual time: a second model plays you and chats with her for days (memory, corrected facts, reminders, moods, proactive messages, a silent day, a difficult user, every persona); a judge model reads each day and lists problems with quotes. Runs on a throw-away copy of the data, never touches her real chats |
| `decision_eval.py` | The memory step and the proactive decision with fewer tokens: the same result? |
| `memory_conflicts.py`, `memory_restore_decayed.py` | Lists conflicting memories (read-only); recovers event memories that an older version deleted by mistake |
| `schedule_rollback.py` | Drops the timetable tables from the database (used when rolling back the timetable feature) |
| `fake_mcp_server.py` | An MCP server with only made-up read-only tools, for testing |
| `legacy.py` | Reads test records written before the project was renamed |

To run one: in the `violet/` directory, run `python -m tools.<name> --help`.
