# tools/

[English](README.md) | **简体中文**

这个目录放两类东西：服务器上用的安装和管理脚本，以及开发时用的测试工具。开发测试的完整过程记录（Phase 7–9 的验收、真实模型评估）在 [TESTING-HISTORY.md](TESTING-HISTORY.zh-CN.md)。

## 服务器上用的

用户平时只用 `/opt/violet/violet <命令>`，它会调用下面这些脚本。网页后台的“一键安装”按钮调用的也是它们。

| 文件 | 对应命令 | 作用 |
|---|---|---|
| `violet_installer.py` | `install.sh`、`violet update / backup / restore / uninstall / info / reset-password / set-port …` | 安装、修复、升级（失败自动退回）、备份、卸载、访问方式和防火墙。只用 Python 标准库 |
| `violet_setup.py` | （安装时自动运行） | 生成配置和数据目录，可以重复运行 |
| `violet_service.py` | （没有 systemd 时） | Violet 自带的守护进程，用于 Windows、macOS、容器和精简版 Linux |
| `violet_admin.py` | `python -m tools.violet_admin set-password` | 后台管理员密码 |
| `violet_searxng.py` | `violet searxng install / uninstall / status` | 私有的 SearXNG 联网搜索，只在本机监听，不需要 Docker |
| `violet_snowluma.py` | `violet qq install / uninstall / status / restart` | QQ 小号（SnowLuma，Docker 运行）。拉镜像时会自动换镜像站 |
| `violet_voice.py`、`voice_server.py` | `violet voice install / uninstall / status` | 她的声音：CosyVoice 在服务器 CPU 上运行，用一段录音克隆声音 |
| `violet_relay.py` | `violet mcp-relay install / status / uninstall / set-port` | 手机 App Violet MCP 的中继 |
| `make_release.py` | （维护者） | 打包发布文件（`install.sh`、`violet.tar.gz`、`SHA256SUMS`）。GitHub Actions 发布时也用它 |

## 开发时用的

评测工具在隔离的沙盒里跑真实的 Violet 流程：用临时数据库，不连微信等聊天平台，也不读你的聊天记录和记忆。加 `--mock` 用模拟模型，不花钱；不加就用你指定的设置文件里的模型。

| 文件 | 作用 |
|---|---|
| `scenario_eval.py` | 该用的能力她用了吗？约 40 个日常场景（语音、表情包、天气、手机、提醒、心情、闲聊、群聊），每个用你的模型按真实代码跑一遍，自动打分并统计 token，用来比较改动前后 |
| `decision_eval.py` | 记忆整理和主动联系的决策少用一些 token，结果是否一样 |
| `memory_conflicts.py`、`memory_restore_decayed.py` | 只读地列出互相矛盾的记忆；找回被旧版本误删的事件记忆 |
| `schedule_rollback.py` | 从数据库里删掉课表的表（回滚课表功能时用） |
| `fake_mcp_server.py` | 一个只有虚构只读工具的 MCP 服务器，测试用 |
| `legacy.py` | 读取项目改名之前留下的测试记录 |

运行方式：在 `violet/` 目录里执行 `python -m tools.<名字> --help`。
