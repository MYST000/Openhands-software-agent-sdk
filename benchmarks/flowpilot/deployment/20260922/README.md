# 2026-09-22 迁移资料

完整操作说明 `交接文档9.22.md` 由用户本地单独交付，不包含在 GitHub 发布中。

`snapshot/` 收录固定依赖锁、数据来源与划分元数据、下载后校验脚本、检索启动器和 C4 开发配置。没有数据正文、答案、隐藏测试、模型、轨迹、Python 环境或凭据。

在新机克隆仓库后执行 `python3 materialize.py --root /目标项目根目录`，即可按原项目布局还原。脚本先验证 `SNAPSHOT_SHA256SUMS`，仅替换开发配置中的原项目根路径，拒绝覆盖已有不同文件；不联网、不安装软件、不启动服务。仓库约定放在目标根目录下的 `repos/Openhands-software-agent-sdk`。

原样保存的历史锁文件和来源记录可能包含原机器路径；它们不能代替新机的验收记录。检索 serve.py 和 openhands_env.sh 已改为按脚本位置定位根目录，BM25 和官方工具语义保持原样。

prepare_downloaded_data.py 的 repositories 阶段仍核对历史提交 8e8ed52。本次迁移只执行文档指定的 hashes lcb browsecomp hotpot_questions 阶段，Git 来源由单独的提交检查负责，不能无参数调用该历史脚本。

Hotpot 远程 RPC 未实现。本资料提供其全量语料下载、SQLite 索引构建和本地运行配置；不能把改 endpoint 当成已有远程支持。真实代码 UID 隔离和 Qwen C4 容量需在新机验收。

公开版将内部主机/账号和历史绝对路径替换为占位路径；SHA256 清单针对脱敏后的发布快照重新生成。数据本身的校验值、原协议题目分组与 SDK 核心不变。
