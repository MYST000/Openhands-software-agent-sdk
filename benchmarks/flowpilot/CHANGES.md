# 本轮修改文件与原因

所有路径相对 `benchmarks/flowpilot/`。本目录是新增到OpenHands Git仓库的完整扩展代码；包内部分文件沿用已完成真实试跑的外部实现。没有修改SDK的Agent/Conversation/LLM核心文件。

| 文件 | 本轮处理 | 原因 |
| --- | --- | --- |
| `src/benchmark_adapters/code_tasks.py` | 迁入已有QuixBugsAdapter、LiveCodeBenchAdapter，保留ClassEval兼容实现 | 保持公开输入、隐藏评价和提交边界；不把代码生成题伪装成SWE issue |
| `src/benchmark_adapters/code_collection.py` | 新增正式入口 | 支持显式配置/精确任务ID、prepare/validate/run；两种benchmark可独立收集，拒绝重复ID、未知题、非支持协议及覆盖旧run |
| `src/benchmark_adapters/data/lcb_release_v6.lock.json` | 新增固定6个数据分片及checker的SHA256 | 合并到其他机器后仍能确认实际文件对应固定版本，而非仅相信用户填写的revision |
| `src/benchmark_adapters/controller_paths.py` | 新增低权限访问检查 | 防止更换数据路径后把参考解/private cases/历史评价暴露给actor；新输出目录0700，已有可访问目录拒绝运行 |
| `src/benchmark_adapters/sdk_provenance.py` | 新增核心内容校验 | 单独提交适配器后HEAD变化或仓库新增文件不应阻止运行；仍检测核心源码/锁文件与基线的差异、未跟踪核心文件、错误editable安装来源 |
| `src/benchmark_adapters/config.py` | SDK路径改为从包所在仓库推导 | 避免在同门机器上仍固定读取/root/predictor_exp；显式配置仍可覆盖 |
| `src/benchmark_adapters/cli.py` | 使用统一SDK校验 | SWE/旧入口也不会仅因加入本扩展就报告SDK被非法修改 |
| `src/benchmark_adapters/code_campaign.py` | 增加显式的独立阶段调用方式与run ID，改为记录实际SDK来源/模型端点 | 新入口不依赖另一数据集summary；不再无证据硬写服务模型路径或最大上下文；旧阶段默认前置检查保留 |
| `src/benchmark_adapters/runner.py`、`sdk_bridge.py`、`tracing.py` | 原实现迁入 | 每题独立Conversation；同样的两类环境工具和think/finish；保留请求前快照、完整参数、关联ID和实际计时 |
| `src/benchmark_adapters/environment.py`、`local_environment.py`、`code_evaluation.py`、`code_audit.py` | 原实现迁入 | 保留真实执行、低权限UID、已修复的进程清理、独立评价及轨迹审计；不改变历史测量定义 |
| `pyproject.toml` | 扩展版本0.2.0，新增openhands-code-collect命令及锁定元数据打包 | 可从本仓库独立安装，继续使用benchmark_adapters导入名，不依赖旧外部目录 |
| `configs/code/*.example.toml` | 新增两类配置模板 | 固定任务协议，显式声明模型/工具/运行预算；机器路径由本地配置填写 |
| `tests/test_collection_entrypoint.py` | 新增选题、冻结、SDK校验和目录权限测试 | 防止合并导致不能运行、默默换题或隐藏测试泄漏；root专属验证在非root下显式skip |
| 其余`src/`、`tests/`及历史配置 | 迁入共享/兼容依赖 | 保持原包导入与SWE/检索代码兼容；不表示本轮重新运行了SWE/检索/ClassEval |
| `.gitignore`、`task-requirements.txt`、`README.md` | 新增 | 排除运行数据/环境/缓存；给出固定依赖、使用方式与协议边界 |
| SDK根`README.md` | 添加本扩展入口 | 同门从OpenHands仓库可以找到适配器与安装说明 |

完整逐文件hash/迁入差异在本机整合记录的`extension_files.json`。数据、模型、run产物和venv留在Git仓库外。旧外部README增加迁移提示；旧源码和历史run不覆盖。

合并时包含整个 `benchmarks/flowpilot/` 和根README入口即可；尚未自动提交或推送。若同门同时修改SDK核心，不能直接沿用旧基线的通过结论：应明确记录合并后SDK版本，更新基线并重跑适配器和任务验收。
