# OpenHands 中的 FlowPilot 预测轨迹适配器

本目录是指定 OpenHands SDK 仓库内、可独立安装和合并的 benchmark 扩展。当前主入口支持 **QuixBugs Python 修复**与**LiveCodeBench release_v6 Python stdin easy/medium**。运行统一 OpenHands Agent，每题独立 Conversation、工作目录和 HOME，使用低权限本地工具；最终单独评价。SDK 的 Agent、Conversation、LLM 核心代码没有改动。

2026-09-16 将已验证的外部 `benchmark_adapters` 包移入此目录维护，并新增显式配置/任务 ID 入口。保留包名便于既有代码导入；以后修改此目录，不要同步维护旧外部副本。旧实验的精确源码仍在原始 run 的快照中。

## 与官方协议的关系

- **QuixBugs 没有指定一个唯一官方 agent。** 上游提供40个缺陷程序、公开测试和参考解。适配器复制选定 Python 缺陷程序、必要 helper 和对应公开测试；不把参考解提供给 actor。只提交目标源码，最终恢复原测试再检查。沿用上游默认慢测试跳过规则，不能声称覆盖全部边界或隐藏泛化。
- **LiveCodeBench 的代码生成协议不等于一个固定的多工具 agent。** 本扩展保留题面、公开例子、stdin/stdout接口及固定官方checker；最终在独立工作区检查public＋private cases，隐藏反馈不回传actor。本扩展只支持固定release_v6、easy/medium、stdin，不支持functional/LeetCode方法接口或其他release。
- 两者在这里都是明确的 **OpenHands多轮agentic开发协议**；不声称复制其原始单次生成成绩，也不声称兼容所有第三方agent的轨迹分布。若使用原始无工具单次生成协议，就没有下一环境工具的训练序列。

上游来源：[QuixBugs](https://github.com/jkoppel/QuixBugs/tree/4257f44b0ff1181dedaedee6a447e133219fcebf)、[LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench/tree/28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24)、[固定LCB数据](https://huggingface.co/datasets/livecodebench/code_generation_lite/tree/0fe84c3912ea0c4d4a78037083943e8f0c4dd505)。

## FlowPilot 工具复用

当前复用选择为 `search`、`read_document`、`get_document`，对应 FlowPilot 的
`RetrievalObservation` adapter；旧 Tavily / 原生 BrowseComp MCP 复用配置不再用于此入口。
OpenHands 通用工具能力和既有实验快照保留。

| 工具 | 精确匹配 | 语义匹配 |
| --- | --- | --- |
| search | query 与 top_k（有该参数的后端默认 5） | 只允许 query；其余约束必须相同 |
| read_document | doc_id、start_sentence、max_sentences | 禁止 |
| get_document | docid；SQLite 还包含 offset | 禁止 |

backend、语料版本、snippet_chars/read_chars、schema、部署和 namespace 都参与隔离。
BrowseComp MCP 的 search/get_document 没有 top_k/offset 参数，使用独立 adapter_id，
返回服务端原文。不能根据某次调用是否携带 top_k 判断 benchmark。

用**本次运行同一份 TOML**导出网关 registry，支持同时提供多个配置：

```bash
PYTHONPATH=benchmarks/flowpilot/src .venv/bin/python -m benchmark_adapters.reuse_profile \
  --config /absolute/path/hotpot.toml \
  --config /absolute/path/browsecomp.toml \
  --output /absolute/path/benchmark-registry.json
```

`retrieval.corpus_revision` 必须固定。远程 `hotpot_rpc/browsecomp_mcp` 还需声明
`retrieval.server_policy_revision`，标识实际服务端检索器/模型、k、snippet/tokenizer、
读取范围及返回规则；改变服务配置时必须换版本。该声明不能证明远端资产未被修改。
导出器不执行工具；SQLite/RPC 原有准备阶段继续负责实际索引校验。

网关加载 registry 后，benchmark 运行端设置 `FLOWPILOT_PREDICTOR_GATEWAY`、
`FLOWPILOT_INGRESS_API_KEY` 和 `FLOWPILOT_REUSE_ENABLED=1`。
仅设置预测连接地址不会开启复用。若设置 `FLOWPILOT_EXPERIMENT_PROFILE`，
工具白名单和 exact/semantic/DCS 开关以实验配置为准。
benchmark 不显式分配 `job_id` 或 `line_id`，由 `LocalConversation` 根据自身 UUID
派生 `job-<conversation_id>` 和 `line-<conversation_id>`。同一 campaign 的独立任务
属于不同 Job；恢复同一 conversation 则保持身份。`run_id/task_id/attempt_id` 仍用于实验记录。
默认语义模式 shadow 只观察；active 才允许替代。active 可与 DCS 同时开启：语义命中立即交付，DCS 隐藏续跑仍只接收 exact 命中。

本机全流程实验要求保留工具全文：SDK 的 Chat（list/string）与 Responses 序列化均不按字符数截断，DCS 和缓存同样保留全文。原生 BrowseComp MCP 适配器直接转交完整工具返回文本。模型上下文窗口及 DCS 字节容量仍按配置生效；超限不得靠裁剪结果继续。此规则与早期 50,000 字符截断实验不同，耗时不可作为受控性能对比。
检索工具显式声明 `readOnlyHint=True`，供 SDK 核验委托资格。适配器保留历史 actor 请求的 `summary/security_risk` LLM schema；只读声明不改变实际 MCP 参数或预测器的训练 schema 身份。
直接构造 FlowPilotConfig 时使用 `retrieval_scope(config)` 作为 data_source_constraints。

命中只交付给当前调用的 tool_call_id；真实执行后的 Observation 才能发布。
缓存命中不制造 tool_end 或 RTT 反馈，预测器的实现仍由外部提供。
SDK 现在同时发送本地 ToolDefinition 与 MCP 的真实输入 schema 摘要。
正式采集仍需通过原有 SDK provenance 检查；registry 导出不绕过这一检查。

## 工具与预测时点

两类均有两个环境工具：`code_terminal`、`code_file_editor`，加SDK内置`think`、`finish`两个控制动作，即当前模型看到4个工具schema。`pytest`、`python`、文件查找等是终端参数中的操作，不单独算工具；最终离线评价也不算actor工具。

`llm_request_prepared`是预测T0：已经形成、尚未返回本轮LLM回复的请求快照。保存messages、tool schemas和生成配置作为当时可见输入；事后保存真实提出/执行的工具、完整参数、耗时、超时和未执行状态。`think/finish`、无工具、损坏/截断回复分别处理。不把本轮回复、usage或最终评价混入T0特征。当前仍是采集器，尚未挂在线预测future。

## 安装与依赖

在SDK仓库根目录，使用已安装指定SDK的Python3.12+控制器环境：

```bash
python -m pip install -e ./openhands-sdk
python -m pip install -e ./benchmarks/flowpilot
```

本扩展不加入SDK发布包和根uv workspace，也不修改根锁文件；控制器应先安装本地SDK。保留单独安装边界可避免把benchmark的依赖/评价器带入同门的模型服务环境。当前基线为保留工具全文及采样参数的 DCS 修复版 `c4f3ea625f8fbe72f2af12e879651dcff4ba4ef9`；工具 schema 兼容层依赖该版本的 SDK schema 构造器。适配器/文档单独提交后仍可运行，修改SDK核心或根依赖锁则需要明确更新并重新验证基线。

任务执行环境与控制器分开：Python3.10.12及固定依赖见[task-requirements.txt](task-requirements.txt)。当前共享环境也支持历史ClassEval评价，故包含scipy/func-timeout。重建时用同版本Python创建root所有的venv并安装此文件，不允许任务UID写共享环境。

本地后端依赖Linux、root控制器、`/usr/bin/python3`、`setpriv`、Git和可用任务UID63111/63112。**源数据与结果目录必须对任务UID不可遍历**，建议放在root所有的0700目录。新建结果根目录用0700；已有公开可读目录会被拒绝，不自动修改权限。每题仅将允许的公开文件复制到临时工作区。

这不是Docker/namespace/cgroup/网络沙箱；网络与内核仍共享。路径权限检查只保护本地控制器文件，不承诺隔绝网络上的公开答案。当前固定UID后端应串行使用；并发收集需先实现独立UID/沙箱与资源边界，不能直接并发启动多个本地collector。

## 可复用入口

复制`configs/code/*.example.toml`，设置本机`dataset.path`、`runtime.runs_dir`及模型端点。路径相对于配置文件，`runtime.sdk_path`默认自动定位本SDK仓库；可以显式指定，但安装来源必须一致。

使用同一配置、run ID、任务列表和Python路径依次执行：

```bash
python -m benchmark_adapters.code_collection \
  --config /private/configs/quixbugs.toml \
  --ids bitcount flatten --run-id quix_dev_01 \
  --python-bin /opt/task-python/bin --prepare
# 然后以完全相同参数将 --prepare 换为 --validate，最后换为 --run。
```

任务Python路径及父目录必须对任务UID可执行；它与需要私有权限的**原始数据/控制器输出**不是同一目录。`--python-bin`传venv的bin目录，不将venv解释器符号链接resolve到基础Python。

LiveCodeBench使用其配置及题目ID，额外传固定官方`testing_util.py`：

```bash
python -m benchmark_adapters.code_collection \
  --config /private/configs/livecodebench.toml \
  --ids abc301_a abc302_b --run-id lcb_dev_01 \
  --python-bin /opt/task-python/bin \
  --checker /private/assets/lcb_testing_util.reference.py --prepare
```

`--prepare`只固定选择，不执行任务代码/调用模型；`--validate`执行参考解或checker一致性检查；`--run`要求同一输入、源码及依赖环境已验证，然后运行真实模型并评价。两类可以独立运行，不需要另一类的历史summary。所有源数据内容和checker要通过固定hash检查，不因用户填写revision就视为来源已核验。

当前入口有意只支持`split=dev`。正式fit/tune/calibration/test要先冻结任务/相关题家族划分和选择清单，再扩展相应split策略，不能靠改名字把已用开发题变成未见测试。

新run路径为`runtime.runs_dir / run_id / dataset.kind`。同run输入变化会报错；已有validation、manifest和attempt拒绝覆盖。参数变化、重试或实现变化必须另建run并保留旧数据。旧`code_campaign`只是2026-09-15固定三阶段pilot的兼容入口；新采集使用`code_collection`。

示例采样沿用旧pilot的temperature=0、16384输出、600秒/request、900秒/task，**不是宣称Qwen最佳配置**。正式扩大前应在已用开发题对照模型卡采样与thinking设置；不要混合配置后仅报最高通过率。

## 每题产物与合并边界

`public_task.json`、`profile.json`、`events.jsonl`、请求/回复`blobs/`、`sdk_state/`、`artifacts/solution.py`、`submission.json`、`prediction_samples.jsonl`、`trace_audit.json`、`evaluation/evaluation.json`均独立保存。LCB提交同时包含question_id/code_list。评价结果以独立evaluation文件为准，不把actor result中初始pending误读为未运行评价。

- 复制并提交本目录即可携带全部适配器源代码、测试、配置模板和LCB锁定元数据；无需复制旧的外部adapter路径。
- 不把数据集、模型、venv、实验日志或私有机器配置加入Git。
- 与同门对接的是请求快照/事件ID、工具proposal/execution和耗时接口。调度器/预测器接入位置在`sdk_bridge.py`与`tracing.py`边界，不需要把某一benchmark的评价代码塞入SDK的Agent循环。
- benchmark 通过 SDK 接入独立 FlowPilot 的 Tool Cache，不在采集器内实现缓存或 KV 调度。工具执行时间与评价时间分开，编辑操作不能缓存 stdout 后跳过真实修改。

## 验证

```bash
cd benchmarks/flowpilot
python -m pytest -q
python -m ruff check src tests
python -m ruff format --check src tests
```

实际文件变化与原因见[CHANGES.md](CHANGES.md)；本服务器路径、测量结论及验证记录见[服务器说明](/root/flowpilot/OpenHands代码适配器整合说明_2026-09-16.md)。服务器说明是本地资产，不是另一台服务器的必需依赖。

### 实验 SLO 入口

设置 `FLOWPILOT_EXPERIMENT_PROFILE` 后，runner 读取 `workload.baseline_latency_path`、`baseline_latency_sha256` 与 `slo_multiplier`。基线按 dataset ID、revision、task ID 和题目文本哈希匹配；deadline 等于本次任务开始时间加历史 `duration_s` 乘倍率。在 SDK 会话注册前登记同一 `job-<conversation UUID>`，两次运行使用独立 UUID。SLO 记录写入 `slo.json`，结果包含 `slo_met`、`slo_lateness_s`；SLO 是端到端完成目标，不替代 task_timeout。
