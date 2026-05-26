# 前端检索很慢/一直出不来（Embedding 不稳定）问题与修复

## 1. 问题现象

前端提问后出现以下情况：

- 一直显示“Searching…”或长时间无响应
- 偶发直接返回 500
- 同一问题有时 20s+ 才返回

## 2. 直接原因（从日志定位）

后端日志显示 `/api/ask` 在调用 embedding 外部服务时频繁失败，例如：

- `ECONNRESET`
- `fetch failed`
- `Client network socket disconnected before secure TLS connection was established`

embedding 请求目标为：

- `ark.cn-beijing.volces.com` 的 `/embeddings/multimodal`

当该外部依赖抖动时：

- `/api/ask` 会卡在 embedding 获取向量阶段（或直接抛错）
- 前端表现为一直转圈或请求失败

## 3. 根因分析

### A) Node fetch 默认无超时，网络抖动会“拖很久”

embedding 调用使用 `fetch()`，未设置超时与重试策略。外部服务出现连接重置/抖动时，会导致请求长时间悬挂，阻塞整个问答链路。

### B) Embedding 失败时没有可靠降级

当 embedding 不可用时，如果后端仍强依赖向量检索，就会出现：

- 直接 500
- 或返回时间极不稳定

## 4. 修复方案

### 4.1 Embedding 调用增加超时 + 重试 + 降并发

文件：`src/lib/embedding.ts`

- 为 embedding 请求增加超时（默认 8s，可配置）
- 失败后重试 1 次并做轻微 backoff
- 批量 embedding 并发从 20 降到 8，降低外部服务抖动时的雪崩概率

可配置项：

- `ARK_EMBEDDING_TIMEOUT_MS`：embedding 请求超时（默认 `8000`）

### 4.2 LLM 调用增加超时

文件：`src/lib/llm.ts`

- 为 DeepSeek chat completions 增加超时（默认 20s，可配置）

可配置项：

- `DEEPSEEK_TIMEOUT_MS`：LLM 请求超时（默认 `20000`）

### 4.3 `/api/ask` 增加 keyword-only 检索降级路径

文件：`src/app/api/ask/route.ts`

- embedding 成功：走原本的“向量检索 + 混合检索 + 重排”
- embedding 失败：自动降级为 **keyword-only** 检索（不依赖向量服务），保证仍可回答并返回 sources

#### 降级路径中的 SQL 占位符修复

降级初版中曾出现 Postgres 报错：

- “无法确定参数 $1 的数据类型”

原因是 keyword-only 分支下 SQL 的 `$n` 占位符与参数数组顺序不一致。
已修复为：

- vector 分支与 non-vector 分支使用不同的 SQL 模板与参数编号（保证占位符严格匹配）

## 5. 验证方式

### 5.1 正常路径

1. 运行前端并提问任意问题（例如“索引是什么”）
2. 观察请求应在可接受时间内返回（无长时间转圈）

### 5.2 降级路径

在 embedding 服务不可用（或被网络重置）时：

- `/api/ask` 仍应返回 `200`
- `sources` 不为空（至少有 keyword 命中）
- 前端不会一直 loading

## 6. 涉及文件清单

- `src/lib/embedding.ts`：embedding 超时/重试/并发控制
- `src/lib/llm.ts`：LLM 请求超时
- `src/app/api/ask/route.ts`：embedding 失败自动降级 keyword-only + SQL 参数修复

