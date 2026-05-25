# Notes QA — 实施方案（中文版）

## 一、架构总览

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  浏览器 UI   │────▶│  Next.js API     │────▶│  Neon PostgreSQL │
│  (React)    │◀────│  Routes          │     │  + pgvector      │
└─────────────┘     └────┬─────────┬───┘     └─────────────────┘
                         │         │               ▲
                    嵌入API   LLM API         存储/查询
                         │         │         向量+文本块
                         ▼         ▼               │
                  ┌──────────┐ ┌──────────┐        │
                  │  豆包    │ │ DeepSeek │        │
                  │ Embedding│ │ V4 Pro   │────────┘
                  └──────────┘ └──────────┘  (带引用的提示词)
```

**核心流程**：上传笔记 → 解析 & 分块 → 生成嵌入向量 → 存入 pgvector → 用户提问 → 嵌入查询 → 检索 top-k 文本块 → LLM 生成带内联引用的回答 → 展示可点击的来源引用

---

## 二、技术栈选型

| 层级 | 选型 | 理由 |
|------|------|------|
| 框架 | Next.js 14 (App Router) | 部署 Vercel 最短路径；SSR + API 路由一体 |
| UI | Tailwind CSS + shadcn/ui | 构建快，默认样式干净，引用交互组件开箱即用 |
| 嵌入模型 | 豆包/ARK (doubao-embedding-vision-251215, 2048维) | 已在 .env 配置；中英文支持好 |
| LLM | DeepSeek V4 Pro (deepseek-v4-pro) | 已配置 API Key；带推理链，引用遵循能力强，性价比高 |
| 向量数据库 | Neon PostgreSQL + pgvector | 免费层，持久化，原生 SQL，与 Vercel 完美集成 |
| 存储数据库 | Neon PostgreSQL（同一个实例） | 笔记元数据、文本块、用户状态均存于此；无需额外数据库 |
| 文件解析 | pdf-parse (PDF), gray-matter (Markdown), 原生 (TXT) | 轻量级，无外部依赖 |
| 部署 | Vercel | 零配置 Next.js 部署，免费层足够 |

---

## 三、数据库选型详解

### 向量数据库：Neon PostgreSQL + pgvector

**为什么选它**：
- **pgvector** 是 PostgreSQL 的向量扩展，直接在关系型数据库中支持向量存储和余弦相似度搜索
- Neon 提供**无服务器 PostgreSQL**，自动扩缩容、连接池、冷启动快
- 与 Vercel 部署**零摩擦**集成，Vercel 原生支持 Neon 数据库创建
- 免费层：0.5 GB 存储，足够支撑 ~200 篇笔记（约 1000 个文本块 + 2048 维向量）
- 一个数据库同时承担**向量检索 + 结构化存储**，无需维护两套数据库

**其他备选对比**：

| 方案 | 优点 | 缺点 | 结论 |
|------|------|------|------|
| Pinecone | 专用向量数据库，性能好 | 免费层仅 1 个 index、限制多；需额外配结构化数据库 | 过度设计，demo 不需要 |
| Supabase + pgvector | 与 Neon 类似，自带 Dashboard | Vercel 集成不如 Neon 原生；免费层连接数限制更严 | 可行但非最优 |
| Qdrant Cloud | 高性能向量搜索 | 免费层仅 0.5 GB；需额外配结构化数据库 | 多一套系统，增加复杂度 |
| ChromaDB | 本地开发方便 | 无法在 Vercel 无服务器环境运行；无持久化 | 不适合部署 |
| Milvus/Zilliz | 企业级向量数据库 | 重量级，免费层有限 | 远超 demo 需求 |

### 存储数据库：Neon PostgreSQL（同一实例）

**存储内容**：
- `notes` 表：笔记文件元数据（文件名、类型、完整内容、创建时间）
- `chunks` 表：文本块内容 + 嵌入向量 + 位置信息（字符偏移量）
- 未来可扩展：用户表、对话历史表等

**为什么不用独立存储数据库**：
- pgvector 让 PostgreSQL 同时胜任向量检索和结构化查询
- 减少运维复杂度：一个连接串、一套迁移脚本、一个免费层
- 跨表 JOIN 查询自然支持（如：查向量 → JOIN notes 获取文件名）

---

## 四、数据模型

```sql
-- notes: 每个上传文件一行
CREATE TABLE notes (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  filename    TEXT NOT NULL,              -- 原始文件名
  file_type   TEXT NOT NULL,              -- 'md' | 'txt' | 'pdf'
  content     TEXT NOT NULL,              -- 完整原始文本
  created_at  TIMESTAMPTZ DEFAULT now()
);

-- chunks: 每个文本块一行，附带嵌入向量
CREATE TABLE chunks (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  note_id     UUID REFERENCES notes(id) ON DELETE CASCADE,
  chunk_index INT NOT NULL,               -- 在笔记中的顺序
  content     TEXT NOT NULL,              -- 文本块内容（约 300-500 token）
  start_char  INT NOT NULL,              -- 在原始笔记中的字符起始偏移
  end_char    INT NOT NULL,              -- 字符结束偏移
  embedding   vector(2048),              -- pgvector 向量类型
  created_at  TIMESTAMPTZ DEFAULT now()
);

-- 向量索引（IVFFlat，适合 <10万向量规模）
CREATE INDEX ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```

---

## 五、分块策略

- **方法**：优先按段落边界分割。段落超 500 token 时按句子边界分割，带 50 token 重叠。段落太短（<50 token）时与下一段合并。
- **目标大小**：每块 300–500 token（平衡点：足够上下文保证检索准确性，足够小保证引用精度）。
- **元数据**：每块记录 `note_id`、`chunk_index`、`start_char`、`end_char`，用于在源笔记中高亮精确段落。

---

## 六、RAG 流水线

### 1. 笔记导入（POST /api/notes/upload）

```
接收 multipart 文件上传
  → 解析文件（PDF 用 pdf-parse，MD 用 gray-matter，TXT 原生读取）
  → 分块（基于段落，见上方策略）
  → 批量调用豆包 API 生成嵌入向量
  → INSERT 到 notes + chunks 表
  → 返回笔记元数据
```

### 2. 问答查询（POST /api/ask）

```
接收问题字符串
  → 调用豆包 API 生成问题嵌入向量（同一模型）
  → 用 pgvector <=> 算子做余弦相似度检索 top-8 文本块
  → 过滤：仅返回相似度 > 0.5 的文本块
  → 构建提示词，每个文本块标注 [来源 N]
  → 系统提示词指令："仅根据提供的来源回答问题。
     每个论点必须在方括号中标注来源编号 [N]。
     如果来源中没有答案，明确说明无法找到答案。"
  → 调用 DeepSeek V4 Pro 生成回答
  → 解析回答，提取引用索引
  → 返回 { answer, sources: [{ note_id, filename, chunk_text, start_char, end_char, similarity }] }
```

### 3. 无答案处理

- 所有相似度 < 0.5 → 直接返回："在您的笔记中未找到与该问题相关的信息。"不附带来源。
- LLM 回答未引用任何来源 → 仍展示回答，但标注"此回答未基于笔记内容。"
- 两种情况均明确诚实，避免幻觉。

---

## 七、UI 布局

```
┌─────────────────────────────────────────────────────┐
│  Notes QA                          [上传笔记]       │
├──────────────────────┬──────────────────────────────┤
│                      │                              │
│  来源面板            │  对话面板                    │
│  ┌────────────────┐  │  ┌────────────────────────┐  │
│  │ note1.md       │  │  │ Q: 会议的主要结论      │  │
│  │ note2.txt      │  │  │    是什么？             │  │
│  │ paper.pdf      │  │  │                          │  │
│  │                │  │  │ A: 团队决定在Q3发布     │  │
│  │ [点击展开      │  │  │ [1]。预算已批准为       │  │
│  │  并高亮]       │  │  │ 5万元 [2]。            │  │
│  │                │  │  │                          │  │
│  └────────────────┘  │  │ [1] note1.md ¶3  ←点击  │  │
│                      │  │ [2] note1.md ¶5  ←点击  │  │
│                      │  └────────────────────────┘  │
│                      │  ┌────────────────────────┐  │
│                      │  │ 输入问题...        [➤] │  │
│                      │  └────────────────────────┘  │
└──────────────────────┴──────────────────────────────┘
```

**引用交互**：点击 [1] → 来源面板滚动到该文本块并高亮。高亮使用 `start_char`/`end_char` 精确定位原始段落。

---

## 八、API 端点

| 方法 | 路径 | 用途 |
|------|------|------|
| POST | `/api/notes/upload` | 上传一个或多个文件，返回笔记 ID |
| GET | `/api/notes` | 列出所有已上传笔记（id、文件名、文本块数） |
| GET | `/api/notes/[id]` | 获取笔记完整内容 + 文本块边界 |
| DELETE | `/api/notes/[id]` | 删除笔记及其所有文本块 |
| POST | `/api/ask` | 提交问题，获取带引用的回答 |
| GET | `/api/health` | 健康检查 |

---

## 九、分阶段计划（8–12 小时）

### 阶段 1：脚手架 & 数据库搭建（1.5h）
- [ ] `npx create-next-app@latest` 初始化（App Router + Tailwind）
- [ ] 安装依赖：`@neondatabase/serverless`、`pg`、`pgvector`、`pdf-parse`、`gray-matter`、`openai`（兼容 DeepSeek）、`shadcn/ui`
- [ ] 创建 Neon 数据库，执行 schema 迁移
- [ ] 配置 `.env.local` 所有 API Key
- [ ] 测试数据库连接

### 阶段 2：笔记导入（2.5h）
- [ ] 上传 UI：拖拽区域、文件列表、状态指示器
- [ ] 服务端：文件解析（MD/TXT/PDF）
- [ ] 分块逻辑：基于段落的分割 + 重叠
- [ ] 嵌入生成：调用豆包 API，批量处理文本块
- [ ] 存储文本块 + 向量到 pgvector
- [ ] 错误处理：不支持的格式、文件过大、API 失败

### 阶段 3：RAG 查询（2.5h）
- [ ] 对话 UI：问题输入框、回答展示区
- [ ] 服务端：嵌入问题、pgvector 相似度搜索（top-8，阈值 0.5）
- [ ] 构建带引用指令的提示词
- [ ] 调用 DeepSeek V4 Pro
- [ ] 解析回答，提取引用标记
- [ ] 无答案检测与诚实响应

### 阶段 4：引用交互（2h）
- [ ] 来源面板：列出笔记，展开显示内容
- [ ] 回答中内联引用标记 [1]、[2]，可点击
- [ ] 点击 → 滚动到来源文本块，高亮段落
- [ ] 每个来源旁显示相似度分数（透明度）
- [ ] "未找到相关来源"空状态

### 阶段 5：部署 & 文档（1.5h）
- [ ] 推送到 GitHub
- [ ] 部署到 Vercel，配置环境变量
- [ ] 用示例笔记端到端测试
- [ ] 撰写 README（3 个小节，每节 ≤250 词）
- [ ] 录制屏幕操作演示

---

## 十、范围取舍

### 构建的内容
- 单用户模式（无认证）—— 同一时间一套笔记
- 文件上传（MD/TXT/PDF）—— 无实时同步，无 API 导入
- 内联引用 + 来源高亮 —— 诚实的可追溯性
- PostgreSQL 向量搜索 —— 可靠、持久、免费层
- 简洁最小 UI —— 功能优先于美观

### 明确不构建的内容（及原因）

| 砍掉的功能 | 原因 |
|------------|------|
| 用户认证 / 多租户 | Demo 范围；增加复杂度但不证明核心 RAG 价值 |
| 实时笔记同步（Obsidian、Notion） | 文件上传更简单，足以支撑评估 |
| 对话记忆 / 多轮对话 | 单轮 Q&A 即可验证有据回答；多轮增加提示词复杂度 |
| 流式响应 | 体验好但非必须；增加 UI 复杂度，demo 收益小 |
| PDF 图片/表格提取 | pdf-parse 仅处理文本；OCR/表格提取是深坑 |
| 重排序 / 混合搜索 | pgvector 余弦相似度足够；重排序是优化项 |

### 再给 3 天会做的事
- 加入混合搜索（BM25 + 向量）提升关键词密集查询的召回率
- 实现对话记忆支持追问
- 加入流式响应降低感知延迟
- 支持 Obsidian/Notion 笔记库实时同步
- 加入 PDF 图片/表格 OCR（通过视觉模型）

---

## 十一、环境变量

```env
# 数据库
DATABASE_URL=postgresql://...neon.tech/notesqa

# 豆包嵌入 API（已配置）
ARK_API_KEY=ark-...
ARK_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
ARK_EMBEDDING_MODEL=doubao-embedding-vision-251215
ARK_EMBEDDING_DIMENSION=2048

# DeepSeek LLM API（已配置）
DEEPSEEK_API_KEY=sk-...
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_CHAT_MODEL=deepseek-v4-pro
```

---

## 十二、风险与应对

| 风险 | 应对措施 |
|------|----------|
| 豆包嵌入 API 限流或宕机 | 批量处理文本块（每次 10 个），加重试逻辑，可回退到 OpenAI 嵌入 |
| pgvector 索引在大数据集上变慢 | IVFFlat 索引 + 合适的 lists 参数；demo 规模（<200 笔记，~1000 块）无此问题 |
| LLM 幻觉引用 | 严格系统提示词 + 后验证：检查回答中每个 [N] 是否映射到真实来源 |
| PDF 解析失败 | 捕获错误，友好提示，建议重新上传为 TXT |
| Vercel 冷启动延迟 | 使用 Neon 无服务器驱动 + 连接池 |
| DeepSeek API 限流 | 指数退避重试，单次请求超时 30s |
