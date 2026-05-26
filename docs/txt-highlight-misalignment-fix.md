# TXT 溯源高亮错位（总是高亮开头）修复总结

## 1. 问题现象

针对 `.txt` 笔记（例如 `agent.txt`）提问时：

- **回答内容是正确的**（检索与回答链路正常）
- 但点击回答中的引用（如 `[1]`）或 Sources 卡片后：
  - 左侧溯源高亮**总是落在文档开头**（开头第一句或第一段）
  - 没有高亮到与问题/答案对应的段落

## 2. 影响范围

- 主要影响 `.txt` 这类“长文 + 高频通用词”场景（如 “Agent/LLM/模型” 等词在全文多次出现）
- 对 MD/PDF/题库类文档可能也会产生类似错位，但表现更常见于 TXT

## 3. 根因分析

本问题实际由两类原因叠加导致：

### A) 后端高亮裁剪定位策略偏差

后端为了让 UI 高亮更精准，会尝试在 chunk 内按“问题关键词”定位匹配点并裁剪出更小的高亮范围。

早期实现存在一个关键缺陷：

- 使用“**最早出现的关键词**”作为定位点
- 对于 `agent.txt` 这种文本，像 `Agent`、`LLM`、`模型` 属于**极高频的通用词**
- 导致匹配点经常被拉到全文最前面，最终 UI 高亮范围也靠近开头

### B) 前端仅依赖 start/end 偏移（偏移看似有效，但位置不对）

即使 `startChar/endChar` 是合法的数值，仍可能指向“与回答引用不一致”的位置。
若前端只按偏移切片高亮，就会出现“偏移有效但高亮不相关”的错位问题。

## 4. 修复方案

本轮修复采用“双保险”：后端更聪明地算高亮范围，前端再做一次 anchor 校准。

### 4.1 后端：选择“最佳命中窗口”而不是“最早命中”

文件：`src/app/api/ask/route.ts`

- **过滤通用词**：把容易到处出现的词（如 `Agent/LLM/模型`）加入 stop list，避免它们主导定位。
- 新增 `findBestKeywordMatchIndex()`：
  - 收集每个关键词的多处出现位置（限制次数）
  - 对候选位置按窗口（例如 idx 前后一定范围）统计“命中关键词数量 + 关键词长度（更具体）”
  - 选择覆盖更多关键词、信息量更大的“最佳窗口”
  - 对靠近文本开头的候选做轻微惩罚，避免无意义地偏向开头
- `refineRangeByQuestionMatch()` 改为用上述“最佳命中窗口”的 `matchIdx` 来裁剪高亮范围

### 4.2 前端：优先用 anchorText 反查校准高亮位置

文件：

- `src/components/ChatPanel.tsx`
- `src/components/SourcePanel.tsx`
- `src/app/page.tsx`（类型扩展）

实现方式：

- 在点击引用按钮 `[n]` 或 Sources 卡片时，除了传 `noteId/startChar/endChar`，还会传 `anchorText`：
  - `anchorText` 取自当前 source 的内容片段（例如前 600 字符）
- 左侧渲染高亮时：
  - **优先**用 `anchorText` 在全文 `note.content` 中执行 `indexOf(anchor)`，找到真实位置
  - 若匹配成功，使用该位置来设置 `startChar/endChar`（对偏移做校准）
  - 再回退到偏移高亮（当 anchor 不存在或匹配失败）

这使得即使后端偏移偶发漂移，或关键词策略仍有误差，也能可靠定位到真正引用的文本片段。

## 5. 验证方式

建议用 TXT 中“非开头段落”相关的问题验证：

- 例如：`agent.txt 里提到 chunk overlap 的作用是什么？`
- 期望：
  - 回答引用 `[1]` / `[5]` 等
  - 点击引用后左侧应高亮到包含“overlap，避免语义断裂”附近的段落
  - 不再高亮文档第一段

## 6. 涉及改动点清单

- `src/app/api/ask/route.ts`
  - `extractMatchTokensFromQuestion`：增加通用词过滤
  - `findBestKeywordMatchIndex`：选择最佳命中窗口
  - `refineRangeByQuestionMatch`：使用最佳命中点进行裁剪
- `src/components/ChatPanel.tsx`
  - 引用点击与 Sources 卡片点击时附带 `anchorText`
- `src/components/SourcePanel.tsx`
  - 高亮时优先用 `anchorText` 在全文中反查并校准位置
- `src/app/page.tsx`
  - `HighlightRange` 增加 `anchorText?: string`

