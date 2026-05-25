# 引用溯源（高亮/跳转）与 Sources 展示修复总结

## 背景与目标

本轮主要围绕两类体验问题做修复：

- 回答后默认展示 Top 8 检索块（Sources），但用户希望**只展示回答中实际引用到的块**。
- 点击回答中的引用（如 `[1]`）后，左侧溯源不精准：
  - 高亮范围过大（整段 chunk 甚至跨多题）
  - 左侧不自动滚动到高亮位置，需要手动下拉寻找
  - 在 PDF/题库类文档里尤其明显；但并非所有 PDF/MD/TeX 都带数字题号
- 修复高亮后出现副作用：**MD 检索/回答准确性下降**、回答引用与高亮内容不相关。

## 1. 只展示“回答里引用到的 Sources”

### 现象

回答下方固定展示 8 个检索块的匹配情况，但其中很多块并未在回答中被引用，造成前端噪声与干扰。

### 改动

- 在 `src/components/ChatPanel.tsx` 中，从答案文本提取所有 `[\d+]` 引用标记（如 `[1]`、`[3]`），仅渲染这些 `source.index` 对应的 Sources 卡片。
- 若答案没有任何引用，则 Sources 区域不展示。

### 关键点

- 引用按钮仍保持可点击，并把 `noteId/startChar/endChar` 传递给左侧溯源面板。

## 2. PDF 上传报错：`pdfParse is not a function` / `pdf-parse export is not a function`

### 现象

上传 PDF 时后端解析报错，提示 `pdfParse is not a function` 或 `pdf-parse export is not a function`。

### 根因

当前依赖的 `pdf-parse@2.4.5` **不再导出旧版函数式 API**，而是导出 `PDFParse` 类（以及异常类型等）。因此把 `pdf-parse` 当函数调用会失败。

### 改动

- 在 `src/lib/parser.ts` 中改为新版用法：
  - 动态 `import("pdf-parse")` 取 `PDFParse`
  - `new PDFParse({ data: buffer }).getText()` 获取文本
  - `finally` 中 `destroy()` 释放资源

## 3. 点击引用后的溯源体验：自动展开与滚动定位

### 现象

- 点击引用后左侧会高亮，但：
  - 没有自动跳转到高亮位置
  - 高亮可能在很长的笔记内容中，需要手动滚动查找

### 改动

- 在 `src/components/SourcePanel.tsx` 中新增滚动逻辑：
  - 当 `highlight.noteId` 变化并加载完对应 note 后：
    - 先把展开的 note header `scrollIntoView({ block: "nearest" })`
    - 再把 `<mark>` 高亮元素 `scrollIntoView({ block: "center" })`

## 4. 高亮范围精准化策略（兼容“有题号 / 无题号”的文档）

### 需求特点

- PDF/题库类常见格式：`14.`、`14、`、`Q14:`、`第14题`、`(14)` 等。
- 但用户问题本身可能不带“第14题”，例如只问：“什么是国歌？为什么听到国歌要站好？”
- 同时，MD/TeX 等并不总是题号结构，不能用“题号边界”去裁剪。

### 策略 A：题号明确时（问题中带题号）

在 `src/app/api/ask/route.ts`：

- 从问题（或改写后的检索问题）里解析题号（`extractQuestionNumber`）。
- 在 chunk 内定位该题的题头，并向下找到下一题题头作为结束边界（`refineRangeByNumber`）。
- 将 UI 的 `content/startChar/endChar` 收窄到该题块。

### 策略 B：问题不带题号时（靠关键词定位 + 回溯题头）

在 `src/app/api/ask/route.ts`：

- 从问题中抽取少量高价值关键词（例如“国歌”）（`extractMatchTokensFromQuestion`）。
- 在 chunk 内找到关键词命中点。
- 若 chunk 看起来像“题号问答结构”，则：
  - 向上回溯到最近的“题头”作为起点
  - 向下找到下一题头作为终点
- 若 chunk 不像题库结构，则回退为“围绕命中点的窗口截取”（避免误判）。

### 策略 C：定义类问题（X 是什么）优先走 MD 结构边界

为了解决 MD 文档中“索引是什么”这类定义题高亮容易跑偏的问题：

- 从问题中提取定义主题（`extractDefinitionTopic`），支持纯中文结尾（如 `索引是什么`）。
- UI 高亮优先锚定到：
  - `什么是X` / `X是什么`
  - `## X` / `### X`
  - 以及加粗标题形态
- 再按 Markdown heading / 段落边界收窄范围（`refineRangeByDefinitionTopic`）

## 5. 避免“高亮裁剪影响回答准确性”：拆分 LLM sources 与 UI sources

### 问题

为了精准高亮而对 sources 做了裁剪后，会影响 LLM 看到的上下文，导致：

- MD 检索/回答准确性下降
- 回答引用与高亮内容不相关（引用指向被裁剪后不完整/偏离的片段）

### 修复

在 `src/app/api/ask/route.ts` 做了**两套 sources 分离**：

- `focusedSources`（给 LLM）：始终使用**完整 chunk**（不做 UI 裁剪），保证回答稳定、准确。
- `uiSources`（返回前端用于展示/高亮）：在 `focusedSources` 的基础上做题号/关键词/定义题的收窄，提升溯源体验。

## 6. 验证方式（建议）

- **Sources 展示**：
  - 提问后检查回答中引用了哪些 `[n]`，Sources 区域仅出现这些编号的卡片。
- **溯源跳转**：
  - 点击 `[n]` 后左侧自动展开对应 note，并滚到 `<mark>` 高亮位置。
- **PDF 解析**：
  - 上传 PDF 不再报 `pdfParse is not a function` / `export is not a function`。
- **题库类 PDF 高亮**：
  - 用户不带题号提问，但 PDF 有 `14.`/`14、` 等题号时，高亮尽量收敛到单题块。
- **MD 定义题稳定性**：
  - 提问“索引是什么”，答案应稳定命中 `### 索引是什么` 的定义段，引用与高亮一致。

## 涉及文件清单

- `src/components/ChatPanel.tsx`：只渲染回答引用到的 Sources
- `src/components/SourcePanel.tsx`：自动滚动定位到高亮
- `src/lib/parser.ts`：PDF 解析改为 `PDFParse` 新 API
- `src/app/api/ask/route.ts`：
  - 高亮精准化（题号/关键词/定义题）
  - LLM sources 与 UI sources 拆分

