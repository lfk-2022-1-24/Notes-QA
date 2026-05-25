# 检索未命中（Q4）问题总结与修复

## 问题现象

上传一个包含问答段落的 Markdown 笔记（例如包含：

> `### Q4: 如何保证审核结果的准确性？`

以及对应回答），在前端提问：

> “如何保证审核结果的准确性？”

出现以下情况之一：

- 返回 `hasAnswer=false` / “找不到相关信息”
- 或者 `sources` 命中了其它不相关段落（弱匹配），没有命中 Q4 段落

---

## 排查结论

### 1) 数据本身是正确入库的

通过将 `GET /api/notes/[id]` 的响应保存为文件（避免 PowerShell 控制台编码干扰），确认笔记内容与 chunks 中包含完整中文 Q4 段落。

推荐验证方式（PowerShell）：

```powershell
$id = "<noteId>"
$out = Join-Path $env:TEMP "note.json"
curl.exe -s "http://localhost:3000/api/notes/$id" -o $out
notepad $out
```

### 2) “看起来没命中”可能与终端显示有关

PowerShell 直接 `Invoke-RestMethod | ConvertTo-Json` 输出时，可能因为控制台编码/字体导致：

- 中文显示乱码
- 输出截断

从而误以为没检索到正确来源。因此在诊断检索问题时，优先用 `curl.exe` 保存 JSON 再查看。

---

## 根因分析（为什么向量检索会漏掉 Q4）

在“问答标题 + 引用块（blockquote）”这种 Markdown 结构里：

- 真实答案可能分散在较长 chunk 的中后部
- 问题是短句，Embedding 相似度不稳定
- 纯向量 TopK + 固定阈值/过滤策略，容易出现：
  - top1 相似度偏低
  - 过滤后只保留弱匹配
  - Q4 chunk 没被召回或没进入最终 sources

---

## 修复策略（已落地）

### A. Hybrid 检索：向量召回 + 关键词/短语兜底

在 `src/app/api/ask/route.ts` 中增加了关键词/短语召回逻辑（`ILIKE`）：

- 先做向量检索（pgvector `<=>`）
- 再用 `ILIKE` 做“关键词/短语”召回，尤其针对 Q/A 风格：
  - 精确问题短语（含 `？` 与去标点版本）
  - `Q4:` 前缀变体
  - 中文 2~3 字 n-gram（提升召回率）
- 合并两路候选，按“精确短语命中优先 + 相似度”排序

效果：即使 embedding 相似度偏低，仍能稳定把包含 `Q4: 如何保证…` 的 chunk 召回到 `sources` 中。

### B. 控制进入 LLM 的 sources 数量

保留较多候选用于 UI 展示/调试，但传入 LLM 的 sources 只取前若干条（例如 Top8），避免把无关 chunk 塞进上下文稀释注意力。

### C. Markdown 清洗仅用于 embedding（不破坏高亮 offset）

上传入库时：

- `chunks` 的 `content/start_char/end_char` 仍基于**原文**分块，保证前端点击引用高亮不偏移
- 仅在计算 embedding 输入时做 `normalizeForEmbedding()` 清洗（去 frontmatter/标题符号/列表符号/链接等）

---

## 如何验证修复是否生效

### 1) 用 curl 保存 ask 响应（避免 PowerShell 显示干扰）

```powershell
$payload = Join-Path $env:TEMP "ask-q4-payload.json"
$out = Join-Path $env:TEMP "ask-q4.json"
Set-Content -Path $payload -Value '{"question":"如何保证审核结果的准确性？"}' -Encoding utf8
curl.exe -s -X POST "http://localhost:3000/api/ask" -H "Content-Type: application/json" --data-binary "@$payload" -o $out
notepad $out
```

检查 `sources[].filename` 是否包含目标 md，且 `sources[].content` 中能看到 Q4 段落。

---

## 后续可进一步提升的方向（可选）

- **QA-aware 分块**：上传时把 `### Q\d+`、`Q\d+:` 识别为 chunk 边界，让“问题+答案”尽量落在同一个 chunk 中，进一步提升向量检索稳定性。
- **Hybrid Search (BM25 + 向量)**：对“短问句/精确术语”效果更稳，比手写 n-gram 更通用。
- **Rerank**：对 TopN 候选用更强的 cross-encoder 或 LLM 做重排序，提高精度并减少误匹配。

