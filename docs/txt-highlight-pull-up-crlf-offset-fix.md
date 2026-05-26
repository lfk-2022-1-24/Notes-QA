# TXT 溯源高亮“往上拉”（包含上一段无关内容）问题与修复

## 1. 问题现象

在 `.txt` 笔记中提问并点击引用溯源时：

- **引用块（source.content）内容是正确的**（例如从 `6. 为什么 Agent 需要 Tool Use？` 开始）
- 但左侧笔记高亮区域会出现“往上拉”的现象：
  - 高亮包含了题目/答案之前的上一段模板文字或无关内容
  - 造成“上半部分也被扫黄”的体验

示例表现：

- `source.content` 开头正确：`6. 为什么 Agent 需要 Tool Use？ ...`
- 但 `note.content.slice(startChar, endChar)` 却包含：`不要遗漏关键步骤... 用户目标... 6. 为什么...`

## 2. 根因定位（关键结论）

根因在 **后端裁剪引用块时使用了“换行归一化文本”的索引，但返回偏移却仍按原始字符串计算**，导致 `startChar/endChar` 相对于原文被“提前”，从而覆盖到上一段。

具体来说：

- 后端为了统一匹配逻辑，经常把 chunk 文本做了：
  - `chunkContent.replace(/\\r\\n/g, "\\n")`
- 在这个“归一化后的文本”上计算出来的 `startIdx/endIdx` 是基于 **LF** 的索引
- 但 `note.content`/DB 中保存的是原始文本，常见为 **CRLF**
  - 同样的视觉位置，在原始字符串中的索引会更大（因为每个换行多一个 `\\r`）
- 因此把“LF 索引”直接当成原文索引返回给前端，会产生偏移错位，表现为高亮“往上拉”

这也是为什么：

- 引用块内容看起来正确（因为 content 是在归一化文本里截出来的）
- 但高亮位置不正确（因为 start/end 对原文错位）

## 3. 修复方案

### 3.1 引入“归一化索引 → 原文索引”映射

文件：`src/app/api/ask/route.ts`

新增两类工具函数：

- `buildLfTextAndMap(orig)`
  - 输出 `text`：把 CRLF 归一化成 LF 的文本（用于匹配/regex）
  - 输出 `map`：建立 `map[normIndex] = origIndex` 的索引映射
- `mapNormRangeToOrig(map, startNorm, endNorm, origLen)`
  - 把归一化文本中的区间 `[startNorm, endNorm)` 映射回原文区间 `[startOrig, endOrig)`

### 3.2 将所有裁剪函数统一改成“在归一化文本上算位置，再映射回原文”

涉及的裁剪函数（均在 `src/app/api/ask/route.ts`）：

- `refineRangeByQuestionMatch`
- `refineRangeBySentenceMatch`
- `refineRangeByMarkdownLineMatch`
- `refineRangeByDefinitionTopic`
- `refineRangeByNumber`

修改原则：

1. 用 `buildLfTextAndMap(chunkContent)` 得到 `text/map`
2. 所有 `indexOf/regex` 都在 `text` 上做
3. 最终把 `[startIdx, endIdx)` 通过 `mapNormRangeToOrig` 映射成原文区间
4. 返回的：
   - `content` 使用 `chunkContent.slice(mapped.start, mapped.end)`
   - `startChar/endChar` 使用 `chunkStartAbs + mapped.start/end`

这样保证：

- `sources[i].content` 与 `note.content.slice(startChar,endChar)` 一致
- 高亮不会再“往上拉”

## 4. 验证方式

用同一个 `.txt` 笔记验证（例如 `agent.txt`）：

1. 提问：`为什么 Agent 需要 Tool Use？`
2. 点击引用 `[1]`
3. 期望：
   - 高亮从 `6. 为什么 Agent 需要 Tool Use？` 开始
   - 不再包含上一段“不要遗漏关键步骤/用户目标...”等模板内容

建议做一个后端校验（用于排查类似问题）：

- 验证 `note.content.slice(startChar,endChar)` 的开头是否与 `source.content` 的开头一致

## 5. 影响范围与说明

- 该修复主要针对 **TXT/Windows CRLF** 场景最明显的问题，但同样适用于任何包含 CRLF 的笔记内容。
- 不改变检索/回答逻辑，只修正“引用块裁剪与偏移回传”的一致性。

