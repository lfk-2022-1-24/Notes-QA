# 会议记录 003：检索失败与引用修复总结

## 现象（用户反馈）

对以下问法：

- `会议记录 003主要包含哪几部分内容`
- `介绍会议记录 003`
- `会议记录 003`

系统会出现：

- 回答不准确或直接提示 `【需要补充上下文】`
- 引用块（sources）为空或引用到不正确的块

## 复现与定位

本地测试时可以确认：

- `meeting_会议记录.md` 这类文件是“多条会议记录拼接在同一份 note 内容”结构
- `# 会议记录 003` 在 note 原文中确实存在
- 但 API 返回中 `sources` 为空，说明“003 的 chunk 被过滤掉”，而不是笔记本身缺内容

## 根因

我们为了避免同一份 note 里混入其他会议记录的 chunk，会先计算“会议记录 003 在 note 原文中的 section 起止范围”：

- sectionStart：`# 会议记录 003` 在 `notes.content` 中的位置
- sectionEnd：下一条 `# 会议记录 XXX` 的位置（或文档末尾）

然后对候选 chunk 做 section 过滤。

问题在于：**chunk 的边界不一定与会议记录的 section 对齐**。

常见情况是 chunk 会跨越边界（例如 chunk 从会议记录 002 的末尾开始，但内容一直延伸到 `# 会议记录 003` 的 header 与后续正文）。

此前的过滤条件是：

- 仅保留 `startChar` 落在 `[sectionStart, sectionEnd)` 范围内的 chunk

这会把“跨界但包含 003 header”的 chunk 误删，导致 `sources` 为空，最终触发两阶段“补充上下文”提示。

## 修复策略

将 section 过滤从“startChar 在范围内”改为“chunk 与 section 范围有重叠即可”：

- **旧逻辑**：`startChar >= start && startChar < end`
- **新逻辑**：`endChar > start && startChar < end`

该逻辑能保留任何与会议记录 section 有交集的 chunk，从而保证 `# 会议记录 003` 的 header chunk 不会被误删。

## 验证结果（本地）

修复后本地复测：

- `会议记录 003`
- `介绍会议记录 003`
- `会议记录 003主要包含哪几部分内容`

均可：

- `hasAnswer=true`
- `sources` 至少返回 1 个且包含 `# 会议记录 003` 开头的引用块

## 相关实现位置

- 后端主逻辑：`src/app/api/ask/route.ts`
- 修复点：record section 过滤的 `sectionScopedSources` 逻辑（由 start-in-range 改为 overlap-with-range）

