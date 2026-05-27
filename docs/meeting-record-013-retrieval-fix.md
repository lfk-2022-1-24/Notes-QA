# 会议记录 013：两次检索未命中修复总结

## 现象（用户反馈）

对以下问法：

- `会议记录 013`

用户反馈连续查询两次都未命中，接口返回：

- `sources: []`
- `answer` 进入两阶段追问：`【需要补充上下文】...`

## 复现与定位（本地）

本地复现 `会议记录 013` 后确认：

- `meeting_会议记录.md` 中 **确实存在** `# 会议记录 013`（并包含日期/部门/主持人/参会人/议题等字段）
- 但检索链路在某些情况下会出现 **recordRows=0 / recordNoteIds 为空**，导致后续“按会议记录分段裁剪（section scoping）”无法启动，最终 `sources` 为空并触发追问

## 根因

### 1) 会议记录定位依赖 chunk 召回，可能拿不到记录 header

当前“会议记录 ID”召回的第一优先级来自 `chunks` 表的关键词匹配（`recordRows`）：

- 如果 chunks 的切分没有覆盖 `# 会议记录 013` 那一行（只覆盖了正文的某一段），就可能出现：
  - `detectMeetingRecordHeadingId(recordRows[i].content)` 识别不到 `013`
  - `recordTargetNoteId` 为空 / `recordNoteIds` 不完整

进而无法拿到该 note 的完整 `notes.content` 来计算精确 section bounds。

### 2) “record 概览问法”在 LLM 偶发拒答时，会被当成无资料

即使已经有 record 相关的 `uiSources`，LLM 仍可能输出 `__NO_RELEVANT_INFO__`（比如 embedding 不稳定、上下文裁剪后信息不足等），从而走到“补充上下文”的分支。

## 修复策略（通用）

### 1) note 级定位兜底：recordRows 为空也能找到对应 note 并切片

当识别到 `recordHint = 013` 但 chunk 召回不足以定位 note/section 时：

- 直接在 `notes` 表里用 `notes.content/filename ILIKE` 搜索 `会议记录 013` 的多种 token
- 找到 note 后用 `computeMeetingRecordSectionBoundsFromNoteContent()` 在 **原文** 里计算 section `[start,end)`
- 生成 record section 的 sources（必要时合并到后续的 section scoping / focusedSources 逻辑）

这让“会议记录 013”不再依赖 chunk 边界是否刚好覆盖 header。

### 2) record 概览证据兜底：LLM 拒答时直接抽取头部字段作答

当满足：

- 指定了 `recordHint.id`
- **没有指定子议题**（record overview）
- `uiSources.length > 0` 但 `result.hasAnswer == false`

则从 record header chunk 中直接抽取：

- `日期/部门/主持人/参会人/时长/议题`

拼成答案并带引用（例如 `[1]`），避免用户反复被追问上下文。

## 验证结果（本地）

在 embedding 服务不稳定（日志中出现 `Embedding failed, falling back to keyword-only retrieval`）的情况下，本地复测：

- `会议记录 013`

现在可以稳定返回：

- `hasAnswer=true`
- `sources` 至少 1 个，且包含 `# 会议记录 013` 的基本信息片段

## 相关实现位置

- 后端主逻辑：`src/app/api/ask/route.ts`
  - `recordRows` 为空时的 `notes` 级定位与 section bounds 兜底
  - `record overview` 的拒答证据兜底（从 header 抽取字段）

