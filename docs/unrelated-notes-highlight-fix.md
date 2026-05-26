# 引用溯源后“其它文档也出现高亮”的问题与修复

## 1. 问题现象

当用户在回答中点击引用（如 `[1]`）进行溯源后：

- 被引用的笔记可以**精准定位并高亮**（主流程正确）
- 但在高亮仍然存在的时间窗口内，如果用户去展开/查看其它笔记（其它 md/txt/pdf），会看到：
  - 其它笔记内容中也出现一小段 `<mark>` 高亮
  - 高亮内容与本次问答**完全无关**

该问题会造成“系统在胡乱高亮”的观感，即使引用笔记本身定位正确，也会干扰用户判断。

## 2. 根因分析

当前 UI 的高亮状态由页面级 state 管理：

- `src/app/page.tsx` 中 `highlight` 被设置后，会在 **8 秒后**清除（`setTimeout(() => setHighlight(null), 8000)`）
- 在这 8 秒内，用户如果展开其它笔记，`SourcePanel` 仍会收到同一个 `highlight` 值

在旧逻辑中，`SourcePanel` 的 `renderHighlightedContent` 只要 `highlight` 不为空，就会渲染 `<mark>`：

- 没有判断当前渲染的笔记是否就是 `highlight.noteId` 对应的笔记
- 导致高亮被“带到”其它笔记内容上（因为同一个 `highlight` 被复用）

## 3. 修复方案

### 3.1 只在被引用的笔记里渲染高亮（硬约束）

文件：`src/components/SourcePanel.tsx`

在 `renderHighlightedContent` 开头增加判断：

- 仅当 `highlight.noteId === noteDetail.note.id` 时，才渲染 `<mark>`
- 否则直接渲染普通文本，不做高亮

该约束确保：

- 高亮只会出现在“被引用的那篇笔记”中
- 用户展开其它笔记时，不会出现高亮残留/串台

## 4. 验证方式

1. 提问并点击某个引用 `[n]`，确认左侧展开并高亮正确位置
2. 在高亮未消失（8 秒内）时，手动展开另一个笔记
3. 期望：另一个笔记**不出现任何高亮**

## 5. 涉及文件

- `src/app/page.tsx`：高亮 state 生命周期（8 秒清除）为触发背景
- `src/components/SourcePanel.tsx`：修复点（仅在引用笔记中高亮）

