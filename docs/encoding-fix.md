# 乱码问题修复总结（文本解析编码）

## 背景

在 Windows 环境下，用户上传的 `.txt` / `.md` 文件可能不是 UTF-8 编码（常见为 **GBK/GB18030**，或带 **UTF-16 BOM**）。原实现使用 `buffer.toString("utf-8")` 固定按 UTF-8 解码，导致：

- 笔记内容入库即乱码（出现 `???`、`�` 等替换字符）
- 后续检索能命中 chunk，但内容不可读，LLM 也可能因为上下文乱码而无法可靠作答

另外，PowerShell 控制台对 UTF-8 的显示也可能造成“看起来乱码”，即使服务端返回的 JSON 实际是正确的 UTF-8。

---

## 根因

- **服务端解码策略过于单一**：无 BOM 且非 UTF-8 的中文文本（GBK/GB18030）会被错误按 UTF-8 解码。
- **终端显示与实际数据可能不一致**：`Invoke-RestMethod | ConvertTo-Json` 直接输出到 PowerShell 时，可能因控制台代码页/字体导致显示异常。

---

## 修复方案

### 1) 服务端：实现“自动识别 + 回退解码”

修改文件：`src/lib/parser.ts`

- 新增 `decodeTextBuffer(buffer)`：
  - 识别 UTF-8/UTF-16（LE/BE）BOM
  - 先尝试 UTF-8
  - 若 UTF-8 结果中 `\uFFFD`（replacement char）比例异常，回退用 **GB18030** 解码
- `parsePlainText()` 与 `parseMarkdown()` 改为使用 `decodeTextBuffer()`

### 2) 依赖：引入 `iconv-lite`

用于支持 `gb18030`、`utf16-be` 等解码：

- 新增依赖：`iconv-lite`

---

## 如何验证（推荐方式）

### A. 验证服务端返回的 JSON 内容是否正常

PowerShell 控制台可能显示不准，建议用 `curl.exe` 保存响应到文件后查看：

```powershell
$id = "<noteId>"
$out = Join-Path $env:TEMP "note.json"
curl.exe -s "http://localhost:3000/api/notes/$id" -o $out
notepad $out
```

若文件中中文正常，说明 **服务端解码与 API 返回均正常**，控制台显示问题可忽略或另行调整终端编码。

### B. 端到端：上传中文文件后问答检索

1. 上传包含中文内容的 `.txt`/`.md`
2. `GET /api/notes/[id]` 检查 `note.content` 与 `chunks[].content` 可读
3. `POST /api/ask` 用中文问题检索，看 `sources[].content` 是否正常中文

---

## 影响范围

- **改善**：上传的中文文本（GBK/GB18030/UTF-16）在入库前会被正确解码，避免“入库即乱码”。
- **不涉及**：PDF 的文本提取由 `pdf-parse` 提供（其解码/提取质量取决于 PDF 本身）。

