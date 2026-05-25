# Notes QA — Implementation Plan

## Architecture Overview

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Browser UI  │────▶│  Next.js API     │────▶│  Neon PostgreSQL │
│  (React)     │◀────│  Routes          │     │  + pgvector      │
└─────────────┘     └────┬─────────┬───┘     └─────────────────┘
                         │         │               ▲
                    embed│    LLM  │          store/query
                    API  │    API  │          vectors+chunks
                         ▼         ▼               │
                  ┌──────────┐ ┌───────┐           │
                  │  Doubao  │ │ OpenAI│           │
                  │ Embedding│ │ GPT-4o│───────────┘
                  └──────────┘ └───────┘  (citation prompt)
```

**Core flow**: Upload notes → parse & chunk → embed → store in pgvector → ask question → embed query → retrieve top-k → LLM generates answer with inline citations → display with clickable source references.

---

## Stack Decisions

| Layer | Choice | Why |
|-------|--------|-----|
| Framework | Next.js 14 (App Router) | Shortest path to Vercel deploy; SSR + API routes in one repo |
| UI | Tailwind CSS + shadcn/ui | Fast to build, clean defaults, citation UX out of the box |
| Embeddings | Doubao/ARK (doubao-embedding-vision-251215, 2048-dim) | Already configured in .env; good Chinese + English support |
| LLM | OpenAI GPT-4o-mini | Cheap (~$0.15/1M input tokens), strong citation-following instruction, widely tested with RAG |
| Vector DB | Neon PostgreSQL + pgvector | Free tier, persistent, native SQL, works perfectly on Vercel |
| File parsing | pdf-parse (PDF), gray-matter (Markdown), raw (TXT) | Lightweight, no external deps |
| Deployment | Vercel | Zero-config Next.js deploy, free tier sufficient |

**Alternative**: If you prefer to avoid OpenAI, the ARK API also supports chat completions — swap the LLM provider in one config line.

---

## Data Model

```sql
-- notes: one row per uploaded file
CREATE TABLE notes (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  filename    TEXT NOT NULL,
  file_type   TEXT NOT NULL,          -- 'md' | 'txt' | 'pdf'
  content     TEXT NOT NULL,          -- full original text
  created_at  TIMESTAMPTZ DEFAULT now()
);

-- chunks: one row per passage, with embedding vector
CREATE TABLE chunks (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  note_id     UUID REFERENCES notes(id) ON DELETE CASCADE,
  chunk_index INT NOT NULL,            -- ordering within the note
  content     TEXT NOT NULL,           -- chunk text (~300-500 tokens)
  start_char  INT NOT NULL,            -- char offset in original note
  end_char    INT NOT NULL,
  embedding   vector(2048),            -- pgvector type
  created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```

---

## Chunking Strategy

- **Method**: Split by paragraph boundaries first. If a paragraph exceeds 500 tokens, split on sentence boundaries with 50-token overlap. If a paragraph is too short (<50 tokens), merge with the next.
- **Target size**: 300–500 tokens per chunk (good balance: enough context for accurate retrieval, small enough for precise citation).
- **Metadata per chunk**: `note_id`, `chunk_index`, `start_char`, `end_char` — enables highlighting the exact passage in the source note.

---

## RAG Pipeline

### 1. Ingestion (POST /api/notes/upload)

```
Accept multipart file upload
  → Parse file (PDF via pdf-parse, MD via gray-matter, TXT raw)
  → Chunk text (paragraph-based, see above)
  → Batch embed chunks via Doubao API
  → INSERT into notes + chunks tables
  → Return note metadata
```

### 2. Query (POST /api/ask)

```
Receive question string
  → Embed question via Doubao API (same model)
  → SELECT top-8 chunks by cosine similarity (pgvector <=> operator)
  → Filter: only return chunks with similarity > 0.5 threshold
  → Build prompt with retrieved chunks, each labeled [Source N]
  → System prompt instructs: "Answer based ONLY on the provided sources.
     For each claim, cite the source number in brackets [N].
     If no source answers the question, say you cannot find an answer."
  → Call OpenAI chat completions
  → Parse response, extract citation indices
  → Return { answer, sources: [{ note_id, filename, chunk_text, start_char, end_char, similarity }] }
```

### 3. No-Answer Handling

- If all similarity scores < 0.5 → short-circuit: return "I couldn't find relevant information in your notes to answer this question." with no sources.
- If LLM response doesn't cite any source → still show answer but flag "This answer is not grounded in your notes."
- Both cases are explicit and honest, avoiding hallucination.

---

## UI Layout

```
┌─────────────────────────────────────────────────────┐
│  Notes QA                          [Upload Notes]   │
├──────────────────────┬──────────────────────────────┤
│                      │                              │
│  Source Panel        │  Chat Panel                  │
│  ┌────────────────┐  │  ┌────────────────────────┐  │
│  │ note1.md       │  │  │ Q: What is the main     │  │
│  │ note2.txt      │  │  │ conclusion of the       │  │
│  │ paper.pdf      │  │  │ meeting?                │  │
│  │                │  │  │                          │  │
│  │ [click to      │  │  │ A: The team decided to  │  │
│  │  expand &      │  │  │ launch in Q3 [1]. The   │  │
│  │  highlight]    │  │  │ budget was approved at   │  │
│  │                │  │  │ $50K [2].               │  │
│  └────────────────┘  │  │                          │  │
│                      │  │ [1] note1.md ¶3  ←click  │  │
│                      │  │ [2] note1.md ¶5  ←click  │  │
│                      │  └────────────────────────┘  │
│                      │  ┌────────────────────────┐  │
│                      │  │ Ask a question...   [➤] │  │
│                      │  └────────────────────────┘  │
└──────────────────────┴──────────────────────────────┘
```

**Citation interaction**: Clicking [1] scrolls the Source Panel to that chunk and highlights it. The highlight uses `start_char`/`end_char` to mark the exact passage.

---

## API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/notes/upload` | Upload one or more files, returns note IDs |
| GET | `/api/notes` | List all uploaded notes (id, filename, chunk count) |
| GET | `/api/notes/[id]` | Get full note content + chunk boundaries |
| DELETE | `/api/notes/[id]` | Delete a note and all its chunks |
| POST | `/api/ask` | Submit question, get answer with citations |
| GET | `/api/health` | Health check |

---

## Phase Plan (8–12 hours)

### Phase 1: Scaffold & DB Setup (1.5h)
- [ ] `npx create-next-app@latest` with App Router + Tailwind
- [ ] Install deps: `@neondatabase/serverless`, `pgvector`, `pdf-parse`, `gray-matter`, `openai`, `shadcn/ui`
- [ ] Create Neon database, run schema migration
- [ ] Configure `.env.local` with all API keys
- [ ] Test DB connection

### Phase 2: Note Ingestion (2.5h)
- [ ] Upload UI: drag-and-drop zone, file list, status indicators
- [ ] Server: file parsing (MD/TXT/PDF)
- [ ] Chunking logic: paragraph-based splitting with overlap
- [ ] Embedding: call Doubao API, batch chunks
- [ ] Store chunks + vectors in pgvector
- [ ] Error handling: unsupported format, too-large files, API failures

### Phase 3: RAG Query (2.5h)
- [ ] Chat UI: question input, answer display area
- [ ] Server: embed question, pgvector similarity search (top-8, threshold 0.5)
- [ ] Construct citation-aware prompt for LLM
- [ ] Call OpenAI GPT-4o-mini
- [ ] Parse response, extract citation references
- [ ] No-answer detection and honest response

### Phase 4: Citation UX (2h)
- [ ] Source panel: list notes, expand to show content
- [ ] Inline citation markers [1], [2] in answers, clickable
- [ ] Click → scroll to source chunk, highlight passage
- [ ] Show similarity score next to each source (transparency)
- [ ] "No relevant sources found" empty state

### Phase 5: Deploy & Document (1.5h)
- [ ] Push to GitHub
- [ ] Deploy to Vercel, configure env vars
- [ ] End-to-end test with sample notes
- [ ] Write README (3 sections, ≤250 words each)
- [ ] Record screen walkthrough

---

## Scope Cuts & Tradeoffs

### What we build
- Single-user mode (no auth) — one set of notes at a time
- File upload (MD/TXT/PDF) — no real-time sync, no API ingestion
- Inline citation with source highlighting — honest traceability
- PostgreSQL vector search — reliable, persistent, free tier
- Clean, minimal UI — functional over pretty

### What we explicitly don't build (and why)
| Cut | Why |
|-----|-----|
| User auth / multi-tenancy | Demo scope; adds complexity without proving the core RAG value |
| Real-time note syncing (Obsidian, Notion) | File upload is simpler and sufficient for evaluation |
| Conversation memory / multi-turn | Single-turn Q&A proves grounded answers; multi-turn adds prompt complexity |
| Streaming responses | Nice but not required; adds UI complexity for minimal gain in a demo |
| PDF with images/tables | pdf-parse handles text extraction only; OCR/table extraction is a deep rabbit hole |
| Re-ranking / hybrid search | Cosine similarity with pgvector is sufficient; re-ranking is an optimization |

### With 3 more days, I would
- Add hybrid search (BM25 + vector) for better recall on keyword-heavy queries
- Implement conversation memory for follow-up questions
- Add streaming responses for better perceived latency
- Support real-time sync from Obsidian/Notion vaults
- Add OCR for PDF images/tables (via vision model)

---

## Environment Variables

```env
# Database
DATABASE_URL=postgresql://...neon.tech/notesqa

# Doubao Embedding API (already configured)
ARK_API_KEY=ark-...
ARK_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
ARK_EMBEDDING_MODEL=doubao-embedding-vision-251215
ARK_EMBEDDING_DIMENSION=2048

# OpenAI LLM
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```

---

## Risk Mitigations

| Risk | Mitigation |
|------|------------|
| Doubao embedding API rate limits or downtime | Batch chunks (10 per call), add retry logic, fall back to OpenAI embeddings |
| pgvector index slow on large note sets | IVFFlat index with appropriate lists parameter; for demo scale (<200 notes, ~1000 chunks) this is not a concern |
| LLM hallucinates citations | Strict system prompt + post-validation: check every [N] in response maps to a real source |
| PDF parsing failures | Catch errors, show user-friendly message, suggest re-uploading as TXT |
| Vercel cold start latency on DB connection | Use Neon serverless driver with connection pooling |
