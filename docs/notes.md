# **Engineering Take-Home: Notes QA**

## What we're asking you to build

Lots of people accumulate big personal note collections (meeting notes, research excerpts, journals) and want an AI assistant that can answer questions grounded in them. Build a working version of this product.

## Hard constraints

- Users can bring in their own notes (you decide the mechanism and supported formats).
- Users can ask questions and get answers grounded in those notes.
- Every claim in the answer must be traceable back to source. At a minimum, a reader can see which note (and ideally which passage) supports each claim and verify it themselves. UX and granularity are your call.
- Deploy to a public URL we can hit (Vercel, Railway, Fly, your call).

Everything else is a design decision you make: stack, persistence, retrieval method, ingestion format, UI, auth (or none), multi-user (or single-user). We expect you to cut scope to fit the time budget. What you cut and why is part of the signal.

## How we'll test it

We'll evaluate by loading our own notes and asking our own questions. To save you guessing:

- ~20 to 200 notes per test, mixed formats (Markdown, plain text, possibly PDF), ranging from a few hundred to a few thousand words each.
- At least one of our questions has no answer in your notes. We're watching how your system behaves there too.

## What we care about most

When we evaluate, in this order:

1. The grounded-answer experience actually works end-to-end on notes we bring.
2. The traceability is honest. Citations point to text that actually supports the claim, not just to a nearby paragraph.
3. Your judgment is legible. The README makes the tradeoffs you made obvious.

We're not grading on feature count or framework choice.

## Stack

Pick whatever you'll be fastest in. A lot of people reach for Next.js + Vercel AI SDK, which is what we run in production and probably the shortest path to a deployed app. A different choice doesn't cost you anything if you ship.

## API costs

We'll reimburse up to $20 in LLM / embedding API spend. Send receipts with your submission. If you expect to go over, ping us before, not after. Don't pick a worse model to save $2. Pick the model you'd pick for a real product.

## What you submit

1. A screen recording walking through your app: importing notes, asking a question, clicking a citation back to source. This is the first thing we watch, before we touch your deploy.
2. A repo link (GitHub).
3. A live URL.
4. A README covering, in three sections of ≤250 words each:
    - What you built
    - What you chose not to build, and why
    - One thing you would do differently with another 3 days

## Time

72 hours from when you accept. We estimate 8 to 14 hours of real work. Stop when the core works, don't try to fill the window. If life gets in the way, tell us and we'll extend.

## After submission

If we want to continue, we schedule a 60-minute live session. You share your screen, run your own app, extend it with a small feature we describe on the call, and walk us through your design choices. Two engineers, no panel.

If your deploy is flaky the day we test, ping us. We'll run it locally with you on the call rather than ding you for infra.

## Tools

Use any AI assistant you want. Claude Code, Codex, Antigravity, Cursor, Copilot, whatever. We care about how you work with them, not whether you used them. The live session is partly about how well you understand what you shipped.

## Ambiguity is intentional

The spec is deliberately under-specified beyond the hard constraints and the list above. Resolving the rest is part of the work. Make calls, document them. Please don't email for spec clarifications. How you decide what to build and what to defer is itself the signal we're reading.