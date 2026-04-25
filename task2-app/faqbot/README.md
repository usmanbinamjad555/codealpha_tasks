# NexaFlow FAQ Chatbot

An interactive FAQ chatbot built with **Next.js 14**, **TypeScript**, and **Tailwind CSS**.  
Uses a hand-rolled **TF-IDF + Cosine Similarity** NLP engine — zero external ML dependencies.

---

## Quick Start

```bash
# 1. Install dependencies
npm install

# 2. Run dev server
npm run dev

# 3. Open in browser
# http://localhost:3000
```

No API keys. No `.env` file. Works offline after install.

---

## Project Structure

```
faqbot/
├── app/
│   ├── globals.css           # Design system + glassmorphism
│   ├── layout.tsx            # Root layout
│   ├── page.tsx              # Full chat UI (client component)
│   └── api/
│       └── chat/
│           └── route.ts      # POST /api/chat endpoint
├── lib/
│   ├── faqs.ts               # 15 FAQ dataset + category metadata
│   └── faq-matcher.ts        # NLP pipeline (tokenize→stem→TF-IDF→cosine)
├── tailwind.config.ts
└── package.json
```

---

## NLP Pipeline (lib/faq-matcher.ts)

The matching engine implements a classic IR pipeline, fully from scratch:

### Step 1 — Tokenization
Splits text on whitespace, strips punctuation, filters tokens ≤1 char.

### Step 2 — Stopword Removal
Removes 80+ common English stopwords ("the", "is", "of", etc.) so only
semantically meaningful words remain.

### Step 3 — Stemming (Porter-lite)
Applies suffix-stripping rules (e.g. "billing" → "bill", "resetting" → "reset")
to normalize word variants into a common root form.

### Step 4 — TF-IDF Vectorization
- **TF (Term Frequency)**: how often a term appears in the document, normalized by doc length.
- **IDF (Inverse Document Frequency)**: penalizes terms that appear in many FAQs (common words get less weight).
- Combined: `TF × IDF` produces a weighted vector per document.

### Step 5 — Cosine Similarity
Measures the angle between the query vector and each FAQ vector.
A score of 1.0 = identical; 0.0 = no overlap.

```
similarity = (A · B) / (|A| × |B|)
```

### Step 6 — Tag Bonus
Each FAQ has a `tags` array. Matching stemmed tags adds a bonus (up to +0.4)
to boost precision for short or ambiguous queries.

### Confidence Thresholds
| Score | Confidence |
|-------|-----------|
| ≥ 0.45 | High |
| ≥ 0.20 | Medium |
| ≥ 0.08 | Low |
| < 0.08 | No Match (fallback) |

---

## Upgrading to an LLM API (Optional)

Replace the `findBestMatch` call in `app/api/chat/route.ts` with a Claude/OpenAI call:

```ts
// .env.local
ANTHROPIC_API_KEY=sk-ant-...

// In route.ts
import Anthropic from "@anthropic-ai/sdk";
const client = new Anthropic();

const faqList = FAQS.map(f => `Q: ${f.question}\nA: ${f.answer}`).join("\n\n");

const response = await client.messages.create({
  model: "claude-opus-4-5",
  max_tokens: 512,
  messages: [{
    role: "user",
    content: `Given these FAQs:\n${faqList}\n\nUser question: "${message}"\n\nAnswer from the FAQs only. If no match, say so.`
  }]
});
```

---

## UI Features

- **Glassmorphic dark UI** — void-black background, violet/iris accents, backdrop blur panels
- **Live typing indicator** — animated dots while processing
- **Confidence badge** — color-coded match quality with a fill bar
- **NLP debug panel** — click "▸ NLP debug" below any bot reply to inspect tokens, score, and matched terms
- **Suggested questions** — quick-tap chips on first load
- **Left sidebar** — session stats, last match details, category legend
- **Copy on hover** — hover any bot message to copy the answer
- **Keyboard** — Enter to send, Shift+Enter for new line
- **Auto-clear** — "Clear" button resets the session

---

## Tech Stack

| Tech | Purpose |
|------|---------|
| Next.js 14 | Framework + API routes |
| TypeScript | Full type safety |
| Tailwind CSS | Utility styling |
| TF-IDF (custom) | Query→FAQ similarity |
| Porter Stemmer (custom) | Word normalization |
| Google Fonts | Outfit + DM Sans + IBM Plex Mono |
