# Lingua — AI Language Translation Tool

A modern, glassmorphism-styled language translation app built with **Next.js 14**, **TypeScript**, and **Tailwind CSS**. Uses the **MyMemory Translation API** — completely free, no API key required.

---

## Features

- 🌍 **40+ languages** with auto-detect
- ⚡ **Auto-translate** as you type (600ms debounce)
- 📋 **Copy to clipboard** button
- 🔄 **Swap languages** with a single click
- ⌨️ **Quick language shortcuts** bar
- 🎨 **Glassmorphism UI** — dark theme, backdrop blur, teal accents
- 🚫 **No API key needed** — uses MyMemory's free tier
- 🛡️ **API route** proxies requests (ready for key injection if needed)

---

## Prerequisites

- **Node.js** v18.17 or newer
- **npm** v9+ (comes with Node.js)

---

## Installation

### 1. Install dependencies

```bash
npm install
```

### 2. Run the development server

```bash
npm run dev
```

### 3. Open the app

Navigate to [http://localhost:3000](http://localhost:3000) in your browser.

---

## Project Structure

```
lingua/
├── app/
│   ├── globals.css          # Global styles + glassmorphism utilities
│   ├── layout.tsx           # Root layout with metadata
│   ├── page.tsx             # Main translation UI (single-page app)
│   ├── languages.ts         # Language definitions (40+ languages)
│   └── api/
│       └── translate/
│           └── route.ts     # Next.js API route (proxies MyMemory API)
├── tailwind.config.ts       # Tailwind config with custom tokens
├── postcss.config.js
├── next.config.mjs
├── tsconfig.json
└── package.json
```

---

## How It Works

1. **User types** text in the source panel
2. After a **600ms debounce**, the frontend POSTs to `/api/translate`
3. The **Next.js API route** calls the MyMemory Translation API:
   ```
   https://api.mymemory.translated.net/get?q={text}&langpair={source}|{target}
   ```
4. The translated text is returned and displayed in the output panel

### Why an API route?

The `/api/translate` route acts as a **server-side proxy**. This means:
- If you later switch to a paid API (DeepL, Google, Azure), you add the API key to `.env.local` and it **never touches the client**.
- Keeps any future credentials safe from browser exposure.

---

## Upgrading to a Paid API (Optional)

### DeepL API (Best Quality)

1. Get a free API key at [deepl.com](https://www.deepl.com/pro-api)
2. Create `.env.local`:
   ```
   DEEPL_API_KEY=your_key_here
   ```
3. Replace the fetch in `app/api/translate/route.ts`:
   ```ts
   const res = await fetch("https://api-free.deepl.com/v2/translate", {
     method: "POST",
     headers: {
       "Authorization": `DeepL-Auth-Key ${process.env.DEEPL_API_KEY}`,
       "Content-Type": "application/json",
     },
     body: JSON.stringify({
       text: [text],
       source_lang: sourceLang.toUpperCase(),
       target_lang: targetLang.toUpperCase(),
     }),
   });
   const data = await res.json();
   const translatedText = data.translations[0].text;
   ```

### Google Translate API

1. Get a key from [Google Cloud Console](https://console.cloud.google.com)
2. Create `.env.local`:
   ```
   GOOGLE_TRANSLATE_API_KEY=your_key_here
   ```
3. Update `route.ts` to call:
   ```
   https://translation.googleapis.com/language/translate/v2?key={key}
   ```

---

## MyMemory API Limits

| Plan | Limit |
|------|-------|
| Free (anonymous) | ~5,000 words/day |
| Free (with email) | ~10,000 words/day |
| Paid | Unlimited |

To increase the free limit, append `&de=your@email.com` to the API URL in `route.ts`.

---

## Build for Production

```bash
npm run build
npm start
```

---

## Tech Stack

| Technology | Version | Purpose |
|-----------|---------|---------|
| Next.js | 14.2.5 | React framework + API routes |
| TypeScript | 5 | Type safety |
| Tailwind CSS | 3.4 | Utility-first styling |
| MyMemory API | Free | Translation engine |
| Google Fonts | — | Sora + JetBrains Mono |
