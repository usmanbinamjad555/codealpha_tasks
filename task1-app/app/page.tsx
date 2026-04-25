"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import { LANGUAGES, SOURCE_LANGUAGES, getLanguageFlag } from "./languages";

/* ─── Types ──────────────────────────────────────────────────── */
type Status = "idle" | "loading" | "success" | "error";

/* ─── Icons ──────────────────────────────────────────────────── */
const IconArrow = () => (
  <svg
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth={1.5}
    className="w-5 h-5"
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      d="M13.5 4.5L21 12m0 0l-7.5 7.5M21 12H3"
    />
  </svg>
);

const IconSwap = () => (
  <svg
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth={1.5}
    className="w-4 h-4"
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      d="M7.5 21L3 16.5m0 0L7.5 12M3 16.5h13.5m0-13.5L21 7.5m0 0L16.5 12M21 7.5H7.5"
    />
  </svg>
);

const IconCopy = () => (
  <svg
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth={1.5}
    className="w-4 h-4"
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      d="M15.75 17.25v3.375c0 .621-.504 1.125-1.125 1.125h-9.75a1.125 1.125 0 01-1.125-1.125V7.875c0-.621.504-1.125 1.125-1.125H6.75a9.06 9.06 0 011.5.124m7.5 10.376h3.375c.621 0 1.125-.504 1.125-1.125V11.25c0-4.46-3.243-8.161-7.5-8.876a9.06 9.06 0 00-1.5-.124H9.375c-.621 0-1.125.504-1.125 1.125v3.5m7.5 10.375H9.375a1.125 1.125 0 01-1.125-1.125v-9.25m12 6.625v-1.875a3.375 3.375 0 00-3.375-3.375h-1.5a1.125 1.125 0 01-1.125-1.125v-1.5a3.375 3.375 0 00-3.375-3.375H9.75"
    />
  </svg>
);

const IconCheck = () => (
  <svg
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth={2}
    className="w-4 h-4"
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      d="M4.5 12.75l6 6 9-13.5"
    />
  </svg>
);

const IconClear = () => (
  <svg
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth={1.5}
    className="w-4 h-4"
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      d="M6 18L18 6M6 6l12 12"
    />
  </svg>
);

const IconGlobe = () => (
  <svg
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth={1.5}
    className="w-5 h-5"
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      d="M12 21a9.004 9.004 0 008.716-6.747M12 21a9.004 9.004 0 01-8.716-6.747M12 21c2.485 0 4.5-4.03 4.5-9S14.485 3 12 3m0 18c-2.485 0-4.5-4.03-4.5-9S9.515 3 12 3m0 0a8.997 8.997 0 017.843 4.582M12 3a8.997 8.997 0 00-7.843 4.582m15.686 0A11.953 11.953 0 0112 10.5c-2.998 0-5.74-1.1-7.843-2.918m15.686 0A8.959 8.959 0 0121 12c0 .778-.099 1.533-.284 2.253m0 0A17.919 17.919 0 0112 16.5c-3.162 0-6.133-.815-8.716-2.247m0 0A9.015 9.015 0 013 12c0-1.605.42-3.113 1.157-4.418"
    />
  </svg>
);

/* ─── Language Select Component ──────────────────────────────── */
function LanguageSelect({
  value,
  onChange,
  options,
  label,
}: {
  value: string;
  onChange: (v: string) => void;
  options: typeof LANGUAGES;
  label: string;
}) {
  return (
    <div className="relative flex-1">
      <span
        className="absolute left-3 top-1/2 -translate-y-1/2 text-sm pointer-events-none z-10"
        aria-hidden
      >
        {getLanguageFlag(value)}
      </span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        aria-label={label}
        className="
          w-full h-10 pl-9 pr-8 rounded-lg text-sm font-display font-medium
          bg-white/5 border border-white/10
          text-slate-200 hover:border-teal-500/40
          transition-all duration-200
          focus:border-teal-400/50 focus:ring-2 focus:ring-teal-400/10
          cursor-pointer
        "
        style={{ backgroundImage: "none" }}
      >
        {options.map((lang) => (
          <option key={lang.code} value={lang.code}>
            {lang.name}
          </option>
        ))}
      </select>
      {/* Custom chevron */}
      <svg
        viewBox="0 0 16 16"
        fill="none"
        stroke="currentColor"
        strokeWidth={1.5}
        className="w-3.5 h-3.5 absolute right-2.5 top-1/2 -translate-y-1/2 text-slate-400 pointer-events-none"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          d="M4 6l4 4 4-4"
        />
      </svg>
    </div>
  );
}

/* ─── Character Counter ───────────────────────────────────────── */
function CharCounter({ count, max }: { count: number; max: number }) {
  const pct = count / max;
  const color =
    pct > 0.9
      ? "text-red-400"
      : pct > 0.75
      ? "text-amber-400"
      : "text-slate-500";
  return (
    <span className={`font-mono text-xs tabular-nums ${color}`}>
      {count.toLocaleString()} / {max.toLocaleString()}
    </span>
  );
}

/* ─── Typing Indicator ────────────────────────────────────────── */
function TypingIndicator() {
  return (
    <div className="flex items-center gap-1.5 px-1">
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className="w-1.5 h-1.5 rounded-full bg-teal-400 pulse-dot"
          style={{ animationDelay: `${i * 0.2}s` }}
        />
      ))}
    </div>
  );
}

/* ─── Main Page ──────────────────────────────────────────────── */
const MAX_CHARS = 5000;

export default function Home() {
  const [sourceText, setSourceText] = useState("");
  const [translatedText, setTranslatedText] = useState("");
  const [sourceLang, setSourceLang] = useState("en");
  const [targetLang, setTargetLang] = useState("es");
  const [status, setStatus] = useState<Status>("idle");
  const [errorMsg, setErrorMsg] = useState("");
  const [copied, setCopied] = useState(false);
  const [charCount, setCharCount] = useState(0);

  const abortRef = useRef<AbortController | null>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  /* Auto-translate after 600ms debounce */
  useEffect(() => {
    if (!sourceText.trim()) {
      setTranslatedText("");
      setStatus("idle");
      return;
    }
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => {
      handleTranslate(sourceText);
    }, 600);
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sourceText, sourceLang, targetLang]);

  const handleTranslate = useCallback(
    async (text: string) => {
      if (!text.trim()) return;

      // Cancel any in-flight request
      if (abortRef.current) abortRef.current.abort();
      abortRef.current = new AbortController();

      setStatus("loading");
      setErrorMsg("");

      try {
        const res = await fetch("/api/translate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text, sourceLang, targetLang }),
          signal: abortRef.current.signal,
        });

        const data = await res.json();

        if (!res.ok) {
          throw new Error(data.error || "Translation failed");
        }

        setTranslatedText(data.translatedText);
        setStatus("success");
      } catch (err: unknown) {
        if (err instanceof Error && err.name === "AbortError") return;
        setErrorMsg(
          err instanceof Error ? err.message : "Something went wrong"
        );
        setStatus("error");
      }
    },
    [sourceLang, targetLang]
  );

  /* Swap languages */
  const handleSwap = () => {
    if (sourceLang === "auto") return;
    const prevSource = sourceLang;
    const prevTarget = targetLang;
    const prevTranslated = translatedText;
    setSourceLang(prevTarget);
    setTargetLang(prevSource);
    setSourceText(prevTranslated);
    setTranslatedText(sourceText);
    setCharCount(prevTranslated.length);
  };

  /* Clear source */
  const handleClear = () => {
    setSourceText("");
    setTranslatedText("");
    setCharCount(0);
    setStatus("idle");
    setErrorMsg("");
  };

  /* Copy to clipboard */
  const handleCopy = async () => {
    if (!translatedText) return;
    try {
      await navigator.clipboard.writeText(translatedText);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Fallback
      const ta = document.createElement("textarea");
      ta.value = translatedText;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      document.body.removeChild(ta);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const handleSourceChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const val = e.target.value;
    if (val.length > MAX_CHARS) return;
    setSourceText(val);
    setCharCount(val.length);
  };

  const isRTL = (lang: string) =>
    ["ar", "he", "fa", "ur"].includes(lang);

  return (
    <div className="relative min-h-screen flex flex-col z-10">
      {/* ── Header ── */}
      <header className="flex items-center justify-between px-6 py-5 md:px-10">
        <div className="flex items-center gap-3">
          <div
            className="w-8 h-8 rounded-lg flex items-center justify-center"
            style={{
              background:
                "linear-gradient(135deg, rgba(45,212,191,0.3) 0%, rgba(45,212,191,0.1) 100%)",
              border: "1px solid rgba(45,212,191,0.3)",
            }}
          >
            <IconGlobe />
          </div>
          <span
            className="font-display font-semibold text-lg tracking-tight text-slate-100"
          >
            Lingua
          </span>
          <span
            className="hidden sm:inline-block text-[10px] font-mono font-medium px-2 py-0.5 rounded-full uppercase tracking-widest"
            style={{
              background: "rgba(45,212,191,0.1)",
              border: "1px solid rgba(45,212,191,0.25)",
              color: "#2DD4BF",
            }}
          >
            Neural Translate
          </span>
        </div>

        {/* Status indicator */}
        <div className="flex items-center gap-2 text-xs font-mono text-slate-400">
          <span
            className="w-1.5 h-1.5 rounded-full"
            style={{
              background:
                status === "loading"
                  ? "#EAB308"
                  : status === "error"
                  ? "#EF4444"
                  : "#2DD4BF",
              boxShadow:
                status === "loading"
                  ? "0 0 8px rgba(234,179,8,0.6)"
                  : status === "error"
                  ? "0 0 8px rgba(239,68,68,0.6)"
                  : "0 0 8px rgba(45,212,191,0.6)",
            }}
          />
          <span>
            {status === "loading"
              ? "Translating"
              : status === "error"
              ? "Error"
              : status === "success"
              ? "Ready"
              : "Standby"}
          </span>
        </div>
      </header>

      {/* ── Hero label ── */}
      <div className="px-6 md:px-10 pb-6">
        <h1
          className="font-display text-2xl md:text-3xl font-semibold text-slate-100 tracking-tight"
          style={{ letterSpacing: "-0.02em" }}
        >
          Translate anything.{" "}
          <span
            style={{
              background: "linear-gradient(90deg, #2DD4BF, #5EEAD4)",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
            }}
          >
            Instantly.
          </span>
        </h1>
        <p className="mt-1.5 text-sm text-slate-400 font-display">
          {LANGUAGES.length} languages · Free · No account required
        </p>
      </div>

      {/* ── Main translation area ── */}
      <main className="flex-1 px-4 md:px-10 pb-10">
        <div className="max-w-6xl mx-auto">

          {/* Language bar */}
          <div
            className="glass-panel rounded-2xl p-3 mb-4 flex items-center gap-3"
          >
            <LanguageSelect
              value={sourceLang}
              onChange={setSourceLang}
              options={SOURCE_LANGUAGES}
              label="Source language"
            />

            {/* Swap button */}
            <button
              onClick={handleSwap}
              disabled={sourceLang === "auto"}
              title="Swap languages"
              className="
                flex-shrink-0 w-9 h-9 rounded-lg flex items-center justify-center
                text-slate-400 hover:text-teal-400
                transition-all duration-200
                disabled:opacity-30 disabled:cursor-not-allowed
              "
              style={{
                background: "rgba(255,255,255,0.05)",
                border: "1px solid rgba(255,255,255,0.08)",
              }}
            >
              <IconSwap />
            </button>

            <LanguageSelect
              value={targetLang}
              onChange={setTargetLang}
              options={LANGUAGES}
              label="Target language"
            />
          </div>

          {/* Translation panels */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">

            {/* ── Source panel ── */}
            <div className="glass-panel rounded-2xl overflow-hidden flex flex-col">
              {/* Panel header */}
              <div
                className="flex items-center justify-between px-4 py-3"
                style={{ borderBottom: "1px solid rgba(255,255,255,0.06)" }}
              >
                <span className="text-xs font-mono text-slate-400 uppercase tracking-widest">
                  Source
                </span>
                {sourceText && (
                  <button
                    onClick={handleClear}
                    className="flex items-center gap-1.5 text-xs text-slate-400 hover:text-slate-200 transition-colors duration-150"
                  >
                    <IconClear />
                    Clear
                  </button>
                )}
              </div>

              {/* Textarea */}
              <div className="relative flex-1">
                <textarea
                  value={sourceText}
                  onChange={handleSourceChange}
                  placeholder="Type or paste text to translate…"
                  dir={isRTL(sourceLang) ? "rtl" : "ltr"}
                  className="
                    w-full h-64 lg:h-80 p-4
                    bg-transparent text-slate-200
                    placeholder:text-slate-600
                    font-mono text-sm leading-relaxed
                  "
                  aria-label="Source text input"
                />
              </div>

              {/* Source footer */}
              <div
                className="flex items-center justify-between px-4 py-2.5"
                style={{ borderTop: "1px solid rgba(255,255,255,0.05)" }}
              >
                <CharCounter count={charCount} max={MAX_CHARS} />
                <button
                  onClick={() => handleTranslate(sourceText)}
                  disabled={!sourceText.trim() || status === "loading"}
                  className="
                    flex items-center gap-2 px-4 py-1.5 rounded-lg
                    text-xs font-display font-medium
                    transition-all duration-200
                    disabled:opacity-40 disabled:cursor-not-allowed
                    hover:scale-105 active:scale-95
                  "
                  style={{
                    background:
                      "linear-gradient(135deg, rgba(45,212,191,0.25) 0%, rgba(45,212,191,0.1) 100%)",
                    border: "1px solid rgba(45,212,191,0.35)",
                    color: "#2DD4BF",
                  }}
                >
                  <IconArrow />
                  Translate
                </button>
              </div>
            </div>

            {/* ── Output panel ── */}
            <div
              className="glass-panel rounded-2xl overflow-hidden flex flex-col transition-all duration-300"
              style={{
                borderColor:
                  status === "success"
                    ? "rgba(45,212,191,0.15)"
                    : undefined,
              }}
            >
              {/* Panel header */}
              <div
                className="flex items-center justify-between px-4 py-3"
                style={{ borderBottom: "1px solid rgba(255,255,255,0.06)" }}
              >
                <span className="text-xs font-mono text-slate-400 uppercase tracking-widest">
                  Translation
                </span>
                {status === "loading" && <TypingIndicator />}
              </div>

              {/* Output content */}
              <div className="relative flex-1 p-4 h-64 lg:h-80 overflow-y-auto">
                {/* Loading shimmer */}
                {status === "loading" && !translatedText && (
                  <div className="space-y-2 pt-1">
                    {[100, 80, 90, 60].map((w, i) => (
                      <div
                        key={i}
                        className="h-4 rounded shimmer"
                        style={{ width: `${w}%` }}
                      />
                    ))}
                  </div>
                )}

                {/* Error state */}
                {status === "error" && (
                  <div
                    className="flex items-start gap-3 p-4 rounded-xl"
                    style={{
                      background: "rgba(239,68,68,0.08)",
                      border: "1px solid rgba(239,68,68,0.2)",
                    }}
                  >
                    <svg
                      viewBox="0 0 20 20"
                      fill="currentColor"
                      className="w-4 h-4 text-red-400 flex-shrink-0 mt-0.5"
                    >
                      <path
                        fillRule="evenodd"
                        d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-8-5a.75.75 0 01.75.75v4.5a.75.75 0 01-1.5 0v-4.5A.75.75 0 0110 5zm0 10a1 1 0 100-2 1 1 0 000 2z"
                        clipRule="evenodd"
                      />
                    </svg>
                    <div>
                      <p className="text-sm text-red-400 font-display font-medium">
                        Translation failed
                      </p>
                      <p className="text-xs text-red-400/70 mt-0.5 font-mono">
                        {errorMsg}
                      </p>
                    </div>
                  </div>
                )}

                {/* Translated text */}
                {translatedText && status !== "error" && (
                  <p
                    className="font-mono text-sm leading-relaxed text-slate-100 animate-in"
                    dir={isRTL(targetLang) ? "rtl" : "ltr"}
                    style={{ whiteSpace: "pre-wrap", wordBreak: "break-word" }}
                  >
                    {translatedText}
                  </p>
                )}

                {/* Empty state */}
                {!translatedText && status === "idle" && (
                  <div className="h-full flex flex-col items-center justify-center gap-3 text-center">
                    <div
                      className="w-12 h-12 rounded-2xl flex items-center justify-center"
                      style={{
                        background: "rgba(45,212,191,0.06)",
                        border: "1px solid rgba(45,212,191,0.12)",
                      }}
                    >
                      <IconGlobe />
                    </div>
                    <div>
                      <p className="text-sm text-slate-500 font-display">
                        Translation will appear here
                      </p>
                      <p className="text-xs text-slate-600 mt-1 font-mono">
                        Start typing to auto-translate
                      </p>
                    </div>
                  </div>
                )}
              </div>

              {/* Output footer */}
              <div
                className="flex items-center justify-between px-4 py-2.5"
                style={{ borderTop: "1px solid rgba(255,255,255,0.05)" }}
              >
                {/* Word / char count */}
                {translatedText ? (
                  <span className="text-xs font-mono text-slate-500 tabular-nums">
                    {translatedText.split(/\s+/).filter(Boolean).length} words ·{" "}
                    {translatedText.length} chars
                  </span>
                ) : (
                  <span />
                )}

                {/* Copy button */}
                <button
                  onClick={handleCopy}
                  disabled={!translatedText || status === "error"}
                  title="Copy to clipboard"
                  className="
                    flex items-center gap-1.5 px-3 py-1.5 rounded-lg
                    text-xs font-display font-medium
                    transition-all duration-200
                    disabled:opacity-30 disabled:cursor-not-allowed
                    hover:scale-105 active:scale-95
                  "
                  style={{
                    background: copied
                      ? "rgba(45,212,191,0.15)"
                      : "rgba(255,255,255,0.05)",
                    border: copied
                      ? "1px solid rgba(45,212,191,0.4)"
                      : "1px solid rgba(255,255,255,0.08)",
                    color: copied ? "#2DD4BF" : "#94a3b8",
                  }}
                >
                  {copied ? (
                    <>
                      <IconCheck />
                      Copied!
                    </>
                  ) : (
                    <>
                      <IconCopy />
                      Copy
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>

          {/* ── Quick language shortcuts ── */}
          <div className="mt-4 flex flex-wrap gap-2 items-center">
            <span className="text-xs font-mono text-slate-600 mr-1">
              Quick switch:
            </span>
            {[
              { src: "en", tgt: "es", label: "EN → ES" },
              { src: "en", tgt: "fr", label: "EN → FR" },
              { src: "en", tgt: "de", label: "EN → DE" },
              { src: "en", tgt: "ja", label: "EN → JA" },
              { src: "en", tgt: "zh", label: "EN → ZH" },
              { src: "en", tgt: "ar", label: "EN → AR" },
              { src: "en", tgt: "ur", label: "EN → UR" },
            ].map(({ src, tgt, label }) => (
              <button
                key={label}
                onClick={() => {
                  setSourceLang(src);
                  setTargetLang(tgt);
                }}
                className="
                  px-3 py-1 rounded-full text-xs font-mono
                  transition-all duration-150
                  hover:scale-105 active:scale-95
                "
                style={{
                  background:
                    sourceLang === src && targetLang === tgt
                      ? "rgba(45,212,191,0.15)"
                      : "rgba(255,255,255,0.04)",
                  border:
                    sourceLang === src && targetLang === tgt
                      ? "1px solid rgba(45,212,191,0.35)"
                      : "1px solid rgba(255,255,255,0.07)",
                  color:
                    sourceLang === src && targetLang === tgt
                      ? "#2DD4BF"
                      : "#64748b",
                }}
              >
                {label}
              </button>
            ))}
          </div>
        </div>
      </main>

      {/* ── Footer ── */}
      <footer
        className="px-6 md:px-10 py-4 flex items-center justify-between"
        style={{ borderTop: "1px solid rgba(255,255,255,0.04)" }}
      >
        <p className="text-xs font-mono text-slate-600">
          Powered by{" "}
          <a
            href="https://mymemory.translated.net"
            target="_blank"
            rel="noopener noreferrer"
            className="text-slate-500 hover:text-teal-400 transition-colors"
          >
            MyMemory API
          </a>
        </p>
        <p className="text-xs font-mono text-slate-600">
          Free · 5,000 chars/request · ~5k req/day
        </p>
      </footer>
    </div>
  );
}
