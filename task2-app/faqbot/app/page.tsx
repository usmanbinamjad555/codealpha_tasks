"use client";

import {
  useState,
  useRef,
  useEffect,
  useCallback,
  KeyboardEvent,
} from "react";
import { SUGGESTED_QUESTIONS, CATEGORY_COLORS } from "@/lib/faqs";
import type { ChatResponse } from "@/app/api/chat/route";

/* ─── Types ──────────────────────────────────────────────────── */
interface Message {
  id: string;
  role: "user" | "bot";
  text: string;
  timestamp: Date;
  meta?: ChatResponse;
}

/* ─── Icons ──────────────────────────────────────────────────── */
const SendIcon = () => (
  <svg viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4">
    <path d="M3.105 2.289a.75.75 0 00-.826.95l1.414 4.926A1.5 1.5 0 005.135 9.25h6.115a.75.75 0 010 1.5H5.135a1.5 1.5 0 00-1.442 1.085l-1.414 4.926a.75.75 0 00.826.95 28.896 28.896 0 0015.293-7.154.75.75 0 000-1.115A28.897 28.897 0 003.105 2.289z" />
  </svg>
);

const BotIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} className="w-4 h-4">
    <path strokeLinecap="round" strokeLinejoin="round"
      d="M8.25 3v1.5M4.5 8.25H3m18 0h-1.5M4.5 12H3m18 0h-1.5m-15 3.75H3m18 0h-1.5M8.25 19.5V21M12 3v1.5m0 15V21m3.75-18v1.5m0 15V21M6.75 19.5h10.5a2.25 2.25 0 002.25-2.25V6.75a2.25 2.25 0 00-2.25-2.25H6.75A2.25 2.25 0 004.5 6.75v10.5a2.25 2.25 0 002.25 2.25zm3-6.75a1.5 1.5 0 113 0 1.5 1.5 0 01-3 0z"
    />
  </svg>
);

const UserIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} className="w-4 h-4">
    <path strokeLinecap="round" strokeLinejoin="round"
      d="M15.75 6a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0zM4.501 20.118a7.5 7.5 0 0114.998 0A17.933 17.933 0 0112 21.75c-2.676 0-5.216-.584-7.499-1.632z"
    />
  </svg>
);

const CopyIcon = ({ done }: { done: boolean }) =>
  done ? (
    <svg viewBox="0 0 20 20" fill="currentColor" className="w-3.5 h-3.5 text-emerald-400">
      <path fillRule="evenodd" d="M16.704 4.153a.75.75 0 01.143 1.052l-8 10.5a.75.75 0 01-1.127.075l-4.5-4.5a.75.75 0 011.06-1.06l3.894 3.893 7.48-9.817a.75.75 0 011.05-.143z" clipRule="evenodd" />
    </svg>
  ) : (
    <svg viewBox="0 0 20 20" fill="currentColor" className="w-3.5 h-3.5">
      <path d="M7 3.5A1.5 1.5 0 018.5 2h3.879a1.5 1.5 0 011.06.44l3.122 3.12A1.5 1.5 0 0117 6.622V12.5a1.5 1.5 0 01-1.5 1.5h-1v-3.379a3 3 0 00-.879-2.121L10.5 5.379A3 3 0 008.379 4.5H7v-1z" />
      <path d="M4.5 6A1.5 1.5 0 003 7.5v9A1.5 1.5 0 004.5 18h7a1.5 1.5 0 001.5-1.5v-5.879a1.5 1.5 0 00-.44-1.06L9.44 6.439A1.5 1.5 0 008.378 6H4.5z" />
    </svg>
  );

const ClearIcon = () => (
  <svg viewBox="0 0 20 20" fill="currentColor" className="w-3.5 h-3.5">
    <path d="M6.28 5.22a.75.75 0 00-1.06 1.06L8.94 10l-3.72 3.72a.75.75 0 101.06 1.06L10 11.06l3.72 3.72a.75.75 0 101.06-1.06L11.06 10l3.72-3.72a.75.75 0 00-1.06-1.06L10 8.94 6.28 5.22z" />
  </svg>
);

/* ─── Confidence badge ───────────────────────────────────────── */
const CONFIDENCE_CONFIG = {
  high:   { label: "High Match",   color: "#34D399", bg: "rgba(52,211,153,0.1)",  border: "rgba(52,211,153,0.25)"  },
  medium: { label: "Partial Match",color: "#FBBF24", bg: "rgba(251,191,36,0.1)",  border: "rgba(251,191,36,0.25)"  },
  low:    { label: "Weak Match",   color: "#F87171", bg: "rgba(248,113,113,0.1)", border: "rgba(248,113,113,0.25)" },
  none:   { label: "No Match",     color: "#94A3B8", bg: "rgba(148,163,184,0.1)", border: "rgba(148,163,184,0.2)" },
};

function ConfidenceBadge({ confidence, score }: { confidence: ChatResponse["confidence"]; score: number }) {
  const cfg = CONFIDENCE_CONFIG[confidence];
  const pct = Math.round(score * 100);
  return (
    <div className="flex items-center gap-2 mt-3 pt-3" style={{ borderTop: "1px solid rgba(255,255,255,0.06)" }}>
      <div
        className="flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[10px] font-mono font-medium"
        style={{ background: cfg.bg, border: `1px solid ${cfg.border}`, color: cfg.color }}
      >
        <span className="w-1 h-1 rounded-full" style={{ background: cfg.color }} />
        {cfg.label}
      </div>
      <div className="flex-1 h-1 rounded-full" style={{ background: "rgba(255,255,255,0.06)" }}>
        <div
          className="h-full rounded-full fill-bar"
          style={{ width: `${pct}%`, background: cfg.color, opacity: 0.7 }}
        />
      </div>
      <span className="text-[10px] font-mono" style={{ color: cfg.color }}>{pct}%</span>
    </div>
  );
}

/* ─── Typing indicator ───────────────────────────────────────── */
function TypingBubble() {
  return (
    <div className="flex items-end gap-2 msg-in">
      <BotAvatar />
      <div
        className="flex items-center gap-1.5 px-4 py-3 rounded-2xl rounded-bl-sm"
        style={{
          background: "rgba(139,92,246,0.08)",
          border: "1px solid rgba(139,92,246,0.15)",
        }}
      >
        {[0, 1, 2].map((i) => (
          <span
            key={i}
            className="typing-dot w-1.5 h-1.5 rounded-full"
            style={{ background: "#8B5CF6", animationDelay: `${i * 0.15}s` }}
          />
        ))}
      </div>
    </div>
  );
}

/* ─── Bot avatar ─────────────────────────────────────────────── */
function BotAvatar() {
  return (
    <div
      className="relative flex-shrink-0 w-7 h-7 rounded-lg flex items-center justify-center overflow-hidden"
      style={{
        background: "linear-gradient(135deg, rgba(139,92,246,0.3) 0%, rgba(99,102,241,0.2) 100%)",
        border: "1px solid rgba(139,92,246,0.35)",
      }}
    >
      <BotIcon />
      <div
        className="scan-line absolute inset-x-0 h-1/3"
        style={{
          background:
            "linear-gradient(transparent, rgba(139,92,246,0.25), transparent)",
        }}
      />
    </div>
  );
}

/* ─── Message bubble ─────────────────────────────────────────── */
function MessageBubble({ msg }: { msg: Message }) {
  const [copied, setCopied] = useState(false);
  const isBot = msg.role === "bot";

  const handleCopy = () => {
    navigator.clipboard.writeText(msg.text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const pad = (n: number) => String(n).padStart(2, "0");
  const raw = msg.timestamp;
  const h = raw.getHours();
  const ts = `${pad(h % 12 || 12)}:${pad(raw.getMinutes())} ${h >= 12 ? "PM" : "AM"}`;

  if (!isBot) {
    return (
      <div className="flex items-end justify-end gap-2 msg-in">
        <div className="max-w-[72%] flex flex-col items-end gap-1">
          <div
            className="px-4 py-2.5 rounded-2xl rounded-br-sm text-sm leading-relaxed"
            style={{
              background: "linear-gradient(135deg, rgba(139,92,246,0.3) 0%, rgba(99,102,241,0.2) 100%)",
              border: "1px solid rgba(139,92,246,0.25)",
              color: "#E2E8F0",
            }}
          >
            {msg.text}
          </div>
          <span className="text-[10px] font-mono mr-1" style={{ color: "rgba(148,163,184,0.4)" }}>
            {ts}
          </span>
        </div>
        <div
          className="flex-shrink-0 w-7 h-7 rounded-lg flex items-center justify-center"
          style={{
            background: "rgba(255,255,255,0.05)",
            border: "1px solid rgba(255,255,255,0.1)",
          }}
        >
          <UserIcon />
        </div>
      </div>
    );
  }

  const meta = msg.meta;
  const catColor = meta?.categoryColor ?? "#818CF8";

  return (
    <div className="flex items-end gap-2 msg-in">
      <BotAvatar />
      <div className="max-w-[78%] flex flex-col gap-1">
        {/* Matched question label */}
        {meta && !meta.isFallback && meta.matchedQuestion && (
          <div className="flex items-center gap-1.5 mb-0.5 ml-1">
            <span
              className="text-[9px] font-mono uppercase tracking-widest px-1.5 py-0.5 rounded"
              style={{
                color: catColor,
                background: `${catColor}18`,
                border: `1px solid ${catColor}30`,
              }}
            >
              {meta.category}
            </span>
            <span
              className="text-[10px] font-mono truncate max-w-[260px]"
              style={{ color: "rgba(148,163,184,0.5)" }}
            >
              matched: {meta.matchedQuestion}
            </span>
          </div>
        )}

        {/* Main bubble */}
        <div
          className="group relative px-4 py-3 rounded-2xl rounded-bl-sm"
          style={{
            background: "rgba(14,14,31,0.85)",
            border: "1px solid rgba(139,92,246,0.15)",
          }}
        >
          <p className="text-sm leading-relaxed" style={{ color: "#CBD5E1" }}>
            {msg.text}
          </p>

          {/* Confidence bar */}
          {meta && !meta.isFallback && (
            <ConfidenceBadge confidence={meta.confidence} score={meta.score} />
          )}

          {/* Top terms */}
          {meta && meta.topTerms.length > 0 && (
            <div className="flex flex-wrap gap-1 mt-2">
              {meta.topTerms.map((t) => (
                <span
                  key={t}
                  className="text-[10px] font-mono px-1.5 py-0.5 rounded"
                  style={{
                    background: "rgba(99,102,241,0.1)",
                    border: "1px solid rgba(99,102,241,0.2)",
                    color: "#818CF8",
                  }}
                >
                  {t}
                </span>
              ))}
            </div>
          )}

          {/* Copy button */}
          <button
            onClick={handleCopy}
            className="absolute top-2.5 right-2.5 opacity-0 group-hover:opacity-100 transition-opacity duration-150 p-1 rounded"
            style={{ color: "rgba(148,163,184,0.5)", background: "rgba(255,255,255,0.05)" }}
            title="Copy answer"
          >
            <CopyIcon done={copied} />
          </button>
        </div>

        <span className="text-[10px] font-mono ml-1" style={{ color: "rgba(148,163,184,0.35)" }}>
          {ts}
        </span>
      </div>
    </div>
  );
}

/* ─── NLP debug panel ────────────────────────────────────────── */
function DebugPanel({ meta }: { meta: ChatResponse }) {
  const [open, setOpen] = useState(false);
  if (meta.isFallback) return null;

  return (
    <div className="mx-3 mb-1">
      <button
        onClick={() => setOpen((o) => !o)}
        className="text-[10px] font-mono px-2 py-0.5 rounded transition-all"
        style={{
          color: "rgba(129,140,248,0.5)",
          border: "1px solid rgba(99,102,241,0.12)",
          background: open ? "rgba(99,102,241,0.07)" : "transparent",
        }}
      >
        {open ? "▾" : "▸"} NLP debug
      </button>
      {open && (
        <div
          className="mt-1 p-2.5 rounded-lg text-[10px] font-mono space-y-1"
          style={{
            background: "rgba(9,9,20,0.8)",
            border: "1px solid rgba(99,102,241,0.15)",
          }}
        >
          <div>
            <span style={{ color: "rgba(148,163,184,0.5)" }}>query tokens: </span>
            <span style={{ color: "#818CF8" }}>[{meta.preprocessedQuery.join(", ")}]</span>
          </div>
          <div>
            <span style={{ color: "rgba(148,163,184,0.5)" }}>top terms: </span>
            <span style={{ color: "#34D399" }}>[{meta.topTerms.join(", ")}]</span>
          </div>
          <div>
            <span style={{ color: "rgba(148,163,184,0.5)" }}>cosine score: </span>
            <span style={{ color: "#FBBF24" }}>{meta.score.toFixed(4)}</span>
          </div>
          <div>
            <span style={{ color: "rgba(148,163,184,0.5)" }}>confidence: </span>
            <span style={{ color: "#F87171" }}>{meta.confidence}</span>
          </div>
        </div>
      )}
    </div>
  );
}

/* ─── Suggested question chip ────────────────────────────────── */
function SuggestionChip({
  text,
  onClick,
}: {
  text: string;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className="text-left text-xs px-3 py-2 rounded-xl transition-all duration-150 hover:scale-[1.02] active:scale-[0.98] shrink-0"
      style={{
        background: "rgba(139,92,246,0.07)",
        border: "1px solid rgba(139,92,246,0.18)",
        color: "#A78BFA",
        fontFamily: "'DM Sans', sans-serif",
      }}
    >
      {text}
    </button>
  );
}

/* ─── Main Page ──────────────────────────────────────────────── */
const WELCOME: Message = {
  id: "welcome",
  role: "bot",
  text: "Hi! I'm the NexaFlow AI assistant. I can instantly answer your questions about billing, account management, features, security, and more. What would you like to know?",
  timestamp: new Date(),
};

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([WELCOME]);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [lastMeta, setLastMeta] = useState<ChatResponse | null>(null);
  const [msgCount, setMsgCount] = useState(0);

  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isTyping]);

  const sendMessage = useCallback(async (text: string) => {
    const trimmed = text.trim();
    if (!trimmed || isTyping) return;

    const userMsg: Message = {
      id: `u-${Date.now()}`,
      role: "user",
      text: trimmed,
      timestamp: new Date(),
    };

    setMessages((m) => [...m, userMsg]);
    setInput("");
    setIsTyping(true);
    setMsgCount((c) => c + 1);

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: trimmed }),
      });

      const data: ChatResponse = await res.json();

      const botMsg: Message = {
        id: `b-${Date.now()}`,
        role: "bot",
        text: data.answer,
        timestamp: new Date(),
        meta: data,
      };

      setMessages((m) => [...m, botMsg]);
      setLastMeta(data);
    } catch {
      const errMsg: Message = {
        id: `err-${Date.now()}`,
        role: "bot",
        text: "Something went wrong connecting to the server. Please ensure `npm run dev` is running and try again.",
        timestamp: new Date(),
      };
      setMessages((m) => [...m, errMsg]);
    } finally {
      setIsTyping(false);
      inputRef.current?.focus();
    }
  }, [isTyping]);

  const handleKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage(input);
    }
  };

  const handleClear = () => {
    setMessages([WELCOME]);
    setLastMeta(null);
    setMsgCount(0);
    inputRef.current?.focus();
  };

  const charPct = input.length / 500;
  const charColor = charPct > 0.9 ? "#F87171" : charPct > 0.7 ? "#FBBF24" : "rgba(148,163,184,0.35)";

  return (
    <div className="relative h-screen flex z-10">

      {/* ── Left sidebar ── */}
      <aside
        className="hidden lg:flex flex-col w-64 xl:w-72 h-full flex-shrink-0 glass"
        style={{ borderRight: "1px solid rgba(139,92,246,0.1)" }}
      >
        {/* Logo */}
        <div className="p-5 pb-4" style={{ borderBottom: "1px solid rgba(255,255,255,0.05)" }}>
          <div className="flex items-center gap-2.5">
            <div
              className="w-8 h-8 rounded-lg flex items-center justify-center"
              style={{
                background: "linear-gradient(135deg, rgba(139,92,246,0.35) 0%, rgba(99,102,241,0.25) 100%)",
                border: "1px solid rgba(139,92,246,0.35)",
              }}
            >
              <BotIcon />
            </div>
            <div>
              <p className="text-sm font-display font-semibold leading-none" style={{ color: "#E2E8F0" }}>
                NexaFlow
              </p>
              <p className="text-[10px] font-mono mt-0.5" style={{ color: "rgba(139,92,246,0.7)" }}>
                AI Support
              </p>
            </div>
          </div>
        </div>

        {/* Stats */}
        <div className="p-4 space-y-2">
          <p className="text-[10px] font-mono uppercase tracking-widest mb-3" style={{ color: "rgba(148,163,184,0.4)" }}>
            Session Stats
          </p>
          {[
            { label: "Messages", value: msgCount },
            { label: "FAQ Database", value: "15 entries" },
            { label: "Algorithm", value: "TF-IDF" },
            { label: "Similarity", value: "Cosine" },
          ].map(({ label, value }) => (
            <div
              key={label}
              className="flex items-center justify-between px-3 py-2 rounded-lg"
              style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.05)" }}
            >
              <span className="text-xs" style={{ color: "rgba(148,163,184,0.5)" }}>{label}</span>
              <span className="text-xs font-mono" style={{ color: "#A78BFA" }}>{value}</span>
            </div>
          ))}
        </div>

        {/* Last match detail */}
        {lastMeta && !lastMeta.isFallback && (
          <div className="p-4" style={{ borderTop: "1px solid rgba(255,255,255,0.05)" }}>
            <p className="text-[10px] font-mono uppercase tracking-widest mb-3" style={{ color: "rgba(148,163,184,0.4)" }}>
              Last Match
            </p>
            <div
              className="p-3 rounded-xl space-y-2"
              style={{ background: "rgba(139,92,246,0.06)", border: "1px solid rgba(139,92,246,0.15)" }}
            >
              <div className="flex items-center justify-between">
                <span
                  className="text-[10px] font-mono px-1.5 py-0.5 rounded capitalize"
                  style={{
                    color: lastMeta.categoryColor ?? "#818CF8",
                    background: `${lastMeta.categoryColor ?? "#818CF8"}18`,
                  }}
                >
                  {lastMeta.category}
                </span>
                <span className="text-[11px] font-mono font-medium" style={{ color: "#34D399" }}>
                  {Math.round(lastMeta.score * 100)}%
                </span>
              </div>
              <p className="text-[11px] leading-snug" style={{ color: "rgba(148,163,184,0.7)" }}>
                {lastMeta.matchedQuestion}
              </p>
            </div>
          </div>
        )}

        {/* Categories legend */}
        <div className="p-4 mt-auto" style={{ borderTop: "1px solid rgba(255,255,255,0.05)" }}>
          <p className="text-[10px] font-mono uppercase tracking-widest mb-3" style={{ color: "rgba(148,163,184,0.4)" }}>
            Categories
          </p>
          <div className="space-y-1.5">
            {Object.entries(CATEGORY_COLORS).map(([cat, color]) => (
              <div key={cat} className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full" style={{ background: color }} />
                <span className="text-xs capitalize" style={{ color: "rgba(148,163,184,0.55)" }}>
                  {cat}
                </span>
              </div>
            ))}
          </div>
        </div>
      </aside>

      {/* ── Main chat ── */}
      <main className="flex-1 flex flex-col h-full min-w-0">

        {/* Header */}
        <header
          className="flex items-center justify-between px-5 py-3.5 flex-shrink-0"
          style={{
            background: "rgba(9,9,20,0.6)",
            backdropFilter: "blur(20px)",
            borderBottom: "1px solid rgba(139,92,246,0.1)",
          }}
        >
          <div className="flex items-center gap-3">
            {/* Online dot */}
            <div className="relative w-2 h-2">
              <span
                className="absolute inset-0 rounded-full"
                style={{ background: "#34D399", boxShadow: "0 0 8px rgba(52,211,153,0.7)" }}
              />
              <span
                className="absolute inset-0 rounded-full animate-ping"
                style={{ background: "#34D399", opacity: 0.4 }}
              />
            </div>
            <span className="text-sm font-display font-medium" style={{ color: "#CBD5E1" }}>
              NexaFlow Support
            </span>
            <span
              className="hidden sm:inline text-[10px] font-mono px-2 py-0.5 rounded-full"
              style={{
                background: "rgba(52,211,153,0.08)",
                border: "1px solid rgba(52,211,153,0.2)",
                color: "#34D399",
              }}
            >
              Online
            </span>
          </div>

          <button
            onClick={handleClear}
            className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg transition-all hover:scale-105 active:scale-95"
            style={{
              color: "rgba(148,163,184,0.5)",
              border: "1px solid rgba(255,255,255,0.07)",
              background: "rgba(255,255,255,0.03)",
            }}
          >
            <ClearIcon />
            Clear
          </button>
        </header>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto py-6 px-4 sm:px-6 space-y-4">
          {messages.map((msg) => (
            <div key={msg.id}>
              <MessageBubble msg={msg} />
              {msg.meta && <DebugPanel meta={msg.meta} />}
            </div>
          ))}
          {isTyping && <TypingBubble />}
          <div ref={bottomRef} />
        </div>

        {/* Suggestions */}
        {messages.length <= 2 && !isTyping && (
          <div
            className="px-4 sm:px-6 pb-3 flex gap-2 overflow-x-auto"
            style={{ scrollbarWidth: "none" }}
          >
            {SUGGESTED_QUESTIONS.map((q) => (
              <SuggestionChip key={q} text={q} onClick={() => sendMessage(q)} />
            ))}
          </div>
        )}

        {/* Input area */}
        <div
          className="flex-shrink-0 px-4 sm:px-6 py-4"
          style={{ borderTop: "1px solid rgba(139,92,246,0.08)" }}
        >
          <div
            className="flex items-end gap-3 px-4 py-3 rounded-2xl transition-all duration-200"
            style={{
              background: "rgba(14,14,31,0.8)",
              border: `1px solid ${input.length > 0 ? "rgba(139,92,246,0.3)" : "rgba(139,92,246,0.12)"}`,
              boxShadow: input.length > 0 ? "0 0 20px rgba(139,92,246,0.08)" : "none",
            }}
          >
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value.slice(0, 500))}
              onKeyDown={handleKey}
              placeholder="Ask about billing, features, security…"
              rows={1}
              className="flex-1 text-sm bg-transparent text-slate-200 placeholder:text-slate-600 leading-relaxed"
              style={{
                minHeight: "24px",
                maxHeight: "120px",
                overflowY: input.length > 200 ? "auto" : "hidden",
                fontFamily: "'DM Sans', sans-serif",
              }}
              onInput={(e) => {
                const el = e.currentTarget;
                el.style.height = "auto";
                el.style.height = Math.min(el.scrollHeight, 120) + "px";
              }}
              disabled={isTyping}
              aria-label="Message input"
              autoFocus
            />

            <div className="flex items-center gap-2 flex-shrink-0 self-end pb-0.5">
              {input.length > 200 && (
                <span className="text-[10px] font-mono tabular-nums" style={{ color: charColor }}>
                  {input.length}/500
                </span>
              )}
              <button
                onClick={() => sendMessage(input)}
                disabled={!input.trim() || isTyping}
                className="w-8 h-8 rounded-xl flex items-center justify-center transition-all duration-150 hover:scale-105 active:scale-95 disabled:opacity-30 disabled:cursor-not-allowed"
                style={{
                  background: input.trim()
                    ? "linear-gradient(135deg, rgba(139,92,246,0.4) 0%, rgba(99,102,241,0.3) 100%)"
                    : "rgba(255,255,255,0.04)",
                  border: input.trim()
                    ? "1px solid rgba(139,92,246,0.45)"
                    : "1px solid rgba(255,255,255,0.08)",
                  color: input.trim() ? "#C4B5FD" : "#475569",
                }}
                aria-label="Send message"
              >
                <SendIcon />
              </button>
            </div>
          </div>

          <p className="text-center text-[10px] font-mono mt-2.5" style={{ color: "rgba(148,163,184,0.25)" }}>
            Press Enter to send · Shift+Enter for new line · Powered by TF-IDF cosine similarity
          </p>
        </div>
      </main>
    </div>
  );
}
