import { NextRequest, NextResponse } from "next/server";
import { findBestMatch, getFallbackResponse } from "@/lib/faq-matcher";
import { CATEGORY_COLORS } from "@/lib/faqs";

export interface ChatRequest {
  message: string;
}

export interface ChatResponse {
  answer: string;
  faqId: number | null;
  matchedQuestion: string | null;
  category: string | null;
  categoryColor: string | null;
  confidence: "high" | "medium" | "low" | "none";
  score: number;
  topTerms: string[];
  preprocessedQuery: string[];
  isFallback: boolean;
}

// Simulate a slight processing delay to feel natural (50–120ms)
const delay = (ms: number) => new Promise((res) => setTimeout(res, ms));

export async function POST(req: NextRequest) {
  try {
    const body: ChatRequest = await req.json();
    const { message } = body;

    if (!message || !message.trim()) {
      return NextResponse.json({ error: "Message is required" }, { status: 400 });
    }

    if (message.length > 500) {
      return NextResponse.json(
        { error: "Message too long. Please keep questions under 500 characters." },
        { status: 400 }
      );
    }

    await delay(60 + Math.random() * 80);

    const result = findBestMatch(message);

    // Use the matched FAQ if confidence is not "none", else fallback
    const isFallback = result.confidence === "none";

    const response: ChatResponse = isFallback
      ? {
          answer: getFallbackResponse(message),
          faqId: null,
          matchedQuestion: null,
          category: null,
          categoryColor: null,
          confidence: "none",
          score: result.score,
          topTerms: [],
          preprocessedQuery: result.preprocessedQuery,
          isFallback: true,
        }
      : {
          answer: result.faq.answer,
          faqId: result.faq.id,
          matchedQuestion: result.faq.question,
          category: result.faq.category,
          categoryColor: CATEGORY_COLORS[result.faq.category],
          confidence: result.confidence,
          score: Math.round(result.score * 100) / 100,
          topTerms: result.topTerms.slice(0, 5),
          preprocessedQuery: result.preprocessedQuery.slice(0, 8),
          isFallback: false,
        };

    return NextResponse.json(response);
  } catch (err) {
    console.error("Chat API error:", err);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    );
  }
}
