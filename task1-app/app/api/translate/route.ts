import { NextRequest, NextResponse } from "next/server";

export interface TranslateRequest {
  text: string;
  sourceLang: string;
  targetLang: string;
}

export interface TranslateResponse {
  translatedText: string;
  detectedLanguage?: string;
  characterCount?: number;
}

export async function POST(req: NextRequest) {
  try {
    const body: TranslateRequest = await req.json();
    const { text, sourceLang, targetLang } = body;

    // Validate inputs
    if (!text || !text.trim()) {
      return NextResponse.json(
        { error: "Text is required" },
        { status: 400 }
      );
    }

    if (text.length > 5000) {
      return NextResponse.json(
        { error: "Text exceeds maximum length of 5000 characters" },
        { status: 400 }
      );
    }

    if (!sourceLang || !targetLang) {
      return NextResponse.json(
        { error: "Source and target languages are required" },
        { status: 400 }
      );
    }

    // Build language pair: "auto" source uses just the target
    const langPair =
      sourceLang === "auto"
        ? `${targetLang}|${targetLang}`
        : `${sourceLang}|${targetLang}`;

    // Call MyMemory Translation API (free, no API key required for basic use)
    const encodedText = encodeURIComponent(text.trim());
    const apiUrl = `https://api.mymemory.translated.net/get?q=${encodedText}&langpair=${langPair}`;

    const response = await fetch(apiUrl, {
      method: "GET",
      headers: {
        "User-Agent": "LinguaTranslator/1.0",
      },
    });

    if (!response.ok) {
      throw new Error(`Translation API responded with status ${response.status}`);
    }

    const data = await response.json();

    // MyMemory returns responseStatus 200 on success
    if (data.responseStatus !== 200) {
      throw new Error(data.responseDetails || "Translation failed");
    }

    const translatedText: string = data.responseData?.translatedText || "";

    // MyMemory sometimes returns "MYMEMORY WARNING" messages as translation
    if (translatedText.toUpperCase().includes("MYMEMORY WARNING")) {
      return NextResponse.json(
        {
          error:
            "Daily free translation quota reached. Please try again tomorrow or use a shorter text.",
        },
        { status: 429 }
      );
    }

    const result: TranslateResponse = {
      translatedText,
      detectedLanguage: data.responseData?.detectedLanguage || undefined,
      characterCount: text.length,
    };

    return NextResponse.json(result);
  } catch (error: unknown) {
    console.error("Translation error:", error);
    const message =
      error instanceof Error ? error.message : "An unexpected error occurred";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
