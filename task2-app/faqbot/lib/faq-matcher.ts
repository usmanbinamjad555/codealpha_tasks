/**
 * FAQ Matching Engine
 * Implements: tokenization → stopword removal → stemming → TF-IDF → cosine similarity
 * Zero external dependencies — pure TypeScript.
 */

import { FAQ, FAQS } from "./faqs";

/* ─── 1. Stopwords ──────────────────────────────────────────── */
const STOPWORDS = new Set([
  "a","about","above","after","again","against","all","am","an","and","any",
  "are","as","at","be","because","been","before","being","below","between",
  "both","but","by","can","did","do","does","doing","down","during","each",
  "few","for","from","further","get","got","had","has","have","having","he",
  "her","here","him","his","how","i","if","in","into","is","it","its",
  "itself","just","me","more","most","my","myself","no","not","now","of",
  "off","on","once","only","or","other","our","out","own","same","she","so",
  "some","such","than","that","the","their","them","then","there","these",
  "they","this","those","through","to","too","under","until","up","us",
  "very","was","we","were","what","when","where","which","while","who",
  "whom","why","will","with","would","you","your","yours","yourself","'s",
  "m","t","re","ve","ll","d","s",
]);

/* ─── 2. Tokenizer ──────────────────────────────────────────── */
export function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[''`]/g, "")          // strip apostrophes
    .replace(/[^a-z0-9\s]/g, " ")  // punctuation → space
    .split(/\s+/)
    .filter((t) => t.length > 1 && !STOPWORDS.has(t));
}

/* ─── 3. Porter-lite stemmer ────────────────────────────────── */
// A simplified suffix-stripping stemmer (covers the most common English rules)
export function stem(word: string): string {
  const rules: [RegExp, string][] = [
    [/ational$/, "ate"],
    [/tional$/, "tion"],
    [/enci$/, "ence"],
    [/anci$/, "ance"],
    [/izer$/, "ize"],
    [/ising$/, "ise"],
    [/izing$/, "ize"],
    [/ising$/, "ise"],
    [/ation$/, "ate"],
    [/ations$/, "ate"],
    [/ness$/, ""],
    [/ment$/, ""],
    [/ments$/, ""],
    [/ities$/, "ity"],
    [/ity$/, ""],
    [/ies$/, "i"],
    [/ational$/, "ate"],
    [/ers$/, "er"],
    [/ing$/, ""],
    [/ings$/, ""],
    [/ed$/, ""],
    [/ly$/, ""],
    [/ful$/, ""],
    [/less$/, ""],
    [/able$/, ""],
    [/ible$/, ""],
    [/ous$/, ""],
    [/ive$/, ""],
    [/ize$/, ""],
    [/ise$/, ""],
    [/tion$/, "t"],
    [/al$/, ""],
    [/s$/, ""],
  ];

  for (const [pattern, replacement] of rules) {
    const root = word.replace(pattern, replacement);
    if (root.length >= 3) return root;
  }
  return word;
}

/* ─── 4. Preprocessing pipeline ────────────────────────────── */
export function preprocess(text: string): string[] {
  return tokenize(text).map(stem);
}

/* ─── 5. TF-IDF ─────────────────────────────────────────────── */
type TermFreq = Map<string, number>;

function termFrequency(tokens: string[]): TermFreq {
  const tf: TermFreq = new Map();
  for (const t of tokens) tf.set(t, (tf.get(t) ?? 0) + 1);
  // Normalize by doc length
  for (const [k, v] of tf) tf.set(k, v / tokens.length);
  return tf;
}

function buildIDF(corpus: string[][]): Map<string, number> {
  const N = corpus.length;
  const df: Map<string, number> = new Map();
  for (const doc of corpus) {
    const seen = new Set(doc);
    for (const t of seen) df.set(t, (df.get(t) ?? 0) + 1);
  }
  const idf: Map<string, number> = new Map();
  for (const [term, count] of df) {
    idf.set(term, Math.log((N + 1) / (count + 1)) + 1); // smooth IDF
  }
  return idf;
}

function tfidfVector(tf: TermFreq, idf: Map<string, number>): Map<string, number> {
  const vec: Map<string, number> = new Map();
  for (const [term, tfVal] of tf) {
    vec.set(term, tfVal * (idf.get(term) ?? 1));
  }
  return vec;
}

/* ─── 6. Cosine Similarity ──────────────────────────────────── */
function cosineSimilarity(a: Map<string, number>, b: Map<string, number>): number {
  let dot = 0;
  let magA = 0;
  let magB = 0;

  for (const [term, valA] of a) {
    dot += valA * (b.get(term) ?? 0);
    magA += valA * valA;
  }
  for (const [, valB] of b) magB += valB * valB;

  if (magA === 0 || magB === 0) return 0;
  return dot / (Math.sqrt(magA) * Math.sqrt(magB));
}

/* ─── 7. Tag & Keyword Bonus ────────────────────────────────── */
function tagBonus(query: string[], faq: FAQ): number {
  const querySet = new Set(query);
  let bonus = 0;
  for (const tag of faq.tags) {
    const stemmedTag = stem(tag.toLowerCase().replace(/[^a-z]/g, ""));
    if (querySet.has(stemmedTag)) bonus += 0.15;
  }
  return Math.min(bonus, 0.4); // cap bonus
}

/* ─── 8. Build corpus index (runs once) ─────────────────────── */
interface CorpusEntry {
  faq: FAQ;
  tokens: string[];
  tfVec: TermFreq;
  tfidfVec: Map<string, number>;
}

let _index: { entries: CorpusEntry[]; idf: Map<string, number> } | null = null;

function buildIndex() {
  if (_index) return _index;

  // Build combined text: question + tags (weighted)
  const docs = FAQS.map((faq) => {
    const combined = `${faq.question} ${faq.question} ${faq.tags.join(" ")}`;
    return preprocess(combined);
  });

  const idf = buildIDF(docs);

  const entries: CorpusEntry[] = FAQS.map((faq, i) => {
    const tokens = docs[i];
    const tfVec = termFrequency(tokens);
    const tfidfVec = tfidfVector(tfVec, idf);
    return { faq, tokens, tfVec, tfidfVec };
  });

  _index = { entries, idf };
  return _index;
}

/* ─── 9. Public Match API ────────────────────────────────────── */
export interface MatchResult {
  faq: FAQ;
  score: number;           // 0–1 cosine similarity
  confidence: "high" | "medium" | "low" | "none";
  preprocessedQuery: string[];
  topTerms: string[];       // terms that drove the match
}

export function findBestMatch(userQuery: string): MatchResult {
  const { entries, idf } = buildIndex();

  const queryTokens = preprocess(userQuery);

  // Edge case: empty after preprocessing
  if (queryTokens.length === 0) {
    return {
      faq: FAQS[9], // fallback to "contact support"
      score: 0,
      confidence: "none",
      preprocessedQuery: [],
      topTerms: [],
    };
  }

  const queryTF = termFrequency(queryTokens);
  const queryTFIDF = tfidfVector(queryTF, idf);

  let bestScore = -1;
  let bestEntry = entries[0];

  const scored = entries.map((entry) => {
    const cosine = cosineSimilarity(queryTFIDF, entry.tfidfVec);
    const bonus = tagBonus(queryTokens, entry.faq);
    const total = Math.min(cosine + bonus, 1);
    return { entry, total };
  });

  scored.sort((a, b) => b.total - a.total);
  bestScore = scored[0].total;
  bestEntry = scored[0].entry;

  // Identify top driving terms
  const topTerms: string[] = [];
  for (const term of queryTokens) {
    if (bestEntry.tokens.includes(term)) topTerms.push(term);
  }

  const confidence: MatchResult["confidence"] =
    bestScore >= 0.45
      ? "high"
      : bestScore >= 0.2
      ? "medium"
      : bestScore >= 0.08
      ? "low"
      : "none";

  return {
    faq: bestEntry.faq,
    score: bestScore,
    confidence,
    preprocessedQuery: queryTokens,
    topTerms,
  };
}

/* ─── 10. Fallback response ──────────────────────────────────── */
export function getFallbackResponse(query: string): string {
  const greetings = ["hi", "hello", "hey", "howdy", "good morning", "good evening"];
  const thanks = ["thanks", "thank you", "thx", "ty", "cheers", "appreciate"];
  const bye = ["bye", "goodbye", "cya", "see you", "later"];

  const q = query.toLowerCase().trim();
  if (greetings.some((g) => q.includes(g))) {
    return "Hey there! 👋 I'm the NexaFlow support bot. Ask me anything about billing, features, account settings, or security — I'm here to help!";
  }
  if (thanks.some((t) => q.includes(t))) {
    return "Happy to help! Is there anything else you'd like to know about NexaFlow? 😊";
  }
  if (bye.some((b) => q.includes(b))) {
    return "Take care! Feel free to come back anytime you have questions. 👋";
  }
  return "I couldn't find a strong match for your question in our FAQ database. For personalized help, please contact our support team at support@nexaflow.io or use the live chat. Would you like to try rephrasing your question?";
}
