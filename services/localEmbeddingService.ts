const DEFAULT_DIMENSIONS = 384;

interface RetrieveOptions {
  topK?: number;
  maxChunkChars?: number;
  overlapChars?: number;
}

const normalize = (text: string) =>
  text
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s+\-*/^=().,:]/gu, ' ')
    .replace(/\s+/g, ' ')
    .trim();

const tokenize = (text: string) => normalize(text).split(' ').filter(Boolean);

const hashToken = (token: string): number => {
  let hash = 2166136261;
  for (let i = 0; i < token.length; i += 1) {
    hash ^= token.charCodeAt(i);
    hash +=
      (hash << 1) +
      (hash << 4) +
      (hash << 7) +
      (hash << 8) +
      (hash << 24);
  }
  return hash >>> 0;
};

export const embedTextLocal = (text: string, dimensions = DEFAULT_DIMENSIONS): Float32Array => {
  const vector = new Float32Array(dimensions);
  const tokens = tokenize(text);

  if (tokens.length === 0) return vector;

  for (const token of tokens) {
    const hash = hashToken(token);
    const idx = hash % dimensions;
    const sign = ((hash >>> 1) & 1) === 0 ? 1 : -1;
    const weight = 1 + Math.log(1 + token.length);
    vector[idx] += sign * weight;
  }

  let norm = 0;
  for (let i = 0; i < vector.length; i += 1) {
    norm += vector[i] * vector[i];
  }
  norm = Math.sqrt(norm) || 1;

  for (let i = 0; i < vector.length; i += 1) {
    vector[i] /= norm;
  }

  return vector;
};

const cosineSimilarity = (a: Float32Array, b: Float32Array): number => {
  const len = Math.min(a.length, b.length);
  let dot = 0;
  for (let i = 0; i < len; i += 1) {
    dot += a[i] * b[i];
  }
  return dot;
};

export const chunkText = (text: string, maxChunkChars = 900, overlapChars = 160): string[] => {
  const source = text.trim();
  if (!source) return [];
  if (source.length <= maxChunkChars) return [source];

  const chunks: string[] = [];
  let start = 0;

  while (start < source.length) {
    let end = Math.min(source.length, start + maxChunkChars);

    if (end < source.length) {
      const breakpoints = [
        source.lastIndexOf('\n\n', end),
        source.lastIndexOf('\n', end),
        source.lastIndexOf('. ', end),
        source.lastIndexOf(' ', end),
      ];
      const best = Math.max(...breakpoints);
      if (best > start + Math.floor(maxChunkChars * 0.6)) {
        end = best + 1;
      }
    }

    const chunk = source.slice(start, end).trim();
    if (chunk) chunks.push(chunk);

    if (end >= source.length) break;

    const nextStart = Math.max(0, end - overlapChars);
    start = nextStart > start ? nextStart : end;
  }

  return chunks;
};

export const retrieveRelevantChunks = (
  query: string,
  documentText: string,
  options: RetrieveOptions = {}
): string[] => {
  const topK = options.topK ?? 5;
  const maxChunkChars = options.maxChunkChars ?? 900;
  const overlapChars = options.overlapChars ?? 160;

  const chunks = chunkText(documentText, maxChunkChars, overlapChars);
  if (chunks.length === 0) return [];

  const queryVec = embedTextLocal(query);
  const ranked = chunks.map((chunk, index) => {
    const chunkVec = embedTextLocal(chunk);
    return {
      index,
      chunk,
      score: cosineSimilarity(queryVec, chunkVec),
    };
  });

  ranked.sort((a, b) => b.score - a.score);
  return ranked.slice(0, topK).map(item => item.chunk);
};
