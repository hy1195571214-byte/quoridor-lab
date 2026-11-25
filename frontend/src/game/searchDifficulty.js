export const SEARCH_DIFFICULTIES = {
  easy: "简单",
  medium: "中等",
  hard: "困难",
};

export function normalizeSearchDifficulty(value) {
  return Object.hasOwn(SEARCH_DIFFICULTIES, value) ? value : "medium";
}

export function savedSearchDifficulty(entry) {
  const legacyLevel = entry.endpoints?.search?.match(
    /\/agents\/search_(easy|medium|hard)\/move(?:[?#]|$)/,
  )?.[1];
  return normalizeSearchDifficulty(entry.searchDifficulty ?? legacyLevel);
}

export function resolveSearchEndpoint(endpoints, difficulty, apiBase = "/api") {
  const level = normalizeSearchDifficulty(difficulty);
  const key = `search_${level}`;
  if (Object.hasOwn(endpoints, key)) return endpoints[key]?.trim() || "";
  const legacy = endpoints.search?.trim();
  if (legacy && /\/agents\/search(?:_(?:easy|medium|hard))?\/move(?:[?#]|$)/.test(legacy)) {
    return legacy.replace(
      /\/agents\/search(?:_(?:easy|medium|hard))?\/move/,
      `/agents/${key}/move`,
    );
  }
  // Existing custom search services remain the default medium service.
  if (legacy && level === "medium") return legacy;
  return `${apiBase}/agents/${key}/move`;
}
