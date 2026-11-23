const historyCue = /\b(?:last time|last (?:see|saw|seen)|previous(?:ly)?|earlier|yesterday|past|history|archives?|when did i|did i (?:ever )?see|have i (?:ever )?seen|derniere fois|hier|precedemment|auparavant|quand ai[- ]je|ai[- ]je (?:deja )?vu|j['\u2019]ai (?:deja )?vu)\b/;

export function questionSource(question, now = new Date()) {
  const spoken = question.toLowerCase().normalize("NFD").replace(/\p{M}/gu, "");
  if (!historyCue.test(spoken)) return { scope: "current", live_only: true };
  const source = { scope: "history" };
  let offset;
  if (/\b(?:day before yesterday|avant[- ]hier)\b/.test(spoken)) offset = -2;
  else if (/\b(?:yesterday|hier)\b/.test(spoken)) offset = -1;
  else if (/\b(?:today|aujourd['\u2019]hui)\b/.test(spoken)) offset = 0;
  if (offset !== undefined) {
    const start = new Date(now);
    start.setHours(0, 0, 0, 0);
    start.setDate(start.getDate() + offset);
    const next = new Date(start);
    next.setDate(next.getDate() + 1);
    source.start = start.toISOString();
    // Archive timestamps have microsecond precision; include the whole final millisecond.
    source.end = new Date(next.getTime() - 1).toISOString().slice(0, -1) + "999Z";
  }
  return source;
}

export function answerError(status, scope) {
  if (status === 404) return scope === "history" ? "noHistory" : "noLiveFrame";
  if (status === 429) return "budget";
  if (status === 409) return "busy";
  if (status === 503 && scope === "history") return "historyUnavailable";
  return status ? "answerUnavailable" : "connectionLost";
}
