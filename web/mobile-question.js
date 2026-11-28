const historyCue = /\b(?:first (?:time|see|recorded)|earliest|last (?:time|see|saw|seen|recorded)|latest|previous(?:ly)?|earlier|yesterday|past|history|archives?|when (?:did|was|have)|did (?:i|you) (?:ever )?see|have (?:i|you) (?:ever )?seen|what was written|etait ecrit|this morning|hours? ago|premiere fois|derniere fois|hier|precedemment|auparavant|quand|(?:ai[- ]je|as[- ]tu) (?:deja )?vu|j['\u2019]ai (?:deja )?vu|ce matin|il y a .{1,15} heures?)\b/;

export function questionSource(question) {
  const spoken = question.toLowerCase().normalize("NFD").replace(/\p{M}/gu, "");
  return historyCue.test(spoken) ? { scope: "history" } : { scope: "current", live_only: true };
}

export function answerError(status, scope) {
  if (status === 404) return scope === "history" ? "noHistory" : "noLiveFrame";
  if (status === 429) return "budget";
  if (status === 409) return "busy";
  if (status === 503 && scope === "history") return "historyUnavailable";
  return status ? "answerUnavailable" : "connectionLost";
}
