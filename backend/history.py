"""Deterministic archive questions and Beirut-local time, independent of the host."""

from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
import re
import unicodedata
from zoneinfo import ZoneInfo

from .frame_store import utc_timestamp

BEIRUT = ZoneInfo("Asia/Beirut")
ALIASES = {
    "laptop": ("ordinateur portable", "laptop", "notebook computer"),
    "cell phone": ("telephone portable", "telephone", "smartphone", "phone", "cell phone"),
    "chair": ("chair", "chaise"), "couch": ("couch", "sofa", "canape"),
    "backpack": ("backpack", "sac a dos"), "handbag": ("handbag", "sac a main"),
    "dining table": ("dining table", "table"), "tv": ("television", "tv"),
    "book": ("book", "livre"), "car": ("car", "voiture"), "bus": ("bus", "autobus"),
    "keyboard": ("keyboard", "clavier"), "bottle": ("bottle", "bouteille"),
    "bicycle": ("bicycle", "velo"), "person": ("person", "personne"),
}
HISTORY = re.compile(r"\b(?:first (?:time|see|recorded)|earliest|last (?:time|see|saw|seen|recorded)|latest|previous(?:ly)?|earlier|yesterday|"
                     r"past|history|archives?|when (?:did|was|have)|did (?:you|i) see|have (?:you|i) seen|"
                     r"what was written|etait ecrit|(?:this morning|hours? ago)|"
                     r"premiere? fois|derniere fois|hier|precedemment|auparavant|quand|(?:as|ai)[- ](?:tu|je) (?:deja )?vu|"
                     r"ce matin|il y a .{1,15} heures?)\b")
OCR = re.compile(r"\b(?:written|read|say|said|text|label|sign|ecrit|inscrit|lire|texte|pancarte|ecriteau)\b")
QUALIFIED = re.compile(r"\b(?:red|blue|green|black|white|yellow|orange|pink|purple|silver|gold|gray|grey|brown|"
                       r"large|small|open|closed|broken|on (?:a|the|my)|with|beside|next to|"
                       r"variant|rouge|bleu|vert|noir|blanc|jaune|rose|violet|argent|gris|marron|"
                       r"grand|petit|ouvert|ferme|sur (?:la|le|une|un)|avec)\w*\b")
NUMBERS = dict(zip(("one", "two", "three", "four", "five", "six", "un", "deux", "trois", "quatre", "cinq", "six"),
                   (1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6)))
NAMED_PLACE = re.compile(r"\b(?:at (?:home|work|(?:the |my |your )?(?:university|school|office))|"
                         r"a (?:la maison|l'universite|l'ecole|mon bureau|ton bureau))\b")


def normalized(text):
    return "".join(ch for ch in unicodedata.normalize("NFD", text.casefold().replace("\u2019", "'"))
                   if not unicodedata.combining(ch))


@dataclass(frozen=True)
class ArchiveQuery:
    history: bool
    label: str | None
    mode: str
    qualified: bool
    text: str | None
    named_place: bool


def interpret(question):
    phrase = normalized(question)
    labels = [label for label, aliases in ALIASES.items()
              if any(re.search(rf"\b{re.escape(alias)}s?\b", phrase) for alias in aliases)]
    label = labels[0] if len(labels) == 1 else None
    earliest = re.search(r"\b(?:first|earliest|premiere? fois|plus ancienne)\b", phrase)
    quoted = re.search(r'["\u00ab\u201c]([^"\u00bb\u201d]{2,100})["\u00bb\u201d]', question)
    named_text = re.search(r"\b(?:room|salle)\s+[\w-]+(?:\s*\d+)?\b", question, re.IGNORECASE)
    if OCR.search(phrase):
        mode = "ocr"
    elif re.search(r"\b(?:where|ou)\b.*\b(?:see|saw|seen|recorded|vu|vue|observe)", phrase):
        mode = "earliest_context" if earliest else "latest_context"
    elif earliest:
        mode = "earliest"
    elif re.search(r"\b(?:when|quand|last time|latest sighting|derniere fois)\b", phrase):
        mode = "latest"
    elif re.match(r"(?:have (?:you|i) (?:ever )?seen|did (?:you|i) (?:ever )?see|(?:as|ai)[- ](?:tu|je) (?:deja )?vu)\b", phrase):
        mode = "presence"
    else:
        mode = "general"
    return ArchiveQuery(bool(HISTORY.search(phrase)), label, mode, bool(QUALIFIED.search(phrase) or len(labels) > 1),
                        quoted[1].strip() if quoted else named_text[0] if named_text else None,
                        bool(NAMED_PLACE.search(phrase)))


def parse_bound(value, last=False):
    if len(value) == 10:
        day = datetime.fromisoformat(value).date()
        if last:
            return midnight(day + timedelta(days=1)) - timedelta(microseconds=1)
        return midnight(day)
    stamp = datetime.fromisoformat(value)
    if stamp.utcoffset() is None:
        raise ValueError("Timestamp must include an offset")
    return stamp.astimezone(UTC)


def midnight(day):
    # Convert each boundary independently: a Beirut civil day need not be 24 hours.
    return datetime.combine(day, time.min, tzinfo=BEIRUT).astimezone(UTC)


def relative_window(question, now=None):
    now = now or datetime.now(UTC)
    local_day = now.astimezone(BEIRUT).date()
    phrase = normalized(question)
    elapsed = re.search(r"\b(?:(\w+) hours? ago|il y a (\w+) heures?)\b", phrase)
    if elapsed:
        count = elapsed[1] or elapsed[2]
        hours = int(count) if count.isascii() and count.isdecimal() and len(count) <= 3 else NUMBERS.get(count)
        if hours is not None:
            target = now.astimezone(UTC) - timedelta(hours=hours)
            return utc_timestamp(target - timedelta(minutes=30)), utc_timestamp(min(target + timedelta(minutes=30), now))
    if re.search(r"\b(?:day before yesterday|avant[- ]hier)\b", phrase):
        local_day -= timedelta(days=2)
    elif re.search(r"\b(?:yesterday|hier)\b", phrase):
        local_day -= timedelta(days=1)
    elif re.search(r"\b(?:this morning|ce matin)\b", phrase):
        noon = datetime.combine(local_day, time(12), tzinfo=BEIRUT)
        return utc_timestamp(midnight(local_day)), utc_timestamp(noon.astimezone(UTC) - timedelta(microseconds=1))
    elif not re.search(r"\b(?:today|aujourd'hui)\b", phrase):
        return None, None
    return utc_timestamp(midnight(local_day)), utc_timestamp(midnight(local_day + timedelta(days=1)) - timedelta(microseconds=1))


FR_MONTHS = "janvier f\u00e9vrier mars avril mai juin juillet ao\u00fbt septembre octobre novembre d\u00e9cembre".split()
EN_MONTHS = "January February March April May June July August September October November December".split()
FR_OBJECTS = dict(zip(ALIASES, ("un ordinateur portable", "un t\u00e9l\u00e9phone", "une chaise", "un canap\u00e9", "un sac \u00e0 dos",
                              "un sac \u00e0 main", "une table", "une t\u00e9l\u00e9vision", "un livre", "une voiture", "un bus",
                              "un clavier", "une bouteille", "un v\u00e9lo", "une personne")))


def beirut_time(stamp):
    parsed = datetime.fromisoformat(stamp)
    if parsed.utcoffset() is None:
        raise ValueError("Observation timestamp must include an offset")
    return parsed.astimezone(BEIRUT)


def spoken_time(stamp, language, now=None):
    local = beirut_time(stamp)
    today = (now or datetime.now(UTC)).astimezone(BEIRUT).date()
    days = (today - local.date()).days
    if language == "fr":
        day = "aujourd'hui" if days == 0 else "hier" if days == 1 else f"le {local.day} {FR_MONTHS[local.month - 1]} {local.year}"
        return f"{day} \u00e0 {local.hour} h {local.minute:02d}, heure de Beyrouth"
    day = "today" if days == 0 else "yesterday" if days == 1 else f"on {EN_MONTHS[local.month - 1]} {local.day}, {local.year}"
    hour = local.hour % 12 or 12
    return f"{day} at {hour}:{local.minute:02d} {'AM' if local.hour < 12 else 'PM'}, Beirut time"


def local_answer(query, row, language):
    label = query.label
    fr = language == "fr"
    article = FR_OBJECTS.get(label) if fr else (f"{'an' if label and label[0] in 'aeiou' else 'a'} {label}" if label else None)
    if not row:
        if query.mode == "ocr":
            return ("Je n'ai trouv\u00e9 aucun texte enregistr\u00e9 correspondant." if fr else
                    "I couldn't find matching text in the archive.")
        return (f"Je n'ai trouv\u00e9 aucune observation enregistr\u00e9e correspondant \u00e0 {article}." if fr else
                f"I couldn't find a recorded sighting of {article}.")
    captured = spoken_time(row["timestamp"], language)
    if query.mode == "ocr":
        text = " ".join(row["ocr_text"].split())[:240]
        return (f"Le texte enregistr\u00e9 {captured} semble indiquer : {text}. Je ne peux pas confirmer le support exact." if fr else
                f"Text recorded {captured} appears to read: {text}. I can't confirm the exact sign or surface.")
    if fr:
        order = "premi\u00e8re" if query.mode == "earliest" else "derni\u00e8re"
        return (f"La {order} observation enregistr\u00e9e correspondant \u00e0 {article} date de ce moment : {captured}. "
                "Je ne peux pas confirmer l'identit\u00e9 de l'objet.")
    order = "earliest" if query.mode == "earliest" else "latest"
    return f"The {order} recorded sighting of {article} was {captured}. I can't confirm the item's identity."
