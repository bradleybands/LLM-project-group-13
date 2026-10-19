"""Route conversation tasks and resolve the job a follow-up refers to."""
import re

from jobmatch.runtime import clock_kind
from jobmatch.data import THESIS_PATTERN

CV_TASKS = {"tailor", "rewrite", "cv_match", "cv_advice"}


def previous_answer(history: list[dict] | None) -> dict:
    return next((m for m in reversed(history or []) if m.get("role") == "assistant"), {})


def thesis_opportunity_query(question: str) -> bool:
    """Distinguish seeking a thesis opening from discussing a completed degree."""
    match = THESIS_PATTERN.search(question)
    if not match:
        return False
    if re.search(r"\b(?:opportunities|openings|vacancies)\b", question, re.IGNORECASE):
        return True
    return not re.search(r"\b(?:requir\w*|complet\w*|finish\w*|wrote|written|write|writing|"
                         r"defend\w*|submit\w*|structur\w*|format\w*)\b",
                         question[:match.start()], re.IGNORECASE)


def intent_for(question: str, history: list[dict] | None = None) -> str:
    q = question.casefold().strip().replace("’", "'")
    if clock_kind(question):
        return "clock"
    if re.fullmatch(r"(?:so |and |but )?(?:how|why|how (?:do|did|can) you know(?: (?:this|that|it))?|"
                    r"what (?:is|was) (?:your|the) source(?: for (?:this|that))?|sources?|"
                    r"where did (?:you get (?:this|that)|(?:this|that) come from)|"
                    r"are you sure(?: about (?:this|that))?|explain (?:this|that)(?: please)?|"
                    r"tell me more|why is that|what do you mean)[?.! ]*", q):
        return "follow_up"
    if history and re.fullmatch(r"(?:the\s+)?(?:first|second|third|fourth|last)(?:\s+(?:one|job|role|ad))?(?:\s+please)?[.!?]?|\[\d+\]", q):
        previous = next((m["content"] for m in reversed(history) if m.get("role") == "user"), "")
        previous_task = intent_for(previous)
        if previous_task in {"tailor", "rewrite", "cv_match", "job_question"}:
            return previous_task
    if re.search(r"\b(rewrite|rephrase|reword|draft|shorten|cover letter|write.*(?:bullet|summary|letter)|make.*(?:shorter|concise|professional|clearer))\b", q):
        return "rewrite"
    if re.search(r"\b(tailor\w*|customi[sz]\w*|adapt\w*|align\w*)\b", q):
        return "tailor"
    if re.search(r"\b(?:improv\w*|change\w*|strengthen\w*|optimi[sz]\w*)\b", q) and re.search(r"\b(?:cv|resume|résumé)\b", q) and re.search(r"\b(?:job|role|position|application)\b", q):
        return "tailor"
    if history and re.search(r"\b(?:what do you mean|you (?:already )?have my (?:cv|resume|résumé)|"
                             r"i (?:already )?(?:uploaded|shared|provided|sent) (?:it|my (?:cv|resume|résumé)))\b", q):
        return "follow_up"
    thesis_reference = thesis_opportunity_query(q)
    if thesis_reference and re.search(r"\b(?:opportunities|openings|available)\b", q) and not re.search(r"\b(?:my|cv|resume|résumé)\b", q):
        return "job_search"
    if history and re.match(r"(?:what|how) about\b|(?:and )?which (?:ones|of (?:these|those))\b", q):
        return "follow_up"
    cv_reference = bool(re.search(r"\b(?:cv|resume|résumé)\b|\bmy\s+(?:experience|background|skills|profile)\b", q))
    job_reference = bool(re.search(r"\b(?:jobs?|roles?|positions?)\b", q) or thesis_reference)
    matching = bool(re.search(r"\b(?:match\w*|fit\w*|compar\w*|gap\w*|missing|qualif\w*|suited|suitable)\b", q))
    personal = bool(re.search(r"\b(?:my|me|i)\b", q))
    personal_details = bool(re.search(r"\b(?:my|this)\s+(?:cv|resume|résumé|experience|background|skills|profile|education|projects|qualifications|strengths|weaknesses)\b", q))
    ad_reference = bool(re.search(r"\b(?:ads?|requirements?|vacanc\w*)\b", q))
    comparison = bool(re.search(r"\bcompar\w*\b", q) and cv_reference)
    if personal_details and not (job_reference or ad_reference or comparison):
        return "cv_advice"
    if (matching and (cv_reference or job_reference and personal)) or (job_reference and cv_reference) or (job_reference and re.search(r"\b(?:for me|find me)\b", q)):
        return "cv_match"
    if re.search(r"\b(?:my|this)\s+(?:cv|resume|résumé)\b", q):
        return "cv_advice"
    if re.search(r"\b(?:hi|hello|hey|thanks|thank you)\b", q) and len(q.split()) <= 5:
        return "chat"
    if history and not re.search(r"\b(?:jobs?|roles?|ads?|positions?|companies|company|employer)\b", q) and re.search(r"\b(?:this|that|these|those|them|they|their|it)\b", q):
        return "follow_up"
    if re.search(r"\b(?:how (?:do|can|should|to)|tips|advice|explain|what is|what does)\b", q) and not re.search(r"\b(?:this|that|these|those|ad|ads|job|jobs|role|roles|company|employer)\b", q):
        return "advice"
    if re.search(r"\b(?:find|show|list|search|hiring|vacanc\w*)\b|\b(?:what|which)\s+(?:jobs|roles|companies|positions)\b", q):
        return "job_search"
    if re.search(r"\b(?:job\w*|role\w*|ad|ads|position\w*|requir\w*|salary|benefits|language|company|employer)\b|\[\d+\]", q):
        return "job_question"
    if thesis_reference:
        return "job_question"
    return "advice"


def active_request(question: str, history: list[dict] | None = None, task: str | None = None) -> str:
    """Keep the user's substantive question through clarification/correction chains."""
    if (task or intent_for(question, history)) != "follow_up":
        return question
    saved = previous_answer(history).get("active_request")
    if saved:
        return saved
    # Backward-compatible with conversations created before request tracking.
    for message in reversed(history or []):
        if message.get("role") == "user" and intent_for(message["content"]) != "follow_up":
            return message["content"]
    return ""


def uses_cv(question: str, history: list[dict] | None = None) -> bool:
    task = intent_for(question, history)
    return task in CV_TASKS or (task == "follow_up" and intent_for(active_request(question, history, task)) in CV_TASKS)


def latest_sources(history: list[dict]) -> list[dict]:
    return next((m.get("sources", []) for m in reversed(history)
                 if m.get("role") == "assistant" and m.get("sources")), [])


def resolve_job(question: str, ads: list[dict], history: list[dict], selected: str | None = None) -> str | None:
    """Explicit IDs/names/references beat the currently selected target."""
    q = question.casefold()
    sources = latest_sources(history)
    reference = re.search(r"\[(\d+)\]|\bsource\s*#?\s*(\d+)\b", q)
    if reference:
        number = int(reference.group(1) or reference.group(2))
        if 1 <= number <= len(sources):
            return sources[number - 1]["metadata"]["ad_id"]
        return None
    for ad in ads:
        if re.search(r"(?<!\w)" + re.escape(ad["ad_id"]) + r"(?!\w)", q):
            return ad["ad_id"]
    generic_names = {"region", "group", "sweden", "sverige", "stockholm", "uppsala", "university", "consulting"}
    named = []
    for ad in ads:
        brand = ad.get("employer", "").split()[0:1]
        values = [ad.get("headline", ""), ad.get("employer", "")]
        if brand and brand[0].casefold() not in generic_names:
            values.extend(brand)
        if any(len(value) >= 5 and re.search(r"(?<!\w)" + re.escape(value.casefold()) + r"(?!\w)", q) for value in values):
            named.append(ad)
    if len(named) == 1:
        return named[0]["ad_id"]
    if named:
        ids = {a["ad_id"] for a in named}
        if selected in ids:
            return selected
        recent_ids = {c["metadata"]["ad_id"] for c in sources} & ids
        return next(iter(recent_ids)) if len(recent_ids) == 1 else None
    previous_ids = list(dict.fromkeys(c["metadata"]["ad_id"] for c in sources))
    ordinal = re.search(r"\b(first|second|third|fourth|last)(?:\s+(?:job|role|one|ad))?\b", q)
    if ordinal and previous_ids:
        number = {"first": 0, "second": 1, "third": 2, "fourth": 3, "last": -1}[ordinal.group(1)]
        if -len(previous_ids) <= number < len(previous_ids):
            return previous_ids[number]
        return None
    if selected and any(a["ad_id"] == selected for a in ads):
        return selected
    if len(previous_ids) == 1 and re.search(r"\b(?:this|that|the)\s+(?:job|role|ad|position)\b", q):
        return previous_ids[0]
    return None


def focused_sources(ads: list[dict], ad_id: str) -> list[dict]:
    ad = next((a for a in ads if a["ad_id"] == ad_id), None)
    if not ad:
        return []
    return [{"chunk_id": f"{ad_id}:focused", "text": ad["text"][:18000],
             "metadata": {key: value for key, value in ad.items() if key != "text"}}]
