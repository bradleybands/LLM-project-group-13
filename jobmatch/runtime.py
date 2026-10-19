"""Current date/time comes from the app's clock, never model memory or ads."""
import re
from datetime import datetime
from zoneinfo import ZoneInfo

APP_TIMEZONE = "Europe/Stockholm"


def clock_kind(question: str) -> str | None:
    q = question.casefold().replace("’", "'")
    # Publication dates and deadlines must still come from the job sources.
    if re.search(r"\b(?:ads?|jobs?|roles?|deadline|published|posted|listed|application|closing|interview)\b", q):
        return None
    if re.search(r"\b(?:today'?s date|date today|current date|date is it|day(?: of the week)? is (?:it|today))\b", q):
        return "date"
    if re.search(r"\b(?:current time|time is it|time (?:right )?now)\b", q):
        return "time"
    if re.fullmatch(r"(?:so |and |then )?(?:what is|what's|tell me) (?:the )?(date|day|time)[?.! ]*", q.strip()):
        return "time" if "time" in q else "date"
    return None


def runtime_context(now: datetime | None = None) -> dict:
    zone = ZoneInfo(APP_TIMEZONE)
    current = datetime.now(zone) if now is None else now.astimezone(zone)
    return {"current_datetime": current.isoformat(timespec="seconds"),
            "current_date": current.date().isoformat(), "timezone": APP_TIMEZONE}


def is_clock_followup(history: list[dict] | None) -> bool:
    """Inspect only the last exchange, including conversations predating this fix."""
    messages = history or []
    if not messages or messages[-1].get("role") != "assistant":
        return False
    if messages[-1].get("stats", {}).get("answer_origin") == "runtime_clock":
        return True
    return len(messages) >= 2 and messages[-2].get("role") == "user" and bool(clock_kind(messages[-2]["content"]))


def clock_reply(question: str, history: list[dict] | None, context: dict,
                follow_up: bool = False) -> str | None:
    kind = clock_kind(question)
    if not kind and not (follow_up and is_clock_followup(history)):
        return None
    current = datetime.fromisoformat(context["current_datetime"])
    date = f"{current:%A}, {current.day} {current:%B %Y}"
    zone = context["timezone"]
    if kind == "time":
        return f"The current time is {current:%H:%M:%S} on {date} ({zone})."
    if kind:
        return f"Today is {date} ({zone})."
    return (f"I read the system clock on the computer running this app, using {zone}. "
            f"It currently reports {date} at {current:%H:%M:%S}. "
            "That is the source for the current date and time; job publication dates are separate. "
            "If I gave a different date earlier, that answer was incorrect.")
