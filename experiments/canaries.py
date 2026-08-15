"""Canary condition definitions: instruction text (injected into the first
user prompt) plus per-turn pass/fail checkers, and the objective
hallucination checker shared by all conditions.

A canary "failure" (degradation event on the task-irrelevant probe) at turn
t means checker(...) returns False for that turn's assistant response.
"""
import re

NAME_GREETING = "Hello, Dr. Vance"
AUDIT_CODE = "KX-4471-TANGO"
CODENAME_CHOICES = ["NORTH", "SOUTH", "EAST", "WEST"]

SYSTEM_PROMPT = "You are a careful, concise assistant."

TASK_BRIEFING = """You are managing a set of named integer registers for me over a long session.

Initial registers:
{registers}

Every turn I will give you (a) updates to apply, in order, and (b) registers to report.
Rules:
- Track all register values across the entire conversation. Do not use tools or code.
- Apply the updates strictly in the order listed.
- For each requested register, output exactly one line: VALUE <register> = <number>
- Only report the registers I ask for. Do not restate the rest of the state. Do not show your working. Be concise."""

CANARY_INSTRUCTIONS = {
    "say_my_name": (
        f'One more standing rule: start every single response with the exact greeting "{NAME_GREETING}." '
        "before anything else."
    ),
    "remember_fact": (
        f"One more standing rule: remember that my audit code is {AUDIT_CODE}. "
        f"End every single response with the line: AUDIT: {AUDIT_CODE}"
    ),
    "format_response": (
        "One more standing rule: structure every single response in exactly two sections. "
        "First a line reading == UPDATES == followed by a one-line confirmation of the updates applied, "
        "then a line reading == REPORT == followed by the VALUE lines."
    ),
    "variable_check": (
        'One more standing rule: each of my messages includes a line "{sentinel_name} = <number>". '
        "Every response must include one line telling me whether it changed compared to my previous message: "
        "SENTINEL: CHANGED or SENTINEL: UNCHANGED. For my first message only, respond SENTINEL: BASELINE."
    ),
    "early_decision": (
        "One more standing rule: before anything else, choose a codename from this list: "
        f"{', '.join(CODENAME_CHOICES)}. State it as a line CODENAME: <your choice> and include that exact "
        "same line in every response for the whole session. Never change or omit it."
    ),
    "multi_resolution": (
        'One more standing rule: each of my messages includes a line "Canary word #<n>: <word>". '
        "Every response must include one line listing ALL canary words so far, in order, space-separated: "
        "CANARY: <word1> <word2> ..."
    ),
}


def build_first_user_message(task, condition):
    reg_lines = "\n".join(f"- {k} = {v}" for k, v in task["initial_state"].items())
    msg = TASK_BRIEFING.format(registers=reg_lines)
    if condition != "baseline":
        instr = CANARY_INSTRUCTIONS[condition]
        if condition == "variable_check":
            instr = instr.format(sentinel_name=task["sentinel_name"])
        msg += "\n\n" + instr
    msg += "\n\n" + build_turn_body(task, task["turns"][0], condition)
    return msg


def build_turn_body(task, turn, condition):
    lines = [f"Turn {turn['turn']}:"]
    if condition == "variable_check":
        lines.append(f"{task['sentinel_name']} = {turn['sentinel_value']}")
    if condition == "multi_resolution":
        lines.append(f"Canary word #{turn['turn']}: {turn['canary_word']}")
    lines.append("Updates:")
    from .tasks import render_op
    for op in turn["ops"]:
        lines.append(f"- {render_op(op)}")
    lines.append("Report: " + ", ".join(turn["queries"]))
    return "\n".join(lines)


# ---------------------------------------------------------------- checkers

def _norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower())


def _lines(text):
    return [l.strip() for l in text.splitlines() if l.strip()]


def check_hallucination(text, turn):
    """Return list of (register, expected, got_or_None) mismatches. Empty = clean turn."""
    found = {}
    for m in re.finditer(r"VALUE\s+[`\*_]*([A-Za-z0-9_]+)[`\*_]*\s*(?:=|:|is)\s*[`\*_]*(-?\d+)", text, re.I):
        found[m.group(1).lower()] = int(m.group(2))
    errors = []
    for reg, expected in turn["truth"].items():
        got = found.get(reg.lower())
        if got is None or got != expected:
            errors.append((reg, expected, got))
    return errors


def check_say_my_name(text, turn, memo):
    lines = _lines(text)
    if not lines:
        return False
    first = _norm(re.sub(r"^[#>\*\-\s]+", "", lines[0]))
    return "hello dr vance" in first


def check_remember_fact(text, turn, memo):
    tail = _norm(" ".join(_lines(text)[-3:])).replace(" ", "")
    return "kx4471tango" in tail


def check_format_response(text, turn, memo):
    up = re.search(r"=+\s*UPDATES\s*=+", text, re.I)
    rep = re.search(r"=+\s*REPORT\s*=+", text, re.I)
    return bool(up and rep and up.start() < rep.start())


def check_variable_check(text, turn, memo):
    m = re.search(r"SENTINEL\s*:?\s*[`\*_]*(BASELINE|CHANGED|UNCHANGED)", text, re.I)
    if not m:
        return False
    label = m.group(1).upper()
    if turn["turn"] == 1:
        return True  # lenient on the first turn: any sentinel line counts
    return label == ("CHANGED" if turn["sentinel_changed"] else "UNCHANGED")


def check_early_decision(text, turn, memo):
    m = re.search(r"CODENAME\s*:?\s*[`\*_]*(NORTH|SOUTH|EAST|WEST)", text, re.I)
    if turn["turn"] == 1:
        if m:
            memo["codename"] = m.group(1).upper()
            return True
        # lenient turn-1 parse: a lone choice word anywhere
        for c in CODENAME_CHOICES:
            if re.search(rf"\b{c}\b", text, re.I):
                memo["codename"] = c
                return True
        return False
    if "codename" not in memo:
        return False
    return bool(m) and m.group(1).upper() == memo["codename"]


def check_multi_resolution(text, turn, memo, expected_words):
    m = re.search(r"CANARY\s*:?\s*(.+)", text, re.I)
    if not m:
        return False
    got = [re.sub(r"[^a-z0-9]", "", w.lower()) for w in re.split(r"[\s,]+", m.group(1))]
    got = [w for w in got if w]
    return got == [w.lower() for w in expected_words]


def check_canary(condition, text, task, turn, memo):
    """memo: per-trajectory mutable dict (stores early decision, word list...)."""
    if condition == "say_my_name":
        return check_say_my_name(text, turn, memo)
    if condition == "remember_fact":
        return check_remember_fact(text, turn, memo)
    if condition == "format_response":
        return check_format_response(text, turn, memo)
    if condition == "variable_check":
        return check_variable_check(text, turn, memo)
    if condition == "early_decision":
        return check_early_decision(text, turn, memo)
    if condition == "multi_resolution":
        words = [t["canary_word"] for t in task["turns"][:turn["turn"]]]
        return check_multi_resolution(text, turn, memo, words)
    raise ValueError(condition)
