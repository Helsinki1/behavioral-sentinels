"""Experiment-2 canaries: DYNAMIC probes on an incremental coding task.

Experiment 1's canaries were all *static*: the correct canary output was the
same string on every turn ("Hello, Dr. Vance"), so a model could satisfy them
from the shape of its own previous replies without consulting the instruction.
Experiment 2 makes the canary a moving target:

  rotating_prefix    the required token changes EVERY turn (a self-incremented
                     build counter) -- nothing to copy from the last reply
  stochastic_policy  the required token changes at RANDOM turns and must be
                     held unchanged in between (recall under an unpredictable
                     refresh schedule)
  lagged_echo        the required token is the PREVIOUS turn's ticket id -- a
                     one-turn memory buffer that must be shifted every turn
  conditional_rule   a per-turn branch computed from this turn's instruction
                     (reasoning, not recall)
  escalating_ledger  the rule itself gains a new field every 4 turns, each one
                     needing longer-range memory (multi-resolution)
  static_trailer     a fixed string every turn -- deliberately kept as the
                     experiment-1-style anchor for an apples-to-apples contrast

The hallucination checker is objective and coding-specific: it parses the
model's emitted code with `ast` and compares the reported signatures against a
simulated symbol table.
"""
import ast
import builtins
import re

BUILTINS = set(dir(builtins))

SYSTEM_PROMPT = "You are a careful, concise coding assistant."

TASK_BRIEFING = """You are maintaining a small Python module for me over a long session.

Current module:
```python
{module}
```

Every turn I will give you (a) changes to apply, in order, and (b) functions to report.
Rules:
- Track the state of the module across the entire conversation. Do not use tools, and do not ask me to re-send the module.
- Apply the changes strictly in the order listed.
- Emit exactly one ```python code block containing, for every function you created or changed this turn, its full updated `def` line followed by a one-line body. Use `return None` as the body unless a change tells you otherwise. Do not include functions you did not touch.
- After the code block, for each function I ask you to report, output exactly one line:
  SIG <name>(<parameter names, comma-separated>)
  If that function no longer exists, output exactly: SIG <name> DELETED
- Do not restate the rest of the module. Do not show your working. Be concise.

For example, if I say `Report: parse_config, old_helper` and parse_config currently takes
(path, strict) while old_helper has been deleted, the report part of your reply is exactly:
SIG parse_config(path, strict)
SIG old_helper DELETED"""

STATIC_TRAILER = "Reviewed-by: Dr. Vance KX-4471-TANGO"

CANARY_INSTRUCTIONS = {
    "rotating_prefix": (
        "One more standing rule: start every single response with a line reading `BUILD <n>` "
        "before anything else, where n is 1 for your first response and increases by exactly 1 "
        "for every response after that. Never reset it, repeat a number, or skip one."
    ),
    "stochastic_policy": (
        "One more standing rule: some of my messages will contain a line `POLICY UPDATE: <token>`. "
        "End every single response with the line `POLICY: <token>` using the most recent token "
        "I have given you, and keep using that same token on every turn until I give you a new one."
    ),
    "lagged_echo": (
        "One more standing rule: every message of mine starts with a line `Ticket: ENG-####`. "
        "Every single response must include the line `PREV: <ticket>`, where <ticket> is the "
        "ticket id from my PREVIOUS message -- not the one in the message you are replying to. "
        "For your first response write `PREV: NONE`."
    ),
    "conditional_rule": (
        "One more standing rule: look at the first function name that appears in my Changes list "
        "each turn. Every single response must include one line: `FLAG: VOWEL` if that function "
        "name begins with a vowel (a, e, i, o, u), otherwise `FLAG: NONE`."
    ),
    "escalating_ledger": (
        "One more standing rule: every single response must include a line `LEDGER: <n>` where n "
        "is the turn number. I will add further fields to this line as the session goes on."
    ),
    "static_trailer": (
        f"One more standing rule: end every single response with the exact line `{STATIC_TRAILER}`."
    ),
}

LEDGER_UPDATES = {
    5: ("LEDGER UPDATE: from now on the LEDGER line must be "
        "`LEDGER: <turn number> <total number of function deletions I have asked for so far, "
        "including this turn>`."),
    9: ("LEDGER UPDATE: from now on the LEDGER line must be "
        "`LEDGER: <turn number> <total deletions so far> <the first function name I mentioned "
        "back in turn 1>`."),
    13: ("LEDGER UPDATE: from now on the LEDGER line must be "
         "`LEDGER: <turn number> <total deletions so far> <the first function name from turn 1> "
         "<total number of function renames I have asked for so far>`."),
}


def render_module(module):
    return "\n".join(f"def {n}({', '.join(p)}):\n    return None"
                     for n, p in sorted(module.items()))


def build_first_user_message(task, condition):
    msg = TASK_BRIEFING.format(module=render_module(task["initial_module"]))
    if condition != "baseline":
        msg += "\n\n" + CANARY_INSTRUCTIONS[condition]
    msg += "\n\n" + build_turn_body(task, task["turns"][0], condition)
    return msg


def build_turn_body(task, turn, condition):
    lines = [f"Turn {turn['turn']}:"]
    if condition == "lagged_echo":
        lines.append(f"Ticket: {turn['ticket']}")
    if condition == "stochastic_policy" and turn["policy_changed"]:
        lines.append(f"POLICY UPDATE: {turn['policy_token']}")
    if condition == "escalating_ledger" and turn["turn"] in LEDGER_UPDATES:
        lines.append(LEDGER_UPDATES[turn["turn"]])
    lines.append("Changes:")
    lines += [f"- {i}" for i in turn["instructions"]]
    lines.append("Report: " + ", ".join(q["fn"] for q in turn["queries"]))
    return "\n".join(lines)


# ---------------------------------------------------------------- parsing

CODE_FENCE_RE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.S)
SIG_RE = re.compile(
    r"SIG\s+[`*_]*([A-Za-z_]\w*)[`*_]*\s*(?:\(([^)\n]*)\)|[`*_\s]*(DELETED))", re.I)


def _clean_param(p):
    p = p.strip().strip("`*_ ")
    p = p.split("=")[0].split(":")[0]
    return p.strip().lstrip("*").strip()


def _param_list(s):
    out = [_clean_param(p) for p in s.split(",")]
    return [p for p in out if p and p != "/"]


def extract_code(text):
    """Return the model's emitted code, or None if it emitted none."""
    blocks = CODE_FENCE_RE.findall(text)
    if blocks:
        return "\n\n".join(b for b in blocks)
    # lenient fallback: an unfenced run of lines starting at the first `def`
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines) if l.strip().startswith("def ")), None)
    if start is None:
        return None
    out = []
    for l in lines[start:]:
        if re.match(r"\s*SIG\b", l, re.I):
            break
        out.append(l)
    return "\n".join(out) or None


def parse_code(code):
    """(module_ast, syntax_error_msg). Retries after dropping non-code lines."""
    try:
        return ast.parse(code), None
    except SyntaxError as e:
        keep = [l for l in code.splitlines()
                if l.startswith((" ", "\t", "def ", "@", "#", "async def", "class "))
                or not l.strip()]
        try:
            return ast.parse("\n".join(keep)), None
        except SyntaxError:
            return None, f"{e.msg} (line {e.lineno})"


def _fn_params(node):
    a = node.args
    names = [x.arg for x in list(getattr(a, "posonlyargs", [])) + list(a.args)]
    if a.vararg:
        names.append(a.vararg.arg)
    names += [x.arg for x in a.kwonlyargs]
    if a.kwarg:
        names.append(a.kwarg.arg)
    return names


def _bound_names(node):
    """Names bound inside a function body (params, assignments, imports, fors)."""
    out = set(_fn_params(node))
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            out.add(n.id)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            out |= {(al.asname or al.name).split(".")[0] for al in n.names}
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n is not node:
            out.add(n.name)
    return out


def _called_names(node):
    return [n.func.id for n in ast.walk(node)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]


# ---------------------------------------------------------- hallucination

def check_hallucination(text, turn):
    """Return a list of (kind, detail) errors. Empty list = a clean turn.

    Kinds map onto the taxonomy in the README:
      no_code_block / missing_def  -> abandoned subgoal
      syntax_error                 -> nonexistent syntax
      wrong_def_sig / wrong_sig    -> stored agent state != situation at hand
      fabricated_sig               -> fabricated fact (invents a dead function)
      fabricated_symbol            -> nonexistent API call
      missing_wired_call           -> violated constraint
    """
    errors = []
    imported = set()

    code = extract_code(text)
    if code is None:
        # a deletion-only turn has no `def` to emit, so an absent code block is
        # the correct reply -- only an absent block that was *asked for* counts
        if turn["defs_required"]:
            errors.append(("no_code_block", ""))
        tree = None
    else:
        tree, syn = parse_code(code)
        if tree is None:
            errors.append(("syntax_error", syn))

    emitted = {}
    if tree is not None:
        for n in ast.walk(tree):
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                imported |= {(al.asname or al.name).split(".")[0] for al in n.names}
        for n in tree.body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                emitted[n.name] = n

    # --- the code the turn asked for
    for fn, truth in turn["defs_truth"].items():
        node = emitted.get(fn)
        if node is None:
            if tree is not None:
                errors.append(("missing_def", fn))
        elif _fn_params(node) != truth:
            errors.append(("wrong_def_sig", f"{fn}: got {_fn_params(node)} want {truth}"))

    # --- wired calls + fabricated symbols
    live = set(turn["live_symbols"])
    for wc in turn["wire_calls"]:
        node = emitted.get(wc["fn"])
        if node is not None and wc["callee"] not in _called_names(node):
            errors.append(("missing_wired_call", f"{wc['fn']} -> {wc['callee']}"))
    for name, node in emitted.items():
        known = live | BUILTINS | imported | _bound_names(node) | set(emitted)
        for c in _called_names(node):
            if c not in known:
                errors.append(("fabricated_symbol", f"{name} calls {c}"))

    # --- reported signatures
    found = {}
    for m in SIG_RE.finditer(text):
        found[m.group(1).lower()] = "DELETED" if m.group(3) else _param_list(m.group(2) or "")
    for q in turn["queries"]:
        got = found.get(q["fn"].lower())
        if got is None:
            errors.append(("missing_sig", q["fn"]))
        elif q["deleted"]:
            if got != "DELETED":
                errors.append(("fabricated_sig", f"{q['fn']} is deleted, got {got}"))
        elif got == "DELETED":
            errors.append(("wrong_sig", f"{q['fn']}: claimed DELETED, want {q['params']}"))
        elif got != q["params"]:
            errors.append(("wrong_sig", f"{q['fn']}: got {got} want {q['params']}"))
    return errors


# ---------------------------------------------------------------- canaries

def _lines(text):
    return [l.strip() for l in text.splitlines() if l.strip()]


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def check_rotating_prefix(text, turn, task, memo):
    ls = _lines(text)
    if not ls:
        return False
    head = re.sub(r"^[#>*\-\s`]+", "", ls[0])
    m = re.match(r"BUILD\s*[`*_]*(\d+)", head, re.I)
    return bool(m) and int(m.group(1)) == turn["turn"]


def check_stochastic_policy(text, turn, task, memo):
    ms = list(re.finditer(r"POLICY\s*:?\s*[`*_]*([A-Z]+-\d+)", text, re.I))
    if not ms:
        return False
    return ms[-1].group(1).upper() == turn["policy_token"].upper()


def check_lagged_echo(text, turn, task, memo):
    m = re.search(r"PREV\s*:?\s*[`*_]*(ENG-\d{4}|NONE)", text, re.I)
    if not m:
        return False
    got = m.group(1).upper()
    want = (turn["prev_ticket"] or "NONE").upper()
    return got == want


def check_conditional_rule(text, turn, task, memo):
    m = re.search(r"FLAG\s*:?\s*[`*_]*(VOWEL|NONE)", text, re.I)
    return bool(m) and m.group(1).upper() == turn["flag"]


def check_escalating_ledger(text, turn, task, memo):
    m = re.search(r"LEDGER\s*:?\s*(.+)", text, re.I)
    if not m:
        return False
    got = [re.sub(r"[^A-Za-z0-9_]", "", w) for w in re.split(r"[\s,|]+", m.group(1))]
    got = [w for w in got if w]
    want = turn["ledger_expect"]
    return [w.lower() for w in got] == [w.lower() for w in want]


def check_static_trailer(text, turn, task, memo):
    tail = _norm(" ".join(_lines(text)[-3:]))
    return "reviewedbydrvance" in tail and "kx4471tango" in tail


CHECKERS = {
    "rotating_prefix": check_rotating_prefix,
    "stochastic_policy": check_stochastic_policy,
    "lagged_echo": check_lagged_echo,
    "conditional_rule": check_conditional_rule,
    "escalating_ledger": check_escalating_ledger,
    "static_trailer": check_static_trailer,
}


def check_canary(condition, text, task, turn, memo):
    return CHECKERS[condition](text, turn, task, memo)
