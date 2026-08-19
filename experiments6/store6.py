"""The external state store for experiment 6: a deterministic reducer that
plays the file system.

Each domain's store applies every user-issued instruction (the generator's
machine-readable `ops` / story lines -- the same content the user message
renders as text) to a state object, exactly as a repo materialises an
agent's edits. Nothing else enters the store: no model output, no generator
fields that are not implied by the instructions shown to the agent. The
`verify_pool` function proves, offline over every turn of every task in the
data, that the store's state equals the generator's own truth fields -- that
proof is what licenses the R1 "reground" reset to serve the store's state as
ground truth.

Reset operators (both deterministic; no LLM call at reset time):

  resume_reground (R1): original task briefing with the CURRENT state
      substituted for the initial state, plus tombstones for deletions
      (a fresh session can see a repo's current files and its history).
  resume_replay (R2): the verbatim log of every prior user message,
      assistant turns dropped. Zero harness intelligence.
"""
import json

from experiments.canaries import TASK_BRIEFING as REG_BRIEFING
from experiments2.canaries2 import TASK_BRIEFING as COD_BRIEFING, render_module
from experiments3.canaries3 import (BABI_BRIEFING, NEW_STORY_LINE,
                                    build_first_user_message, build_turn_body)

from .config6 import DATA_DIR, DOMAINS


class CodingStore:
    def __init__(self, task):
        self.fns = {n: list(p) for n, p in task["initial_module"].items()}
        self.deleted = set()

    def apply(self, turn):
        for op in turn["ops"]:
            o = op["op"]
            if o == "add_fn":
                self.fns[op["fn"]] = list(op["params"])
                self.deleted.discard(op["fn"])
            elif o == "add_param":
                self.fns[op["fn"]].append(op["param"])
            elif o == "remove_param":
                self.fns[op["fn"]].remove(op["param"])
            elif o == "rename_param":
                ps = self.fns[op["fn"]]
                ps[ps.index(op["old"])] = op["new"]
            elif o == "rename_fn":
                self.fns[op["new"]] = self.fns.pop(op["old"])
                self.deleted.discard(op["new"])
            elif o == "delete_fn":
                del self.fns[op["fn"]]
                self.deleted.add(op["fn"])
            elif o == "wire_call":
                pass          # body-level; signatures unaffected
            else:
                raise ValueError(o)

    def state_text(self):
        dead = ", ".join(sorted(self.deleted)) if self.deleted else "none"
        return (render_module(self.fns)
                + f"\n\n# functions deleted earlier in this session: {dead}")

    def verify(self, turn):
        errors = []
        if set(self.fns) != set(turn["live_symbols"]):
            errors.append(f"live set mismatch at turn {turn['turn']}")
        for fn, params in turn["defs_truth"].items():
            if self.fns.get(fn) != params:
                errors.append(f"{fn} params {self.fns.get(fn)} != {params}")
        for q in turn["queries"]:
            if q["deleted"]:
                if q["fn"] in self.fns or q["fn"] not in self.deleted:
                    errors.append(f"query {q['fn']} deleted-flag mismatch")
            elif self.fns.get(q["fn"]) != q["params"]:
                errors.append(f"query {q['fn']} params mismatch")
        return errors


class RegistersStore:
    def __init__(self, task):
        self.regs = dict(task["initial_state"])
        self.deleted = set()

    def apply(self, turn):
        for op in turn["ops"]:
            o = op["op"]
            if o == "set":
                self.regs[op["k"]] = op["v"]
            elif o == "add":
                self.regs[op["k"]] += op["n"]
            elif o == "sub":
                self.regs[op["k"]] -= op["n"]
            elif o == "swap":
                a, b = op["a"], op["b"]
                self.regs[a], self.regs[b] = self.regs[b], self.regs[a]
            elif o == "copyplus":
                self.regs[op["a"]] = self.regs[op["b"]] + op["n"]
            elif o == "create":
                self.regs[op["k"]] = op["v"]
                self.deleted.discard(op["k"])
            elif o == "delete":
                del self.regs[op["k"]]
                self.deleted.add(op["k"])
            else:
                raise ValueError(o)

    def state_text(self):
        lines = "\n".join(f"- {k} = {v}" for k, v in self.regs.items())
        if self.deleted:
            lines += ("\n(registers deleted earlier in this session: "
                      + ", ".join(sorted(self.deleted)) + ")")
        return lines

    def verify(self, turn):
        errors = []
        for reg, want in turn["truth"].items():
            if self.regs.get(reg) != want:
                errors.append(f"{reg}: store {self.regs.get(reg)} != {want}")
        for reg in turn["queries"]:
            if reg not in self.regs:
                errors.append(f"queried register {reg} not live in store")
        return errors


class BabiStore:
    def __init__(self, task):
        self.story = []

    def apply(self, turn):
        if turn["new_story"] and turn["turn"] > 1:
            self.story = []
        self.story.extend(turn["story_lines"])

    def state_text(self):
        return "\n".join(self.story)

    def verify(self, turn):
        words = {w.lower() for line in self.story
                 for w in line.replace(".", " ").split()}
        if turn["answer"].lower() not in words:
            return [f"answer {turn['answer']!r} not in current story"]
        return []


STORES = {"coding": CodingStore, "registers": RegistersStore,
          "babi": BabiStore}


def make_store(domain, task):
    return STORES[domain](task)


# ------------------------------------------------------------ reset messages

PREAMBLE = """This is a fresh session resuming an in-progress task. The
previous session ended after task turn {last_turn}; the next message will be
Turn {next_turn}. The task rules and the {what} follow. Do not perform any
task work until Turn {next_turn} arrives."""

REGROUND_WHAT = "CURRENT state, read back from the session's external record"
REPLAY_WHAT = ("original briefing and a verbatim log of every task message "
               "so far (already handled -- do not answer them again)")


def _briefing_with_state(domain, store):
    if domain == "coding":
        return COD_BRIEFING.format(module=store.state_text())
    if domain == "registers":
        return REG_BRIEFING.format(registers=store.state_text())
    story = store.state_text()
    return (BABI_BRIEFING + "\n\nThe current story so far:\n"
            + (story if story else "(no sentences yet)"))


def resume_reground(domain, task, store, next_turn):
    """R1: briefing + materialised current state. One user message."""
    return (PREAMBLE.format(last_turn=next_turn - 1, next_turn=next_turn,
                            what=REGROUND_WHAT)
            + "\n\n" + _briefing_with_state(domain, store))


def resume_replay(domain, task, next_turn):
    """R2: briefing + verbatim prior user messages (assistant turns dropped)."""
    parts = [build_first_user_message(domain, task, "baseline")]
    for turn in task["turns"][1:next_turn - 1]:
        parts.append(build_turn_body(domain, task, turn, "baseline"))
    return (PREAMBLE.format(last_turn=next_turn - 1, next_turn=next_turn,
                            what=REPLAY_WHAT)
            + "\n\n" + "\n\n---\n\n".join(parts))


# ----------------------------------------------------------------- verifier

def verify_pool(limit=None):
    """Prove store == generator truth on every turn of every task. Returns
    {domain: {"tasks": n, "turns": n, "errors": [...]}}."""
    report = {}
    for domain in DOMAINS:
        tasks = json.loads((DATA_DIR / f"tasks3_{domain}.json").read_text())
        if limit:
            tasks = tasks[:limit]
        errors, turns = [], 0
        for task in tasks:
            store = make_store(domain, task)
            for turn in task["turns"]:
                store.apply(turn)
                turns += 1
                for e in store.verify(turn):
                    errors.append(f"task {task['task_id']} turn {turn['turn']}: {e}")
        report[domain] = {"tasks": len(tasks), "turns": turns, "errors": errors}
    return report


if __name__ == "__main__":
    for domain, r in verify_pool().items():
        status = "OK" if not r["errors"] else f"{len(r['errors'])} ERRORS"
        print(f"{domain}: {r['tasks']} tasks, {r['turns']} turns -- {status}")
        for e in r["errors"][:10]:
            print("   ", e)
