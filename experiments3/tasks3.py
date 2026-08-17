"""Experiment-3 task builders: three task sets, one canary payload machinery.

  coding    -- data2/tasks2.json AUGMENTED with exp-3 payloads (reusing the
               validated exp-2 tasks means zero new generator bugs and keeps
               the hallucination checker identical to experiment 2).
  registers -- data/tasks.json (first 100) augmented the same way.
  babi      -- published bAbI QA items (tasks 1-3) reconstructed into stories
               and concatenated 3-5 per session with an explicit NEW STORY
               reset; one question per turn, single-word objective answers.

Each task gains `task["c3"]` (task-level payload) and `turn["c3"]` (per-turn
payload); nothing already present is modified.
"""
import argparse
import json
import random

from experiments.tasks import COLORS as REG_COLORS, NOUNS as REG_NOUNS
from experiments2.tasks2 import OBJECTS as COD_OBJECTS, VERBS as COD_VERBS

from .config3 import DATA_DIR, N_TASKS, RAW_DIR, ROOT, SEED, task_file
from .payloads3 import gen_payloads

BABI_TASKS = (1, 2, 3)
BABI_STORIES_PER_SESSION = (3, 5)
# look-alikes for bAbI's Mary/John/Sandra/Daniel cast -- the interference twin
BABI_SHADOW_NAMES = ["Marla", "Jon", "Sandro", "Daniella", "Marek", "Joana",
                     "Samir", "Dana", "Mario", "Johan", "Sanna", "Danilo",
                     "Marta", "Jonah", "Sandrine", "Darek", "Maren", "Jona",
                     "Sander", "Dario"]


# ------------------------------------------------------------------ vocab

def _shadow_vocab_coding(rng, task):
    used = set(task["initial_module"])
    for turn in task["turns"]:
        for op in turn["ops"]:
            for key in ("fn", "old", "new"):
                if key in op and isinstance(op[key], str):
                    used.add(op[key])
    combos = [f"{v}_{o}" for v in COD_VERBS for o in COD_OBJECTS if f"{v}_{o}" not in used]
    rng.shuffle(combos)
    return combos[:20]


def _shadow_vocab_registers(rng, task):
    used = set(task["initial_state"])
    for turn in task["turns"]:
        for op in turn["ops"]:
            for key in ("k", "a", "b"):
                if key in op and isinstance(op[key], str):
                    used.add(op[key])
    combos = [f"{c}_{n}" for c in REG_COLORS for n in REG_NOUNS if f"{c}_{n}" not in used]
    rng.shuffle(combos)
    return combos[:20]


def _augment(task, rng, shadow_vocab):
    task_level, per_turn = gen_payloads(rng, task["horizon"], shadow_vocab)
    task["c3"] = task_level
    for turn, payload in zip(task["turns"], per_turn):
        turn["c3"] = payload
    return task


# ------------------------------------------------------------------ builders

def build_coding():
    src = ROOT / "data2" / "tasks2.json"
    tasks = json.loads(src.read_text())[:N_TASKS["coding"]]
    rng = random.Random(SEED)
    for task in tasks:
        _augment(task, rng, _shadow_vocab_coding(rng, task))
    return tasks


def build_registers():
    src = ROOT / "data" / "tasks.json"
    tasks = json.loads(src.read_text())[:N_TASKS["registers"]]
    rng = random.Random(SEED + 1)
    for task in tasks:
        _augment(task, rng, _shadow_vocab_registers(rng, task))
    return tasks


def _babi_stories():
    """Reconstruct (babi_task, [rows]) stories from the cumulative-passage
    jsonl: consecutive rows whose passage extends the previous one belong to
    the same story."""
    raw = RAW_DIR / "babi_train.jsonl"
    if not raw.exists():
        raise SystemExit(
            f"missing {raw}\nDownload it first:\n  curl -L -o {raw} "
            "https://huggingface.co/datasets/Muennighoff/babi/resolve/main/babi_train.jsonl")
    rows = [json.loads(l) for l in raw.read_text().splitlines() if l.strip()]
    stories, cur, prev = [], [], ""
    for r in rows:
        if r["task"] not in BABI_TASKS:
            continue
        if cur and r["task"] == cur[0]["task"] and \
                r["passage"].startswith(prev) and len(r["passage"]) > len(prev):
            cur.append(r)
        else:
            if cur:
                stories.append(cur)
            cur = [r]
        prev = r["passage"]
    if cur:
        stories.append(cur)
    return stories


def build_babi():
    stories = _babi_stories()
    rng = random.Random(SEED + 2)
    by_task = {bt: [s for s in stories if s[0]["task"] == bt] for bt in BABI_TASKS}
    for pool in by_task.values():
        rng.shuffle(pool)
    si = {bt: 0 for bt in BABI_TASKS}
    tasks = []
    for task_id in range(N_TASKS["babi"]):
        n_stories = rng.randint(*BABI_STORIES_PER_SESSION)
        # difficulty ramp, as in the exp-1/exp-2 generators: early stories are
        # bAbI task 1 (trivial per turn), later ones task 2, the last task 3 --
        # so hallucination onset reflects long-horizon accumulation rather
        # than immediate per-turn overload
        ramp = [1] * (n_stories - n_stories // 2 - 1) + \
               [2] * (n_stories // 2) + [3]
        turns, t = [], 0
        for bt in ramp:
            story = by_task[bt][si[bt] % len(by_task[bt])]
            si[bt] += 1
            seen = 0
            for qi, row in enumerate(story):
                lines = row["passage"].splitlines()
                new_lines = lines[seen:]
                seen = len(lines)
                t += 1
                turns.append({
                    "turn": t,
                    "new_story": qi == 0,
                    "story_lines": new_lines,
                    "question": row["question"],
                    "answer": row["answer"],
                    "babi_task": row["task"],
                })
        task = {
            "task_id": task_id,
            "difficulty": round(sum(tr["babi_task"] for tr in turns) / (3 * len(turns)), 3),
            "horizon": len(turns),
            "n_stories": n_stories,
            "turns": turns,
        }
        _augment(task, rng, list(BABI_SHADOW_NAMES))
        tasks.append(task)
    return tasks


BUILDERS = {"coding": build_coding, "registers": build_registers, "babi": build_babi}


def main(sets=None):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for ts in (sets or list(BUILDERS)):
        tasks = BUILDERS[ts]()
        out = task_file(ts)
        out.write_text(json.dumps(tasks))
        n_turns = sum(t["horizon"] for t in tasks)
        print(f"[{ts}] wrote {len(tasks)} tasks, {n_turns} turns -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", type=str, default=None)
    args = ap.parse_args()
    main(args.sets.split(",") if args.sets else None)
