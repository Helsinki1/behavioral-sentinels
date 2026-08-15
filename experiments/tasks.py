"""Generator for 200 synthetic state book-keeping tasks.

Each task: a set of named integer registers. Every turn the user issues
update operations (applied in order) and asks for the current value of
1-2 registers. Ground truth is tracked programmatically, so hallucination
labels (reported value != true value, or missing report) are objective.

The same 200 tasks are reused across every condition and model so that
comparisons are paired.
"""
import json
import random

from .config import (DATA_DIR, HORIZON_RANGE, N_TASKS, NUM_KEYS_RANGE, SEED)

COLORS = ["red", "blue", "green", "gray", "gold", "teal", "pink", "black",
          "white", "amber", "coral", "olive", "plum", "cyan", "rust"]
NOUNS = ["box", "bin", "crate", "tray", "rack", "shelf", "drawer", "pod",
         "cell", "tank", "vault", "slot", "case", "hub", "node", "port",
         "cart", "jar", "kit", "pack"]
CANARY_WORDS = ["ember", "quartz", "willow", "falcon", "cedar", "onyx",
                "harbor", "tundra", "maple", "cobalt", "raven", "juniper",
                "granite", "lagoon", "aspen", "salmon", "topaz", "birch",
                "meadow", "canyon", "frost", "dune", "orchid", "pebble",
                "reef", "sable", "thorn", "umber", "vine", "wharf",
                "yarrow", "zephyr", "basil", "clover", "delta", "fjord"]
SENTINEL_NAMES = ["sentinel", "tracer", "beacon", "marker"]


def _make_names(rng, n, difficulty):
    """Generate n unique register names; higher difficulty -> more confusable."""
    n_colors = max(2, int(round(len(COLORS) * (1.0 - 0.75 * difficulty))))
    n_nouns = max(2, int(round(len(NOUNS) * (1.0 - 0.75 * difficulty))))
    colors = rng.sample(COLORS, n_colors)
    nouns = rng.sample(NOUNS, n_nouns)
    combos = [f"{c}_{s}" for c in colors for s in nouns]
    rng.shuffle(combos)
    if len(combos) < n:  # fall back: add numeric suffixes
        combos += [f"{c}_{s}2" for c in colors for s in nouns]
    return combos[:n]


def _gen_ops(rng, state, n_ops, spare_names):
    """Generate n_ops operations valid against `state` (applied in order)."""
    ops = []
    for _ in range(n_ops):
        live = sorted(state.keys())
        kinds = ["set", "add", "sub", "add", "sub", "copyplus", "swap"]
        if spare_names and rng.random() < 0.10:
            kinds.append("create")
        if len(live) > 5 and rng.random() < 0.08:
            kinds.append("delete")
        kind = rng.choice(kinds)
        if kind == "set":
            k = rng.choice(live)
            v = rng.randint(0, 99)
            ops.append({"op": "set", "k": k, "v": v})
            state[k] = v
        elif kind == "add":
            k = rng.choice(live)
            n = rng.randint(1, 12)
            ops.append({"op": "add", "k": k, "n": n})
            state[k] += n
        elif kind == "sub":
            k = rng.choice(live)
            n = rng.randint(1, 12)
            ops.append({"op": "sub", "k": k, "n": n})
            state[k] -= n
        elif kind == "swap" and len(live) >= 2:
            a, b = rng.sample(live, 2)
            ops.append({"op": "swap", "a": a, "b": b})
            state[a], state[b] = state[b], state[a]
        elif kind == "copyplus" and len(live) >= 2:
            a, b = rng.sample(live, 2)
            n = rng.randint(1, 15)
            ops.append({"op": "copyplus", "a": a, "b": b, "n": n})
            state[a] = state[b] + n
        elif kind == "create" and spare_names:
            k = spare_names.pop()
            v = rng.randint(0, 99)
            ops.append({"op": "create", "k": k, "v": v})
            state[k] = v
        elif kind == "delete" and len(live) > 5:
            k = rng.choice(live)
            ops.append({"op": "delete", "k": k})
            del state[k]
        else:
            k = rng.choice(live)
            n = rng.randint(1, 12)
            ops.append({"op": "add", "k": k, "n": n})
            state[k] += n
    return ops


def render_op(op):
    o = op["op"]
    if o == "set":
        return f"Set {op['k']} to {op['v']}."
    if o == "add":
        return f"Add {op['n']} to {op['k']}."
    if o == "sub":
        return f"Subtract {op['n']} from {op['k']}."
    if o == "swap":
        return f"Swap the values of {op['a']} and {op['b']}."
    if o == "copyplus":
        return f"Set {op['a']} to the current value of {op['b']} plus {op['n']}."
    if o == "create":
        return f"Create a new register named {op['k']} with value {op['v']}."
    if o == "delete":
        return f"Delete the register {op['k']} entirely."
    raise ValueError(o)


def generate_tasks():
    rng = random.Random(SEED)
    tasks = []
    for task_id in range(N_TASKS):
        difficulty = rng.random()
        num_keys = NUM_KEYS_RANGE[0] + int(round(difficulty * (NUM_KEYS_RANGE[1] - NUM_KEYS_RANGE[0])))
        horizon = rng.randint(*HORIZON_RANGE)
        base_ops = 1 + int(round(difficulty * 3))  # 1..4 ops per turn

        names = _make_names(rng, num_keys + 8, difficulty)
        initial_keys = names[:num_keys]
        spare_names = names[num_keys:]
        init_state = {k: rng.randint(0, 99) for k in initial_keys}

        state = dict(init_state)
        turns = []
        sentinel_val = rng.randint(10, 99)
        for t in range(1, horizon + 1):
            # difficulty ramp: early turns are light, load grows with turn number,
            # so hallucination onset reflects long-horizon accumulation rather
            # than immediate per-turn overload
            ramp = base_ops * (0.3 + 0.7 * t / horizon)
            n_ops = max(1, int(round(ramp)) + rng.choice([-1, 0, 0, 1]))
            ops = _gen_ops(rng, state, n_ops, spare_names)
            live = sorted(state.keys())
            n_q = min(len(live), rng.choice([1, 2, 2]))
            queries = rng.sample(live, n_q)
            truth = {q: state[q] for q in queries}
            # sentinel variable for the variable_check canary
            changed = t > 1 and rng.random() < 0.5
            if changed:
                new_val = sentinel_val
                while new_val == sentinel_val:
                    new_val = rng.randint(10, 99)
                sentinel_val = new_val
            turns.append({
                "turn": t,
                "ops": ops,
                "queries": queries,
                "truth": truth,
                "sentinel_value": sentinel_val,
                "sentinel_changed": changed,   # vs previous turn (turn 1: False)
                "canary_word": rng.choice(CANARY_WORDS),
            })

        tasks.append({
            "task_id": task_id,
            "difficulty": round(difficulty, 3),
            "horizon": horizon,
            "num_keys": num_keys,
            "initial_state": init_state,
            "sentinel_name": rng.choice(SENTINEL_NAMES),
            "turns": turns,
        })
    return tasks


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tasks = generate_tasks()
    out = DATA_DIR / "tasks.json"
    out.write_text(json.dumps(tasks))
    n_turns = sum(t["horizon"] for t in tasks)
    print(f"Wrote {len(tasks)} tasks, {n_turns} total turns -> {out}")


if __name__ == "__main__":
    main()
