"""Generator for the experiment-2 CODING tasks.

Experiment 1 asked the model to track integer registers. Experiment 2 asks it
to maintain a small Python module across a long session: functions are added,
renamed, deleted, and have parameters added/removed/renamed. Every turn the
model must (a) emit the code for the requested change and (b) report the
current signature of one or two functions.

Ground truth (the live symbol table) is simulated programmatically, so the
hallucination labels stay fully objective -- exactly as in experiment 1 --
while the surface task is real code:

  * a wrong/missing `SIG` line          -> lost track of state
  * a signature for a DELETED function  -> fabricated fact
  * code that does not parse            -> nonexistent syntax
  * no code block at all                -> abandoned subgoal
  * a call to a symbol that was never
    defined, imported or bound          -> nonexistent API call

Every canary condition reuses the same task set, so all comparisons are paired.
"""
import json
import random
import string

from .config2 import DATA_DIR, HORIZON_RANGE, N_TASKS, NUM_FNS_RANGE, SEED

VERBS = ["parse", "load", "fetch", "build", "render", "resolve", "collect",
         "encode", "flush", "merge", "scan", "emit", "verify", "index",
         "apply", "expand", "align", "order", "unpack", "inspect"]
OBJECTS = ["config", "record", "payload", "session", "token", "manifest",
           "bucket", "cursor", "digest", "entry", "header", "packet",
           "route", "schema", "socket", "target", "vector", "window"]
PARAMS = ["path", "strict", "timeout", "retries", "conn", "query", "limit",
          "offset", "verbose", "encoding", "cache", "session", "token",
          "dry_run", "depth", "fmt", "key", "mode", "backend", "chunk",
          "prefix", "salt", "window", "cursor", "flush", "seed"]
POLICY_TOKENS = ["AMBER", "COBALT", "DELTA", "EMBER", "FLINT", "GRANITE",
                 "HALON", "INDIGO", "JASPER", "KRYPTON", "LUMEN", "MARBLE",
                 "NIMBUS", "OPAL", "PUMICE", "QUARTZ", "RIPPLE", "SABLE"]

VOWELS = set("aeiouAEIOU")


def _make_fn_names(rng, n, difficulty):
    """Pool of unique function names. Higher difficulty -> fewer distinct
    verbs/objects, so names collide on prefixes and become confusable."""
    n_v = max(2, int(round(len(VERBS) * (1.0 - 0.75 * difficulty))))
    n_o = max(2, int(round(len(OBJECTS) * (1.0 - 0.75 * difficulty))))
    verbs = rng.sample(VERBS, n_v)
    objs = rng.sample(OBJECTS, n_o)
    combos = [f"{v}_{o}" for v in verbs for o in objs]
    rng.shuffle(combos)
    if len(combos) < n:
        combos += [f"{v}_{o}_v2" for v in verbs for o in objs]
    return combos[:n]


def _render_op(op):
    o = op["op"]
    if o == "add_fn":
        return f"Add a new function `{op['fn']}({', '.join(op['params'])})`."
    if o == "add_param":
        return f"Append a parameter `{op['param']}` to the end of `{op['fn']}`'s parameter list."
    if o == "remove_param":
        return f"Remove the parameter `{op['param']}` from `{op['fn']}`."
    if o == "rename_param":
        return f"In `{op['fn']}`, rename the parameter `{op['old']}` to `{op['new']}`."
    if o == "rename_fn":
        return f"Rename the function `{op['old']}` to `{op['new']}` (keep its parameters unchanged)."
    if o == "delete_fn":
        return f"Delete the function `{op['fn']}` from the module entirely."
    if o == "wire_call":
        return (f"Rewrite `{op['fn']}`'s body so that it calls `{op['callee']}` "
                f"and returns the result.")
    raise ValueError(o)


def _op_target(op):
    """First function name mentioned by the rendered instruction -- this is the
    'target function' the conditional_rule canary branches on."""
    return op.get("fn") or op["old"]


def _gen_ops(rng, fns, deleted, spare_names, spare_params, n_ops):
    """Mutate `fns` (name -> ordered param list) in place, return op records."""
    ops = []
    for _ in range(n_ops):
        live = sorted(fns)
        kinds = ["add_param", "rename_param", "add_fn", "rename_fn"]
        if len(live) > 3:
            kinds += ["remove_param", "delete_fn"]
        kinds += ["add_param", "rename_param"]  # weight the cheap edits
        kind = rng.choice(kinds)

        if kind == "add_fn" and spare_names:
            name = spare_names.pop()
            params = rng.sample(PARAMS, rng.randint(1, 3))
            fns[name] = list(params)
            ops.append({"op": "add_fn", "fn": name, "params": list(params)})
        elif kind == "add_param" and live:
            fn = rng.choice(live)
            avail = [p for p in PARAMS if p not in fns[fn]]
            if not avail:
                continue
            p = rng.choice(avail)
            fns[fn].append(p)
            ops.append({"op": "add_param", "fn": fn, "param": p})
        elif kind == "remove_param" and live:
            cands = [f for f in live if len(fns[f]) > 1]
            if not cands:
                continue
            fn = rng.choice(cands)
            p = rng.choice(fns[fn])
            fns[fn].remove(p)
            ops.append({"op": "remove_param", "fn": fn, "param": p})
        elif kind == "rename_param" and live:
            fn = rng.choice(live)
            old = rng.choice(fns[fn])
            avail = [p for p in PARAMS if p not in fns[fn]] or spare_params
            new = rng.choice(avail)
            fns[fn][fns[fn].index(old)] = new
            ops.append({"op": "rename_param", "fn": fn, "old": old, "new": new})
        elif kind == "rename_fn" and live and spare_names:
            old = rng.choice(live)
            new = spare_names.pop()
            fns[new] = fns.pop(old)
            deleted.discard(new)
            ops.append({"op": "rename_fn", "old": old, "new": new})
        elif kind == "delete_fn" and len(live) > 3:
            fn = rng.choice(live)
            del fns[fn]
            deleted.add(fn)
            ops.append({"op": "delete_fn", "fn": fn})
    return ops


def _maybe_wire(rng, fns, ops):
    """Append at most one wire_call, chosen from the symbol table as it stands
    AFTER every other edit this turn, so the instruction can never be
    invalidated by a later edit in the same turn."""
    live = sorted(fns)
    if len(live) >= 2 and rng.random() < 0.25:
        fn, callee = rng.sample(live, 2)
        ops.append({"op": "wire_call", "fn": fn, "callee": callee})
    return ops


def generate_tasks():
    rng = random.Random(SEED)
    tasks = []
    for task_id in range(N_TASKS):
        difficulty = rng.random()
        num_fns = NUM_FNS_RANGE[0] + int(round(difficulty * (NUM_FNS_RANGE[1] - NUM_FNS_RANGE[0])))
        horizon = rng.randint(*HORIZON_RANGE)
        base_ops = 1 + int(round(difficulty * 2))  # 1..3 edits per turn

        names = _make_fn_names(rng, num_fns + horizon + 6, difficulty)
        initial = names[:num_fns]
        spare_names = names[num_fns:]
        rng.shuffle(spare_names)

        fns = {n: rng.sample(PARAMS, rng.randint(1, 3)) for n in initial}
        initial_module = {n: list(p) for n, p in fns.items()}
        deleted = set()

        policy_token = f"{rng.choice(POLICY_TOKENS)}-{rng.randint(10, 99)}"
        prev_ticket = None
        n_deletes = n_renames = 0
        first_target = None
        turns = []

        for t in range(1, horizon + 1):
            # difficulty ramp: light early turns, heavier later, so onset
            # reflects long-horizon accumulation rather than per-turn overload
            ramp = base_ops * (0.35 + 0.65 * t / horizon)
            n_ops = max(1, int(round(ramp)) + rng.choice([-1, 0, 0, 1]))
            ops = _gen_ops(rng, fns, deleted, spare_names, PARAMS, n_ops)
            if not ops:  # degenerate turn -- force one safe edit
                fn = rng.choice(sorted(fns))
                avail = [p for p in PARAMS if p not in fns[fn]]
                p = rng.choice(avail)
                fns[fn].append(p)
                ops = [{"op": "add_param", "fn": fn, "param": p}]
            ops = _maybe_wire(rng, fns, ops)

            n_deletes += sum(1 for o in ops if o["op"] == "delete_fn")
            n_renames += sum(1 for o in ops if o["op"] == "rename_fn")
            target_fn = _op_target(ops[0])
            if first_target is None:
                first_target = target_fn

            # ---- report queries: 1-2 signatures, sometimes a deleted one (a
            # fabrication trap: the honest answer is "DELETED")
            live = sorted(fns)
            queries = []
            n_q = min(len(live), rng.choice([1, 2, 2]))
            for q in rng.sample(live, n_q):
                queries.append({"fn": q, "deleted": False, "params": list(fns[q])})
            if deleted and rng.random() < 0.22:
                d = rng.choice(sorted(deleted))
                queries.append({"fn": d, "deleted": True, "params": None})
            rng.shuffle(queries)

            # ---- canary ground truth ------------------------------------
            ticket = f"ENG-{rng.randint(1000, 9999)}"
            policy_changed = t == 1 or rng.random() < 0.25
            if policy_changed and t > 1:
                new_tok = policy_token
                while new_tok == policy_token:
                    new_tok = f"{rng.choice(POLICY_TOKENS)}-{rng.randint(10, 99)}"
                policy_token = new_tok

            flag = "VOWEL" if target_fn[0] in VOWELS else "NONE"

            if t <= 4:
                stage, ledger = 1, [str(t)]
            elif t <= 8:
                stage, ledger = 2, [str(t), str(n_deletes)]
            elif t <= 12:
                stage, ledger = 3, [str(t), str(n_deletes), first_target]
            else:
                stage, ledger = 4, [str(t), str(n_deletes), first_target, str(n_renames)]

            turns.append({
                "turn": t,
                "ops": ops,
                "instructions": [_render_op(o) for o in ops],
                "target_fn": target_fn,
                "wire_calls": [{"fn": o["fn"], "callee": o["callee"]}
                               for o in ops if o["op"] == "wire_call"],
                "queries": queries,
                "live_symbols": live,
                "defs_required": sorted({o.get("fn") or o["new"] for o in ops
                                         if o["op"] != "delete_fn"} & set(live)),
                "defs_truth": {n: list(fns[n]) for n in
                               ({o.get("fn") or o["new"] for o in ops
                                 if o["op"] != "delete_fn"} & set(live))},
                "deletes_required": sorted({o["fn"] for o in ops
                                            if o["op"] == "delete_fn"}),
                "deleted_symbols": sorted(deleted),
                "ticket": ticket,
                "prev_ticket": prev_ticket,
                "policy_token": policy_token,
                "policy_changed": policy_changed,
                "flag": flag,
                "ledger_stage": stage,
                "ledger_expect": ledger,
            })
            prev_ticket = ticket

        tasks.append({
            "task_id": task_id,
            "difficulty": round(difficulty, 3),
            "horizon": horizon,
            "num_fns": num_fns,
            "initial_module": initial_module,
            "turns": turns,
        })
    return tasks


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tasks = generate_tasks()
    out = DATA_DIR / "tasks2.json"
    out.write_text(json.dumps(tasks))
    n_turns = sum(t["horizon"] for t in tasks)
    print(f"Wrote {len(tasks)} coding tasks, {n_turns} total turns -> {out}")


if __name__ == "__main__":
    main()
