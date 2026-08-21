"""Build the experiment-9 session pool from the lost_in_conversation `math`
split: 3 sharded GSM8K problems per session, one shard revealed per turn,
plus the lag_span ticket payloads (user-issued content the active arms'
probe needs). Deterministic under SEED9.

  python -m experiments9.build9        # writes data9/tasks9_shardmath.json
"""
import json
import random
import re

from .config9 import DATA_DIR, EPISODES_PER_SESSION, N_SESSIONS, SEED9

SOURCE = DATA_DIR / "sharded_instructions_600.json"
OUT = DATA_DIR / "tasks9_shardmath.json"


def gold_number(answer_text):
    m = re.search(r"####\s*([-\d,\.]+)", answer_text)
    assert m, f"no #### answer in: {answer_text[-80:]}"
    return m.group(1).replace(",", "").rstrip(".")


def lag_payload(rng, tickets, turn_no):
    ticket = f"ENG-{rng.randrange(1000, 10000)}"
    tickets.append(ticket)

    def lag(k):
        i = len(tickets) - 1 - k
        return tickets[i].upper() if i >= 0 else "NONE"
    return ticket, {"ticket": ticket, "lag1": lag(1), "lag3": lag(3),
                    "lag6": lag(6)}


def build():
    math = [x for x in json.loads(SOURCE.read_text()) if x["task"] == "math"]
    rng = random.Random(SEED9)
    rng.shuffle(math)
    sessions = []
    for s in range(N_SESSIONS):
        eps = math[s * EPISODES_PER_SESSION:(s + 1) * EPISODES_PER_SESSION]
        if len(eps) < EPISODES_PER_SESSION:
            break
        turns, tickets = [], []
        t = 0
        for e_idx, ep in enumerate(eps):
            shards = sorted(ep["shards"], key=lambda x: x["shard_id"])
            for i, sh in enumerate(shards):
                t += 1
                _, c3 = lag_payload(rng, tickets, t)
                turns.append({
                    "turn": t, "episode": e_idx,
                    "source_id": ep["task_id"],
                    "shard_no": i + 1, "n_shards": len(shards),
                    "new_problem": i == 0, "final": i == len(shards) - 1,
                    "shard": sh["shard"],
                    "answer": gold_number(ep["answer"]) if i == len(shards) - 1
                              else None,
                    "c3": c3,
                })
        sessions.append({
            "task_id": s, "horizon": t,
            "difficulty": round(t / (EPISODES_PER_SESSION * 12), 3),
            "n_episodes": len(eps),
            "source_ids": [ep["task_id"] for ep in eps],
            "turns": turns,
        })
    OUT.write_text(json.dumps(sessions, indent=1))
    hz = [s["horizon"] for s in sessions]
    print(f"wrote {OUT}: {len(sessions)} sessions, horizons "
          f"{min(hz)}-{max(hz)} (mean {sum(hz)/len(hz):.1f})")
    return sessions


if __name__ == "__main__":
    build()
