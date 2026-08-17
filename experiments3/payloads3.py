"""Domain-independent canary payload synthesis for experiment 3.

Every task set (coding / registers / babi) gets the SAME canary machinery: a
per-turn payload dict computed here at generation time, so every canary answer
is objective and machine-checkable.  Only the shadow-item vocabulary is
domain-specific (interference requires names that look like the task's own).
"""
from .config3 import (CHECKSUM_MOD, EVENT_P, EVENT_P_HEAVY, LAGS,
                      N_AUDIT_PROBES, N_COUNTERS, N_COUNTERS_HEAVY,
                      SHADOW_RENAME_P, SHADOW_SLOTS, STAIR_PERIOD, TAG_NOTE_P,
                      TAG_QUERY_P)

COLORS = ["red", "blue", "gold", "plum", "teal", "rust"]
TAG_WORDS = ["maple", "quartz", "falcon", "harbor", "cedar", "onyx", "tundra",
             "lagoon", "aspen", "topaz", "birch", "canyon", "frost", "orchid",
             "pebble", "reef", "thorn", "umber", "wharf", "zephyr"]
AUDIT_WORDS = ["TANGO", "VICTOR", "OSCAR", "SIERRA", "BRAVO", "DELTA", "ECHO",
               "KILO", "LIMA", "NOVEMBER"]


def _even_last_digit(ticket):
    return int(ticket[-1]) % 2 == 0


def _contains_seven(ticket):
    return "7" in ticket.split("-")[1]


def stair_expect(t, tickets, period):
    """Ledger fields expected at turn t (tickets = tickets for turns 1..t)."""
    fields = [str(t)]
    if t > period:
        fields.append(str(sum(1 for k in tickets if _even_last_digit(k))))
    if t > 2 * period:
        fields.append(tickets[0])
    if t > 3 * period:
        fields.append(str(sum(1 for k in tickets if _contains_seven(k))))
    return fields


def gen_payloads(rng, horizon, shadow_vocab):
    """Returns (task_level_dict, [per-turn payload dicts])."""
    colors = COLORS[:N_COUNTERS]
    colors_heavy = COLORS[:N_COUNTERS_HEAVY]
    audit_code = f"XK-{rng.randint(1000, 9999)}-{rng.choice(AUDIT_WORDS)}"

    # sparse_recall probe turns: spread over the horizon, never before turn 3
    probes = set()
    for frac in [0.30, 0.50, 0.72, 0.92][:N_AUDIT_PROBES]:
        t = max(3, min(horizon, round(frac * horizon) + rng.choice([-1, 0, 1])))
        while t in probes:
            t = min(horizon, t + 1)
        probes.add(t)

    shadow_names = list(shadow_vocab)
    rng.shuffle(shadow_names)
    shadow_init = [shadow_names.pop() for _ in range(SHADOW_SLOTS)]
    shadow_cur = list(shadow_init)

    tickets = []
    counts = {c: 0 for c in colors}
    counts_heavy = {c: 0 for c in colors_heavy}
    check = 0
    tagged = {}      # ticket -> word
    untagged = []    # past tickets with no tag
    turns = []

    for t in range(1, horizon + 1):
        ticket = f"ENG-{rng.randint(1000, 9999)}"
        tickets.append(ticket)
        p = {"ticket": ticket}

        # --- lag_span
        for k in LAGS:
            p[f"lag{k}"] = tickets[t - 1 - k] if t - 1 - k >= 0 else "NONE"

        # --- multi_counter (light + heavy streams are independent draws)
        p["event"] = rng.choice(colors) if rng.random() < EVENT_P else None
        if p["event"]:
            counts[p["event"]] += 1
        p["counts"] = dict(counts)
        p["event_heavy"] = (rng.choice(colors_heavy)
                            if rng.random() < EVENT_P_HEAVY else None)
        if p["event_heavy"]:
            counts_heavy[p["event_heavy"]] += 1
        p["counts_heavy"] = dict(counts_heavy)

        # --- chain_checksum
        p["key"] = rng.randint(1, 40)
        check = (check + p["key"]) % CHECKSUM_MOD
        p["check"] = check

        # --- interference_twin
        p["shadow_rename"] = None
        if t > 1 and rng.random() < SHADOW_RENAME_P and shadow_names:
            slot = rng.randrange(SHADOW_SLOTS)
            new = shadow_names.pop()
            shadow_cur[slot] = new
            p["shadow_rename"] = {"slot": slot + 1, "new": new}
        p["shadow_query"] = rng.randrange(SHADOW_SLOTS) + 1
        p["shadow_truth"] = shadow_cur[p["shadow_query"] - 1]

        # --- confab_trap (note applies to the CURRENT ticket, query targets a
        # strictly PAST ticket, half of them never-tagged: the correct answer
        # is then exactly NONE and anything else is a fabrication)
        p["tag_note"] = None
        if rng.random() < TAG_NOTE_P:
            word = rng.choice(TAG_WORDS)
            tagged[ticket] = word
            p["tag_note"] = {"ticket": ticket, "word": word}
        else:
            untagged.append(ticket)
        p["tag_query"] = p["tag_truth"] = None
        p["tag_is_trap"] = False
        if t >= 3 and rng.random() < TAG_QUERY_P:
            past_tagged = [k for k in tagged if k != ticket]
            past_untagged = [k for k in untagged if k != ticket]
            want_trap = rng.random() < 0.5
            pool = past_untagged if (want_trap and past_untagged) else past_tagged
            if not pool:
                pool = past_untagged or past_tagged
            if pool:
                target = rng.choice(pool)
                p["tag_query"] = target
                p["tag_truth"] = tagged.get(target, "NONE")
                p["tag_is_trap"] = p["tag_truth"] == "NONE"

        # --- sparse_recall
        p["audit_probe"] = t in probes

        # --- staircase
        p["stair_expect"] = stair_expect(t, tickets, STAIR_PERIOD)
        p["stair_update"] = None
        for stage, ut in enumerate([STAIR_PERIOD + 1, 2 * STAIR_PERIOD + 1,
                                    3 * STAIR_PERIOD + 1], start=2):
            if t == ut:
                p["stair_update"] = stage

        turns.append(p)

    task_level = {
        "audit_code": audit_code,
        "shadow_init": shadow_init,
        "colors": colors,
        "colors_heavy": colors_heavy,
    }
    return task_level, turns
