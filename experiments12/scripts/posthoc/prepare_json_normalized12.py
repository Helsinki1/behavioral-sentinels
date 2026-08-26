"""Launch deployment preparation with a representation-only JSON normalization.

The frozen extractor returns tuples inside ``signal_traces[*].checkpoints``.
Writing and reloading the extract through JSON turns those tuples into lists,
so the stock preparation CLI rejects its own byte-valid artifact.  This
launcher normalizes the freshly rebuilt object through JSON before the exact
comparison.  It changes no values, tasks, thresholds, scores, or paid calls.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Sequence


def _json_value(value: Any) -> Any:
    return json.loads(
        json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] not in {"adaptive", "two-pass"}:
        raise SystemExit("usage: prepare_json_normalized12.py {adaptive|two-pass} ...")
    mode = args.pop(0)

    import experiments12.prepare_deployment12 as preparation

    frozen_extract_run = preparation.extract_run

    def normalized_extract_run(*call_args: Any, **call_kwargs: Any) -> Any:
        return _json_value(frozen_extract_run(*call_args, **call_kwargs))

    preparation.extract_run = normalized_extract_run
    if mode == "adaptive":
        from experiments12 import adaptive_deployment12

        return adaptive_deployment12.main(args)
    return preparation.main(args)


if __name__ == "__main__":
    raise SystemExit(main())
