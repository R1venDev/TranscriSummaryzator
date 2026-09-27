#!/usr/bin/env python3
"""Review or commit one saved v3 Luna summary without network inference."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from summary.luna_v1.recovery import RecoveryExpected, recover_saved_v3, recovery_code_sha256


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-code-sha256", action="store_true")
    parser.add_argument("--private-root", type=Path)
    parser.add_argument("--root-job-id")
    parser.add_argument("--audit-job-id")
    parser.add_argument("--source-sha256")
    parser.add_argument("--root-manifest-sha256")
    parser.add_argument("--draft-sha256")
    parser.add_argument("--audit-sha256")
    parser.add_argument("--code-sha256")
    parser.add_argument("--error-code")
    parser.add_argument("--apply", action="store_true", help="publish locally after preflight")
    args = parser.parse_args()
    if args.print_code_sha256:
        print(recovery_code_sha256())
        return 0
    required = ("private_root", "root_job_id", "audit_job_id", "source_sha256",
                "root_manifest_sha256", "draft_sha256", "audit_sha256",
                "code_sha256", "error_code")
    missing = ["--" + key.replace("_", "-") for key in required if getattr(args, key) is None]
    if missing:
        parser.error("missing arguments: " + ", ".join(missing))
    result = recover_saved_v3(
        private_root=args.private_root, root_job_id=args.root_job_id,
        audit_job_id=args.audit_job_id,
        expected=RecoveryExpected(
            source_sha256=args.source_sha256,
            root_manifest_sha256=args.root_manifest_sha256,
            draft_sha256=args.draft_sha256,
            audit_sha256=args.audit_sha256,
            code_sha256=args.code_sha256,
            error_code=args.error_code,
        ),
        apply=args.apply,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
