#!/usr/bin/env python3
"""Run the unchanged installed speech release with scoped voice ownership."""
from pathlib import Path
import os
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
# Reuse the already deployed reversible record-delete adapter.
import record_dashboard_wrapper as records
from scripts.project_speech_adapter import install

if __name__ == '__main__':
    pipeline = records.load_pipeline(Path(os.environ.get('TRANSCRI_RECORD_ROOT','/srv/meeting-transcript')))
    records.install_handler(pipeline)
    install(pipeline,os.environ['TRANSCRI_PROJECT_APP_DATA'])
    for outcome in records.recover_staged_deletions(pipeline):
        print("Record deletion recovery:", outcome, flush=True)
    pipeline.main()
