"""Read completed native speech jobs into the existing summary queue."""
from pathlib import Path
import json
import hashlib
import os
import shutil
import sqlite3
import uuid

# Speech export bytes are copied, never rewritten or re-transcribed.
EXPORTS = ('transcript.json','transcript.md','transcript.txt','transcript.html',
           'subtitles.srt','speakers.json','result.json','diarization.rttm','result.rttm')


def sync(pipeline):
    source_root = os.environ.get('TRANSCRI_NATIVE_SPEECH_ROOT')
    if not source_root:
        return
    root = Path(source_root)
    speech = sqlite3.connect('file:'+str(root/'state/queue.sqlite3')+'?mode=ro',uri=True,timeout=10)
    speech.row_factory = sqlite3.Row
    try:
        with pipeline.connect() as db:
            rows = db.execute("SELECT * FROM jobs WHERE native_job_id IS NOT NULL AND status IN ('queued','running') ORDER BY id").fetchall()
            for row in rows:
                try:
                    _sync_row(pipeline,db,speech,root,row)
                except (ValueError,OSError,KeyError,sqlite3.Error) as exc:
                    print('Speech import job {}: {}'.format(row['id'],type(exc).__name__),flush=True)
    finally:
        speech.close()


def request(pipeline, job_id, kind, speaker_count=None):
    """Durable explicit UI operation; never mutates an active summary source."""
    from summary.project_profiles import CURRENT
    with pipeline.connect() as db:
        db.execute('BEGIN IMMEDIATE')
        row = db.execute('SELECT * FROM jobs WHERE id=?',(job_id,)).fetchone()
        if (row is None or row['project_id'] != CURRENT.get() or not row['native_job_id']
                or row['status'] != 'done'):
            raise ValueError('Для этой записи нет готового задания службы речи')
        if row['summary_status'] in ('running','queued','queued_force','pending_batch','submission_unknown','credential_required') or pipeline._luna_output_has_active_batch(row['output_dir']):
            raise ValueError('Дождитесь завершения текущего конспекта')
        if row['native_operation']:
            prior=json.loads(row['native_operation'])
            if prior['status'] in ('requested','applied','unknown'):
                raise ValueError('Предыдущее изменение ещё выполняется')
        operation={'id':uuid.uuid4().hex,'kind':kind,'status':'requested','speaker_count':speaker_count}
        pipeline.update_job(db,job_id,native_operation=json.dumps(operation),status='queued',stage='speech_operation',detail='Изменение голосов поставлено в очередь')
    return {'ok':True,'job_id':job_id,'queued':True}


def _sync_row(pipeline,db,speech,root,row):
    if row['native_operation'] and json.loads(row['native_operation'])['status'] != 'applied':
        return
    remote = speech.execute('SELECT * FROM jobs WHERE id=?',(row['native_job_id'],)).fetchone()
    if remote is None or remote['project_id'] != row['project_id'] or remote['content_sha256'] != row['content_sha256']:
        return  # wrong/missing identity never imports another recording
    if remote['status'] == 'done':
        original = Path(remote['output_dir'])
        # Host paths are mounted at their original absolute location.
        if root.resolve() not in original.resolve().parents:
            raise ValueError('Speech export outside mounted source root')
        revision=hashlib.sha256((original/'transcript.json').read_bytes()).hexdigest()
        final = pipeline.OUTPUTS / ('speech-'+str(row['id'])+'-'+revision[:16])
        if not final.exists():
            temporary = final.with_name('.'+final.name+'-'+uuid.uuid4().hex)
            temporary.mkdir(mode=0o700)
            try:
                for name in EXPORTS:
                    src = original / name
                    if src.is_file():
                        shutil.copyfile(src,temporary/name)
                if not (temporary/'transcript.json').is_file():
                    raise ValueError('Speech transcript missing')
                if hashlib.sha256((temporary/'transcript.json').read_bytes()).hexdigest() != revision:
                    raise ValueError('Speech transcript changed during capture')
                # Commit complete files before exposing this directory.
                for file in temporary.iterdir():
                    with file.open('rb') as handle: os.fsync(handle.fileno())
                os.rename(temporary, final)
                descriptor=os.open(final.parent,os.O_RDONLY|os.O_DIRECTORY)
                try: os.fsync(descriptor)
                finally: os.close(descriptor)
            finally:
                if temporary.exists(): shutil.rmtree(temporary)
        pipeline.update_job(db,row['id'],status='done',stage='done',progress=100,detail='Готово',output_dir=str(final),error=None,summary_status='queued',summary_stage='summary_queued',summary_progress=0,summary_detail='В очереди суммаризатора',finished_at=pipeline.now(),native_operation=None,speaker_count=remote['speaker_count'])
    elif remote['status'] == 'failed':
        pipeline.update_job(db,row['id'],status='failed',stage=remote['stage'],progress=remote['progress'],detail=remote['detail'],error=remote['error'])
    else:
        pipeline.update_job(db,row['id'],status=remote['status'],stage=remote['stage'],progress=remote['progress'],detail=remote['detail'])
