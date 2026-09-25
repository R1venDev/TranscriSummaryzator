"""Selected-generation task view and bounded local edit service.

The caller supplies the application's verified generation resolver and protects
the HTTP route with the summary administrator guard. No model, provider, or
external task system is contacted here. Every view is rendered from the same
sealed model document and versioned human overrides.
"""

from __future__ import annotations

from dataclasses import dataclass
import fcntl
import json
from pathlib import Path
import re
from typing import Callable

from . import load_source, render_document
from .publication import is_luna_generation
from .tasks import RevisionConflict, TaskStore


_ACTION_ID = re.compile(r"A-[0-9a-f]{24}\Z")
_GENERATION_ID = re.compile(r"[0-9]{8}-[0-9]{6}-[0-9a-f]{12}\Z")


class TaskViewUnavailable(ValueError):
    """The selected generation cannot safely serve the Luna card editor."""


@dataclass(frozen=True)
class TaskView:
    generation_id: str
    source_sha256: str
    tasks: list[dict]
    rendered: dict
    source_refs: dict[str, list[dict]]

    def public(self) -> dict:
        """Admin API payload; source quotations are already in the transcript."""
        return {
            "generation_id": self.generation_id,
            "source_sha256": self.source_sha256,
            "tasks": self.tasks,
            "source_refs": self.source_refs,
        }


def _stamp(milliseconds: int) -> str:
    seconds = milliseconds // 1000
    return f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


def _read_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def read_current(output_dir: Path, transcript_path: Path, task_db: Path,
                 package_resolver: Callable[[Path], Path | None]) -> TaskView:
    """Render the currently accepted Luna generation with human edits.

    ``package_resolver`` must verify the selected generation manifest and all
    artifact digests (the app's ``current_summary_output`` does so). A legacy
    generation stays readable via its legacy reader, but has no Luna cards.
    """
    output_dir = Path(output_dir)
    package = package_resolver(output_dir)
    if package is None:
        raise TaskViewUnavailable("Нет проверенной опубликованной версии конспекта")
    package = Path(package)
    if not _GENERATION_ID.fullmatch(package.name) or not is_luna_generation(package):
        raise TaskViewUnavailable("Редактирование карточек доступно для нового конспекта Luna")
    if package.resolve().parent != (output_dir / "summary_generations").resolve():
        raise TaskViewUnavailable("Папка конспекта не относится к выбранной записи")
    source_text, source_index, source_sha256 = load_source(transcript_path)
    del source_text  # The full transcript is neither returned nor logged here.
    sealed_manifest = _read_json(package / "generation_manifest.json")
    run_manifest = _read_json(package / "run_manifest.json")
    if (sealed_manifest.get("source_sha256") != source_sha256
            or run_manifest.get("source_sha256") != source_sha256):
        raise TaskViewUnavailable("Исходная стенограмма изменилась после публикации")
    document = _read_json(package / "model_document.json")
    sealed_tasks = _read_json(package / "tasks.json")
    if not isinstance(document, dict) or not isinstance(sealed_tasks, list):
        raise TaskViewUnavailable("Карточки опубликованной версии повреждены")
    action_ids = [item.get("action_id") if isinstance(item, dict) else None for item in sealed_tasks]
    store = TaskStore(task_db)
    effective = store.effective_for_sealed(source_sha256, document["tasks"], action_ids)
    rendered = render_document(document, source_index, effective)
    refs = {}
    for task in effective:
        refs[task["action_id"]] = [
            {
                "source_id": source_id,
                "start_ms": source_index["by_id"][source_id]["start_ms"],
                "timecode": _stamp(source_index["by_id"][source_id]["start_ms"]),
                "speaker": source_index["by_id"][source_id]["speaker"],
                "text": source_index["by_id"][source_id]["text"],
            }
            for source_id in task["source_ids"]
        ]
    return TaskView(package.name, source_sha256, effective, rendered, refs)


def edit_current(output_dir: Path, transcript_path: Path, task_db: Path,
                 package_resolver: Callable[[Path], Path | None], *,
                 action_id: str, expected_generation_id: str,
                 expected_revision: int, changes: dict, actor: str) -> TaskView:
    """CAS edit only a card in the selected sealed generation.

    Publication and edits use one per-meeting lock. A stale browser therefore
    cannot quietly apply an edit to a different generation after regeneration.
    TaskStore provides the independent per-card revision CAS and edit history.
    """
    if not isinstance(action_id, str) or not _ACTION_ID.fullmatch(action_id):
        raise ValueError("Неверный ID карточки")
    if not isinstance(expected_generation_id, str) or not _GENERATION_ID.fullmatch(expected_generation_id):
        raise ValueError("Неверная версия конспекта")
    output_dir = Path(output_dir)
    lock_path = output_dir / ".summary_publication.lock"
    if not output_dir.is_dir():
        raise TaskViewUnavailable("Папка записи недоступна")
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            selected = read_current(output_dir, transcript_path, task_db, package_resolver)
            if selected.generation_id != expected_generation_id:
                raise RevisionConflict("Конспект обновился; загрузите карточку заново")
            if action_id not in {task["action_id"] for task in selected.tasks}:
                raise TaskViewUnavailable("Карточка не входит в выбранный конспект")
            TaskStore(task_db).update(action_id, expected_revision, changes, actor)
            return read_current(output_dir, transcript_path, task_db, package_resolver)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
