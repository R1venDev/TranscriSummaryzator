"""Idempotent layout patch for the installed optional Plane video extension.

No editor document mutation: native source stays editable and visible if the
player cannot bind it. This only hides the redundant ready source in view mode.
Back up the original script before applying; rollback restores that file.
"""
from pathlib import Path
import sys

MARKER = 'transcri-single-timecodes-v1'


def patch_source(value):
    if MARKER in value:
        return value
    edits = [
        ('    block.setAttribute("data-plane-timecodes-state", video ? "ready" : "orphan");',
         '    block.setAttribute("data-plane-timecodes-state", video ? "ready" : "orphan");\n'
         '    block.setAttribute("data-plane-timecodes-single", video && chapters.length ? "true" : "false");'),
        ('    panel.appendChild(list);',
         '    panel.appendChild(list);\n'
         '    var source = video.__planeVideoTimecodesBlock;\n'
         '    if (source) {\n'
         '      var edit = document.createElement("button");\n'
         '      edit.type = "button"; edit.textContent = "Редактировать таймкоды";\n'
         '      edit.className = "plane-video-timecodes__edit";\n'
         '      edit.addEventListener("click", function () {\n'
         '        var show = source.getAttribute("data-plane-timecodes-edit") !== "true";\n'
         '        source.setAttribute("data-plane-timecodes-edit", show ? "true" : "false");\n'
         '        edit.textContent = show ? "Скрыть редактор таймкодов" : "Редактировать таймкоды";\n'
         '      });\n'
         '      panel.appendChild(edit);\n'
         '    }'),
    ]
    for before, after in edits:
        if value.count(before) != 1:
            raise ValueError('Неизвестная версия video extension; изменения не применены')
        value = value.replace(before, after)
    # Extra presentation rule kept outside document content, with source fallback.
    value += '\n/* ' + MARKER + ' */\n(function(){var s=document.createElement("style");s.textContent=\'[data-plane-timecodes-single="true"]:not([data-plane-timecodes-edit="true"]){display:none!important}.plane-video-timecodes__edit{font-size:12px;color:#94a3b8;margin-top:8px}\';document.head.appendChild(s);})();\n'
    return value


if __name__ == '__main__':
    path = Path(sys.argv[1])
    before = path.read_text()
    after = patch_source(before)
    if after != before:
        backup = path.with_suffix(path.suffix + '.before-single-timecodes')
        if backup.exists():
            raise SystemExit('Резервная копия уже существует; проверьте состояние')
        backup.write_text(before)
        path.write_text(after)
    print(MARKER)
