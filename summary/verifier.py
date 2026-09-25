"""Deterministic plan and atomic/relation surface guards."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
import hashlib
import re
from semantics.graph import cross_episode_allowed
from semantics.equivalence import equivalent
from semantics.relation_resolver import is_explicit_acceptance_reply, is_weak_backchannel
from summary.policy import RULE_KINDS, TECHNICAL_KINDS

CAUSAL_RE = re.compile(r"(?iu)\b(?:из-за|поэтому|привел[оа]? к|в результате|для этого)\b")
NUMBER_RE = re.compile(r"(?<!\w)\d+(?:[.,:]\d+)*(?:\s*%)?")
NEGATION_RE = re.compile(r"(?iu)\b(?:не|нет|нельзя|никогда)\b")
CERTAIN_RE = re.compile(r"(?iu)\b(?:точно|обязательно|гарантированно|решено|утверждено)\b")
COMPLETED_RE = re.compile(r"(?iu)\b(?:проверен[аоы]?|завершен[аоы]?|готов[аоы]?|выполнен[аоы]?|сделан[аоы]?)\b")
CONDITION_RE = re.compile(r"(?iu)\b(?:если|когда|после|перед|пока|при|до тех пор)\b")
ROLE_RELATION_RE = re.compile(r"(?iu)(@[\w.-]+)\s+(?:долж\w*|сдела\w*|подготов\w*|отправ\w*|переда\w*|покаж\w*|размет\w*|провер\w*|анализ\w*)[^@]{0,100}(@[\w.-]+)")
INTERNAL_LABEL_RE = re.compile(
    r"(?iu)\b(?:self_committed|assigned_pending|explicit_self_commitment|"
    r"additional_tools|rhythmic_entry_implementation|high_tf_result|"
    r"stop_loss_options|should[_ ]\w+|[a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b"
)
ACRONYM_EXPANSION_RE = re.compile(r"\b(?P<acronym>[A-Z]{2,})\s*\(\s*[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)+\s*\)")
ENGLISH_WORD_RE = re.compile(r"(?i)\b[a-z]{3,}\b")
MIXED_SCRIPT_TOKEN_RE = re.compile(r"(?iu)\b(?:[а-яё]+[a-z]+|[a-z]+[а-яё]+)[a-zа-яё]*\b")
GENERIC_QUESTION_RESIDUAL_RE = re.compile(
    r"(?iu)^\s*(?:проверить\s+статус\s+конкретной\s+реализации|"
    r"подтвердить\s+точное\s+время\s+или\s+период|"
    r"уточнить\s+участника\s+или\s+объект|"
    r"подтвердить\s+действие\s+и\s+исполнителя|"
    r"определить\s+проверяемое\s+значение\s+и\s+его\s+объект|"
    r"проверить\s+объяснение|подтвердить\s+ответ|"
    r"уточнить\s+недостающ(?:ий\s+аспект|ий\s+результат))\s*[.?!]*\s*$"
)
QUESTION_SIGNAL_RE = re.compile(
    r"(?iu)(?:\?|\b(?:кто|что|где|куда|откуда|когда|почему|зачем|как|какой|"
    r"какая|какие|сколько|ли|вопрос|уточнить|подтвердить)\b)"
)
WORK_ACTION_SURFACE_RE = re.compile(
    r"(?iu)\b(?:сдела\w*|созда\w*|подготов\w*|переда\w*|отправ\w*|"
    r"предостав\w*|разме[тч]\w*|встраива\w*|встро\w*|внес\w*|перенес\w*|провер\w*|"
    r"исправ\w*|продолж\w*|эксперимент\w*|разработ\w*|реализ\w*|"
    r"обработ\w*|собра\w*|запуст\w*|добав\w*|подключ\w*|скин\w*|выгруз\w*|покаж\w*|"
    r"проанализ\w*|исслед\w*|настро\w*|обнов\w*|заверш\w*|выполн\w*)\b"
)
PROPOSAL_SURFACE_RE = re.compile(
    r"(?iu)\b(?:предлага(?:ется|ет|лось)|можно|возможно|рассматрива\w+\s+возможност|"
    r"рассматрива(?:ется|ют|лся|лась|лись)|обсуждалась\s+необходимость|"
    r"стоит\s+попробовать|планируется)\b"
)
TITLE_ACTION_FRAGMENT_RE = re.compile(
    r"(?iu)^\s*(?:попробовать|пытаться|предлагается|предложено|нужно|надо|следует|"
    r"сделать|создать|подготовить|передать|предоставить|проверить|провести|"
    r"разметить|размечать|реализовать|встроить|подключить|исправить|продолжить)\b"
)
TITLE_DANGLING_RE = re.compile(
    r"(?iu)(?:\b(?:и|или|что|чтобы|из-за|после|для|при|по|с|без)|"
    r"\b(?:\d+|один|одна|два|две|три|тр[её]х|четыре|пять|несколько)\s+"
    r"[а-яё]{4,}(?:ых|их|ого|его|ой|ую|юю))\s*$"
)
PUBLIC_SECTION_LIMITS = {
    "overview": 4,
    "decisions": 6,
    "rules": 6,
    "tasks": 10,
    "questions": 5,
    "technical": 6,
    "experiments": 5,
    "requires_verification": 8,
}
PROTECTED_TASK_STATES = {
    "self_committed", "explicit_self_commitment", "assigned", "accepted",
    "in_progress", "blocked", "completed",
}
CONTEXT_STOPWORDS = {
    "участник", "говорящий", "который", "которая", "можно", "нужно", "будет",
}


def normalize_mixed_script_confusables(text):
    """Repair only visually identical minority-script letters in one token."""
    cyr_to_lat = str.maketrans({
        "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H",
        "О": "O", "Р": "P", "С": "C", "Т": "T", "У": "Y", "Х": "X",
        "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x",
    })
    lat_to_cyr = str.maketrans({
        "A": "А", "B": "В", "E": "Е", "K": "К", "M": "М", "H": "Н",
        "O": "О", "P": "Р", "C": "С", "T": "Т", "Y": "У", "X": "Х",
        "a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "y": "у", "x": "х",
    })

    def repair(match):
        token = match.group(0)
        latin = len(re.findall(r"[A-Za-z]", token))
        cyrillic = len(re.findall(r"[А-Яа-яЁё]", token))
        if latin == cyrillic:
            return token
        candidate = token.translate(cyr_to_lat if latin > cyrillic else lat_to_cyr)
        return candidate if not MIXED_SCRIPT_TOKEN_RE.search(candidate) else token

    return MIXED_SCRIPT_TOKEN_RE.sub(repair, str(text or ""))


def public_context_tokens(value):
    """Return identity-neutral lexical tokens for public-context comparisons."""
    # Handles identify people, not topics. Removing every handle avoids both
    # recording-specific stop lists and false similarity between unrelated
    # statements made by the same participant.
    surface = re.sub(r"@[\w.-]+", " ", str(value or "").casefold())
    return {
        token
        for token in re.findall(r"(?iu)[a-zа-яё0-9]+", surface)
        if len(token) > 3 and token not in CONTEXT_STOPWORDS
    }


def public_context_stems(value):
    """Return coarse topic stems without embedding meeting-specific names."""
    result = set()
    for token in public_context_tokens(value):
        if re.fullmatch(r"[а-яё]+", token):
            token = re.sub(r"[аяоеуыию]$", "", token)
        result.add(token[:5] if len(token) >= 5 else token)
    return result


def public_context_duplicate(left, right, threshold=.72):
    """Apply the publication gate's one canonical context-overlap rule."""
    left_tokens = public_context_tokens(left)
    right_tokens = public_context_tokens(right)
    return bool(
        left_tokens and right_tokens
        and len(left_tokens & right_tokens)
        / max(1, min(len(left_tokens), len(right_tokens))) >= threshold
    )


def has_english_prose(text):
    value = re.sub(r"https?://\S+|@[\w.-]+", "", str(text or ""))
    words = ENGLISH_WORD_RE.findall(value)
    # A few source terms or product names are not prose.  Four lower-case
    # words form a useful language signal without a domain-specific allowlist.
    return len([word for word in words if word.islower()]) >= 4


def sanitize_public_surface(text):
    value = str(text or "").strip()
    # Dialogue acknowledgements and hesitation prefaces are not part of the
    # proposition that follows them. Remove only a closed set of leading
    # discourse markers; never strip a negative answer such as ``Нет``.
    value = re.sub(
        r"(?iu)^\s*(?:(?:угу|ага|ладно|ок(?:ей)?|хорошо)\s*[,.!?;:—-]+\s*|"
        r"ну\s*,?\s*как\s+сказать\s*[?!.]+\s*)+",
        "", value,
    )
    value = re.sub(
        r"(?iu)^\s*просто\s+объясняю\s*,?\s*объясняю\s*,?\s*почему\s+",
        "", value,
    )
    value = re.sub(r"(?iu),?\s*понимаешь\s*\?", ".", value)
    value = re.sub(r"(?iu)(?<!\w)(?:э(?:-э)+|х+м+)(?!\w)[,.;:]?\s*", "", value)
    # Domain translations belong in a configured vocabulary profile.  Core
    # sanitization is deliberately limited to typography/known morphology and
    # never rewrites a meeting-specific proposition.
    value = re.sub(r"(?iu)\bсвичных\b", "свечных", value)
    # A self-correction can survive inside an otherwise useful action (for
    # example, ``сделать А, ну то есть макет``).  It is dialogue scaffolding,
    # not part of the deliverable, and may appear in overview or chronology as
    # well as in the canonical task card.
    value = re.sub(
        r"(?iu)(\b[а-яё-]+(?:ть|ться))\s+(?:[аa]\s*,?\s*)?ну\s+то\s+есть\s+",
        r"\1 ", value,
    )
    # Editorial models sometimes repeat a task status both as a structured
    # suffix and as a parenthetical gloss.  Collapse only an exact repeated
    # label; differing statuses must remain visible for the conflict gate.
    value = re.sub(
        r"(?iu)(—\s*статус:\s*(?P<label>[^()\n.;]+?))\s*"
        r"\(\s*статус:\s*(?P=label)\s*\)",
        r"\1", value,
    )
    # ASR/model output can mix a single visually identical Cyrillic letter
    # into a Latin name ("Мisha") or vice versa. Repairing only confusables in
    # the minority script is typographic normalization, not translation.
    value = normalize_mixed_script_confusables(value)
    # An extractor may expand a source acronym from model knowledge.  The
    # acronym itself is source material; an English parenthetical expansion is
    # not.  Keeping only the original acronym is deterministic and cannot add
    # meeting semantics.
    value = ACRONYM_EXPANSION_RE.sub(lambda match: match.group("acronym"), value)
    return value


def unresolved_public_reference(text):
    """Detect a public fragment whose grammatical referent is outside it."""
    return bool(re.match(
        r"(?iu)^\s*(?:пока\s+что\s+)?(?:я|мы|ты|вы|он|она|оно|они|это|этот|эта|эти|тот|та|те)\b",
        sanitize_public_surface(text),
    ))


RAW_DIALOGUE_RE = re.compile(
    r"(?iu)(?:\bч[её]\b|\bну\s*,|\bну\s+то\s+есть\b|\bтипа\b|\bсобственно\b|"
    r"\bчто\s+даю\b|\bдавай\b.{0,32}\b(?:кин|скин)\w*|"
    r"(?:^|[.!?]\s*)(?:угу|ага|ок(?:ей)?|хорошо)(?:\W|$))"
)
TITLE_LOW_INFORMATION_RE = re.compile(
    r"(?iu)(?:^|;)\s*(?:есть|имеется|существует|бывает|происходит)\b"
)
INCOMPLETE_PUBLIC_FRAGMENT_RE = re.compile(
    r"(?iu)(?:\.{3}|…)$|\b(?:и|или|что|чтобы|из-за|после)\s*$"
)
NAVIGATION_PERSON_LED_RE = re.compile(
    r"(?iu)^\s*(?:@[\w.-]+|[А-ЯЁ][а-яё-]{1,30}(?:\s+(?:и|/)\s+"
    r"[А-ЯЁ][а-яё-]{1,30})?)\s+(?:пытал\w*|решил\w*|говор\w*|"
    r"сказал\w*|отмеча\w*|указыва\w*|предлага\w*|попросил\w*)\b"
)
NAVIGATION_MODAL_RE = re.compile(
    r"(?iu)^\s*(?:необходимо|нужно|надо|следует|стоит)\b|"
    r"^\s*.{0,48}\b(?:необходимо|нужно|надо|следует)\b"
)
NAVIGATION_TOPICLESS_STATE_RE = re.compile(
    r"(?iu)^\s*(?:изменения|правки|обновления|корректировки)\s+"
    r"(?:не\s+)?(?:были\s+)?(?:внесены|сделаны|применены|добавлены)\b"
)


def navigation_label_needs_repair(value):
    """Return whether a grounded chapter label is still poor navigation."""
    # Evaluate the reader-visible label.  ``_public_text`` may add Markdown
    # emphasis around a participant handle, while ``_chapter_label`` removes
    # that formatting before publication.  Looking at the pre-render form
    # made ``**@Alice** says ...`` evade the person-led-label guard and then
    # fail only after an expensive full production run.
    text = re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", str(value or ""))
    text = re.sub(r"[*_`]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = re.findall(r"(?iu)[a-zа-яё0-9]+", text)
    return bool(
        not text
        or len(text) > 160
        or len(words) < 2
        or "?" in text
        or unresolved_public_reference(text)
        or RAW_DIALOGUE_RE.search(text)
        or TITLE_LOW_INFORMATION_RE.search(text)
        or INCOMPLETE_PUBLIC_FRAGMENT_RE.search(text)
        or NAVIGATION_PERSON_LED_RE.search(text)
        or NAVIGATION_MODAL_RE.search(text)
        or NAVIGATION_TOPICLESS_STATE_RE.search(text)
        or re.match(r"(?iu)^\s*(?:спрашивается|возникает\s+вопрос)\b", text)
    )


TASK_REFERENCE_RE = re.compile(r"(?iu)\b(?:это|этот|эта|эти|его|е[её]|их|тебе|вам)\b")
TASK_RESOURCE_RE = re.compile(r"(?iu)\b(?:данн\w*|выгрузк\w*|выборк\w*|файл\w*|запис\w*|образц\w*)\b")


def _task_candidate_score(text):
    value = sanitize_public_surface(text)
    return (
        100 * bool(RAW_DIALOGUE_RE.search(value))
        + 80 * unresolved_public_reference(value)
        + 40 * bool(INTERNAL_LABEL_RE.search(value) or has_english_prose(value))
        + 20 * bool(len(value) > 220)
        + 5 * len(TASK_REFERENCE_RE.findall(value))
    )


def _resolve_task_reference(value, dialogue):
    """Resolve a local ``это`` only from an explicit adjacent antecedent.

    The rule is grammatical rather than domain-specific: a correction of a
    category (``не типы, а варианты``), its noun complement and an optional
    count are carried into the action.  If those pieces are not all present,
    the fragment remains unresolved and the publication gate abstains.
    """
    action = re.match(
        r"(?iu)^\s*(?P<verb>встро\w*|внес\w*|добав\w*|перенес\w*)\s+это\s+"
        r"(?P<target>(?:в|на|к)\s+.+?)\s*$",
        value,
    )
    if not action:
        action = re.match(
            r"(?iu)^\s*.+?\s+(?:нужно|надо|следует)(?:\s+будет)?\s+"
            r"(?P<verb>встро\w*|внес\w*|добав\w*|перенес\w*)\s+"
            r"(?P<target>(?:в|на|к)\s+.+?)\s*$",
            value,
        )
    if not action:
        return value
    correction = re.search(r"(?iu)\bне\s+([а-яё-]+)\w*\s*,?\s*а\s+([а-яё-]+)\w*", dialogue)
    complement = re.search(
        r"(?iu)\b(?:тип\w*|вид\w*|вариант\w*|категори\w*|форм\w*)\s+"
        r"([a-zа-яё][a-zа-яё0-9_-]*(?:\s+[a-zа-яё][a-zа-яё0-9_-]*){0,2})",
        dialogue,
    )
    if not correction or not complement:
        return value
    corrected_category = correction.group(2)
    referent = re.split(r"(?iu)\b(?:их|не|потом|затем|дальше)\b", complement.group(1))[0].strip(" ,.;:—-")
    if not referent:
        return value
    count = re.search(r"(?iu)\bих\s+(\d+|один|одна|одно|два|две|три|четыре|пять|шесть|семь|восемь|девять|десять)\b", dialogue)
    # Keep the resolved surface extractive.  Labels such as ``внешний вид``
    # or ``количество`` may be reasonable editorial paraphrases, but they are
    # not present in the cited turns and therefore make the post-render
    # entailment gate reject an otherwise valid canonical task.
    quantity = f"; их {count.group(1)}" if count else ""
    source_verb = action.group("verb").casefold()
    verb = next(
        rendered for stem, rendered in (
            ("встро", "Встроить"), ("внес", "Внести"),
            ("добав", "Добавить"), ("перенес", "Перенести"),
        ) if source_verb.startswith(stem)
    )
    target = action.group("target").strip(" ,.;:—-")
    return f"{verb} {target}: {corrected_category} {referent}{quantity}"


def _duration_surface(text):
    match = re.search(
        r"(?iu)\b(?:(\d+|один|одна|два|две|три|четыре|пять|пара|несколько)\s+)?"
        r"(месяц\w*|недел\w*|д(?:ень|ня|ней)|час\w*|минут\w*)\b",
        text,
    )
    if not match:
        return None
    amount, unit = match.groups()
    if amount:
        return f"{amount} {unit}"
    roots = (("месяц", "1 месяц"), ("недел", "1 неделя"), ("д", "1 день"),
             ("час", "1 час"), ("минут", "1 минута"))
    return next((surface for root, surface in roots if unit.casefold().startswith(root)), unit)


def _compose_resource_scope(value, dialogue):
    """Add source period and sample length to a resource-delivery task.

    The composition is based on generic temporal dimensions and a nearby
    named resource, not on a product, market or recording-specific lexicon.
    """
    combined = f"{value} {dialogue}"
    if not TASK_RESOURCE_RE.search(combined):
        return value
    year = re.search(r"(?<!\d)(19\d{2}|20\d{2}|21\d{2})(?!\d)", combined)
    duration = _duration_surface(combined)
    if not year or not duration:
        return value
    year_context = next((part for part in re.split(r"[.!?]", dialogue) if year.group(1) in part), "")
    candidates = [token for token in re.findall(r"(?u)\b[A-ZА-ЯЁ][A-Za-zА-Яа-яЁё0-9._-]{1,30}\b", year_context)
                  if token.casefold() not in {"я", "мы", "данные", "год", "года"}]
    resource = candidates[-1] if candidates else None
    base = value
    if re.match(r"(?iu)^готов\w*\s+предоставить", base):
        base = re.sub(r"(?iu)^готов\w*\s+предоставить", "Предоставить", base)
    if resource and resource.casefold() not in base.casefold():
        base = re.sub(r"(?iu)\bданн\w*\b", lambda match: f"{match.group(0)} {resource}", base, count=1)
    base = re.sub(
        r"(?iu)[,;]?\s*(?:(?:достаточно|объ[её]м(?:ом)?|длительност\w*)\s+)?"
        r"(?:за\s+)?(?:\d+|один|одна|два|две|три|четыре|пять|пара|несколько)?\s*"
        r"(?:месяц\w*|недел\w*|д(?:ень|ня|ней)|час\w*|минут\w*)\b",
        "", base,
    )
    base = re.sub(r"(?iu)\s+(?:за\s+)?(?:19\d{2}|20\d{2}|21\d{2})\s+год\w*\b", "", base).strip(" ,.;:—-")
    return f"{base}; период источника — {year.group(1)} год, объём выборки — {duration}"


def task_surface_text(claim, state):
    """Render a task as an independent action rather than a dialogue quote.

    The transformation is deliberately bounded by the structured task frame
    and its cited dialogue window.  It removes person/tense duplication (the
    assignee is rendered in a separate field), resolves only explicit local
    alternatives, and otherwise keeps the reviewed semantic statement.
    """
    candidates = [
        state.get("deliverable"), state.get("description"),
        claim.get("statement"), claim.get("source_statement"),
    ]
    candidates = [sanitize_public_surface(candidate) for candidate in candidates if candidate]
    value = min(candidates, key=lambda candidate: (_task_candidate_score(candidate), len(candidate))) if candidates else ""
    value = re.sub(r"^\*\*(?=@)", "", value)
    value = re.sub(r"(?<=\w)\*\*", "", value)
    assignee = str(state.get("assignee") or "").strip()
    if assignee:
        value = re.sub(rf"(?iu)^\s*{re.escape(assignee)}\s+", "", value)
    value = re.sub(r"(?iu)^\s*мне\s+", "", value)
    value = re.sub(r"(?iu)^\s*(?:потом|затем|дальше)\s+", "", value)
    value = re.sub(r"(?iu)^\s*пойти\s+(?=[а-яё-]+(?:ть|ться)\b)", "", value)

    dialogue = " ".join(
        sanitize_public_surface(turn.get("text"))
        for turn in claim.get("dialogue_evidence", [])
        if isinstance(turn, dict)
    )
    combined = f"{value} {dialogue}"

    replacements = (
        (r"(?iu)^готов\w*\s+предоставить\b", "Предоставить"),
        (r"(?iu)^(?:я\s+)?подготов(?:лю|ит)\b", "Подготовить"),
        (r"(?iu)^(?:я\s+)?(?:предоставлю|предоставит|дам)\b", "Предоставить"),
        (r"(?iu)^(?:я\s+)?(?:отправлю|отправит|скину|передам|передаст)\b", "Передать"),
        (r"(?iu)^(?:я\s+)?встро(?:ю|ит)\b", "Встроить"),
        (r"(?iu)^(?:я\s+)?буду\s+размечивать\b", "Размечать"),
        (r"(?iu)^(?:я\s+)?попробую\b", "Попробовать"),
        (r"(?iu)^(?:я\s+)?сдела(?:ю|ет)\b", "Сделать"),
    )
    for pattern, replacement in replacements:
        value = re.sub(pattern, replacement, value)
    # Remove a self-correction filler only when it occurs directly between an
    # action infinitive and its object.  Content elsewhere is left untouched.
    value = re.sub(
        r"(?iu)^(\s*[a-яё-]+(?:ть|ться))\s+(?:[аa]\s*,?\s*)?ну\s+то\s+есть\s+",
        r"\1 ", value,
    )
    value = re.sub(
        r"(?iu)\b(или|либо)\s+(?:я\s+)?(?:отправлю|отправит|скину|передам|передаст)\b",
        lambda match: match.group(1) + " передать",
        value,
    )
    value = _resolve_task_reference(value, dialogue)
    value = _compose_resource_scope(value, dialogue)
    recipient = (state.get("action_frame", {}).get("explicit_acceptance_actor")
                 or state.get("action_frame", {}).get("recipient"))
    if recipient and re.search(r"(?iu)\b(?:тебе|вам)\b", value):
        value = re.sub(r"(?iu)\b(?:тебе|вам)\b", "", value)
        value = re.sub(r"\s+", " ", value).strip(" ,.;:!?—-") + f" для {recipient}"
    if claim.get("parallel") and not re.search(r"(?iu)\bпараллельно\b", value):
        value = "Параллельно " + value[:1].lower() + value[1:]
    value = re.sub(r"(?iu)^\s*(?:хорошо|ладно)\s*,?\s*(?:тогда\s+)?", "", value)
    value = re.sub(r"\s+", " ", value).strip(" ,.;:!?—-")
    if value:
        value = value[0].upper() + value[1:]
    return value


def public_surface_text(claim, preferred=None):
    """Return Russian, source-grounded wording for a public surface."""
    value = sanitize_public_surface(preferred if preferred is not None else claim.get("statement"))
    time_contract = claim.get("time_contract", {}) if isinstance(claim.get("time_contract"), dict) else {}
    if time_contract.get("resolution_status") == "ambiguous_clock" and time_contract.get("raw"):
        value = re.sub(
            r"(?iu)\bпосле\s+0{1,2}:0{2}\b",
            sanitize_public_surface(time_contract["raw"]),
            value,
        )
    if not has_english_prose(value) and not INTERNAL_LABEL_RE.search(value):
        return value
    allowed_evidence = set(claim.get("evidence_ids", []))
    candidates = []
    for turn in claim.get("dialogue_evidence", []):
        if turn.get("id") not in allowed_evidence:
            continue
        candidate = sanitize_public_surface(turn.get("text"))
        if (re.search(r"(?iu)[а-яё]{3,}", candidate)
                and not INTERNAL_LABEL_RE.search(candidate)
                and not has_english_prose(candidate)):
            candidates.append(candidate)
    if candidates:
        return sanitize_public_surface(max(candidates, key=len))
    return ""


def substantive_unverified_surface(text):
    """Keep uncertain source material, but not bare acknowledgements or labels."""
    value = str(text or "").strip()
    if not value or INTERNAL_LABEL_RE.search(value) or has_english_prose(value):
        return False
    tokens = re.findall(r"(?iu)[a-zа-яё0-9]+", value)
    # A lone command/label such as ``show`` is not a human-facing claim even
    # though it is too short to trip the English-prose heuristic.
    if len(tokens) < 2 or not re.search(r"(?iu)[а-яё]{3,}", value):
        return False
    return not (
        len(tokens) <= 5
        and re.match(r"(?iu)^\s*(?:угу|ага|да|ладно|ок(?:ей)?)(?:\b|[,.!?])", value)
    )


def has_adjacent_stem_repetition(text):
    """Detect malformed neighbouring repetitions without a domain lexicon."""
    words = re.findall(r"(?iu)[a-zа-яё]{5,}", str(text or ""))
    stems = [re.sub(r"(?iu)(?:иями|ями|ами|ого|ему|ыми|ими|ая|яя|ое|ее|ие|ые|ий|ый|ой|ов|ев|ам|ям|ах|ях|а|я|о|е|ы|и|у|ю)$", "", word.casefold()) for word in words]
    return any(len(left) >= 5 and left == right for left, right in zip(stems, stems[1:]))


def normalize_question_surface(text):
    """Remove nested reporting boilerplate and keep only genuine questions."""
    value = sanitize_public_surface(text).strip()
    value = re.sub(
        r"(?iu)^\s*(?:спрашивается[, :] *|возникает\s+вопрос(?:\s+о\s+том)?[, :] *|"
        r"вопрос\s+о\s+том[, :] *)",
        "",
        value,
    )
    if not value or not QUESTION_SIGNAL_RE.search(value):
        return ""
    if not value.rstrip().endswith(("?", ".", "!")):
        value += "?"
    elif value.rstrip().endswith(".") and re.search(r"(?iu)\b(?:ли|кто|что|где|когда|почему|зачем|как|како[йея]|сколько)\b", value):
        value = value.rstrip()[:-1] + "?"
    return value


def role_relations(text):
    # Repeating the same actor in a rendered metadata suffix (for example,
    # ``@A подготовит … — исполнитель: @A``) is not a role relation.  Only a
    # pair of distinct people can demonstrate an actor/recipient swap.
    return {(left, right) for left, right in ROLE_RELATION_RE.findall(str(text or "")) if left != right}


@dataclass(frozen=True)
class PublicItem:
    public_id: str
    section: str
    text: str
    claim_ids: list[str]
    evidence_ids: list[str]
    source_word_ids: list[str]
    content_kind: str
    social_state: str
    lifecycle: str = "active"
    polarity: str = "positive"
    modality: str = "unknown"
    temporal_state: str = "unknown"
    commitment_state: str = "unknown"
    decision_status: str | None = None
    task_status: str | None = None
    quantities: list[dict] = field(default_factory=list)
    conditions: list[dict] = field(default_factory=list)
    origin_ids: list[str] = field(default_factory=list)
    relation_ids: list[str] = field(default_factory=list)

    def as_dict(self):
        return asdict(self)


def can_publish_as_decision(item):
    if not (item.get("lifecycle", "active") == "active"
            and item.get("decision_status") == "accepted"
            and item.get("decision_evidence_ids", item.get("evidence_ids"))):
        return False
    explicit_decision = item.get("content_kind") == "decision" or item.get("speech_act") == "decide"
    if explicit_decision:
        return item.get("acceptance_check") in {None, "not_applicable", "entailed"}
    acceptance_ids = set(item.get("acceptance_evidence_ids", []))
    # A proposal cannot prove its own acceptance.  There must be at least one
    # separately cited source turn in addition to the accepting reply.
    proposal_ids = set(item.get("evidence_ids", [])) - acceptance_ids
    if not proposal_ids:
        return False
    acceptance_turns = [
        turn.get("text")
        for turn in item.get("dialogue_evidence", [])
        if turn.get("id") in acceptance_ids
    ]
    if acceptance_turns and not any(
            is_explicit_acceptance_reply(text, item.get("statement"))
            for text in acceptance_turns
    ):
        return False
    return bool(
        item.get("acceptance_check") == "entailed"
        and item.get("acceptance_relation_ids")
        and item.get("acceptance_evidence_ids")
        and item.get("accepted_by")
    )


def decision_surface_text(claim):
    """Make the accepted status explicit without rewriting the proposition."""
    text = public_surface_text(claim)
    if not text:
        return ""
    if claim.get("content_kind") == "proposal" and PROPOSAL_SURFACE_RE.search(text):
        return f"Согласовано предложение: {text[:1].lower() + text[1:]}"
    return text


def build_public_items(meeting_graph, summary_plan):
    """Create the complete public contract before formatting Markdown."""
    claims = {x["claim_id"]: x for x in meeting_graph.get("claims", [])}
    claims_by_source = {x.get("source_record_id"): x for x in claims.values()}
    dialogue_by_id = {}
    for candidate in claims.values():
        for turn in candidate.get("dialogue_evidence", []):
            if isinstance(turn, dict) and turn.get("id"):
                dialogue_by_id.setdefault(turn["id"], turn)
    view_plans = summary_plan.get("view_plans", {})
    task_states = {x["task_id"]: x for x in meeting_graph.get("task_states", [])}
    task_claims = {}
    for candidate in claims.values():
        task_id = candidate.get("canonical_task_state_id")
        if task_id and candidate.get("lifecycle", "active") == "active" and candidate.get("verification_status") not in {"verification_unavailable", "insufficient_evidence", "contradicted"}:
            task_claims.setdefault(task_id, []).append(candidate["claim_id"])
    question_states = {x["proposition_id"]: x for x in meeting_graph.get("question_states", [])}
    planned_relations = {}
    for sentence in summary_plan.get("public_sentence_plans", []):
        for claim_id in sentence.get("claim_ids", []):
            planned_relations.setdefault(claim_id, []).extend(sentence.get("relation_ids", []))
    sections = []
    materialized = {}

    def task_claim_for_render(claim, state):
        rendered = dict(claim)
        support_ids = list(dict.fromkeys(
            state.get("evidence_ids", []) + claim.get("evidence_ids", [])
        ))
        rendered["dialogue_evidence"] = [
            dialogue_by_id[evidence_id]
            for evidence_id in support_ids
            if evidence_id in dialogue_by_id
        ] or list(claim.get("dialogue_evidence", []))
        return rendered

    def add(section, claim, social_state=None, text=None, claim_ids=None, relation_ids=None, extra_evidence=None):
        if claim.get("lifecycle", "active") != "active" or not claim.get("evidence_ids"):
            return
        task_state = task_states.get(claim.get("canonical_task_state_id"), {})
        canonical_task_projection = bool(
            task_state and claim.get("content_kind") in {"action", "follow_up"}
            and section in {"overview", "minutes"}
        )
        use_task_support = section == "tasks" or canonical_task_projection
        evidence_ids = task_state.get("evidence_ids", claim.get("evidence_ids", [])) if use_task_support else claim.get("evidence_ids", [])
        source_word_ids = task_state.get("source_word_ids", claim.get("source_word_ids", [])) if use_task_support else claim.get("source_word_ids", [])
        # Context turns are not automatically supporting evidence. Only cited
        # task/answer support is added to the public provenance contract.
        evidence_ids = list(dict.fromkeys(evidence_ids))
        source_word_ids = list(dict.fromkeys(source_word_ids))
        cited_ids = list(dict.fromkeys(claim_ids or [claim["claim_id"]]))
        if section == "tasks" and task_state:
            cited_ids = list(dict.fromkeys(cited_ids + task_claims.get(task_state["task_id"], [])))
        retained_relations = list(relation_ids or [])
        retained_relations.extend(value for claim_id in cited_ids for value in planned_relations.get(claim_id, []))
        if section == "tasks":
            retained_relations.extend(task_state.get("scope_relation_ids", []))
            retained_relations.extend(task_state.get("acceptance_relation_ids", []))
        if canonical_task_projection:
            text = task_surface_text(task_claim_for_render(claim, task_state), task_state)
        clean_text = public_surface_text(claim, text if text is not None else claim.get("statement"))
        if not clean_text:
            return
        topic_entities = list(dict.fromkeys(
            list(claim.get("topic_entities", [])) +
            [x.get("canonical_name") or x.get("name") for x in claim.get("entities", []) if isinstance(x, dict) and (x.get("canonical_name") or x.get("name"))]
        ))
        public = PublicItem(
            public_id=f"PI{len(sections)+1:05d}", section=section,
            text=clean_text, claim_ids=cited_ids,
            evidence_ids=list(dict.fromkeys(evidence_ids + list(extra_evidence or []))), source_word_ids=list(source_word_ids), content_kind=claim.get("content_kind") or claim.get("kind"),
            social_state=social_state or claim.get("social_state", "candidate"), lifecycle=claim.get("lifecycle", "active"),
            polarity=claim.get("polarity", "positive"), modality=claim.get("modality", "unknown"),
            temporal_state=claim.get("temporal_state", "unknown"), commitment_state=claim.get("commitment_state", "unknown"),
            decision_status=claim.get("decision_status"), task_status=claim.get("task_status"),
            quantities=list(claim.get("quantities", [])), conditions=list(claim.get("conditions", [])),
            origin_ids=list(claim.get("origin_ids", [])), relation_ids=list(dict.fromkeys(retained_relations)),
        ).as_dict() | {
            "start": claim.get("primary_evidence_start", claim.get("start", 0)),
            "end": claim.get("end", claim.get("start", 0)),
            "episode_id": claim.get("episode_id"),
            "aspect_id": claim.get("aspect_id") or (
                ("ownership:" + str(task_state.get("task_id")))
                if section == "tasks" and task_state
                else ("decision:" + claim["claim_id"] if section == "decisions" else None)
            ),
            "task_state_id": claim.get("canonical_task_state_id"),
            "task_state": task_state,
            "action_frame": task_state.get("action_frame", {}),
            "question_state": question_states.get(claim.get("proposition_id"), {}),
            "topic_entities": topic_entities,
            "context_ids": claim.get("context_ids", []),
            "verification_status": claim.get("verification_status") or "verification_unavailable",
            # Preserve the acceptance contract through the public boundary so
            # the runtime gate can independently detect a proposal that was
            # mislabeled as a decision.
            "speech_act": claim.get("speech_act"),
            "acceptance_check": claim.get("acceptance_check"),
            "acceptance_relation_ids": list(claim.get("acceptance_relation_ids", [])),
            "acceptance_evidence_ids": list(claim.get("acceptance_evidence_ids", [])),
            "accepted_by": list(claim.get("accepted_by", [])),
        }
        # One public sentence is enough when several typed claims project to
        # the same meaning in the same reader view.  Merge provenance rather
        # than spending the section budget on repeated bullets.  Task and
        # question identities remain protected so genuinely separate work or
        # asks are never collapsed just because their wording is similar.
        duplicate = next((
            existing for existing in sections
            if existing.get("section") == section
            and existing.get("social_state") == public.get("social_state")
            and existing.get("verification_status") == public.get("verification_status")
            and (section != "tasks" or existing.get("task_state_id") == public.get("task_state_id"))
            and (section != "questions" or existing.get("question_state", {}).get("proposition_id") == public.get("question_state", {}).get("proposition_id"))
            and (
                (
                    normalize_mixed_script_confusables(existing.get("text", "")).casefold().strip()
                    == normalize_mixed_script_confusables(public.get("text", "")).casefold().strip()
                    and existing.get("episode_id") == public.get("episode_id")
                )
                or (
                    # A proposal that was accepted can legitimately project
                    # both as the proposed decision and as the resulting
                    # action.  Public readers must see that semantic event
                    # once, even though the typed source claims differ.  The
                    # episode guard prevents similar recurring work from
                    # being collapsed across different parts of a meeting.
                    existing.get("episode_id") == public.get("episode_id")
                    and equivalent(existing, public, .8)
                )
            )
        ), None)
        if duplicate is not None:
            for key in ("claim_ids", "evidence_ids", "source_word_ids", "origin_ids", "relation_ids", "topic_entities", "context_ids"):
                duplicate[key] = list(dict.fromkeys(duplicate.get(key, []) + public.get(key, [])))
            duplicate["start"] = min(float(duplicate.get("start", 0)), float(public.get("start", 0)))
            duplicate["end"] = max(float(duplicate.get("end", duplicate["start"])), float(public.get("end", public["start"])))
            for claim_id in cited_ids:
                materialized.setdefault(section, {})[claim_id] = {
                    "status": "published", "public_id": duplicate["public_id"],
                    "reason": "merged_equivalent_public_surface_with_provenance",
                }
            return duplicate
        limit = PUBLIC_SECTION_LIMITS.get(section)
        if limit is not None and sum(item["section"] == section for item in sections) >= limit:
            return
        sections.append(public)
        for claim_id in cited_ids:
            materialized.setdefault(section, {})[claim_id] = {"status": "published", "public_id": public["public_id"]}
        return public
    def selected(view):
        return [claims[x] for x in view_plans.get(view, {}).get("selected_claim_ids", []) if x in claims]

    quarantine = sorted(
        selected("requires_verification"),
        key=lambda claim: (
            {"contradicted": 0, "insufficient_evidence": 1, "verification_unavailable": 2}.get(claim.get("verification_status"), 3),
            0 if claim.get("content_kind") in {"action", "follow_up", "decision", "proposal", "question", "schedule"} else 1,
            float(claim.get("start", 0)),
        ),
    )
    for claim in quarantine:
        if claim.get("verification_status") in {"verification_unavailable", "insufficient_evidence", "contradicted"}:
            # Quarantine is still a public surface.  An internal machine label
            # is omitted with a planner disposition rather than expanded into
            # a long context utterance that may contain several propositions.
            text = str(claim.get("statement") or "")
            if (substantive_unverified_surface(text)
                    and not re.search(r"(?iu)\b(?:шутк|dow\s*jones|s&p|столет|тысячелет)\w*", text)):
                task_state = task_states.get(claim.get("canonical_task_state_id"), {})
                frame = task_state.get("action_frame", {})
                if frame.get("state") == "reported_plan" and frame.get("reporter"):
                    text = re.sub(r"(@[\w.-]+)\s*/\s*(@[\w.-]+)", r"один из \1 или \2", text)
                    if frame["reporter"] not in text:
                        text = f"По словам {frame['reporter']}, {text[:1].lower() + text[1:]}"
                add("requires_verification", claim, "needs_verification", text=text)

    def attributed_text(claim):
        text = public_surface_text(claim)
        if re.search(r"(?iu)(?:\b100\s*%|\b100\s+процент|\b(?:всегда|никогда|невозможно|нельзя|гарантированно)\b)", text) and len(claim.get("speaker_refs", [])) == 1:
            text = f"По словам {claim['speaker_refs'][0]}, {text[:1].lower() + text[1:]}"
        if claim.get("risk", {}).get("recognition", 0) >= .65:
            text += " ⚠ Формулировка или термин требуют проверки."
        return text
    # Overview is a compact mix of state, obstacle and next step. Repeating a
    # short task outcome across sections is useful; verbatim duplicates are not.
    overview = []
    executive_claims = selected("executive")
    category_order = (
        {"problem", "blocker", "constraint"},
        {"action", "follow_up", "decision"},
        set(TECHNICAL_KINDS),
        {"current_state", "observation", "experimental_result"},
    )
    overview_order, overview_seen = [], set()
    for kinds in category_order:
        candidate = next((x for x in executive_claims if x.get("content_kind") in kinds and x["claim_id"] not in overview_seen), None)
        if candidate:
            overview_order.append(candidate); overview_seen.add(candidate["claim_id"])
    overview_order.extend(x for x in executive_claims if x["claim_id"] not in overview_seen)
    for claim in overview_order:
        if claim.get("verification_status") == "verification_unavailable":
            continue
        if claim.get("content_kind") in {"question", "schedule"} or claim.get("risk", {}).get("recognition", 0) >= .65:
            continue
        if claim.get("risk", {}).get("number", 0) >= .45:
            continue
        if re.search(r"(?iu)^\s*(?:это|так|вот\s+эт\w+|они|он|она)\b", claim.get("statement", "")) and not claim.get("entities"):
            continue
        tokens = set(re.findall(r"(?iu)[a-zа-яё0-9]+", claim.get("statement", "").casefold()))
        if any(len(tokens & old) / max(1, min(len(tokens), len(old))) >= .55 for old in (x[1] for x in overview)):
            continue
        overview.append((claim, tokens))
        if len(overview) == 4: break
    for claim, _ in overview:
        add("overview", claim, text=attributed_text(claim))
    for claim in selected("executive"):
        if claim.get("verification_status") == "verification_unavailable":
            continue
        # Canonical work is rendered once as a task.  Repeating the same
        # accepted assignment under "decisions" spends attention without
        # adding a distinct meeting outcome.
        if can_publish_as_decision(claim) and not claim.get("canonical_task_state_id"):
            add("decisions", claim, "accepted", decision_surface_text(claim))
    # Rules have their own protected reader view.  They no longer compete
    # with ordinary technical details for the same editorial budget.
    for claim in selected("rules"):
        if claim.get("verification_status") == "verification_unavailable":
            continue
        if claim.get("content_kind") in RULE_KINDS:
            add("rules", claim, "accepted" if can_publish_as_decision(claim) else "described", attributed_text(claim))
    for claim in selected("technical"):
        if claim.get("verification_status") == "verification_unavailable":
            continue
        if claim.get("content_kind") in TECHNICAL_KINDS and claim.get("content_kind") not in RULE_KINDS:
            if not re.search(r"(?iu)\b(?:ширина\s*[—–-]\s*ширина|называется|определяется)\b", str(claim.get("statement") or "")):
                candidate_text = attributed_text(claim)
                # Deictic fragments without their referent are grounded but
                # not standalone technical conclusions for a reader.
                if not re.search(r"(?iu)^\s*(?:он|она|они|это|этот|эта|эти)\b", candidate_text):
                    add("technical", claim, "observation", candidate_text)
    emitted_tasks = set()
    confirmed = {"self_committed", "explicit_self_commitment", "assigned", "accepted", "in_progress", "blocked", "completed"}
    for claim in selected("tasks"):
        if claim.get("verification_status") == "verification_unavailable":
            continue
        state = task_states.get(claim.get("canonical_task_state_id"), {})
        if not state or state.get("task_id") in emitted_tasks: continue
        emitted_tasks.add(state["task_id"])
        status = state.get("status", "idea")
        task_claim = task_claim_for_render(claim, state)
        description = task_surface_text(task_claim, state)
        # A source can contain both proposal and agreement wording while the
        # canonical task still awaits acceptance. Preserve the proposal only.
        if status == "assigned_pending":
            description = re.sub(r"(?iu)^предлагалось\s+участники\s+договорились\s+", "Предлагалось ", description)
        if re.search(r"(?iu)^\s*(?:вопрос|уточнение|метаописание)\b", description):
            continue
        # A task card must have a clean structured deliverable. Falling back
        # from an internal label to a long dialogue turn converts observations
        # and speculation into apparent work items.
        description = sanitize_public_surface(description)
        if (not description or INTERNAL_LABEL_RE.search(description) or has_english_prose(description)
                or has_adjacent_stem_repetition(description)
                or RAW_DIALOGUE_RE.search(description)
                or not WORK_ACTION_SURFACE_RE.search(description)):
            continue
        details = [description]
        if state.get("assignee"):
            details.append(f"исполнитель: {state['assignee']}")
        normalized_description = re.sub(r"\s+", " ", str(description or "")).strip().casefold()
        if state.get("current_scope") and re.sub(r"\s+", " ", str(state["current_scope"])).strip().casefold() not in normalized_description:
            qualifier = " (предложен, не подтверждён)" if state.get("scope_confidence") == "proposed" else ""
            details.append(f"объём: {state['current_scope']}{qualifier}")
        if state.get("data_origin") and re.sub(r"\s+", " ", str(state["data_origin"])).strip().casefold() not in normalized_description:
            details.append(f"период данных: {state['data_origin']}")
        if state.get("deadline"):
            due = state["deadline"].get("text") if isinstance(state["deadline"], dict) else state["deadline"]
            if due: details.append(f"срок: {due}")
        conditions = [x.get("antecedent") or x.get("text") for x in state.get("conditions", []) if isinstance(x, dict) and (x.get("antecedent") or x.get("text"))]
        if conditions:
            details.append("условие: " + "; ".join(conditions))
        labels = {"self_committed": "участник взял на себя", "intent_to_attempt": "участник намерен попробовать", "in_progress": "в работе", "past_attempt": "ранее выполнялось", "proposed": "предложено, не подтверждено", "idea": "идея, не подтверждена", "assigned_pending": "назначение ожидает подтверждения", "assigned": "назначено", "accepted": "согласовано", "completed": "выполнено", "blocked": "заблокировано", "needs_verification": "требует проверки источника"}
        if status in labels:
            details.append(f"статус: {labels[status]}")
        task_text = " — ".join(details)
        if status in confirmed or status == "needs_verification":
            add("tasks", claim, status, task_text)
        elif status in {"proposed", "idea", "assigned_pending", "intent_to_attempt", "in_progress"}:
            # The planner already bounds this view. A second hidden cap loses
            # selected, evidence-backed work candidates without disposition.
            add("tasks", claim, status, task_text)
    question_candidates = sorted(
        selected("questions"),
        key=lambda claim: (
            0 if claim.get("content_kind") == "schedule" else 1,
            0 if "?" in str(claim.get("statement") or "") else 1,
            float(claim.get("start", 0)),
        ),
    )
    for claim in question_candidates:
        if claim.get("verification_status") == "verification_unavailable":
            continue
        if claim.get("content_kind") in {"question", "schedule"} and claim.get("question_status") not in {"answered", "rhetorical", "superseded", "answer_not_verified", "answer_retrieval_failed"}:
            state = question_states.get(claim.get("proposition_id"), {})
            question_text = state.get("residual_question_text") or state.get("remaining_question") or attributed_text(claim)
            if GENERIC_QUESTION_RESIDUAL_RE.fullmatch(str(question_text or "")):
                # Slot labels are diagnostics, not self-contained questions.
                # Prefer the source-grounded original so two unrelated asks do
                # not collapse into the same generic public bullet.
                question_text = public_surface_text(
                    claim, state.get("original_question") or claim.get("statement")
                )
            if not question_text or GENERIC_QUESTION_RESIDUAL_RE.fullmatch(str(question_text)):
                continue
            if re.search(r"(?iu)^\s*уточнить\s+недостающ\w+\s+результат", str(question_text or "")):
                continue
            question_text = normalize_question_surface(question_text)
            if not question_text:
                continue
            speakers = list(claim.get("speaker_refs", []))
            if len(speakers) == 1 and speakers[0] not in question_text:
                question_text = re.sub(r"(?iu)^\s*(?:участник\s+)?(?:спрашивает|зада[её]т\s+вопрос)(?:\s+о\s+том)?[, :] *", "", question_text)
                question_text = f"{speakers[0]} спрашивает: {question_text[:1].lower() + question_text[1:]}"
            answer_claim_ids = [claims_by_source[value]["claim_id"] for value in state.get("answer_record_ids", []) if value in claims_by_source]
            add("questions", claim, claim.get("question_status", "unanswered"), question_text,
                claim_ids=[claim["claim_id"]] + answer_claim_ids,
                relation_ids=state.get("answer_relation_ids", []), extra_evidence=state.get("answer_evidence_ids", []))
    for claim in selected("experiments"):
        if claim.get("verification_status") == "verification_unavailable":
            continue
        experiment_text = public_surface_text(claim)
        goal_only = bool(
            re.search(r"(?iu)\b(?:обсуждалась\s+цель|целевой\s+ориентир|базов\w*\s+решени\w*)\b", experiment_text)
            and not re.search(r"(?iu)\b(?:проверить|протестировать|обучить|эксперимент|гипотез|предсказывать|детектировать)\b", experiment_text)
        )
        if goal_only:
            continue
        has_testable_method = bool(re.search(
            r"(?iu)\b(?:провер\w*|протест\w*|обуч\w*|предсказыва\w*|детект\w*|"
            r"размеч\w*|эксперимент\w*|сравн\w*|бэктест\w*|апроб\w*)\b",
            experiment_text,
        ))
        if claim.get("content_kind") == "hypothesis" and (
            not has_testable_method or has_adjacent_stem_repetition(experiment_text)
            or re.search(r"(?iu)\bобсуждалась\s+возможность\s+инструмент\b", experiment_text)
        ):
            continue
        if claim.get("content_kind") == "hypothesis" and re.search(r"(?iu)\b(?:нельзя|невозможно|ограничен)\b", claim.get("statement", "")):
            add("technical", claim, "constraint", attributed_text(claim))
        else:
            candidate_text = attributed_text(claim)
            candidate = {"text": candidate_text, "content_kind": claim.get("content_kind"), "claim_ids": [claim["claim_id"]]}
            if not any(item["section"] == "technical" and equivalent(candidate, item, .78) for item in sections):
                add("experiments", claim, text=candidate_text)
    seen_minutes = set()
    for claim in sorted(selected("minutes"), key=lambda x: (float(x.get("start", 0)), x.get("claim_id", ""))):
        if claim.get("verification_status") == "verification_unavailable":
            continue
        if claim.get("content_kind") in {"question", "schedule"} and claim.get("question_status") in {"answered", "rhetorical", "superseded"}:
            continue
        minute_text = attributed_text(claim)
        if unresolved_public_reference(minute_text):
            continue
        key = (claim.get("proposition_id"), claim.get("social_state"), tuple(claim.get("evidence_ids", [])))
        if key not in seen_minutes:
            add("minutes", claim, text=minute_text); seen_minutes.add(key)
    # A separate source claim may describe the same tentative deliverable as
    # an already published canonical task envelope. Keep the richer wording
    # and union exact provenance instead of printing a near-duplicate line.
    task_items = [item for item in sections if item["section"] == "tasks"]
    redundant = set()
    task_tokens = lambda value: {v for v in re.findall(r"(?iu)[a-zа-яё0-9]+", str(value or "").casefold()) if len(v) > 3}
    for item in task_items:
        if item["public_id"] in redundant: continue
        current = task_tokens(item["text"])
        matches = [prior for prior in task_items if prior is not item and prior["public_id"] not in redundant
                   and prior.get("social_state") == item.get("social_state")
                   and prior.get("task_state", {}).get("assignee") == item.get("task_state", {}).get("assignee")
                   and abs(float(prior.get("start", 0)) - float(item.get("start", 0))) <= 120
                   and len(prior["text"]) > len(item["text"])
                   and current and len(task_tokens(prior["text"]) & current) / max(1, len(current)) >= .8]
        if not matches: continue
        keeper = max(matches, key=lambda prior: len(prior["text"]))
        for key in ("claim_ids", "evidence_ids", "source_word_ids", "relation_ids"):
            keeper[key] = list(dict.fromkeys(keeper.get(key, []) + item.get(key, [])))
        redundant.add(item["public_id"])
        for cid in item["claim_ids"]:
            materialized.setdefault("tasks", {})[cid] = {"status": "published", "public_id": keeper["public_id"],
                                                        "reason": "merged_equivalent_task_with_provenance"}
    sections = [item for item in sections if item["public_id"] not in redundant]
    compatible_sections = {
        "executive": {"overview", "decisions"},
        "rules": {"rules"},
        "technical": {"technical", "rules"},
        "tasks": {"tasks"},
        "experiments": {"experiments", "technical"},
        "questions": {"questions"},
        "minutes": {"minutes"},
        "requires_verification": {"requires_verification"},
    }
    for view, view_plan in view_plans.items():
        for claim_id in view_plan.get("selected_claim_ids", []):
            published = next(
                (
                    materialized[section][claim_id]
                    for section in compatible_sections.get(view, {view})
                    if claim_id in materialized.get(section, {})
                ),
                None,
            )
            view_plan.setdefault("dispositions", {})[claim_id] = published or {"status": "excluded", "reason": "editorial_route_or_dedup"}
    return sections


def validate_public_items_contract(items):
    """Validate the exact public objects that will cross the publication boundary."""
    from contracts.meeting import PublicItemContract
    validated = []
    for item in items:
        validated.append(PublicItemContract.model_validate(item).model_dump(mode="json"))
    return validated


def canonicalize_public_item_order(items):
    """Order chronology by its final, evidence-anchored timestamps.

    ``build_public_items`` initially orders minute claims by the semantic
    record timestamp.  The production worker subsequently snaps every item to
    the closest cited source utterance.  A correction can therefore move past
    an adjacent claim after the initial sort.  Preserve the editorial order of
    every other view, but re-sort the chronology slots against the timestamps
    that will actually be published and audited.
    """
    ordered_minutes = iter(sorted(
        (item for item in items if item.get("section") == "minutes"),
        key=lambda item: (
            float(item.get("start", 0)),
            float(item.get("end", item.get("start", 0))),
            str(item.get("public_id") or ""),
        ),
    ))
    return [
        next(ordered_minutes) if item.get("section") == "minutes" else item
        for item in items
    ]


def relation_markers(text):
    """Return relation wording that was already present in a source claim."""
    return {match.casefold() for match in CAUSAL_RE.findall(text or "")}


def source_aware_plan(plan, claims):
    """Allow participant references that occur verbatim in cited source claims."""
    result = dict(plan)
    source_text = " ".join(str(claim.get("statement") or "") for claim in claims)
    result["allowed_speakers"] = sorted(
        set(plan.get("allowed_speakers", [])) | set(re.findall(r"@[\w.-]+", source_text))
    )
    source_polarity = {"negative" if NEGATION_RE.search(str(claim.get("statement") or "")) else "positive"
                       for claim in claims}
    result["polarity"] = sorted(source_polarity)
    # This pre-write check realizes the verbatim source claims, not the later
    # task envelope. Scope added by a cited revision belongs to task metadata
    # and is checked against the rendered PublicItem in the next gate.
    result["time_scope"] = [scope for scope in plan.get("time_scope", [])
                            if isinstance(scope, str) and scope.casefold() in source_text.casefold()]
    return result


def verify_sentence_plan(plan, claims, relations):
    by_id = {x.get("claim_id"): x for x in claims}
    errors = []
    if any(x not in by_id for x in plan.get("claim_ids", [])):
        errors.append("unknown_claim")
    if not cross_episode_allowed(plan.get("claim_ids", []), plan.get("relation_ids", []), claims, relations):
        errors.append("cross_episode_without_relation")
    relation_ids = {x.get("relation_id") for x in relations}
    if any(x not in relation_ids for x in plan.get("relation_ids", [])):
        errors.append("unknown_relation")
    return {"passed": not errors, "errors": errors}


def audit_realization(text, plan):
    allowed_numbers = {x.replace(" ", "") for x in plan.get("allowed_numbers", [])}
    found_numbers = {x.replace(" ", "") for x in NUMBER_RE.findall(text or "")}
    errors = []
    if not found_numbers.issubset(allowed_numbers):
        errors.append("unplanned_number")
    found_relations = relation_markers(text)
    allowed_relations = {str(x).casefold() for x in plan.get("allowed_relation_markers", [])}
    if found_relations and not plan.get("relation_ids") and not found_relations.issubset(allowed_relations):
        errors.append("unsupported_relation_language")
    polarities = set(plan.get("polarity", []))
    semantic_text = re.sub(r"(?iu)\b(?:не\s+уточнено|не\s+подтвержд[её]н[оаы]?|ожидает\s+подтверждения)\b", "", text or "")
    if "negative" in polarities and not NEGATION_RE.search(text or ""):
        errors.append("negation_not_preserved")
    if polarities == {"positive"} and NEGATION_RE.search(semantic_text):
        errors.append("unsupported_negation")
    modalities = set(plan.get("modality", []))
    if modalities & {"possible", "hypothetical", "unknown", "tentative"} and CERTAIN_RE.search(text or ""):
        errors.append("modality_upgraded")
    if modalities & {"possible", "hypothetical", "unknown", "tentative"} and COMPLETED_RE.search(text or ""):
        errors.append("completion_status_upgraded")
    if plan.get("conditions") and not CONDITION_RE.search(text or ""):
        errors.append("condition_not_preserved")
    number_words = {"один": "1", "одного": "1", "одну": "1", "два": "2", "две": "2", "три": "3", "четыре": "4"}
    normalize_scope = lambda value: re.sub(r"(?iu)\b(?:один|одного|одну|два|две|три|четыре)\b", lambda m: number_words[m.group(0).casefold()], str(value).casefold()).replace("ё", "е")
    allowed_scopes = [normalize_scope(x) for x in plan.get("time_scope", []) if isinstance(x, str) and x.strip()]
    scope_variants = {
        variant
        for scope in allowed_scopes
        for variant in (scope, re.sub(r"^1\s+", "", scope))
        if variant
    }
    if allowed_scopes and not any(scope in normalize_scope(text or "") for scope in scope_variants):
        errors.append("time_scope_not_preserved")
    allowed_values = list(plan.get("allowed_speakers", [])) + list(plan.get("allowed_assignees", []))
    allowed_speakers = set(allowed_values) | set(re.findall(r"@[\w.-]+", " ".join(allowed_values)))
    mentioned = set(re.findall(r"@[\w.-]+", text or ""))
    if not mentioned.issubset(allowed_speakers):
        errors.append("speaker_or_assignee_not_preserved")
    return {"passed": not errors, "errors": sorted(set(errors)), "atomic_claims": list(plan.get("claim_ids", [])), "relations": list(plan.get("relation_ids", [])), "status": "SUPPORTED" if not errors else "ABSTAIN"}


def verify_generated_items(items, sentence_plans, claims):
    """Audit actual generated text against the union contract of cited claims."""
    by_claim = {x.get("claim_id"): x for x in claims}
    plans_by_claim = {}
    for plan in sentence_plans:
        for claim_id in plan.get("claim_ids", []):
            plans_by_claim.setdefault(claim_id, []).append(plan)
    audits = []
    for item in items:
        text = str(item.get("text") or item.get("statement") or "")
        requested_claim_ids = list(item.get("claim_ids") or item.get("fact_ids", []))
        unknown_claim_ids = [x for x in requested_claim_ids if x not in by_claim]
        claim_ids = [x for x in requested_claim_ids if x in by_claim]
        # A claim can occur in an individual sentence plan and later in a
        # grouped chronology plan.  The last-write-wins map used previously
        # leaked another claim's quantity/scope into the individual item.
        plans = [
            min(plans_by_claim[x], key=lambda value: len(value.get("claim_ids", [])))
            for x in claim_ids if x in plans_by_claim
        ]
        if not text or not claim_ids or not plans:
            errors = (["empty_public_text"] if not text else []) + (["orphan_public_item"] if not claim_ids else []) + (["claim_outside_plan"] if claim_ids and not plans else []) + (["unknown_claim"] if unknown_claim_ids else [])
            audits.append({"text": text, "claim_ids": claim_ids, "passed": False, "errors": errors, "atomic_claims": claim_ids, "relations": [], "status": "ABSTAIN", "qa": {"passed": False, "checks": {}}})
            continue
        cited = [by_claim[x] for x in claim_ids]
        single_claim_contracts = bool(plans) and all(len(p.get("claim_ids", [])) == 1 for p in plans)
        source_mentions = {mention for source in cited for mention in re.findall(r"@[\w.-]+", str(source.get("statement") or ""))}
        claim_scopes = []
        claim_conditions = []
        for source in cited:
            scope = source.get("time_scope")
            claim_scopes.extend(scope if isinstance(scope, list) else [scope] if scope else [])
            claim_conditions.extend(source.get("conditions", []) or [])
        plan_scopes = [v for p in plans for v in p.get("time_scope", [])]
        # A multi-claim plan stores a flattened union.  Only claim-bound scope
        # may constrain an individual realization; a single-claim contract is
        # still accepted for backwards-compatible callers/tests.
        bound_scopes = claim_scopes or (plan_scopes if single_claim_contracts else [])
        claim_polarity = [source.get("polarity") for source in cited if source.get("polarity")]
        claim_modality = [source.get("modality") for source in cited if source.get("modality")]
        claim_assignees = [person for source in cited for person in source.get("assignees", [])]
        merged = {
            "claim_ids": claim_ids,
            "relation_ids": list(item.get("relation_ids", [])),
            "allowed_numbers": [n for p in plans for n in p.get("allowed_numbers", [])] if single_claim_contracts else [],
            "allowed_relation_markers": sorted({r for p in plans for r in p.get("allowed_relation_markers", [])}) if single_claim_contracts else sorted(set().union(*(relation_markers(source.get("statement")) for source in cited))),
            "allowed_speakers": sorted(source_mentions | {s for source in cited for s in source.get("speaker_refs", [])} | ({s for p in plans for s in p.get("allowed_speakers", [])} if single_claim_contracts else set())),
            "allowed_assignees": sorted(set(claim_assignees) | ({s for p in plans for s in p.get("allowed_assignees", [])} if single_claim_contracts else set())),
            "polarity": claim_polarity or ([v for p in plans for v in p.get("polarity", [])] if single_claim_contracts else []),
            "modality": claim_modality or ([v for p in plans for v in p.get("modality", [])] if single_claim_contracts else []),
            "conditions": claim_conditions or ([v for p in plans for v in p.get("conditions", [])] if single_claim_contracts else []),
            "time_scope": list(dict.fromkeys(bound_scopes)),
        }
        # Canonical tasks are intentionally projected into the overview and
        # chronology as well as the task view.  Those projections use the
        # same structured assignee/recipient and evidence closure, so they
        # must be audited against the same task contract.  Restricting this
        # to the task section caused valid recipient resolutions to be
        # rejected only in the chronology.
        task_state = item.get("task_state", {}) if item.get("task_state_id") else {}
        question_state = item.get("question_state", {}) if item.get("section") == "questions" else {}
        if task_state and item.get("task_state_id") in {by_claim[x].get("canonical_task_state_id") for x in claim_ids}:
            metadata_text = " ".join(str(task_state.get(field) or "") for field in ("description", "current_scope", "data_origin", "deadline", "assignee"))
            merged["allowed_numbers"].extend(NUMBER_RE.findall(metadata_text))
            merged["allowed_speakers"].extend(re.findall(r"@[\w.-]+", metadata_text))
            merged["allowed_assignees"].extend(re.findall(r"@[\w.-]+", metadata_text))
            frame = task_state.get("action_frame", {}) if isinstance(task_state.get("action_frame"), dict) else {}
            support = task_state.get("field_support", {}) if isinstance(task_state.get("field_support"), dict) else {}
            supported_participants = []
            if task_state.get("acceptance_evidence_ids") and frame.get("explicit_acceptance_actor"):
                supported_participants.append(frame["explicit_acceptance_actor"])
            for role in ("recipient", "beneficiary"):
                if frame.get(role) and support.get(role):
                    supported_participants.append(frame[role])
            merged["allowed_speakers"].extend(
                participant for participant in supported_participants
                if re.fullmatch(r"@[\w.-]+", str(participant))
            )
            metadata_text += " " + " ".join(map(str, supported_participants))
            if task_state.get("current_scope"):
                merged["time_scope"].append(str(task_state["current_scope"]))
        else:
            metadata_text = ""
        if question_state:
            metadata_text += " " + " ".join(str(question_state.get(field) or "") for field in
                                                ("original_question", "known_answer", "remaining_question"))
            metadata_text += " " + " ".join(map(str, question_state.get("missing_slot_labels", [])))
            merged["polarity"] = []  # Question-state labels are not predicate polarity.
            primary_scope = by_claim[claim_ids[0]].get("time_scope") if claim_ids else None
            merged["time_scope"] = primary_scope if isinstance(primary_scope, list) else [primary_scope] if primary_scope else []
        # Complete the source-derived contract before auditing the realization.
        # Previously these fields were appended after audit_realization(), so
        # verbatim source negation and source numbers could be rejected as new.
        exact_source_surfaces = []
        semantic_source_surfaces = []
        exact_evidence_surfaces = []
        for source in cited:
            exact_ids = set(source.get("evidence_ids", []))
            exact_turns = [
                str(turn.get("text") or "") for turn in source.get("dialogue_evidence", [])
                if turn.get("id") in exact_ids
            ]
            semantic_surface = " ".join([str(source.get("statement") or ""), public_surface_text(source)])
            semantic_source_surfaces.append(semantic_surface)
            exact_evidence_surfaces.extend(exact_turns)
            exact_source_surfaces.append(" ".join([semantic_surface, *exact_turns]))
        merged["allowed_numbers"].extend(NUMBER_RE.findall(" ".join(exact_source_surfaces)))
        merged["allowed_speakers"].extend(s for x in cited for s in x.get("speaker_refs", []))
        # A long evidence turn can contain a different, unrelated negated
        # clause. It may license negation actually rendered from that exact
        # turn, but it must not force every shorter realization to be negative.
        source_has_negation = (
            any(NEGATION_RE.search(value) for value in semantic_source_surfaces)
            or bool(NEGATION_RE.search(text) and any(NEGATION_RE.search(value) for value in exact_evidence_surfaces))
        )
        if item.get("section") == "questions":
            # A residual question is not an assertion of its cited answer or
            # schedule state. Keep its source-backed wording and provenance
            # checks, but do not copy predicate polarity from those claims.
            merged["polarity"] = []
        elif source_has_negation and "negative" not in merged["polarity"]:
            merged["polarity"].append("negative")
        allowed_evidence = {value for source in cited for value in source.get("evidence_ids", [])}
        allowed_evidence.update(task_state.get("evidence_ids", []))
        allowed_evidence.update(question_state.get("answer_evidence_ids", []))
        allowed_evidence.update(question_state.get("residual_support", []))
        if not set(item.get("evidence_ids", [])) <= allowed_evidence:
            realization = audit_realization(text, merged)
            realization["errors"].append("evidence_outside_closure")
        else:
            realization = audit_realization(text, merged)
        if unknown_claim_ids:
            realization["errors"].append("unknown_claim")
        if INCOMPLETE_PUBLIC_FRAGMENT_RE.search(text.strip()):
            realization["errors"].append("incomplete_public_fragment")
        source_tokens = {v for value in exact_source_surfaces for v in re.findall(r"(?iu)[a-zа-яё0-9]+", value.casefold()) if len(v) > 2}
        source_tokens.update(v for v in re.findall(r"(?iu)[a-zа-яё0-9]+", metadata_text.casefold()) if len(v) > 2)
        comparison_text = re.sub(
            r"(?iu)\s*\(статус:\s*(?:проверка недоступна|недостаточно доказательств для интерпретации|источник содержит противоречие)\)\s*$",
            "", text,
        )
        # Task-card suffixes are structured metadata, not an added semantic
        # clause.  Actor/status/scope are checked separately below and by the
        # QA slots; including their prose labels in lexical entailment caused
        # source-backed tasks to be silently removed.
        semantic_comparison_text = re.sub(
            r"(?iu)(?:\s+[—–-]\s+(?:исполнитель|объ[её]м|период\s+данных|срок|условие|статус):.*)$",
            "",
            comparison_text,
        ) if item.get("section") == "tasks" else comparison_text
        text_tokens = {v for v in re.findall(r"(?iu)[a-zа-яё0-9]+", semantic_comparison_text.casefold()) if len(v) > 2}
        editorial_tokens = {
            "спрашивает", "исполнитель", "статус", "объём", "срок", "условие",
            "период", "источника", "выборки", "данных",
            "участник", "назначение", "ожидает", "подтверждения", "предложен",
            "подтверждён", "уточнено", "известно", "осталось", "уточнить", "взял", "себя",
        }
        content_tokens = text_tokens - editorial_tokens
        overlap = len(source_tokens & content_tokens)
        if item.get("section") and overlap / max(1, len(content_tokens)) < .45:
            realization["errors"].append("cleaned_text_semantic_drift")
        unsupported = content_tokens - source_tokens
        if len(unsupported) >= 3 and overlap / max(1, len(content_tokens)) < .68:
            realization["errors"].append("unsupported_added_clause")
        task_like = item.get("section") == "tasks" or bool(re.search(r"(?iu)\b(?:должен|должна|сделает|подготовит|отправит|передаст|покажет|разметит)\b", text))
        plan_assignees = {person for person in merged.get("allowed_assignees", []) if person}
        expected_actor = task_state.get("assignee") if task_state else (next(iter(plan_assignees)) if len(plan_assignees) == 1 else None)
        rendered_actor = re.search(r"(?iu)исполнитель:\s*(@[\w.-]+)", text)
        if task_like and rendered_actor and (not expected_actor or rendered_actor.group(1) != expected_actor):
            realization["errors"].append("actor_recipient_swap")
        # Check the grammatical subject in every public view, not only the
        # optional `исполнитель:` suffix.  A/B swaps can otherwise retain all
        # source words and pass a lexical overlap gate.
        source_actor = expected_actor
        source_actor_mentions = {person for x in cited for person in re.findall(r"@[\w.-]+", str(x.get("statement") or ""))}
        if not source_actor and len(source_actor_mentions) == 1:
            source_actor = next(iter(source_actor_mentions))
        public_subject = re.search(r"(?iu)(@[\w.-]+)\s+(?:долж\w*|сдела\w*|подготов\w*|отправ\w*|переда\w*|покаж\w*|размет\w*)", text)
        if task_like and source_actor and public_subject and public_subject.group(1) != source_actor:
            realization["errors"].append("actor_recipient_swap")
        source_relations = set().union(*(role_relations(x.get("statement")) for x in cited))
        rendered_relations = role_relations(text)
        if rendered_relations and not rendered_relations <= source_relations:
            realization["errors"].append("actor_recipient_swap")
        if item.get("section") == "tasks" and task_state.get("action_frame", {}).get("state") == "reported_plan" and item.get("social_state") not in {"proposed", "reported_plan", "requires_confirmation"}:
            realization["errors"].append("reported_plan_promoted")
        if any(x.get("lifecycle", "active") != "active" for x in cited):
            realization["errors"].append("inactive_claim_published")
        if item.get("section") == "decisions" and not all(can_publish_as_decision(x) for x in cited):
            realization["errors"].append("status_upgrade")
        if item.get("section") == "tasks" and any(x.get("task_status") in {"idea", "proposed", "superseded"} for x in cited) and item.get("social_state") in {"accepted", "self_committed", "assigned"}:
            realization["errors"].append("task_status_upgrade")
        qa = qa_verify(text, merged)
        realization["errors"] = sorted(set(realization["errors"]))
        realization["passed"] = not realization["errors"] and qa["passed"]
        if not qa["passed"]:
            realization["errors"] = sorted(set(realization["errors"] + ["qa_slot_failure"]))
            realization["status"] = "ABSTAIN"
        audits.append({"text": text, "claim_ids": claim_ids, **realization, "qa": qa})
    return {"passed": all(x["passed"] for x in audits), "audits": audits, "abstentions": [x for x in audits if not x["passed"]]}


def partition_verified_public_items(items, report):
    """Quarantine rejected items while preserving an auditable disposition."""
    audits = list(report.get("audits", []))
    if len(items) != len(audits):
        raise ValueError("PublicItem verification result does not match input length")
    retained, rejected = [], []
    for item, audit in zip(items, audits):
        if audit.get("passed"):
            retained.append(item)
        else:
            rejected.append({"public_item": item, "audit": audit})
    return retained, rejected


def protected_public_item_abstentions(items, report):
    """Return mandatory outcomes that a lossy post-render filter would hide.

    Weak context may be safely omitted after a failed realization audit, but
    a confirmed canonical task, accepted decision, correction, blocker or
    open question is part of the meeting contract.  Publication must fail
    visibly instead of producing a green but incomplete summary.
    """
    audits = list(report.get("audits", []))
    if len(items) != len(audits):
        raise ValueError("PublicItem verification result does not match input length")
    protected = []
    for item, audit in zip(items, audits):
        if audit.get("passed"):
            continue
        mandatory = (
            item.get("section") == "tasks"
            and item.get("social_state") in PROTECTED_TASK_STATES
        ) or (
            item.get("section") == "decisions"
            and item.get("social_state") == "accepted"
        ) or item.get("content_kind") in {
            "blocker", "correction", "experimental_result", "schedule",
        } or item.get("section") == "questions"
        if mandatory:
            protected.append({"public_item": item, "audit": audit})
    return protected


def publication_audit(report, artifact_text, items=None, summary_plan=None, verified_hash=None, document=None):
    items = items or []
    summary_plan = summary_plan or {}
    audits = report.get("audits", [])
    counters = {
        "unsupported_public_items": sum(not x.get("passed") for x in audits),
        "orphan_public_items": sum("orphan_public_item" in x.get("errors", []) for x in audits),
        "status_upgrades": sum(bool({"status_upgrade", "task_status_upgrade"} & set(x.get("errors", []))) for x in audits),
        "superseded_items_published": sum("inactive_claim_published" in x.get("errors", []) for x in audits),
        "number_or_negation_mismatches": sum(bool({"unplanned_number", "negation_not_preserved", "unsupported_negation"} & set(x.get("errors", []))) for x in audits),
        "cross_episode_merges_without_relation": sum("cross_episode_without_relation" in x.get("errors", []) for x in audits),
        "unknown_assignee_publications": sum("speaker_or_assignee_not_preserved" in x.get("errors", []) for x in audits),
    }
    def tokens(value): return {x for x in re.findall(r"(?iu)[a-zа-яё0-9]+", str(value or "").casefold()) if len(x) > 3}
    duplicates = 0
    cross_view_repetitions = 0
    technical_experiment_duplicates = 0
    for index, left in enumerate(items):
        for right in items[index + 1:]:
            similar = equivalent(left, right, .8)
            if left.get("section") == right.get("section"):
                duplicates += similar
            else:
                # Overview, canonical task and chronology intentionally reuse
                # one supported claim in different views; track, do not reject.
                cross_view_repetitions += similar
                if {left.get("section"), right.get("section")} == {"technical", "experiments"}:
                    technical_experiment_duplicates += similar
    minute_starts = [float(x.get("start", 0)) for x in items if x.get("section") == "minutes"]
    task_ids = [x.get("task_state_id") for x in items if x.get("section") == "tasks"]
    internal = INTERNAL_LABEL_RE
    overview_tokens = set().union(*(tokens(x.get("text")) for x in items if x.get("section") == "overview"))
    task_tokens = set().union(*(tokens(x.get("text")) for x in items if x.get("section") == "tasks"))
    rendered_counts, current = {}, None
    heading_sections = {"Главное": "overview", "Краткое описание — что изменилось после встречи": "overview", "Принятые решения": "decisions", "Упомянутые действующие правила": "rules", "Договорённости и следующие шаги": "tasks", "Действия и планы на подтверждение": "tasks", "Задачи и следующие шаги": "tasks", "Что осталось уточнить": "questions", "Открытые вопросы": "questions", "Технические выводы и ограничения": "technical", "Идеи и эксперименты": "experiments", "Идеи и эксперименты, ещё не проверенные": "experiments", "Гипотезы и эксперименты": "experiments", "Требует проверки источника": "requires_verification", "Хронология встречи": "minutes", "Подробная хронология встречи": "minutes"}
    for line in artifact_text.splitlines():
        if line.startswith("## "): current = heading_sections.get(line[3:].strip())
        elif line.startswith("- ") and current: rendered_counts[current] = rendered_counts.get(current, 0) + 1
    item_counts = {section: sum(x.get("section") == section for x in items) for section in sorted({x.get("section") for x in items})}
    title_line = next((line for line in artifact_text.splitlines() if line.startswith("# ")), "")
    authored_title = title_line.removeprefix("# ").split("—", 1)[-1].strip()
    overview_match = re.search(r"(?s)## (?:Главное|Краткое описание[^\n]*)\n(.*?)(?=\n## |\Z)", artifact_text)
    authored_overview = overview_match.group(1) if overview_match else ""
    authored_surface = authored_title + "\n" + authored_overview
    chronology_match = re.search(r"(?s)## Подробная хронология встречи\n(.*)\Z", artifact_text)
    chronology_surface = chronology_match.group(1) if chronology_match else ""
    # Exact source-bound details remain available for auditability inside
    # collapsed disclosure blocks.  Reading-cost measures the default visible
    # summary, not text that the reader explicitly chooses to expand.
    visible_chronology_surface = re.sub(
        r"(?is)<details>.*?</details>", "", chronology_surface
    )
    allowed_document_numbers = {x.replace(" ", "") for item in items for x in NUMBER_RE.findall(str(item.get("text") or ""))}
    found_document_numbers = {x.replace(" ", "") for x in NUMBER_RE.findall(authored_surface)}
    state_conflicts = sum(bool(
        x.get("section") == "tasks" and (not x.get("task_state_id") or x.get("social_state") != x.get("task_state", {}).get("status") or (x.get("task_state", {}).get("current_scope") and str(x["task_state"]["current_scope"]).casefold() not in str(x.get("text") or "").casefold()))
    )
        for x in items
    )
    states_by_claim = {}
    for item in items:
        for claim_id in item.get("claim_ids", []):
            states_by_claim.setdefault(claim_id, []).append((item.get("aspect_id"), item.get("social_state"), item.get("section")))
    incompatible = {("accepted", "assigned_pending"), ("accepted", "proposed"), ("completed", "assigned_pending"), ("rejected", "accepted")}
    cross_view_conflicts = 0
    for values in states_by_claim.values():
        if len({x[1] for x in values}) > 1 and not all(x[0] for x in values):
            states = {x[1] for x in values}
            cross_view_conflicts += any({a, b} <= states for a, b in incompatible)
    double_modality = re.compile(r"(?iu)\b(?:предлагалось|предложено)\b.{0,40}\b(?:договорились|решили|принято)\b")
    mixed_token = re.compile(r"(?iu)\b(?:[а-яё]+[a-z]+|[a-z]+[а-яё]+)\b")
    readability_lint = sum(bool(INCOMPLETE_PUBLIC_FRAGMENT_RE.search(str(x.get("text") or "").strip()) or double_modality.search(str(x.get("text") or "")) or mixed_token.search(str(x.get("text") or ""))) for x in items)
    overview_surface = re.sub(r"[*`]", "", authored_overview).casefold()
    overview_surface_tokens = tokens(overview_surface)
    confirmed_tasks = [
        x for x in items
        if x.get("section") == "tasks" and x.get("social_state") in {"accepted", "self_committed", "assigned", "completed"}
    ]
    overview_claim_ids = {
        claim_id
        for node in (document or {}).get("overview", [])
        for claim_id in node.get("claim_ids", [])
    }
    overview_has_committed_next_step = any(
        bool(set(task.get("claim_ids", [])) & overview_claim_ids)
        or (
            bool(task_words := tokens(
                task.get("task_state", {}).get("deliverable")
                or task.get("task_state", {}).get("description")
                or task.get("text")
            ))
            and len(task_words & overview_surface_tokens)
            / max(1, min(len(task_words), len(overview_surface_tokens))) >= .45
        )
        for task in confirmed_tasks
    )
    overview_claim_ids_for_utility = {
        claim_id for node in (document or {}).get("overview", [])
        for claim_id in node.get("claim_ids", [])
    }
    overview_backing_items = [
        item for item in items
        if overview_claim_ids_for_utility & set(item.get("claim_ids", []))
    ]
    available_constraint_items = [
        item for item in items
        if item.get("content_kind") in {"problem", "blocker", "constraint"}
        and item.get("verification_status") == "supported"
    ]
    action_field_checks = []
    for item in [value for value in items if value.get("section") == "tasks"]:
        state = item.get("task_state", {})
        support = state.get("field_support", {})
        explicit = bool(state.get("source_commitment_evidence_ids"))
        accepted = bool(state.get("acceptance_evidence_ids"))
        action_field_checks.append(bool(support.get("predicate") or explicit or accepted))
        if state.get("assignee"):
            action_field_checks.append(bool(support.get("actor") or explicit or accepted))
        frame = state.get("action_frame", {})
        if frame.get("recipient"):
            action_field_checks.append(bool(support.get("recipient")))
        if state.get("completion_criterion"):
            action_field_checks.append(bool(support.get("object") or state.get("evidence_ids")))
    verified_action_field_rate = sum(action_field_checks) / max(1, len(action_field_checks))
    counters.update({
        "duplicate_items": duplicates,
        "cross_view_repetitions": cross_view_repetitions,
        "answered_questions_published_as_open": sum(x.get("section") == "questions" and x.get("question_state", {}).get("status") in {"answered", "rhetorical", "superseded"} for x in items),
        "answered_questions_in_minutes": sum(x.get("section") == "minutes" and x.get("content_kind") == "question" and x.get("question_state", {}).get("status") in {"answered", "rhetorical", "superseded"} for x in items),
        "unconfirmed_tasks_published_as_committed": sum(x.get("section") == "tasks" and x.get("social_state") == "self_committed" and x.get("task_state", {}).get("commitment_strength") != "explicit" for x in items),
        "duplicate_task_states": len([x for x in task_ids if x]) - len({x for x in task_ids if x}),
        "chronology_inversions": sum(a > b for a, b in zip(minute_starts, minute_starts[1:])),
        "internal_labels_exposed": sum(bool(internal.search(str(x.get("text") or ""))) for x in items) + int(bool(internal.search(artifact_text))),
        "rendered_english_prose": sum(has_english_prose(re.sub(r"\]\([^)]+\)", "]", line)) for line in artifact_text.splitlines() if line.strip()),
        "invented_acronym_expansions": len(re.findall(r"\b[A-Z]{2,}\s*\(\s*[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)+\s*\)", artifact_text)),
        "zero_duration_chapters": len(re.findall(r"(?m)^(?:- |### ).*?(\d{2}:\d{2}:\d{2})(?:\]\([^)]+\))?[–—-]\1\b", artifact_text)),
        "excessive_chapter_count": int(sum(line.startswith("### ") for line in artifact_text.splitlines()) > int((document or {}).get("metadata", {}).get("max_chapters", 12))),
        "missing_public_provenance": sum(not x.get("evidence_ids") or not x.get("source_word_ids") for x in items),
        "planner_budget_violations": sum(int(v.get("selected_count", len(v.get("selected_claim_ids", [])))) > int(v.get("budget", 0)) for v in summary_plan.get("view_plans", {}).values()),
        "state_conflicts": state_conflicts,
        "cross_view_state_conflicts": cross_view_conflicts,
        "readability_lint_failures": readability_lint,
        "title_unresolved_reference": int(
            unresolved_public_reference(authored_title)
            or "@" in authored_title
            or bool(RAW_DIALOGUE_RE.search(authored_title))
        ),
        "overview_unresolved_reference": sum(
            unresolved_public_reference(line)
            for line in authored_overview.splitlines()
            if line.strip() and not line.lstrip().startswith(("<", "**"))
        ),
        "overview_raw_dialogue": int(bool(RAW_DIALOGUE_RE.search(authored_overview))),
        "task_raw_dialogue": sum(
            x.get("section") == "tasks" and bool(RAW_DIALOGUE_RE.search(str(x.get("text") or "")))
            for x in items
        ),
        "invalid_decision_acceptance": sum(
            x.get("section") == "decisions" and not can_publish_as_decision(x)
            for x in items
        ),
        "task_without_deliverable": sum(x.get("section") == "tasks" and not str(x.get("task_state", {}).get("deliverable") or "").strip() for x in items),
        "vague_focus_tasks": sum(x.get("section") == "tasks" and bool(re.search(r"(?iu)\b(?:сделать\s+упор|сосредоточиться|ещ[её]\s+над\s+этим\s+посидеть)\b", str(x.get("text") or ""))) for x in items),
        "reported_plan_assignee_leaks": sum(
            x.get("task_state", {}).get("action_frame", {}).get("state") == "reported_plan"
            and bool(x.get("task_state", {}).get("owner") or x.get("task_state", {}).get("assignee") or x.get("task_state", {}).get("assignees"))
            for x in items
        ),
        "overview_missing_committed_next_step": int(bool(confirmed_tasks) and not overview_has_committed_next_step),
        "overview_task_overlap": (len(overview_tokens & task_tokens) / max(1, len(overview_tokens))) if overview_tokens else 0,
        # Overview items are intentionally merged into prose paragraphs rather
        # than rendered one bullet per PublicItem.
        "section_count_mismatches": sum(item_counts.get(k, 0) != rendered_counts.get(k, 0) for k in (set(item_counts) | set(rendered_counts)) - {"overview", "minutes"}),
        "unplanned_document_numbers": len(found_document_numbers - allowed_document_numbers),
        "navigation_missing": int(bool(item_counts.get("minutes")) and "## Таймкоды" not in artifact_text),
        "chronology_missing": int(bool(item_counts.get("minutes")) and "## Подробная хронология встречи" not in artifact_text and "## Хронология встречи" not in artifact_text),
        "weak_navigation_labels": sum(
            navigation_label_needs_repair(chapter.get("label"))
            for chapter in (document or {}).get("navigation", [])
        ),
        "title_missing": int(not title_line),
        "title_too_long": int(len(authored_title.strip()) > 110),
        "title_action_fragment": int(bool(TITLE_ACTION_FRAGMENT_RE.search(authored_title))),
        "title_dangling_fragment": int(bool(TITLE_DANGLING_RE.search(authored_title))),
        "title_low_information_clause": int(bool(TITLE_LOW_INFORMATION_RE.search(authored_title))),
        "participant_only_title": int(document is not None and bool(authored_title.strip()) and not public_context_tokens(authored_title)),
        "generic_fallback_title": int(document is not None and bool(re.fullmatch(r"(?iu)\s*(?:рабочие\s+итоги|итоги\s+рабочей\s+встречи)(?:\s+и\s+открытые\s+вопросы)?\s*", authored_title.strip()))),
        "excessive_residual_questions": int(item_counts.get("questions", 0) > 5),
        "excessive_technical_items": int(sum(x.get("section") == "technical" and not (x.get("content_kind") == "hypothesis" and x.get("social_state") == "constraint") for x in items) > 6),
        "excessive_verification_items": int(item_counts.get("requires_verification", 0) > 8),
        "technical_experiment_duplicates": technical_experiment_duplicates,
        "tentative_decision_surfaces": sum(
            x.get("section") == "decisions"
            and x.get("content_kind") == "proposal"
            and bool(PROPOSAL_SURFACE_RE.search(str(x.get("text") or "")))
            and not str(x.get("text") or "").casefold().startswith("согласовано предложение:")
            for x in items
        ),
        "non_action_task_surfaces": sum(
            x.get("section") == "tasks"
            and (not WORK_ACTION_SURFACE_RE.search(str(x.get("text") or ""))
                 or bool(INTERNAL_LABEL_RE.search(str(x.get("text") or "")))
                 or has_adjacent_stem_repetition(x.get("text")))
            for x in items
        ),
        "verification_status_repeated_in_text": sum(
            x.get("section") == "requires_verification"
            and bool(re.search(r"(?iu)\(статус:\s*(?:проверка|недостаточно|источник)", str(x.get("text") or "")))
            for x in items
        ),
        "overview_repeated_status": int(bool(re.search(
            r"(?iu)—\s*статус:\s*([^().;\n]+?)\s*"
            r"\(\s*статус:\s*\1\s*\)",
            authored_overview,
        ))),
        "definition_only_technical_items": sum(x.get("section") == "technical" and bool(re.search(r"(?iu)\b(?:называется|определяется|ширина\s*[—–-]\s*ширина)\b", str(x.get("text") or ""))) for x in items),
        "open_questions_with_unverified_answer": sum(
            x.get("section") == "questions"
            and x.get("question_state", {}).get("status") in {"answer_not_verified", "answer_retrieval_failed"}
            for x in items
        ),
        "overview_missing_main_constraint": int(
            bool(available_constraint_items)
            and not any(item.get("content_kind") in {"problem", "blocker", "constraint"} for item in overview_backing_items)
        ),
        "title_too_narrow": int(
            len((document or {}).get("title", {}).get("claim_ids", [])) < 2
            and len({item.get("episode_id") for item in items if item.get("episode_id")}) >= 4
        ),
        "chronology_excessive_reading_cost": int(
            bool(chronology_surface)
            and len(re.findall(r"(?iu)[a-zа-яё0-9]+", visible_chronology_surface))
                > 110 * max(1, len((document or {}).get("chronology", [])))
        ),
        "low_verified_action_field_rate": int(bool(action_field_checks) and verified_action_field_rate < .75),
        "selected_rules_missing": int(
            bool(summary_plan.get("view_plans", {}).get("rules", {}).get("selected_claim_ids"))
            and not item_counts.get("rules")
        ),
    })
    if document is not None:
        contextual_sections = {"tasks", "questions", "technical", "experiments"}
        contextual_items = [item for section, values in document.get("sections", {}).items() if section in contextual_sections for item in values]
        counters["section_items_missing_context"] = 0  # absence is honest when no explicit relation exists
        counters["section_context_repetitions"] = sum(
            public_context_duplicate(context.get("text"), item.get("text"))
            for item in contextual_items for context in item.get("context", [])
        )
        counters["section_context_low_relevance"] = sum(
            not context.get("directly_linked")
            and not (public_context_stems(item.get("text")) & public_context_stems(context.get("text")))
            for item in contextual_items for context in item.get("context", [])
        )
        # Keep this vocabulary aligned with build_public_document() and the
        # renderer.  A renderer-supported role is not an internal label.
        allowed_context_roles = {
            "known_answer", "known_context", "purpose", "related_step",
            "importance", "application", "explanation", "motivation",
            "observation", "test_detail", "condition", "dependency",
            "refinement", "correction", "acceptance",
        }
        counters["section_context_missing_role"] = sum(
            context.get("role") not in allowed_context_roles
            for item in contextual_items for context in item.get("context", [])
        )
        semantic_abstentions = (document.get("semantic_audit") or {}).get("abstentions", [])
        abstained_answer_contexts = set()
        for abstention in semantic_abstentions:
            node = abstention.get("node", abstention) if isinstance(abstention, dict) else {}
            match = re.fullmatch(r"context:questions:(\d+):\d+", str(node.get("node_id") or ""))
            if match and node.get("role") == "known_answer":
                abstained_answer_contexts.add(int(match.group(1)))
        counters["question_context_missing_known_answer"] = sum(
            bool(question.get("answer_record_ids"))
            and question.get("status") in {"answered", "partially_answered"}
            and question.get("answer_verification", {}).get("status")
                not in {"insufficient_evidence", "contradicted", "not_evaluated"}
            and not any(context.get("role") == "known_answer" for context in item.get("context", []))
            and index not in abstained_answer_contexts
            for index, item in enumerate(document.get("sections", {}).get("questions", []), 1)
            for question in [item.get("question_state", {})]
        )
        counters["goal_only_experiments"] = sum(
            bool(re.search(r"(?iu)\b(?:обсуждалась\s+цель|целевой\s+ориентир|базов\w*\s+решени\w*)\b", str(item.get("text") or "")))
            for item in document.get("sections", {}).get("experiments", [])
        )
        chronology_duplicates = 0
        for block in re.split(r"(?m)^### ", artifact_text.split("## Подробная хронология встречи", 1)[-1] if "## Подробная хронология встречи" in artifact_text else ""):
            chronology_values = []
            values = [normalize.group(1).casefold() for line in block.splitlines() if (normalize := re.match(r"^\*\*[^*]+:\*\*\s*(.+)$", line.strip()))]
            for value in values:
                value_tokens = tokens(value)
                chronology_duplicates += any(
                    value_tokens and len(value_tokens & tokens(previous)) / max(1, min(len(value_tokens), len(tokens(previous)))) >= .82
                    for previous in chronology_values
                )
                chronology_values.append(value)
        counters["chronology_duplicate_fields"] = chronology_duplicates
    else:
        counters.update({"section_items_missing_context": 0, "section_context_repetitions": 0, "section_context_low_relevance": 0, "section_context_missing_role": 0, "question_context_missing_known_answer": 0, "goal_only_experiments": 0, "chronology_duplicate_fields": 0})
    counters["section_counts"] = item_counts
    counters["rendered_section_counts"] = rendered_counts
    counters["planner_overflow"] = {name: value.get("overflow_count", 0) for name, value in summary_plan.get("view_plans", {}).items()}
    artifact_hash = hashlib.sha256(artifact_text.encode("utf-8")).hexdigest()
    counters["verified_artifact_hash"] = verified_hash or artifact_hash
    counters["final_artifact_hash"] = artifact_hash
    unexplained = sum(1 for view in summary_plan.get("view_plans", {}).values() for claim_id in view.get("selected_claim_ids", []) if claim_id not in view.get("dispositions", {}))
    public_claims = {claim_id for item in items for claim_id in item.get("claim_ids", [])}
    candidate_ids = set(summary_plan.get("commitment_candidate_ids", []))
    routed_work = {claim_id for item in items if item.get("section") in {"tasks", "requires_verification"} for claim_id in item.get("claim_ids", [])}
    # A verifier abstention is a terminal, auditable disposition rather than a
    # silently lost commitment. Only exclusion-like dispositions carrying an
    # explicit reason may satisfy the accounting invariant.
    explained_candidates = {
        claim_id
        for view in summary_plan.get("view_plans", {}).values()
        for claim_id, disposition in view.get("dispositions", {}).items()
        if disposition.get("status") in {"excluded", "rejected", "abstained", "quarantined"}
        and str(disposition.get("reason") or "").strip()
    }
    counters.update({
        "unexplained_selected_claims": unexplained,
        "explained_commitment_candidates": len((candidate_ids & explained_candidates) - routed_work),
        "unexplained_commitment_candidates": len(candidate_ids - routed_work - explained_candidates),
        "published_unique_claims": len(public_claims),
    })
    integrity_keys = {"unsupported_public_items", "orphan_public_items", "status_upgrades", "superseded_items_published", "number_or_negation_mismatches", "cross_episode_merges_without_relation", "unknown_assignee_publications", "duplicate_items", "answered_questions_published_as_open", "answered_questions_in_minutes", "unconfirmed_tasks_published_as_committed", "duplicate_task_states", "chronology_inversions", "internal_labels_exposed", "rendered_english_prose", "invented_acronym_expansions", "zero_duration_chapters", "excessive_chapter_count", "missing_public_provenance", "planner_budget_violations", "state_conflicts", "cross_view_state_conflicts", "readability_lint_failures", "task_without_deliverable", "vague_focus_tasks", "reported_plan_assignee_leaks", "section_context_repetitions", "section_context_low_relevance", "section_context_missing_role", "question_context_missing_known_answer", "goal_only_experiments", "chronology_duplicate_fields", "section_count_mismatches", "unplanned_document_numbers", "navigation_missing", "chronology_missing", "title_missing", "invalid_decision_acceptance"}
    integrity = all(counters.get(key, 0) == 0 for key in integrity_keys) and counters["verified_artifact_hash"] == artifact_hash
    grounding = counters["unsupported_public_items"] == counters["number_or_negation_mismatches"] == counters["missing_public_provenance"] == 0
    coverage = bool(items) and unexplained == 0 and counters["unexplained_commitment_candidates"] == 0
    readability = all(counters[key] == 0 for key in ("duplicate_items", "internal_labels_exposed", "rendered_english_prose", "invented_acronym_expansions", "zero_duration_chapters", "excessive_chapter_count", "readability_lint_failures", "title_unresolved_reference", "overview_unresolved_reference", "overview_raw_dialogue", "task_raw_dialogue"))
    utility_keys = {
        "title_too_long", "participant_only_title", "generic_fallback_title",
        "title_action_fragment", "title_dangling_fragment", "title_low_information_clause",
        "overview_missing_committed_next_step", "excessive_residual_questions",
        "excessive_technical_items", "excessive_verification_items",
        "technical_experiment_duplicates", "tentative_decision_surfaces",
        "non_action_task_surfaces", "verification_status_repeated_in_text", "overview_repeated_status",
        "definition_only_technical_items",
        "open_questions_with_unverified_answer", "overview_missing_main_constraint",
        "title_too_narrow", "chronology_excessive_reading_cost",
        "low_verified_action_field_rate", "selected_rules_missing",
        "title_unresolved_reference", "overview_unresolved_reference",
        "overview_raw_dialogue", "task_raw_dialogue", "invalid_decision_acceptance",
        "weak_navigation_labels",
    }
    utility = all(counters.get(key, 0) == 0 for key in utility_keys)
    semantic_source = (document or {}).get("semantic_audit")
    semantic_status = semantic_source.get("status", "not_evaluated") if semantic_source else ("passed" if audits and all(x.get("passed") for x in audits) else "not_evaluated")
    required_semantic = document is not None
    semantic_ok = semantic_status == "passed" if required_semantic else semantic_status in {"passed", "not_evaluated"}
    reports = {
        "integrity": {"status": "passed" if integrity else "failed", "source": "deterministic"},
        "grounding": {"status": "passed" if grounding else "failed", "source": "claim_contract"},
        "coverage": {"status": "passed" if coverage else "failed", "source": "candidate_lineage"},
        "semantic": {"status": semantic_status, "source": "independent_final_document_audit" if semantic_source else "not_applicable_without_document", "required": required_semantic},
        "readability": {"status": "passed" if readability else "failed", "source": "lint"},
        "utility": {"status": "passed" if utility else "failed", "source": "bounded_document_contract"},
    }
    dimensions = {"integrity": integrity, "grounding": grounding, "candidate_disposition_integrity": coverage, "semantic": semantic_ok, "readability": readability, "utility": utility}
    semantic_reviews = (semantic_source or {}).get("reviews", [])
    relation_reviews = [item for item in semantic_reviews if item.get("relation_id")]
    task_items = [item for item in items if item.get("section") == "tasks"]
    verified_action_fields = sum(action_field_checks)
    action_field_total = len(action_field_checks)
    supported_outcomes = len({claim_id for item in items if item.get("verification_status") == "supported" for claim_id in item.get("claim_ids", [])})
    audit_metrics = {
        "verified_action_field_rate": {"value": verified_action_fields / max(1, action_field_total), "source": "field_evidence", "denominator": action_field_total},
        "critical_event_retention": {"value": 1 - counters["unexplained_commitment_candidates"] / max(1, len(candidate_ids)), "source": "candidate_lineage", "denominator": len(candidate_ids)},
        "raw_source_supported_claim_rate": {"value": sum(item.get("verdict") == "supported" for item in semantic_reviews) / max(1, len(semantic_reviews)), "source": "final_document_semantic_audit", "denominator": len(semantic_reviews)},
        "relation_support_rate": {"value": sum(item.get("relation_supported") is True for item in relation_reviews) / max(1, len(relation_reviews)), "source": "final_document_semantic_audit", "denominator": len(relation_reviews)},
        "rendered_node_coverage": {"value": (document or {}).get("verification", {}).get("rendered_node_coverage"), "source": "render_trace"},
        "nonredundant_information": {"value": 1 - counters["duplicate_items"] / max(1, len(items)), "source": "typed_public_item_equivalence", "denominator": len(items)},
        "cost_per_published_supported_outcome": {"value": None, "source": "unavailable_at_publication_gate", "supported_outcomes": supported_outcomes, "reason": "usage ledger is finalized after publication"},
    }
    return {"schema": "PublicationAudit", "schema_version": 5, "passed": integrity and grounding and coverage and readability and utility and semantic_ok, "dimensions": dimensions, "reports": reports, "audit_metrics": audit_metrics, "unknown_semantic_checks": (semantic_source or {}).get("unknown_semantic_checks", 0) if semantic_source else None, **counters}


def runtime_quality_gates(report, artifact_text, verified_hash=None, items=None, summary_plan=None, document=None):
    return publication_audit(report, artifact_text, items, summary_plan, verified_hash, document)


def verify_public_document(document, artifact_text, items, graph=None):
    """Bind the final title, prose, navigation and sections to their sources."""
    errors = []
    source_ids = {claim for item in items for claim in item.get("claim_ids", [])}
    graph_claims = {
        claim.get("claim_id"): claim for claim in (graph or {}).get("claims", [])
        if claim.get("claim_id") and claim.get("lifecycle", "active") == "active"
        and claim.get("verification_status") not in {"verification_unavailable", "insufficient_evidence"}
    }
    context_source_ids = source_ids | set(graph_claims)
    graph_relations = {relation.get("relation_id"): relation for relation in (graph or {}).get("relations", []) if relation.get("relation_id")}
    semantic_reviews_by_node = {
        review.get("node_id"): review
        for review in (document.get("semantic_audit") or {}).get("reviews", [])
        if isinstance(review, dict) and review.get("node_id")
    }
    render_trace = []
    by_claim = {}
    for item in items:
        for claim in item.get("claim_ids", []):
            by_claim.setdefault(claim, []).append(item)
    tokens = lambda value: set(re.findall(r"(?iu)[a-zа-яё0-9]+", re.sub(r"[*`]", "", str(value or "")).casefold()))
    evidence_for = lambda claim_ids: {evidence for claim_id in claim_ids for item in by_claim.get(claim_id, []) for evidence in item.get("evidence_ids", [])}
    context_evidence_for = lambda claim_ids: evidence_for(claim_ids) | {
        evidence for claim_id in claim_ids for evidence in graph_claims.get(claim_id, {}).get("evidence_ids", [])
    }
    surface = lambda value: re.sub(
        r"\s+", " ",
        re.sub(
            r"(?iu)\bтаймфрем(?:ы|ов|ами)?\b", "таймфрейм",
            re.sub(r"(?iu)\bbaseline\b", "ориентир", re.sub(r"[*`]", "", sanitize_public_surface(value))),
        ),
    ).strip().casefold()
    heading_sections = {"Главное": "overview", "Таймкоды": "navigation", "Принятые решения": "decisions", "Упомянутые действующие правила": "rules", "Задачи и следующие шаги": "tasks", "Что осталось уточнить": "questions", "Технические выводы и ограничения": "technical", "Идеи и эксперименты, ещё не проверенные": "experiments", "Требует проверки источника": "requires_verification", "Подробная хронология встречи": "chronology"}
    section_text = {}
    current = None
    for line in artifact_text.splitlines():
        if line.startswith("## "):
            current = heading_sections.get(line[3:].strip())
        elif current:
            section_text[current] = section_text.get(current, "") + "\n" + line
    chapter_blocks = {}
    chronology = document.get("chronology", [])
    chapter_index = -1
    for line in artifact_text.splitlines():
        if line.startswith("### "):
            chapter_index += 1
            if chapter_index < len(chronology):
                for outcome_id in chronology[chapter_index].get("outcome_ids", [chronology[chapter_index].get("outcome_id")]):
                    chapter_blocks[outcome_id] = line
        elif chapter_index >= 0 and chapter_index < len(chronology) and not line.startswith("## "):
            for outcome_id in chronology[chapter_index].get("outcome_ids", [chronology[chapter_index].get("outcome_id")]):
                chapter_blocks[outcome_id] = chapter_blocks.get(outcome_id, "") + "\n" + line
    if INTERNAL_LABEL_RE.search(artifact_text):
        errors.append("internal_label_rendered")
    if re.search(r"\b[A-Z]{2,}\s*\(\s*[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)+\s*\)", artifact_text):
        errors.append("invented_acronym_expansion")
    if any(has_english_prose(re.sub(r"\]\([^)]+\)", "]", line)) for line in artifact_text.splitlines() if line.strip()):
        errors.append("english_prose_rendered")
    if len(document.get("navigation", [])) > int(document.get("metadata", {}).get("max_chapters", 12)):
        errors.append("excessive_chapter_count")
    title = document.get("title", {})
    if not title.get("text") or not set(title.get("claim_ids", [])) <= source_ids or not title.get("evidence_ids"):
        errors.append("unsupported_title")
    if not artifact_text.splitlines() or not artifact_text.splitlines()[0].endswith(" — " + str(title.get("text") or "")):
        errors.append("title_not_rendered")
        render_trace.append({"node_id": "title", "rendered": False, "reason": "surface_missing"})
    else:
        render_trace.append({"node_id": "title", "rendered": True})
    if not set(title.get("evidence_ids", [])) <= evidence_for(title.get("claim_ids", [])):
        errors.append("title_evidence_outside_closure")
    title_source = set().union(*(tokens(item.get("text")) | tokens(" ".join(item.get("topic_entities", [])))
                                for claim in title.get("claim_ids", []) for item in by_claim.get(claim, [])))
    title_words = tokens(title.get("text")) - {"следующие", "шаги", "итоги", "встречи", "результаты", "результат", "проверки", "ограничения", "и"}
    if {word for word in title_words if not any(word == source or (len(word) >= 5 and len(source) >= 5 and word[:5] == source[:5]) for source in title_source)}:
        errors.append("title_semantic_drift")
    title_source_relations = set().union(*(role_relations(item.get("text")) for claim in title.get("claim_ids", []) for item in by_claim.get(claim, [])))
    if role_relations(title.get("text")) and not role_relations(title.get("text")) <= title_source_relations:
        errors.append("title_role_swap")
    for overview_index, node in enumerate(document.get("overview", []), 1):
        if not node.get("claim_ids") or not set(node["claim_ids"]) <= source_ids or not node.get("evidence_ids"):
            errors.append("unsupported_overview")
        if not set(node.get("evidence_ids", [])) <= evidence_for(node.get("claim_ids", [])):
            errors.append("overview_evidence_outside_closure")
        if surface(node.get("text")) not in surface(section_text.get("overview", "")):
            errors.append("overview_not_rendered")
            render_trace.append({"node_id": f"overview:{overview_index}", "rendered": False, "reason": "surface_missing"})
        else:
            render_trace.append({"node_id": f"overview:{overview_index}", "rendered": True})
        backing = [item for claim in node.get("claim_ids", []) for item in by_claim.get(claim, [])]
        source_words = set().union(*(
            tokens(item.get("text")) |
            tokens(" ".join(str(item.get("task_state", {}).get(field) or "")
                            for field in ("description", "deliverable", "status", "task_status", "current_scope")))
            for item in backing
        )) if backing else set()
        independently_supported = semantic_reviews_by_node.get(f"overview:{overview_index}", {}).get("verdict") == "supported"
        if not backing or (len(tokens(node.get("text")) - source_words - {"статус"}) > 2 and not independently_supported):
            errors.append("overview_semantic_drift")
    navigation = document.get("navigation", [])
    if document.get("chronology") and not navigation:
        errors.append("missing_navigation")
    if navigation and "## Таймкоды" not in artifact_text:
        errors.append("navigation_not_rendered")
    for chapter_index, chapter in enumerate(navigation, 1):
        if float(chapter.get("end", 0)) <= float(chapter.get("start", 0)):
            errors.append("zero_duration_chapter")
        if not chapter.get("claim_ids") or not set(chapter["claim_ids"]) <= source_ids or not chapter.get("evidence_ids"):
            errors.append("unsupported_navigation")
        if surface(chapter.get("label")) not in surface(section_text.get("navigation", "")):
            errors.append("navigation_not_rendered")
            render_trace.append({"node_id": f"navigation:{chapter_index}", "rendered": False, "reason": "surface_missing"})
        else:
            render_trace.append({"node_id": f"navigation:{chapter_index}", "rendered": True})
        source_words = set().union(*(tokens(item.get("text")) for claim in chapter.get("claim_ids", []) for item in by_claim.get(claim, [])))
        navigation_independently_supported = semantic_reviews_by_node.get(f"navigation:{chapter_index}", {}).get("verdict") == "supported"
        if len(tokens(chapter.get("label")) - source_words) > 1 and not navigation_independently_supported:
            errors.append("navigation_semantic_drift")
        if re.search(r"(?iu)(?:\.{3}|…|\b(?:и|или|что|чтобы|из-за|после))$", str(chapter.get("label") or "").strip()):
            errors.append("truncated_navigation_label")
        if navigation_label_needs_repair(chapter.get("label")):
            errors.append("weak_navigation_label")
    summaries_by_outcome = {}
    chapter_items_by_outcome = {}
    for chapter_index, chapter in enumerate(document.get("chronology", []), 1):
        chapter_item_claims = {
            claim_id for item in chapter.get("items", [])
            for claim_id in item.get("claim_ids", [])
        }
        for outcome_id in chapter.get("outcome_ids", [chapter.get("outcome_id")]):
            chapter_items_by_outcome[outcome_id] = chapter_item_claims
        summaries = chapter.get("summary", [])
        if not summaries:
            continue
        chapter_block = "\n".join(chapter_blocks.get(value, "") for value in chapter.get("outcome_ids", [chapter.get("outcome_id")]))
        for summary_index, node in enumerate(summaries, 1):
            node_id = f"chronology_summary:{chapter_index}:{summary_index}"
            if not node.get("claim_ids") or not set(node.get("claim_ids", [])) <= source_ids or not node.get("evidence_ids"):
                errors.append("unsupported_chronology_summary")
            if not set(node.get("evidence_ids", [])) <= evidence_for(node.get("claim_ids", [])):
                errors.append("chronology_summary_evidence_outside_closure")
            rendered = surface(node.get("text")) in surface(chapter_block)
            if not rendered:
                errors.append("chronology_summary_not_rendered")
            exact_public_item = any(
                surface(node.get("text")) == surface(item.get("text"))
                and set(node.get("claim_ids", [])) == set(item.get("claim_ids", []))
                and set(node.get("evidence_ids", [])) == set(item.get("evidence_ids", []))
                for item in items
            )
            if semantic_reviews_by_node.get(node_id, {}).get("verdict") != "supported" and not exact_public_item:
                errors.append("chronology_summary_not_independently_supported")
            render_trace.append({"node_id": node_id, "rendered": rendered})
        represented_claims = {claim for node in summaries for claim in node.get("claim_ids", [])}
        for outcome_id in chapter.get("outcome_ids", [chapter.get("outcome_id")]):
            summaries_by_outcome[outcome_id] = represented_claims
    for card_index, card in enumerate(document.get("outcome_cards", []), 1):
        if not card.get("claim_ids") or not set(card["claim_ids"]) <= source_ids or not card.get("evidence_ids"):
            errors.append("unsupported_outcome_card")
        if not set(card.get("evidence_ids", [])) <= evidence_for(card.get("claim_ids", [])):
            errors.append("outcome_evidence_outside_closure")
        for name, raw in card.get("fields", {}).items():
            fields = raw if isinstance(raw, list) else [raw] if raw else []
            for field_index, field in enumerate(fields, 1):
                if not field.get("claim_ids") or not set(field["claim_ids"]) <= source_ids:
                    errors.append("unsupported_outcome_field")
                if not set(field.get("evidence_ids", [])) <= evidence_for(field.get("claim_ids", [])):
                    errors.append("outcome_field_evidence_outside_closure")
                chapter_block = chapter_blocks.get(card.get("outcome_id"))
                rendered_card_block = chapter_block or section_text.get("overview", "")
                field_tokens = tokens(field.get("value"))
                rendered_tokens = tokens(rendered_card_block)
                field_labels = {"current_state": "Состояние", "constraint": "Ограничение", "resolution": "Согласованный итог", "work_result": "Полученный результат", "mentioned_resource": "Упомянутый ресурс", "described_rule": "Описанное правило", "next_step": "Дальше", "remaining_unknown": "Осталось уточнить"}
                exact_rendered = surface(field.get("value")) in surface(rendered_card_block)
                represented_by_summary = bool(
                    field.get("claim_ids")
                    and set(field.get("claim_ids", [])) <= summaries_by_outcome.get(card.get("outcome_id"), set())
                )
                represented_by_public_item = bool(
                    field.get("claim_ids")
                    and set(field.get("claim_ids", [])) <= chapter_items_by_outcome.get(card.get("outcome_id"), set())
                )
                # A card projected into chronology must retain its typed
                # public label.  A card used only as the backing structure for
                # an overview sentence has no visible field label by design,
                # but its exact value still has to be present.
                label_rendered = (not chapter_block and exact_rendered) or surface(field_labels.get(name, name)) in surface(rendered_card_block)
                if field.get("value") and not ((label_rendered and exact_rendered) or represented_by_summary or represented_by_public_item):
                    errors.append("outcome_field_not_rendered")
                    render_trace.append({"node_id": f"outcome:{card_index}:{name}:{field_index}", "rendered": False, "reason": "surface_missing"})
                else:
                    reason = "represented_by_chapter_summary" if represented_by_summary and not exact_rendered else "represented_by_verified_detail" if represented_by_public_item and not exact_rendered else None
                    render_trace.append({"node_id": f"outcome:{card_index}:{name}:{field_index}", "rendered": True, "reason": reason})
    for section, section_items in document.get("sections", {}).items():
        for item_index, item in enumerate(section_items, 1):
            target_words = tokens(section_text.get(section, ""))
            original = next((source for source in items if source.get("public_id") == item.get("public_id")), None)
            edited_node_id = f"section:{section}:{item_index}"
            independently_supported = semantic_reviews_by_node.get(edited_node_id, {}).get("verdict") == "supported"
            changed = not original or original.get("text") != item.get("text")
            invalid_edit = (
                not original or not set(item.get("claim_ids", [])) <= source_ids
                or not set(item.get("evidence_ids", [])) <= evidence_for(item.get("claim_ids", []))
                or (changed and not independently_supported)
                or len(tokens(item.get("text")) - target_words) > 2
            )
            if invalid_edit:
                errors.append("section_not_rendered_from_verified_items")
                render_trace.append({"node_id": f"section:{section}:{item_index}", "rendered": False, "reason": "surface_missing_or_changed"})
            else:
                render_trace.append({"node_id": f"section:{section}:{item_index}", "rendered": True})
            for context_index, context in enumerate(item.get("context", []), 1):
                if not context.get("claim_ids") or not set(context["claim_ids"]) <= context_source_ids or not context.get("evidence_ids"):
                    errors.append("unsupported_section_context")
                relation = graph_relations.get(context.get("relation_id"))
                allowed_context_evidence = context_evidence_for(context.get("claim_ids", [])) | set((relation or {}).get("evidence_ids", []))
                if not set(context.get("evidence_ids", [])) <= allowed_context_evidence:
                    errors.append("section_context_evidence_outside_closure")
                if surface(context.get("text")) not in surface(section_text.get(section, "")):
                    errors.append("section_context_not_rendered")
                endpoints = {relation.get("source_claim_id"), relation.get("target_claim_id")} if relation else set()
                if (not relation or context.get("relation_type") != relation.get("type")
                        or not set(item.get("claim_ids", [])) & endpoints
                        or not set(context.get("claim_ids", [])) & endpoints):
                    errors.append("section_context_without_relation")
                render_trace.append({"node_id": f"context:{section}:{item_index}:{context_index}", "rendered": surface(context.get("text")) in surface(section_text.get(section, "")), "relation_id": context.get("relation_id")})
    for chapter in document.get("chronology", []):
        ranges = chapter.get("ranges", [])
        for item in chapter.get("items", []):
            start, end = float(item.get("start", 0)), float(item.get("end", item.get("start", 0)))
            if not any(float(value.get("start", 0)) <= start and end <= float(value.get("end", value.get("start", 0))) + .001 for value in ranges):
                errors.append("chapter_range_excludes_rendered_fact")
    cards_by_id = {card.get("outcome_id"): card for card in document.get("outcome_cards", [])}
    for chapter_index, chapter in enumerate(document.get("chronology", []), 1):
        chapter_cards = [cards_by_id[value] for value in chapter.get("outcome_ids", [chapter.get("outcome_id")]) if value in cards_by_id]
        fields = [field for card in chapter_cards for raw in card.get("fields", {}).values()
                  for field in (raw if isinstance(raw, list) else [raw] if raw else []) if field and field.get("value")]
        covered = {claim for field in fields for claim in field.get("claim_ids", [])}
        summary_covered = {claim for node in chapter.get("summary", []) for claim in node.get("claim_ids", [])}
        block = "\n".join(chapter_blocks.get(value, "") for value in chapter.get("outcome_ids", [chapter.get("outcome_id")]))
        for item_index, item in enumerate(chapter.get("items", []), 1):
            represented = (
                bool(chapter.get("summary"))
                and set(item.get("claim_ids", [])) <= summary_covered
            ) or (
                bool(fields) and (
                    set(item.get("claim_ids", [])) <= covered
                    or any(equivalent(item, field) for field in fields)
                )
            )
            rendered = surface(item.get("text")) in surface(block)
            render_trace.append({"node_id": f"chronology:{chapter_index}:{item_index}", "rendered": rendered,
                                 "reason": "represented_by_chapter_summary" if chapter.get("summary") and represented and not rendered else "represented_by_outcome_field" if represented and not rendered else None})
            if not represented and not rendered:
                errors.append("chronology_source_item_not_rendered")
    chronology_ids = {item.get("public_id") for chapter in document.get("chronology", []) for item in chapter.get("items", [])}
    minute_ids = {item.get("public_id") for item in items if item.get("section") == "minutes"}
    if chronology_ids != minute_ids:
        errors.append("chronology_item_loss")
    return {"passed": not errors, "errors": sorted(set(errors)), "title_claim_ids": title.get("claim_ids", []), "navigation_chapters": len(navigation), "render_trace": render_trace, "rendered_node_coverage": sum(item.get("rendered") for item in render_trace) / max(1, len(render_trace)), "hidden_nodes": [item for item in render_trace if not item.get("rendered")]}


def diff_public_items(previous, current):
    """Non-blocking shadow diff for release review and regression triage."""
    key = lambda x: (x.get("section"), tuple(x.get("claim_ids", [])))
    before, after = {key(x): x for x in previous or []}, {key(x): x for x in current or []}
    return {
        "added": [after[x] for x in sorted(after.keys() - before.keys())],
        "removed": [before[x] for x in sorted(before.keys() - after.keys())],
        "changed": [{"before": before[x], "after": after[x]} for x in sorted(before.keys() & after.keys()) if before[x].get("text") != after[x].get("text") or before[x].get("social_state") != after[x].get("social_state")],
        "section_counts_before": {section: sum(x.get("section") == section for x in previous or []) for section in sorted({x.get("section") for x in previous or []})},
        "section_counts_after": {section: sum(x.get("section") == section for x in current or []) for section in sorted({x.get("section") for x in current or []})},
    }


def alignment_score(premise, hypothesis):
    """Cheap independent alignment stage; ambiguous cases are escalated by caller."""
    tokens = lambda x: {v for v in re.findall(r"(?iu)[a-zа-яё0-9]+", x or "") if len(v) > 2}
    left, right = tokens(premise), tokens(hypothesis)
    entailment = len(left & right) / max(1, len(right))
    contradiction = 1.0 if bool(NEGATION_RE.search(premise or "")) != bool(NEGATION_RE.search(hypothesis or "")) else 0.0
    return {"entailment": entailment, "contradiction": contradiction, "ambiguous": entailment < .72 or contradiction > 0}


def qa_verify(text, plan):
    """Independent slot checks for who/quantity/condition/state questions."""
    assignment_claimed = bool(re.search(r"(?iu)\b(?:поручено|ответственн(?:ый|ая)|должен|владелец)\b", text or ""))
    allowed_values = list(plan.get("allowed_assignees", [])) + list(plan.get("allowed_speakers", []))
    allowed_people = set(re.findall(r"@[\w.-]+", " ".join(allowed_values)))
    mentioned_people = set(re.findall(r"@[\w.-]+", text or ""))
    checks = {"who": not assignment_claimed or (bool(mentioned_people) and mentioned_people.issubset(allowed_people)), "quantity": not plan.get("allowed_numbers") or set(NUMBER_RE.findall(text)).issubset(set(plan["allowed_numbers"])), "condition": not plan.get("conditions") or bool(CONDITION_RE.search(text)), "decision_state": not plan.get("decision_state") or not ("решено" in text.casefold() and "accepted" not in plan["decision_state"])}
    return {"passed": all(checks.values()), "checks": checks}
