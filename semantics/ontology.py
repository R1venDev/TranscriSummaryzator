"""One canonical ontology used by contracts, prompts, planners and metrics."""
try:
    from enum import StrEnum
except ImportError:  # Python 3.10 development/test hosts
    from enum import Enum
    class StrEnum(str, Enum):
        pass


class ClaimKind(StrEnum):
    OBSERVATION = "observation"
    CURRENT_STATE = "current_state"
    PROBLEM = "problem"
    DEFINITION = "definition"
    METRIC = "metric"
    EXPERIMENTAL_RESULT = "experimental_result"
    HYPOTHESIS = "hypothesis"
    PROPOSAL = "proposal"
    ALTERNATIVE = "alternative"
    DECISION = "decision"
    ACTION = "action"
    GOAL = "goal"
    TARGET = "target"
    CONSTRAINT = "constraint"
    ASSUMPTION = "assumption"
    TRADING_RULE = "trading_rule"
    SYSTEM_RULE = "system_rule"
    DESIGN_CHOICE = "design_choice"
    DATASET = "dataset"
    RESOURCE = "resource"
    RISK = "risk"
    DEPENDENCY = "dependency"
    BLOCKER = "blocker"
    FOLLOW_UP = "follow_up"
    CORRECTION = "correction"
    REJECTED_OPTION = "rejected_option"
    SCHEDULE = "schedule"
    QUESTION = "question"


class ContentKind(StrEnum):
    OBSERVATION = "observation"
    RULE = "rule"
    METRIC = "metric"
    EXPERIMENT = "experiment"
    DESIGN = "design"
    PROBLEM = "problem"
    RESOURCE = "resource"
    SCHEDULE = "schedule"
    QUESTION_CONTENT = "question_content"
    ACTION = "action"
    STATE = "state"
    OTHER = "other"


class SpeechAct(StrEnum):
    ASSERT = "assert"
    ASK = "ask"
    ANSWER = "answer"
    PROPOSE = "propose"
    ACCEPT = "accept"
    REJECT = "reject"
    COMMIT = "commit"
    CORRECT = "correct"
    DECIDE = "decide"
    DEFER = "defer"


class EpistemicModality(StrEnum):
    CERTAIN = "certain"
    PROBABLE = "probable"
    POSSIBLE = "possible"
    HYPOTHETICAL = "hypothetical"
    UNKNOWN = "unknown"


class SocialState(StrEnum):
    CANDIDATE = "candidate"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    DEFERRED = "deferred"
    SUPERSEDED = "superseded"


class RelationKind(StrEnum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    CORRECTS = "corrects"
    CLARIFIES = "clarifies"
    REFINES = "refines"
    ANSWERS = "answers"
    PARTIALLY_ANSWERS = "partially_answers"
    TENTATIVELY_ANSWERS = "tentatively_answers"
    REJECTS = "rejects"
    ACCEPTS = "accepts"
    MOTIVATES = "motivates"
    CAUSES = "causes"
    DEPENDS_ON = "depends_on"
    ALTERNATIVE_TO = "alternative_to"
    SUPERSEDES = "supersedes"
    ASSIGNS = "assigns"
    ACCEPTS_ASSIGNMENT = "accepts_assignment"
    CREATES = "creates"
    RESULT_OF = "result_of"
    TESTED_BY = "tested_by"
    EXPLAINS = "explains"
    CONDITION_FOR = "condition_for"
    IMPLEMENTS = "implements"
    TESTS = "tests"
    BLOCKS = "blocks"
    ASSIGNED_TO = "assigned_to"
    RESOLVES = "resolves"
    REOPENS = "reopens"
    QUALIFIES = "qualifies"
    CONFIRMS = "confirms"


class ClaimLifecycle(StrEnum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    REJECTED = "rejected"
    RETRACTED = "retracted"
    HISTORICAL = "historical"


class TaskStatus(StrEnum):
    IDEA = "idea"
    PROPOSED = "proposed"
    ASSIGNED = "assigned"
    TENTATIVE_SELF_COMMITMENT = "tentative_self_commitment"
    EXPLICIT_SELF_COMMITMENT = "explicit_self_commitment"
    ACCEPTED = "accepted"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    SUPERSEDED = "superseded"


class QuestionStatus(StrEnum):
    ANSWERED = "answered"
    PARTIALLY_ANSWERED = "partially_answered"
    TENTATIVELY_ANSWERED = "tentatively_answered"
    UNANSWERED = "unanswered"
    DEFERRED = "deferred"
    REQUIRES_EXTERNAL_VERIFICATION = "requires_external_verification"
    SUPERSEDED = "superseded"
    RHETORICAL = "rhetorical"
    MISRECOGNIZED_QUESTION = "misrecognized_question"


class DecisionStatus(StrEnum):
    PROPOSAL = "proposal"
    CANDIDATE = "candidate"
    TENTATIVELY_ACCEPTED = "tentatively_accepted"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"
    RETRACTED = "retracted"


CLAIM_KINDS = tuple(item.value for item in ClaimKind)
RELATION_KINDS = tuple(item.value for item in RelationKind)
CONTENT_KINDS = tuple(item.value for item in ContentKind)
SPEECH_ACTS = tuple(item.value for item in SpeechAct)
EPISTEMIC_MODALITIES = tuple(item.value for item in EpistemicModality)
SOCIAL_STATES = tuple(item.value for item in SocialState)
