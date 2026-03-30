"""
OpenEnv-compliant typed models for Email Triage Environment.
All models use Pydantic v2 for strict typing.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────

class Priority(str, Enum):
    URGENT = "urgent"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class Category(str, Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    GENERAL = "general"
    SPAM = "spam"
    INTERNAL = "internal"


class Tone(str, Enum):
    FORMAL = "formal"
    FRIENDLY = "friendly"
    APOLOGETIC = "apologetic"
    ESCALATING = "escalating"


class EscalationTarget(str, Enum):
    MANAGER = "manager"
    LEGAL = "legal"
    TECHNICAL_TEAM = "technical_team"
    BILLING_TEAM = "billing_team"


class ArchiveReason(str, Enum):
    RESOLVED = "resolved"
    IRRELEVANT = "irrelevant"
    SPAM = "spam"


class ActionType(str, Enum):
    CLASSIFY = "classify"
    REPLY = "reply"
    ARCHIVE = "archive"
    ESCALATE = "escalate"
    FOCUS = "focus"
    FLAG_SPAM = "flag_spam"
    MARK_READ = "mark_read"
    SNOOZE = "snooze"
    NOOP = "noop"


# ─────────────────────────────────────────────
# Email data structures
# ─────────────────────────────────────────────

class EmailMeta(BaseModel):
    id: str
    subject: str
    sender: str
    timestamp: str
    read: bool = False
    priority_label: Optional[Priority] = None
    category_label: Optional[Category] = None
    is_archived: bool = False
    is_spam_flagged: bool = False
    is_escalated: bool = False
    has_reply: bool = False
    thread_id: Optional[str] = None


class EmailContent(BaseModel):
    id: str
    subject: str
    sender: str
    timestamp: str
    body: str
    thread_id: Optional[str] = None
    attachments: List[str] = Field(default_factory=list)
    labels: List[str] = Field(default_factory=list)


class InboxStats(BaseModel):
    total: int
    unread: int
    urgent: int
    spam_flagged: int
    archived: int
    escalated: int
    replied: int


# ─────────────────────────────────────────────
# OpenEnv: Observation
# ─────────────────────────────────────────────

class Observation(BaseModel):
    """OpenEnv typed Observation model."""
    inbox_summary: List[EmailMeta]
    current_email: Optional[EmailContent] = None
    inbox_stats: InboxStats
    step_number: int
    task_objective: str
    last_action_result: Optional[str] = None
    available_actions: List[str] = Field(default_factory=list)


# ─────────────────────────────────────────────
# OpenEnv: Action
# ─────────────────────────────────────────────

class Action(BaseModel):
    """OpenEnv typed Action model."""
    action_type: ActionType
    # classify
    priority: Optional[Priority] = None
    category: Optional[Category] = None
    # reply
    body: Optional[str] = None
    tone: Optional[Tone] = None
    # archive
    reason: Optional[ArchiveReason] = None
    # escalate
    escalate_to: Optional[EscalationTarget] = None
    note: Optional[str] = None
    # focus
    email_id: Optional[str] = None
    # flag_spam
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    # snooze
    duration_hours: Optional[int] = None


# ─────────────────────────────────────────────
# OpenEnv: Reward
# ─────────────────────────────────────────────

class Reward(BaseModel):
    """OpenEnv typed Reward model."""
    value: float = Field(ge=-1.0, le=1.0)
    reason: str
    components: Dict[str, float] = Field(default_factory=dict)
    cumulative: float = 0.0


# ─────────────────────────────────────────────
# Step result
# ─────────────────────────────────────────────

class StepResult(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────
# State (full internal state, for /state endpoint)
# ─────────────────────────────────────────────

class EnvState(BaseModel):
    task_id: str
    step_number: int
    max_steps: int
    done: bool
    cumulative_reward: float
    emails: List[EmailMeta]
    action_history: List[Dict[str, Any]]
    grader_partial_scores: Dict[str, float]
