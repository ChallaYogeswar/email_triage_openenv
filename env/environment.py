"""
EmailTriageEnv — Core environment implementing the OpenEnv interface.
Exposes step() / reset() / state() methods consumed by the FastAPI layer.
"""

from __future__ import annotations

import copy
import time
from collections import Counter
from typing import Any, Dict, List, Optional

from env.models import (
    Action, ActionType, Observation, Reward, StepResult, EnvState,
    EmailMeta, EmailContent, InboxStats, Priority, Category,
)
from tasks.email_data import TASK_EMAILS, TASK_OBJECTIVES, TASK_MAX_STEPS, GROUND_TRUTH
from graders.graders import GRADERS


AVAILABLE_ACTIONS = [
    "focus", "classify", "reply", "archive", "escalate",
    "flag_spam", "mark_read", "snooze", "noop",
]

# Penalty: repeat same action on same email more than 3 times
LOOP_PENALTY = -0.15
LOOP_THRESHOLD = 3


class EmailTriageEnv:
    """
    OpenEnv-compliant Email Triage environment.
    """

    def __init__(self):
        self._task_id: Optional[str] = None
        self._emails: List[Dict[str, Any]] = []
        self._email_states: List[Dict[str, Any]] = []  # mutable per-episode state
        self._step_number: int = 0
        self._max_steps: int = 0
        self._done: bool = False
        self._cumulative_reward: float = 0.0
        self._action_history: List[Dict[str, Any]] = []
        self._focused_email_id: Optional[str] = None
        self._action_counter: Counter = Counter()

    # ──────────────────────────────────────────────────────────
    # OpenEnv API
    # ──────────────────────────────────────────────────────────

    def reset(self, task_id: str = "task_1_basic_triage") -> Observation:
        """Reset the environment for a given task. Returns initial observation."""
        if task_id not in TASK_EMAILS:
            raise ValueError(f"Unknown task_id: {task_id}. Valid: {list(TASK_EMAILS.keys())}")

        self._task_id = task_id
        self._emails = copy.deepcopy(TASK_EMAILS[task_id])
        self._email_states = [
            {
                "id": e["id"],
                "subject": e["subject"],
                "sender": e["sender"],
                "timestamp": e["timestamp"],
                "read": False,
                "priority_label": None,
                "category_label": None,
                "is_archived": False,
                "is_spam_flagged": False,
                "is_escalated": False,
                "has_reply": False,
                "thread_id": e.get("thread_id"),
            }
            for e in self._emails
        ]
        self._step_number = 0
        self._max_steps = TASK_MAX_STEPS[task_id]
        self._done = False
        self._cumulative_reward = 0.0
        self._action_history = []
        self._focused_email_id = self._emails[0]["id"] if self._emails else None
        self._action_counter = Counter()

        return self._build_observation("Environment reset. Begin processing inbox.")

    def step(self, action: Action) -> StepResult:
        """Execute one action. Returns (observation, reward, done, info)."""
        if self._done:
            obs = self._build_observation("Episode already done.")
            return StepResult(
                observation=obs,
                reward=Reward(value=0.0, reason="Episode is done", cumulative=self._cumulative_reward),
                done=True,
                info={"warning": "step() called after done=True"},
            )

        self._step_number += 1

        # Detect loops
        loop_key = f"{action.action_type}:{action.email_id or self._focused_email_id}"
        self._action_counter[loop_key] += 1
        loop_penalty = 0.0
        if self._action_counter[loop_key] > LOOP_THRESHOLD:
            loop_penalty = LOOP_PENALTY

        # Dispatch action
        reward_value, reward_reason, reward_components = self._dispatch(action)
        reward_value += loop_penalty
        if loop_penalty < 0:
            reward_components["loop_penalty"] = loop_penalty
            reward_reason += f" [Loop penalty: {loop_penalty}]"

        reward_value = round(max(-1.0, min(1.0, reward_value)), 4)
        self._cumulative_reward = round(self._cumulative_reward + reward_value, 4)

        # Record action
        self._action_history.append({
            "step": self._step_number,
            "action_type": action.action_type.value,
            "email_id": action.email_id or self._focused_email_id,
            "priority": action.priority.value if action.priority else None,
            "category": action.category.value if action.category else None,
            "tone": action.tone.value if action.tone else None,
            "body": action.body,
            "escalate_to": action.escalate_to.value if action.escalate_to else None,
            "confidence": action.confidence,
            "reward": reward_value,
            "timestamp": time.time(),
        })

        # Check termination
        done = self._step_number >= self._max_steps or self._check_task_complete()
        self._done = done

        obs = self._build_observation(reward_reason)
        reward = Reward(
            value=reward_value,
            reason=reward_reason,
            components=reward_components,
            cumulative=self._cumulative_reward,
        )

        info: Dict[str, Any] = {
            "step": self._step_number,
            "max_steps": self._max_steps,
            "task_id": self._task_id,
        }
        if done:
            final_grade = self._final_grade()
            info["final_score"] = final_grade["score"]
            info["grade_breakdown"] = final_grade["breakdown"]

        return StepResult(observation=obs, reward=reward, done=done, info=info)

    def state(self) -> EnvState:
        """Return full internal state (for debugging / spec compliance)."""
        return EnvState(
            task_id=self._task_id or "",
            step_number=self._step_number,
            max_steps=self._max_steps,
            done=self._done,
            cumulative_reward=self._cumulative_reward,
            emails=[EmailMeta(**s) for s in self._email_states],
            action_history=self._action_history,
            grader_partial_scores=self._partial_grade(),
        )

    # ──────────────────────────────────────────────────────────
    # Action dispatch
    # ──────────────────────────────────────────────────────────

    def _dispatch(self, action: Action):
        """Route action to handler. Returns (reward_value, reason, components)."""
        atype = action.action_type
        eid = action.email_id or self._focused_email_id

        handlers = {
            ActionType.FOCUS: self._handle_focus,
            ActionType.CLASSIFY: self._handle_classify,
            ActionType.REPLY: self._handle_reply,
            ActionType.ARCHIVE: self._handle_archive,
            ActionType.ESCALATE: self._handle_escalate,
            ActionType.FLAG_SPAM: self._handle_flag_spam,
            ActionType.MARK_READ: self._handle_mark_read,
            ActionType.SNOOZE: self._handle_snooze,
            ActionType.NOOP: self._handle_noop,
        }
        handler = handlers.get(atype, self._handle_noop)
        return handler(action, eid)

    def _get_email_state(self, eid: str) -> Optional[Dict]:
        return next((e for e in self._email_states if e["id"] == eid), None)

    def _get_email_content(self, eid: str) -> Optional[Dict]:
        return next((e for e in self._emails if e["id"] == eid), None)

    def _handle_focus(self, action: Action, eid: str):
        target_id = action.email_id
        if not target_id:
            return 0.0, "focus: no email_id specified", {}
        es = self._get_email_state(target_id)
        if not es:
            return -0.05, f"focus: email {target_id} not found", {"invalid_id": -0.05}
        self._focused_email_id = target_id
        es["read"] = True
        return 0.01, f"Focused on email {target_id}", {"focus": 0.01}

    def _handle_classify(self, action: Action, eid: str):
        if not eid:
            return -0.05, "classify: no email focused", {}
        es = self._get_email_state(eid)
        if not es:
            return -0.05, "classify: invalid email_id", {}

        gt = GROUND_TRUTH.get(eid, {})
        components = {}
        reward = 0.0
        reasons = []

        if action.priority:
            es["priority_label"] = action.priority.value
            if action.priority.value == gt.get("priority"):
                reward += 0.10
                components["priority_correct"] = 0.10
                reasons.append(f"priority={action.priority.value} ✓")
            else:
                reward -= 0.02
                components["priority_wrong"] = -0.02
                reasons.append(f"priority={action.priority.value} ✗")

        if action.category:
            es["category_label"] = action.category.value
            if action.category.value == gt.get("category"):
                reward += 0.10
                components["category_correct"] = 0.10
                reasons.append(f"category={action.category.value} ✓")
            else:
                reward -= 0.02
                components["category_wrong"] = -0.02
                reasons.append(f"category={action.category.value} ✗")

        return reward, f"classify: {', '.join(reasons)}", components

    def _handle_reply(self, action: Action, eid: str):
        if not eid:
            return -0.05, "reply: no email focused", {}
        es = self._get_email_state(eid)
        if not es:
            return -0.05, "reply: invalid email_id", {}

        gt = GROUND_TRUTH.get(eid, {})
        gt_actions = gt.get("gt_actions", [])
        expected_reply = next((a for a in gt_actions if a.startswith("reply:")), None)

        components = {}
        reward = 0.0

        # Replying to spam = bad
        if gt.get("category") == "spam":
            return -0.15, "reply: replying to spam!", {"spam_reply_penalty": -0.15}

        if expected_reply:
            expected_tone = expected_reply.split(":")[1]
            if action.tone and action.tone.value == expected_tone:
                reward += 0.08
                components["tone_correct"] = 0.08
            elif action.tone:
                reward += 0.02  # partial — at least replied
                components["tone_partial"] = 0.02
        else:
            # Reply not needed — slight penalty for noise
            reward -= 0.03
            components["unnecessary_reply"] = -0.03

        if action.body and len(action.body.strip()) > 20:
            reward += 0.04
            components["body_quality"] = 0.04

        es["has_reply"] = True
        return reward, f"reply: sent with tone={action.tone}", components

    def _handle_archive(self, action: Action, eid: str):
        if not eid:
            return -0.05, "archive: no email focused", {}
        es = self._get_email_state(eid)
        if not es:
            return -0.05, "archive: invalid email_id", {}

        gt = GROUND_TRUTH.get(eid, {})
        gt_actions = gt.get("gt_actions", [])
        expects_archive = any(a.startswith("archive:") for a in gt_actions)
        is_spam = gt.get("category") == "spam"

        if expects_archive or is_spam:
            es["is_archived"] = True
            return 0.05, "archive: correctly archived", {"archive": 0.05}
        else:
            es["is_archived"] = True
            return -0.05, "archive: archived email that needs action", {"premature_archive": -0.05}

    def _handle_escalate(self, action: Action, eid: str):
        if not eid:
            return -0.05, "escalate: no email focused", {}
        es = self._get_email_state(eid)
        if not es:
            return -0.05, "escalate: invalid email_id", {}

        gt = GROUND_TRUTH.get(eid, {})
        gt_actions = gt.get("gt_actions", [])
        expected_targets = [a.split(":")[1] for a in gt_actions if a.startswith("escalate:")]

        components = {}
        if not expected_targets:
            return -0.10, f"escalate: unnecessary escalation on {eid}", {"unnecessary_escalation": -0.10}

        target = action.escalate_to.value if action.escalate_to else ""
        if target in expected_targets:
            es["is_escalated"] = True
            return 0.12, f"escalate: correct routing to {target}", {"escalation_correct": 0.12}
        else:
            return 0.02, f"escalate: escalated but wrong team ({target})", {"escalation_wrong_team": 0.02}

    def _handle_flag_spam(self, action: Action, eid: str):
        if not eid:
            return -0.05, "flag_spam: no email focused", {}
        es = self._get_email_state(eid)
        if not es:
            return -0.05, "flag_spam: invalid email_id", {}

        gt = GROUND_TRUTH.get(eid, {})
        is_spam = gt.get("category") == "spam"

        if is_spam:
            es["is_spam_flagged"] = True
            return 0.10, f"flag_spam: true positive on {eid}", {"true_positive": 0.10}
        else:
            es["is_spam_flagged"] = True
            return -0.20, f"flag_spam: false positive! {eid} is NOT spam", {"false_positive": -0.20}

    def _handle_mark_read(self, action: Action, eid: str):
        if not eid:
            return 0.0, "mark_read: no email focused", {}
        es = self._get_email_state(eid)
        if es:
            es["read"] = True
        return 0.0, "mark_read: ok", {}

    def _handle_snooze(self, action: Action, eid: str):
        return 0.0, "snooze: email snoozed", {}

    def _handle_noop(self, action: Action, eid: str):
        return -0.01, "noop: no action taken", {"noop": -0.01}

    # ──────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────

    def _build_observation(self, last_action_result: str) -> Observation:
        stats = InboxStats(
            total=len(self._email_states),
            unread=sum(1 for e in self._email_states if not e["read"]),
            urgent=sum(1 for e in self._email_states if e.get("priority_label") == "urgent"),
            spam_flagged=sum(1 for e in self._email_states if e["is_spam_flagged"]),
            archived=sum(1 for e in self._email_states if e["is_archived"]),
            escalated=sum(1 for e in self._email_states if e["is_escalated"]),
            replied=sum(1 for e in self._email_states if e["has_reply"]),
        )

        current_email = None
        if self._focused_email_id:
            content = self._get_email_content(self._focused_email_id)
            if content:
                current_email = EmailContent(**content)

        return Observation(
            inbox_summary=[EmailMeta(**s) for s in self._email_states],
            current_email=current_email,
            inbox_stats=stats,
            step_number=self._step_number,
            task_objective=TASK_OBJECTIVES.get(self._task_id, ""),
            last_action_result=last_action_result,
            available_actions=AVAILABLE_ACTIONS,
        )

    def _check_task_complete(self) -> bool:
        """Heuristic: task is complete if every email has been classified."""
        return all(
            e.get("priority_label") and e.get("category_label")
            for e in self._email_states
        )

    def _final_grade(self) -> Dict[str, Any]:
        grader = GRADERS.get(self._task_id)
        if not grader:
            return {"score": 0.0, "breakdown": {}}
        return grader(self._email_states, self._action_history)

    def _partial_grade(self) -> Dict[str, float]:
        """Lightweight partial score for /state endpoint."""
        n = len(self._email_states)
        if n == 0:
            return {}
        classified = sum(
            1 for e in self._email_states
            if e.get("priority_label") and e.get("category_label")
        )
        return {"classification_progress": round(classified / n, 4)}
