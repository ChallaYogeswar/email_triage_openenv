---
title: Email Triage OpenEnv
emoji: 📧
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
tags:
  - openenv
  - email
  - triage
  - reinforcement-learning
  - agent
  - real-world
license: mit
pinned: false
---
# 📧 Email Triage OpenEnv

A real-world **email triage environment** for training and evaluating AI agents on the daily workflow of a customer support or operations professional.

[![OpenEnv Compliant](https://img.shields.io/badge/OpenEnv-Compliant-green)](https://github.com/openenv)
[![HF Space](https://img.shields.io/badge/HuggingFace-Space-yellow)](https://huggingface.co/spaces)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue)](https://python.org)

---

## 🌍 Environment Description & Motivation

Every business runs on email. Triaging an inbox — classifying by priority, routing to the right team, replying appropriately, catching spam, and escalating crises — is a high-value cognitive task that humans do daily, yet is tedious and error-prone.

This environment places an AI agent in the role of a **customer support agent** processing a realistic inbox. The agent must:
- **Read** emails (using the `focus` action)
- **Classify** each by urgency and category
- **Reply** with the correct tone to customer queries
- **Escalate** critical issues to the right team (billing, legal, engineering, management)
- **Flag spam** accurately — including distinguishing phishing from legitimate security alerts (the hard task's key trap)
- **Avoid loops** and unnecessary actions

This is a genuinely useful domain for agent training: a strong performing agent here would have real-world deployment value.

---

## 📐 Observation Space

```
Observation
├── inbox_summary: List[EmailMeta]          # All emails + their current labels/status
│   ├── id: str
│   ├── subject: str
│   ├── sender: str  
│   ├── timestamp: str
│   ├── read: bool
│   ├── priority_label: urgent|high|normal|low|null
│   ├── category_label: billing|technical|general|spam|internal|null
│   ├── is_archived: bool
│   ├── is_spam_flagged: bool
│   ├── is_escalated: bool
│   └── has_reply: bool
├── current_email: EmailContent|null        # Full body of focused email
│   ├── id, subject, sender, timestamp
│   ├── body: str
│   ├── thread_id: str
│   └── attachments: List[str]
├── inbox_stats: InboxStats                 # Aggregate counts
├── step_number: int
├── task_objective: str                     # Human-readable task goal
├── last_action_result: str                 # Feedback from previous action
└── available_actions: List[str]
```

---

## ⚡ Action Space

| Action | Parameters | Description |
|--------|-----------|-------------|
| `focus` | `email_id` | Read an email (sets `current_email`) |
| `classify` | `priority`, `category` | Label email by urgency + type |
| `reply` | `body`, `tone` | Send a reply with tone: formal/friendly/apologetic/escalating |
| `escalate` | `escalate_to`, `note` | Route to: manager/legal/technical_team/billing_team |
| `flag_spam` | `confidence` | Mark as spam (penalizes false positives heavily) |
| `archive` | `reason` | Archive: resolved/irrelevant/spam |
| `mark_read` | — | Mark as read |
| `snooze` | `duration_hours` | Snooze for later |
| `noop` | — | Do nothing (small penalty) |

**Action format** (JSON POSTed to `/step`):
```json
{"action_type": "classify", "priority": "urgent", "category": "technical"}
{"action_type": "reply", "body": "We're on it!", "tone": "apologetic"}
{"action_type": "escalate", "escalate_to": "billing_team", "note": "Double charge"}
{"action_type": "focus", "email_id": "t1_e3"}
{"action_type": "flag_spam", "confidence": 0.97}
```

---

## 📋 Tasks

### Task 1 — Basic Inbox Triage *(Easy)*
- **Inbox size:** 10 emails
- **Max steps:** 30
- **Objective:** Focus and classify every email by `priority` + `category`
- **Grader:** Classification accuracy — percentage of (priority, category) pairs correct
- **Expected score range:** 0.4 – 0.9 for a capable LLM
- **Key challenge:** Distinguishing urgent vs. high vs. normal priority from email tone

### Task 2 — Reply and Escalate Critical Issues *(Medium)*
- **Inbox size:** 15 emails (superset of Task 1)
- **Max steps:** 50
- **Objective:** Classify all, reply to customer queries with appropriate tone, escalate critical issues, flag/archive spam
- **Grader:** Weighted composite — classification (40%), reply quality (30%), escalation routing (20%), spam detection (10%)
- **Expected score range:** 0.3 – 0.75
- **Key challenge:** Identifying the correct escalation target and reply tone for time-sensitive emails

### Task 3 — Full Customer Support Workflow *(Hard)*
- **Inbox size:** 20 emails (superset of Task 2)
- **Max steps:** 80
- **Objective:** All of Task 2, plus handling **traps** and **thread continuity**
- **Grader:** Task 2 scoring (70%) + trap handling (15%) + thread continuity (10%) + multi-action completeness (5%)
- **Expected score range:** 0.2 – 0.65
- **Key traps:**
  - `t3_e16`: A **legitimate security alert** from `security-noreply@ourcompany-platform.com` — flagging this as spam is a **hard penalty (-0.30)**
  - `t3_e17`: A **phishing email** disguised as an internal IT message from a `.ru` domain — must be correctly flagged
  - `t3_e18`: A **follow-up thread** to an earlier billing dispute — requires escalation AND reply
  - `t3_e20`: A **server outage** requiring escalation to BOTH `technical_team` AND `manager`

---

## 🏆 Reward Function

Rewards are shaped to provide signal throughout the episode (not just at the end):

| Event | Reward |
|-------|--------|
| Correct priority classification | +0.10 |
| Correct category classification | +0.10 |
| Correct reply tone | +0.08 |
| Non-trivial reply body | +0.04 |
| Correct escalation team | +0.12 |
| True positive spam flag | +0.10 |
| Correct archive of spam/resolved | +0.05 |
| Focus action (reads email) | +0.01 |
| **False positive spam flag** | **-0.20** |
| Wrong classification | -0.02 |
| Unnecessary escalation | -0.10 |
| Replying to spam | -0.15 |
| Premature archive (active email) | -0.05 |
| Loop penalty (>3 same action) | -0.15/repeat |
| Noop | -0.01 |

All rewards are clipped to `[-1.0, 1.0]`.

---

## 🚀 Setup & Usage

### Local Development

```bash
git clone <repo-url>
cd email-triage-env
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

### Docker

```bash
docker build -t email-triage-env .
docker run -p 7860:7860 email-triage-env
```

### Run Baseline Agent

```bash
export HF_TOKEN=hf_your_token_here
export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
export API_BASE_URL=https://router.huggingface.co/v1
export ENV_BASE_URL=http://localhost:7860

python inference.py
```

---

## 🌐 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health + endpoint listing |
| `/reset` | POST | Start new episode: `{"task_id": "task_1_basic_triage"}` |
| `/step` | POST | Execute action, get observation+reward |
| `/state` | GET | Full internal state |
| `/tasks` | GET | List all tasks with metadata |
| `/validate` | GET | OpenEnv spec compliance check |
| `/grade/{task_id}` | GET | Grade current episode |

### Example session

```python
import requests

BASE = "http://localhost:7860"

# Start episode
obs = requests.post(f"{BASE}/reset", json={"task_id": "task_1_basic_triage"}).json()

# Focus on first email
result = requests.post(f"{BASE}/step", json={"action_type": "focus", "email_id": "t1_e1"}).json()

# Classify it
result = requests.post(f"{BASE}/step", json={
    "action_type": "classify",
    "priority": "urgent",
    "category": "technical"
}).json()
print(result["reward"])  # {"value": 0.2, "reason": "classify: priority=urgent ✓, category=technical ✓", ...}

# Get validation
print(requests.get(f"{BASE}/validate").json())
# {"valid": true, "checks": {...}, "env_name": "email-triage-env"}
```

---

## 📊 Baseline Scores

Baseline agent: `meta-llama/Llama-3.1-8B-Instruct` via HF Inference Router

| Task | Score | Difficulty |
|------|-------|------------|
| task_1_basic_triage | ~0.62 | Easy |
| task_2_reply_and_escalate | ~0.48 | Medium |
| task_3_full_workflow | ~0.34 | Hard |
| **Average** | **~0.48** | |

---

## 🗂 Project Structure

```
email-triage-env/
├── app.py                    # FastAPI application (all OpenEnv endpoints)
├── inference.py              # Baseline inference script (mandatory, root level)
├── openenv.yaml              # OpenEnv spec metadata
├── requirements.txt          # Python dependencies
├── pyproject.toml            # Package metadata + entry point definition
├── uv.lock                   # Locked dependency tree (generated by uv)
├── Dockerfile                # Container definition for HF Spaces
├── README.md                 # This file
├── server/
│   ├── __init__.py
│   └── app.py                # OpenEnv multi-mode entry point (main() function)
├── env/
│   ├── __init__.py
│   ├── models.py             # Typed Pydantic models (Observation, Action, Reward)
│   └── environment.py        # Core OpenEnv engine (step/reset/state)
├── tasks/
│   ├── __init__.py
│   └── email_data.py         # Synthetic email dataset + ground truth labels
├── graders/
│   ├── __init__.py
│   └── graders.py            # Deterministic graders for all 3 tasks
└── tests/
    ├── __init__.py
    └── test_env.py           # 23 tests — spec compliance + grader correctness
```

---