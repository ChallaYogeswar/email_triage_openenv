"""
Inference Script — Email Triage OpenEnv
=======================================
Baseline agent using OpenAI API client.
Runs all 3 tasks and produces reproducible scores.

Required environment variables:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

Usage:
    python inference.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import textwrap
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")

MAX_STEPS = {
    "task_1_basic_triage": 30,
    "task_2_reply_and_escalate": 50,
    "task_3_full_workflow": 80,
}
TEMPERATURE = 0.1
MAX_TOKENS = 400

TASKS = ["task_1_basic_triage", "task_2_reply_and_escalate", "task_3_full_workflow"]

# ──────────────────────────────────────────────────────────────
# System prompt
# ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
You are an AI email triage agent. Your job is to process an email inbox efficiently.

## Available Actions (reply with EXACTLY ONE JSON object):

1. Focus on an email to read it:
   {"action_type": "focus", "email_id": "<id>"}

2. Classify an email by priority and category:
   {"action_type": "classify", "priority": "<urgent|high|normal|low>", "category": "<billing|technical|general|spam|internal>"}

3. Reply to an email:
   {"action_type": "reply", "body": "<your reply text>", "tone": "<formal|friendly|apologetic|escalating>"}

4. Flag an email as spam:
   {"action_type": "flag_spam", "confidence": 0.95}

5. Escalate an email to a team:
   {"action_type": "escalate", "escalate_to": "<manager|legal|technical_team|billing_team>", "note": "<reason>"}

6. Archive a resolved email:
   {"action_type": "archive", "reason": "<resolved|irrelevant|spam>"}

7. Do nothing:
   {"action_type": "noop"}

## Strategy:
1. Focus on each email first to read its content.
2. Classify EVERY email (priority + category).
3. For urgent/high priority: reply if it's a customer query, escalate to the right team if critical.
4. Flag and archive spam. Do NOT flag legitimate security alerts as spam.
5. Prioritize urgent emails before normal/low ones.

## Important rules:
- Spam indicators: suspicious domains (.ru, .biz), prize claims, phishing patterns, job scams.
- Legitimate security alerts come from your OWN company domain — do NOT flag those as spam.
- Legal threats → escalate:legal
- Technical emergencies → escalate:technical_team  
- Billing disputes/payment issues → escalate:billing_team
- Anything needing management attention → escalate:manager

Respond ONLY with a valid JSON object. No explanations, no markdown.
""").strip()


# ──────────────────────────────────────────────────────────────
# Env client helpers
# ──────────────────────────────────────────────────────────────

def env_reset(task_id: str) -> Dict[str, Any]:
    r = requests.post(f"{ENV_BASE_URL}/reset", json={"task_id": task_id}, timeout=30)
    r.raise_for_status()
    return r.json()


def env_step(action: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.post(f"{ENV_BASE_URL}/step", json=action, timeout=30)
    r.raise_for_status()
    return r.json()


def env_grade(task_id: str) -> Dict[str, Any]:
    r = requests.get(f"{ENV_BASE_URL}/grade/{task_id}", timeout=30)
    r.raise_for_status()
    return r.json()


# ──────────────────────────────────────────────────────────────
# Agent logic
# ──────────────────────────────────────────────────────────────

def build_user_prompt(obs: Dict[str, Any]) -> str:
    inbox = obs.get("inbox_summary", [])
    stats = obs.get("inbox_stats", {})
    current = obs.get("current_email")
    task_obj = obs.get("task_objective", "")
    last_result = obs.get("last_action_result", "")
    step = obs.get("step_number", 0)

    # Build inbox listing
    inbox_lines = []
    for e in inbox:
        status_parts = []
        if not e.get("read"):
            status_parts.append("UNREAD")
        if e.get("priority_label"):
            status_parts.append(f"priority={e['priority_label']}")
        if e.get("category_label"):
            status_parts.append(f"category={e['category_label']}")
        if e.get("is_spam_flagged"):
            status_parts.append("SPAM")
        if e.get("is_escalated"):
            status_parts.append("ESCALATED")
        if e.get("has_reply"):
            status_parts.append("REPLIED")
        status = " | ".join(status_parts) if status_parts else "untouched"
        inbox_lines.append(f"  [{e['id']}] \"{e['subject']}\" from {e['sender']} — {status}")

    inbox_str = "\n".join(inbox_lines)

    current_str = ""
    if current:
        current_str = (
            f"\n## Currently focused email [{current['id']}]:\n"
            f"Subject: {current['subject']}\n"
            f"From: {current['sender']}\n"
            f"Time: {current['timestamp']}\n"
            f"Body:\n{current['body']}\n"
        )

    unprocessed = [
        e for e in inbox
        if not e.get("priority_label") or not e.get("category_label")
    ]
    next_focus = unprocessed[0]["id"] if unprocessed else None

    prompt = f"""## Task: {task_obj}

## Step: {step}
## Last result: {last_result}

## Inbox ({stats.get('total', 0)} emails, {stats.get('unread', 0)} unread):
{inbox_str}

## Stats: urgent={stats.get('urgent',0)} spam={stats.get('spam_flagged',0)} archived={stats.get('archived',0)} escalated={stats.get('escalated',0)} replied={stats.get('replied',0)}
{current_str}
## Unprocessed emails remaining: {len(unprocessed)}
{"## Suggested next: focus on " + next_focus if next_focus and not current else ""}

What is your next action? Reply with ONE JSON object.
"""
    return prompt.strip()


def parse_action(response_text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON action from model response."""
    text = response_text.strip()
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try extracting from markdown code block
    import re
    match = re.search(r'\{[^{}]+\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


def run_task(client: OpenAI, task_id: str) -> Dict[str, Any]:
    """Run one full episode on a task. Returns final grade."""
    print(f"\n{'='*60}", flush=True)
    print(f"  TASK: {task_id}", flush=True)
    print(f"{'='*60}", flush=True)

    # ── Required structured output: START block ──
    print(f"[START] task={task_id}", flush=True)

    reset_result = env_reset(task_id)
    obs = reset_result["observation"]
    max_steps = reset_result["max_steps"]
    done = False
    step = 0
    cumulative_reward = 0.0

    while not done and step < max_steps:
        step += 1
        user_prompt = build_user_prompt(obs)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
            response_text = completion.choices[0].message.content or ""
        except Exception as exc:
            print(f"  [Step {step}] LLM error: {exc} — using noop", flush=True)
            response_text = '{"action_type": "noop"}'

        action = parse_action(response_text)
        if not action:
            print(f"  [Step {step}] Could not parse action — using noop", flush=True)
            action = {"action_type": "noop"}

        print(f"  [Step {step:2d}] Action: {json.dumps(action)[:80]}", flush=True)

        try:
            result = env_step(action)
        except Exception as exc:
            print(f"  [Step {step}] env.step() error: {exc}", flush=True)
            break

        reward = result.get("reward", {})
        reward_val = reward.get("value", 0.0)
        cumulative_reward += reward_val
        done = result.get("done", False)
        obs = result.get("observation", obs)

        reason = reward.get("reason", "")
        print(f"          Reward: {reward_val:+.3f} | Cumul: {cumulative_reward:+.3f} | {reason[:60]}", flush=True)

        # ── Required structured output: STEP block ──
        print(f"[STEP] step={step} reward={reward_val:.4f}", flush=True)

        if done:
            info = result.get("info", {})
            final_score = info.get("final_score", None)
            if final_score is not None:
                print(f"\n  ✓ Episode done. Final score: {final_score:.4f}", flush=True)
                # ── Required structured output: END block ──
                print(f"[END] task={task_id} score={final_score:.4f} steps={step}", flush=True)
                return {"task_id": task_id, "score": final_score, "breakdown": info.get("grade_breakdown", {}), "steps": step}

    # Episode ended by step limit — get grade
    grade = env_grade(task_id)
    score = grade.get("score", 0.0)
    print(f"\n  ✓ Episode ended (steps={step}). Final score: {score:.4f}", flush=True)
    # ── Required structured output: END block ──
    print(f"[END] task={task_id} score={score:.4f} steps={step}", flush=True)
    return {"task_id": task_id, "score": score, "breakdown": grade.get("breakdown", {}), "steps": step}


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    if not API_KEY:
        print("ERROR: HF_TOKEN or API_KEY environment variable not set.")
        sys.exit(1)

    print(f"Email Triage OpenEnv — Baseline Inference")
    print(f"  Model:   {MODEL_NAME}")
    print(f"  API:     {API_BASE_URL}")
    print(f"  Env:     {ENV_BASE_URL}")
    print()

    # Verify env is up
    try:
        r = requests.get(f"{ENV_BASE_URL}/validate", timeout=10)
        r.raise_for_status()
        val = r.json()
        if not val.get("valid"):
            print(f"WARNING: env validation failed: {val}")
        else:
            print(f"✓ Environment validated: {val['env_name']} v{val['version']}")
    except Exception as e:
        print(f"ERROR: Cannot reach environment at {ENV_BASE_URL}: {e}")
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    results = []
    for task_id in TASKS:
        try:
            result = run_task(client, task_id)
            results.append(result)
        except Exception as e:
            print(f"ERROR running {task_id}: {e}")
            results.append({"task_id": task_id, "score": 0.0, "error": str(e)})
        time.sleep(1)  # brief pause between tasks

    # Summary
    print(f"\n{'='*60}")
    print("  BASELINE SCORES SUMMARY")
    print(f"{'='*60}")
    total = 0.0
    for r in results:
        score = r.get("score", 0.0)
        total += score
        status = "✓" if score >= 0.5 else "✗"
        print(f"  {status} {r['task_id']:<35} score={score:.4f}  steps={r.get('steps','?')}")

    avg = total / len(results) if results else 0.0
    print(f"\n  Average score: {avg:.4f}")
    print(f"{'='*60}\n")

    # Write scores to file for reproducibility
    with open("baseline_scores.json", "w") as f:
        json.dump({"results": results, "average": avg}, f, indent=2)
    print("Scores written to baseline_scores.json")


if __name__ == "__main__":
    main()