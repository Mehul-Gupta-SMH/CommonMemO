"""Build a small, realistic memory file to look at: the README's screenshots come from it.

    python scripts/demo_memory.py demo.db
    commonmemo inspect --store sqlite:///demo.db --review --open

Offline and deterministic apart from timestamps: the local engine, and a
scripted stand-in for the model that learning would normally call.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path

from commonmemo import FactKey, Memory, ModelLearner
from commonmemo.agent_tools import build_agent_tools


def scripted_model(prompt: str) -> str:
    """Turns each correction in the session into a lesson."""
    session = prompt.split("\nSession", 1)[-1]
    return json.dumps({"lessons": [{"text": f"Use {right}, not {wrong}."}
                                   for wrong, right in re.findall(r"not '(.+?)' but '(.+?)'", session)]})


def tools(s):
    return {t.__name__: t for t in build_agent_tools(s)}


def build(path: Path) -> None:
    path.unlink(missing_ok=True)
    mem = Memory(f"sqlite:///{path}", project="shop", learner=ModelLearner(scripted_model), fact_keys=[
        FactKey("refund_window", "How long customers have to return an order", lifetime="90d", pinned=True),
        FactKey("shipping_cutoff", "Order before this time to ship the same day"),
    ])
    mem.learn("refund_window", "30 days")
    mem.learn("shipping_cutoff", "14:00 CET")
    mem.load_procedures([
        {"task": "refund_reply", "kind": "template", "name": "default",
         "use_when": "Replying to a customer about a refund",
         "body": "Thanks for reaching out.\n\nWhat happens next: ...\nWhen you'll see the money: ...\n\n- Support"},
        {"task": "refund_reply", "kind": "rule", "name": "timeline",
         "body": "Always give the refund timeline. Never quote internal ticket ids."},
        {"task": "refund_reply", "kind": "procedure", "name": "flow",
         "body": "1. verify_order: find the order\n2. check_window: is it within the refund window?\n"
                 "3. issue_refund: refund to the original payment method\n4. confirm: email the customer"},
    ])

    @mem.track(name="lookup_order")
    def lookup_order(order_id: str) -> str:
        if not order_id.startswith("A-"):
            raise ValueError(f"unknown order {order_id}: ids look like A-1234")
        return f"{order_id}: delivered 2026-10-02, 59.00 EUR"

    with mem.session("support-ada-1", user_id="ada", agent="support") as s:
        s.message("user", "I'd like to return my order 1042, it doesn't fit.")
        asyncio.run(s.push("I'd like to return my order 1042, it doesn't fit.", part="all"))
        try:
            lookup_order("1042")
        except ValueError:
            pass
        lookup_order("A-1042")
        s.remember("preferred_contact", "email")
        s.never("suggest phone calls", reason="ada asked for email only")
        s.message("assistant", "You can return it within 30 days; the refund takes 5 working days.")
        s.correct("working days", "calendar days", comment="ada counts weekends")
        s.feedback(1, "clear and quick")
        t = tools(s)
        t["memory_note_fact"]("refund_window", "90 days")  # the agent misread a promo banner
        t["memory_save_procedure"]("refund_reply", "rule", "apologise", "Open with an apology when the item "
                                   "arrived damaged.", "Replying to a customer about a damaged item", True)

    with mem.session("support-ada-2", user_id="ada", agent="claude-code") as s:
        s.message("user", "Where's my refund for A-1042?")
        asyncio.run(s.push("Where's my refund for A-1042?", part="all"))
        lookup_order("A-1042")
        s.message("assistant", "It was issued on Oct 6; expect it within 5 calendar days.")
        s.save_exemplar("refund_reply", "Thanks for reaching out, Ada.\n\nYour refund for A-1042 was issued on "
                        "Oct 6 to your card ending 4417. You'll see it within 5 calendar days.\n\n- Support",
                        name="ada-refund-issued", source="extracted")  # a learned candidate: waits for review

    with mem.session("support-bob-1", user_id="bob", agent="support") as s:
        s.message("user", "Can I still return the boots I bought last month?")
        asyncio.run(s.push("Can I still return the boots I bought last month?", part="all"))

    asyncio.run(mem.aclose())


if __name__ == "__main__":
    build(Path(sys.argv[1] if len(sys.argv) > 1 else "demo.db"))
