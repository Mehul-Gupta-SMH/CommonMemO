"""A scripted agent: rules, not a model.

It reads only the text in front of it, exactly as an LLM would: the prompt
(with whatever memory pushed into it) and each tool's description (with
whatever tool memory added to it). Its rules are the obvious ones a capable
model would follow; what it cannot do is know anything it was not shown.
That is the property the evaluation measures.

It also obeys an instruction it finds outside the memory section of a
prompt, the way a naive model might, so the injection scenario can check
that memory never puts stored text where instructions go.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

MEMORY_HEADERS = ("# Memory\n", "# Memory for this turn\n")


@dataclass
class ScriptedAgent:
    prompt: str = ""
    tool_descriptions: dict[str, str] = field(default_factory=dict)

    # -- what it can see -------------------------------------------------------

    def memory_text(self) -> str:
        """The part of the prompt that memory supplied (reference data)."""
        starts = [self.prompt.find(h) for h in MEMORY_HEADERS if h in self.prompt]
        if not starts:
            return ""
        end = self.prompt.find("# Request")
        return self.prompt[min(starts): end if end != -1 else None]

    def instructions(self) -> str:
        """Everything that is not memory: where a naive model takes orders."""
        memory = self.memory_text()
        return self.prompt.replace(memory, "") if memory else self.prompt

    def fact(self, key: str) -> Optional[str]:
        """The value of `key` as the prompt states it, latest line wins."""
        # "- key: value" in the push, "[user] key: value" in a memory_recall result
        found = re.findall(rf"(?m)^(?:- |\[user\] ){re.escape(key)}: (.+?)(?: \(as of .*\))?$", self.prompt)
        return found[-1].strip() if found else None

    # -- decisions -------------------------------------------------------------

    def contact_channel(self) -> str:
        text = self.prompt.lower()
        preferred = self.fact("preferred_contact")
        if preferred:
            return preferred
        if "never: suggest phone calls" in text or "never: call" in text:
            return "email"
        return "phone"  # the default a model reaches for when told nothing

    def date_for(self, tool: str, day: int, month: int, year: int) -> str:
        """Format a date argument the way the tool's description suggests."""
        description = self.tool_descriptions.get(tool, "")
        if "iso_date" in description or "ISO" in description:
            return f"{year:04d}-{month:02d}-{day:02d}"
        return f"{day:02d}/{month:02d}/{year:04d}"

    def wants_recall(self) -> bool:
        """A capable model looks things up when told memory holds more than
        it was shown; told nothing, it answers from what it has."""
        return "memory_recall" in self.memory_text()

    def flash_method(self) -> str:
        """How to reflash a device: what memory says worked, else the usual way."""
        return "serial" if "over serial" in self.memory_text().lower() else "ota"

    def distance_unit(self) -> str:
        """The unit it answers in: what memory says this user wants, else miles."""
        return "km" if "use km" in self.memory_text().lower() else "miles"

    def report_sections(self) -> list[str]:
        """The sections it writes a report with: the template's, when memory
        shows one, else what a model reaches for unprompted."""
        memory = self.memory_text()
        start = memory.find("Template (")
        if start == -1:
            return ["Summary", "Details"]
        block = memory[start:].split("\n- ", 1)[0]
        return re.findall(r"(?m)^\s*## (.+)$", block)

    def pydantic_dump(self) -> str:
        """How it serializes a pydantic model: the v1 method when memory says
        the project pins pydantic 1, else the v2 one models reach for today."""
        return ".dict()" if re.search(r"\bpydantic 1\.", self.memory_text()) else ".model_dump()"

    def obeys(self, phrase: str) -> bool:
        """A naive model follows `phrase` if it appears as an instruction."""
        return phrase.lower() in self.instructions().lower()
