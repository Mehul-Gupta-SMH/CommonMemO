// CommonMemo memory for OpenCode. Save as .opencode/plugin/commonmemo.js
// (`commonmemo setup opencode` prints this file).
//
// A thin shim: each OpenCode event is translated into the hook payload
// Claude Code and Codex send, and handed to `commonmemo hook --harness
// opencode` on stdin, so all the logic stays in Python. Configure it with the
// COMMONMEMO_* environment variables (COMMONMEMO_USER, COMMONMEMO_STORE, ...);
// COMMONMEMO_COMMAND overrides the command, as a JSON array.
//
//   chat.message                        -> UserPromptSubmit (memory for the turn is kept)
//   tool.execute.after                  -> PostToolUse (memory the result brings up is kept for the turn)
//   experimental.chat.system.transform  -> pushes memory into the system prompt
//                                          (SessionStart the first time per session)
//   experimental.session.compacting     -> PreCompact, and keeps stable memory in the summary
//   event session.deleted               -> SessionEnd
//
// Replies are not captured: OpenCode has no event with the final reply text.

import { spawn } from "node:child_process";

const COMMAND = process.env.COMMONMEMO_COMMAND
  ? JSON.parse(process.env.COMMONMEMO_COMMAND)
  : ["commonmemo", "hook", "--harness", "opencode"];

function hook(payload) {
  return new Promise((resolve) => {
    let out = "";
    let child;
    try {
      child = spawn(COMMAND[0], COMMAND.slice(1), { stdio: ["pipe", "pipe", "inherit"] });
    } catch {
      return resolve(null);
    }
    child.stdout.on("data", (chunk) => (out += chunk));
    child.on("error", () => resolve(null)); // memory must never break OpenCode
    child.on("close", () => {
      try {
        resolve(out.trim() ? JSON.parse(out).hookSpecificOutput?.additionalContext ?? null : null);
      } catch {
        resolve(null);
      }
    });
    child.stdin.end(JSON.stringify(payload));
  });
}

export const CommonMemo = async ({ directory }) => {
  const started = new Map(); // sessionID -> what SessionStart pushed
  const turn = new Map(); // sessionID -> memory for the latest prompt
  const steps = new Map(); // sessionID -> memory pushed mid-run since that prompt
  const base = (sessionID, event) => ({ session_id: sessionID, cwd: directory, hook_event_name: event });

  return {
    "chat.message": async (input, output) => {
      const prompt = (output.parts ?? [])
        .filter((p) => p.type === "text" && !p.synthetic)
        .map((p) => p.text)
        .join("\n");
      turn.set(input.sessionID, await hook({ ...base(input.sessionID, "UserPromptSubmit"), prompt }));
      steps.set(input.sessionID, []);
    },
    "tool.execute.after": async (input, output) => {
      const more = await hook({
        ...base(input.sessionID, "PostToolUse"),
        tool_name: input.tool,
        tool_input: input.args ?? {},
        tool_response: output.output,
        tool_use_id: input.callID,
      });
      if (more) steps.set(input.sessionID, [...(steps.get(input.sessionID) ?? []), more]);
    },
    "experimental.chat.system.transform": async (input, output) => {
      if (!input.sessionID) return;
      if (!started.has(input.sessionID)) {
        started.set(input.sessionID, await hook({ ...base(input.sessionID, "SessionStart"), source: "startup" }));
      }
      for (const text of [started.get(input.sessionID), turn.get(input.sessionID), ...(steps.get(input.sessionID) ?? [])]) {
        if (text) output.system.push(text);
      }
    },
    "experimental.session.compacting": async (input, output) => {
      await hook({ ...base(input.sessionID, "PreCompact"), trigger: "auto" });
      const stable = started.get(input.sessionID);
      if (stable) output.context.push(stable);
    },
    event: async ({ event }) => {
      if (event.type === "session.deleted") {
        const id = event.properties.info.id;
        await hook({ ...base(id, "SessionEnd"), reason: "other" });
        started.delete(id);
        turn.delete(id);
        steps.delete(id);
      }
    },
  };
};
