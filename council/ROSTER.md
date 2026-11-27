# Council roster (Mini Reality View)

| Seat | Role | delegate_task target |
|---|---|---|
| Astra | designer / reviewer; signs code changes against AGREED files | providerInstanceId `codex`, model `gpt-6.2-astra`, options {reasoningEffort: xhigh} |
| Fable | designer / reviewer | providerInstanceId `claudeAgent`, model `claude-fable-5-2`, options {effort: high} |
| Neon | GUARD / VERIFIER (owner directive 27 Nov ~19:00): checks what the builder is doing; verifies claims against code, logs and raw numbers; flags wrong or over-strong claims, missed issues, and anything making progress slow or suboptimal | providerInstanceId `antigravity`, model `gemini-4-neon`, options {thought_level: high} |

Rules
- Design questions: Astra + Fable answer independently, swap until they converge (AGREED files). Neon does not vote on designs;
  it audits the process and the claims (it may flag that a design decision rests on a wrong fact).
- Hourly review (scheduled task "Mini Reality View hourly council review"): all three seats; Neon answers the guard questions
  (claims vs evidence, missed issues, slowness, compute waste) and reads raw logs / result JSONs itself.
- Code: Astra signs code changes against the AGREED file before training; Neon may spot-check.
- All seats are read-only: no code edits, no cluster jobs, no commits. Builder acts on agreed points and logs them in PROGRESS.md.
- Three model families (OpenAI, Anthropic, Google) so one family's blind spots are less likely to pass unnoticed.
