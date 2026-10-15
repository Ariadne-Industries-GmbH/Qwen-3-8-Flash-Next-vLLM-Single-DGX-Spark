# Reasoning effort (client side, NOT in the deployment)

The Qwen3.8-Flash-Next chat template accepts **only `xhigh` (default),
`medium` and `low`** — `high` is passed by vLLM request validation but fails
in the Jinja template with `Unexpected reasoning effort high`. Clients must
map `high` → `xhigh` before sending. The engine profiles in
`model_config.json` (other than `qwen-3-8-flash-*`) therefore use only
`xhigh`/`medium`/`low`/`none`:

| Profile | reasoning_effort | enable_thinking | Use |
|---|---|---|---|
| `qwen-3-8-flash-xhigh` | `xhigh` | true | autonomous agents (maximum quality, longest reasoning) |
| `qwen-3-8-flash-medium` | `medium` | true | neutral mode (inserts no effort instruction) |
| `qwen-3-8-flash-low` | `low` | true | short, focused reasoning (fast background/extraction jobs) |
| `qwen-3-8-flash-off` | `none` | false | no thinking (tool-only calls, pure extraction) |

## preserve_thinking

Important: `preserve_thinking` (active by default in the Qwen3.8 template)
works only if the engine sends the earlier `reasoning` back
(`replay_history: true`, `history_field: reasoning`).
