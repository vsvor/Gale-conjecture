# Working locally with Codex

This repository supplies project code and guidance, not a Codex installation or an account.
The setup below is a local workflow for the supplied software. No Codex session was
started or authenticated during repository assembly.

## Human setup

Review `scripts/setup.sh` and the dependency files, run setup, activate `.venv`, and
make a local Git checkpoint. Install/authenticate Codex separately using the official
[CLI instructions](https://developers.openai.com/codex/cli/), then run `codex` in the
repository root. The CLI operates on the local working directory and prompts for
sign-in when necessary. Do not put an API key or login state in this repository.

## Project instructions

Codex reads `AGENTS.md` instructions hierarchically. A more local file may specialize
an enclosing one, and `AGENTS.override.md` can take precedence over `AGENTS.md`. Review
applicable user/global instructions too; the repository cannot override higher-level
policies. The root file is deliberately short and points to detailed documentation.
See the official [AGENTS.md guide](https://developers.openai.com/codex/guides/agents-md/).

## Included project configuration

`.codex/config.toml` requests workspace-write sandboxing, on-request approvals, no
sandbox network access, and disabled web search. It specifies no model, credentials,
plugins, hooks, or MCP servers. Project configuration is loaded only for a trusted
project, and user/organization constraints can supersede it. Review it before trusting
the project. See [configuration basics](https://developers.openai.com/codex/config-basic/).

Dependency installation is a separate human step. Keeping subprocess network access
off does not make the Codex model run offline or eliminate its own service/authentication
requirements. Do not enable unsafe modes to solve a blocked installation or test.

Official guidance above was checked on 2026-09-11. Use the live official pages for
installation changes; this document does not pin a Codex binary or a model version.

## First task

Paste this after opening the project:

> Read AGENTS.md and docs/architecture.md. Run .venv/bin/python scripts/doctor.py,
> then the quick tests and the tiny smoke run. Do not install anything, modify
> source/data, or launch a record search. Report the ownership map, results and
> any gaps. Respect the repository compute limits.

Then choose a task from [tasks/README.md](tasks/README.md). A useful implementation
request specifies a module, invariant, regression case, runtime budget, and deliverable.
Do not start with “improve everything until a new record is found.”

## Separate editing from experiments

For each idea: implement a narrow patch, review the diff, test it, commit it, then
run bounded experiments against that revision. For parallel coding agents use separate
Git branches/worktrees. For parallel scientific runs use distinct output directories
and deterministic distinct seeds. A worker pool inside an experiment is not permission
to launch additional pools from several agents without accounting for the total.

`run_experiment.py` records source hashes before and after execution and flags changes.
It does not freeze a mutable working tree or perform a security sandbox. A clean Git
commit or isolated worktree remains the preferred basis for a comparison.

## Completion report expected from an agent

Report the code change and why it belongs in that layer, tests/smokes actually run,
any skips or failures, and the saved experiment path. Keep scientific conclusions
separate: state lower/upper bounds, unresolved solver queries, and which coordinates
were verified. Do not call a completed process a verified record without its witness
and geometric checks.

Never upload this repository or publish its data merely because Codex can access a
remote tool. No external connector is needed for this local project.
