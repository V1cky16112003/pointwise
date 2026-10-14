# gstack
- For web browsing, use the `/browse` skill from gstack. Never use `mcp__claude-in-chrome__*` tools.
- Available gstack skills: `/office-hours`, `/plan-ceo-review`, `/plan-eng-review`, `/plan-design-review`, `/design-consultation`, `/design-shotgun`, `/design-html`, `/review`, `/ship`, `/land-and-deploy`, `/canary`, `/benchmark`, `/browse`, `/connect-chrome`, `/qa`, `/qa-only`, `/design-review`, `/scrape`, `/setup-browser-cookies`, `/setup-deploy`, `/setup-gbrain`, `/retro`, `/investigate`, `/document-release`, `/document-generate`, `/codex`, `/cso`, `/autoplan`, `/plan-devex-review`, `/devex-review`, `/careful`, `/freeze`, `/guard`, `/unfreeze`, `/gstack-upgrade`, `/learn`

## Skill routing
When a user's request matches an available skill, invoke it via the Skill tool. When in doubt:
- Ideas/brainstorming → `/office-hours`
- Strategy/scope → `/plan-ceo-review`, `/plan-eng-review`
- Design system/plan → `/design-consultation`, `/plan-design-review`, `/autoplan`
- Bugs/errors → `/investigate`
- QA/testing → `/qa`, `/qa-only`
- Review/diff → `/review`, `/design-review`
- Ship/deploy/PR → `/ship`, `/land-and-deploy`, `/context-save`, `/context-restore`
- Backlog-ready spec/issue → `/spec`

## Pointwise (Phase 4+) workflow
- Pointwise work happens in the worktree `../Finetuning-project-jev` on branch `jev` (pushed to `origin/jev`), not on `main`.
- Run log and design notes: `docs/designs/pointwise-phase4-plan.md`, `docs/designs/pointwise-phase4-review.md`.
- Training runs on Kaggle (user `v1cky1611`, kernel `mini-jev-phase-0`, GPU T4 x2, banking77 input). The notebook pulls the latest `jev` code; the user launches it by hand (Save Version → Save & Run All). Never launch or monitor runs unless asked.
- Each run's settings live in `scripts/kaggle_job.py`; give each run a new `DATA_DIR` (`/tmp/pointwise-data-runN`).
- Run `/code-review` before every commit; `git add -N` new files first so the review sees them.
- Gate: pooled held-out out-of-domain ECE (non-banking77, after temperature) ≤ 0.05, with no source's accuracy more than 3 points below run 2. Score held-out predictions with `src.pointwise.train.score_rows`.
- Reference design: Kev (github.com/jaredpalmer/kev). Kev's recorded negative results (label smoothing, Brier, focal loss) are not worth retrying.
