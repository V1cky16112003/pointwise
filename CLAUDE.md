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

## Pointwise: build it production-ready (read SESSION_HANDOFF.md for the live state)
- **Objective (since 2026-10-17):** build Pointwise and make it production-ready. No more comparison or study work (Kev was dropped; the Phase 5 and 6 reports are done). Ship Phase 4 run 3 as it is (`state.pt` step 1541, T 1.516): do not retrain or retune unless the user says so; document limitations instead of fixing them.
- **Where:** all Pointwise code is in the worktree `../Finetuning-project-jev` on branch `jev` (remote `origin/jev`), not on `main`. Status: phases 0-6 done; production hardening steps 1-8 done (operator's guide `docs/production.md`); the production plan is complete. Reports in `reports/`, logs in `docs/designs/pointwise-plan.md`.
- **Tests:** `pytest tests -q` in a venv that has `peft` (`python3 -m venv --system-site-packages V && V/bin/pip install peft==0.21.2`). Without `peft` the serving tests skip silently and the count drops from about 283 to about 141. After writing tests, break the fix on purpose once to see a test fail.
- **Kaggle:** user `v1cky1611`, notebook `mini-jev-phase-0` (T4 x2, banking77 notebook and `pointwise-run3-weights` dataset as inputs). The notebook clones `jev` and runs `scripts/kaggle_job.py`, currently the Phase 6 latency job. **Never launch or monitor runs**: give the script change and a GPU-hour estimate, the user launches by hand and says when it has finished. A job that builds data gets a new `DATA_DIR` (`/tmp/pointwise-data-runN`).
- **Process:** run `/code-review` before every commit and **point it at the worktree path** (it once reviewed the wrong repo); `git add -N` new files first; commit trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; **push only when the user says so**. Score predictions only with `src.pointwise.train.score_rows`.
- **The Phase 4 gate** (pooled out-of-domain ECE after T <= 0.05) was missed at 0.0561; it is now a documented limitation, not a gate.
- Reference design Kev (github.com/jaredpalmer/kev); its recorded negative results (label smoothing, Brier, focal loss) are not worth retrying.
- The user is learning ML: explain a concept briefly the first time it comes up.
