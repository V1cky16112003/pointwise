# The demo page on Modal

`demo/pointwise_app.py` is a Gradio page. `deploy/modal_demo.py` runs it as a small CPU function on Modal
(`pointwise-demo`). It holds no model: each click calls the service from `docs/modal.md` with the key, which only the
page's server sees. The demo never touches the GPU function except through that call.

## Deploy (after `docs/modal.md` works)

```bash
# the service's address, the .modal.run one from `modal deploy deploy/modal_app.py`
modal secret create pointwise-demo POINTWISE_URL=https://<you>--pointwise-serve.modal.run
modal serve deploy/modal_demo.py      # a temporary URL; open it, run each example, Ctrl-C
modal deploy deploy/modal_demo.py     # the permanent URL to share
```

`pointwise-key` already exists from `docs/modal.md`; the demo reuses it. Do not paste the key anywhere.

## Check it

1. Open the URL in a private window. Run each of the three examples.
2. Wait until the GPU has gone idle (over a minute with no requests), then run once more: the page should show
   "Waking up the GPU, about 20 seconds...", then the cards.
3. View the page source and the browser's network tab: neither may contain the key.
4. Run it 7 times in a minute: the 7th must say "Rate limit".

## Limits and costs

- Each visitor IP gets 6 runs a minute (`DEMO_RATE_LIMIT`). The limit is in memory and per container, and Modal's proxy
  may hide the real IP, so treat it as a speed bump. The spend limit in the Modal dashboard is the hard stop.
- At most 4 questions, 10 options per question and 3,000 characters of state per run.
- `max_containers=1`: Gradio keeps a visitor's session in one container. Do not raise it.
- Anyone with the URL can wake the GPU; keep sharing it to people you mean to.
- The three examples were chosen from Pointwise's strongest tasks and were not run before release. If one looks poor,
  edit `PRESETS` in `demo/pointwise_app.py` and redeploy.
