"""Run the server: `python -m src.serve_main`. One process, because the GPU lock, the queue and the state cache live in it.

Why not plain `uvicorn src.serve:app`: uvicorn stops accepting connections when it gets SIGTERM, but /readyz would stay
green until shutdown finished, so a load balancer would keep sending traffic to an instance that is going away. Here the
signal also sets the drain flag, so /readyz fails at once and data requests on open connections get 503 + Retry-After
while work already running finishes. The whole shutdown stays within POINTWISE_SHUTDOWN_GRACE_S (default 25, at least 3):
1 s is held back (uvicorn sleeps a fixed 0.1 s outside its own timeout, and the process needs a moment to close), uvicorn
waits for open connections for 80% of the rest, in whole seconds, and the server's own drain gets exactly what is left.
Keep the grace below the orchestrator's kill timeout (Kubernetes terminationGracePeriodSeconds defaults to 30).

POINTWISE_HOST (default 127.0.0.1; use 0.0.0.0 in a container) and POINTWISE_PORT (default 8000).
"""
from __future__ import annotations

import os
import sys

import uvicorn

from src.serve_ops import DEFAULT_SHUTDOWN_GRACE_S, MIN_GRACE_S, OPS, env_float, env_int, split_grace


class DrainingServer(uvicorn.Server):
    def handle_exit(self, sig, frame) -> None:
        OPS.signalled = OPS.draining = True  # /readyz fails from this moment, before uvicorn stops accepting
        super().handle_exit(sig, frame)


def build_server() -> DrainingServer:
    grace = env_float("POINTWISE_SHUTDOWN_GRACE_S", DEFAULT_SHUTDOWN_GRACE_S, MIN_GRACE_S)  # the same checks as the server's own startup
    port = env_int("POINTWISE_PORT", 8000)
    if not 1 <= port <= 65535:
        sys.exit(f"FATAL: POINTWISE_PORT must be between 1 and 65535, got {port}.")
    uvicorn_s, OPS.drain_wait = split_grace(grace)  # the two waits add up to the grace period
    config = uvicorn.Config(
        "src.serve:app",
        host=os.environ.get("POINTWISE_HOST", "127.0.0.1"),
        port=port,
        workers=1,
        timeout_graceful_shutdown=uvicorn_s,
        access_log=False,  # src.serve logs one metadata line per request, with its request id
    )
    return DrainingServer(config)


if __name__ == "__main__":
    build_server().run()
