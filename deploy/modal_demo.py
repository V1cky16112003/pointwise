"""The Pointwise demo page (demo/pointwise_app.py) as a small CPU function on Modal. It holds no model: it calls the
service in deploy/modal_app.py. Needs the Secrets `pointwise-key` (POINTWISE_API_KEY) and `pointwise-demo`
(POINTWISE_URL, the service's .modal.run address). See docs/modal-demo.md."""
import modal

app = modal.App("pointwise-demo")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("gradio==6.31.0", "requests==2.32.5")
    .add_local_dir("demo", "/root/demo")  # local-file lines stay last
)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("pointwise-key"), modal.Secret.from_name("pointwise-demo")],
    min_containers=0,
    max_containers=1,  # Gradio keeps a visitor's session in one container, so never run two
    scaledown_window=120,
)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def web():
    import sys

    sys.path.insert(0, "/root/demo")
    import gradio as gr
    from fastapi import FastAPI

    import pointwise_app

    return gr.mount_gradio_app(FastAPI(), pointwise_app.build().queue(default_concurrency_limit=2), path="/")
