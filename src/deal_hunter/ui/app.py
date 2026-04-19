import logging
import queue
import threading
import time

import gradio as gr

from deal_hunter.config import settings
from deal_hunter.framework import DealAgentFramework
from deal_hunter.ui.log_formatter import reformat


class QueueHandler(logging.Handler):
    """Logging handler that pushes formatted records onto a queue."""

    def __init__(self, log_queue: queue.Queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record: logging.LogRecord) -> None:
        self.log_queue.put(self.format(record))


def html_for(log_data: list[str]) -> str:
    """Render the last ~18 log lines inside a scrollable div."""
    output = "<br>".join(log_data[-18:])
    return f"""
    <div id="scrollContent" style="height: 400px; overflow-y: auto; border: 1px solid #ccc; background-color: #222229; padding: 10px;">
    {output}
    </div>
    """


def setup_logging(log_queue: queue.Queue) -> None:
    root = logging.getLogger()
    # Drop any QueueHandler from a previous run so we don't leak handlers
    # (and orphaned queues) on every Gradio timer tick.
    for old in [h for h in root.handlers if isinstance(h, QueueHandler)]:
        root.removeHandler(old)

    handler = QueueHandler(log_queue)
    formatter = logging.Formatter(
        "[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S %z",
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)
    root.setLevel(logging.INFO)


class App:
    def __init__(self):
        self.framework: DealAgentFramework | None = None
        self._framework_lock = threading.Lock()

    def get_framework(self) -> DealAgentFramework:
        # Double-checked locking: cheap fast path once the framework is built,
        # serialised slow path so two threads can't both call PersistentClient
        # for the same path (chromadb's SharedSystemClient registry is not
        # safe under that race and leaves a half-initialised RustBindingsAPI
        # behind).
        if self.framework is None:
            with self._framework_lock:
                if self.framework is None:
                    self.framework = DealAgentFramework()
        return self.framework

    def run(self) -> None:
        with gr.Blocks(title="Deal Hunter", fill_width=True) as ui:
            log_data = gr.State([])

            def table_for(opps):
                return [
                    [
                        opp.deal.product_description,
                        f"${opp.deal.price:.2f}",
                        f"${opp.estimate:.2f}",
                        f"${opp.discount:.2f}",
                        opp.deal.url,
                    ]
                    for opp in opps
                ]

            def update_output(log_data, log_queue, result_queue):
                initial_result = table_for(self.get_framework().memory)
                final_result = None
                while True:
                    try:
                        message = log_queue.get_nowait()
                        log_data.append(reformat(message))
                        yield (
                            log_data,
                            html_for(log_data),
                            final_result or initial_result,
                        )
                    except queue.Empty:
                        try:
                            final_result = result_queue.get_nowait()
                            yield (
                                log_data,
                                html_for(log_data),
                                final_result or initial_result,
                            )
                        except queue.Empty:
                            if final_result is not None:
                                break
                            time.sleep(0.1)

            def do_run():
                opportunities = self.get_framework().run()
                return table_for(opportunities)

            def run_with_logging(initial_log_data):
                log_queue: queue.Queue = queue.Queue()
                result_queue: queue.Queue = queue.Queue()
                setup_logging(log_queue)

                def worker():
                    result_queue.put(do_run())

                thread = threading.Thread(target=worker, daemon=True)
                thread.start()

                for log_data, output, final_result in update_output(
                    initial_log_data, log_queue, result_queue
                ):
                    yield log_data, output, final_result

            def do_select(selected_index: gr.SelectData):
                framework = self.get_framework()
                if framework.planner is None:
                    return
                row = selected_index.index[0]
                if row >= len(framework.memory):
                    return
                opportunity = framework.memory[row]
                framework.planner.messenger.alert(opportunity)

            with gr.Row():
                gr.Markdown(
                    '<div style="text-align: center;font-size:24px">'
                    "<strong>Deal Hunter</strong> Autonomous agent framework that hunts for deals"
                    "</div>"
                )
            with gr.Row():
                gr.Markdown(
                    '<div style="text-align: center;font-size:14px">'
                    "A fine-tuned LLM on Modal and a RAG pipeline with a frontier model "
                    "collaborate to send push notifications about online deals."
                    "</div>"
                )
            with gr.Row():
                opportunities_dataframe = gr.Dataframe(
                    headers=[
                        "Deals found so far",
                        "Price",
                        "Estimate",
                        "Discount",
                        "URL",
                    ],
                    wrap=True,
                    column_widths=[6, 1, 1, 1, 3],
                    row_count=10,
                    col_count=5,
                    max_height=400,
                )
            with gr.Row():
                logs = gr.HTML()

            ui.load(
                run_with_logging,
                inputs=[log_data],
                outputs=[log_data, logs, opportunities_dataframe],
            )

            timer = gr.Timer(value=settings.ui_timer_interval, active=True)
            timer.tick(
                run_with_logging,
                inputs=[log_data],
                outputs=[log_data, logs, opportunities_dataframe],
            )

            opportunities_dataframe.select(do_select)

        # Warm the framework on the main thread before Gradio spawns any
        # request workers. Cheap (no planner yet) and removes a startup race.
        self.get_framework()

        ui.launch(share=False, inbrowser=True)


if __name__ == "__main__":
    App().run()
