"""Prediction evaluation helpers.

Do not ``from ... import test`` in notebooks: that shadows the usual ``test``
dataset split. Use ``evaluate`` or ``import ... as testing`` and
``testing.evaluate(...)``.
"""

import math
import warnings
from tqdm import tqdm
import numpy as np

import plotly.graph_objects as go

__all__ = ["Tester", "evaluate"]

# COLOR MAP
GREEN = "\033[92m"
ORANGE = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
COLOR_MAP = {"red": RED, "orange": ORANGE, "green": GREEN}


class Tester:
    def __init__(self, predictor, data, title=None, size=250):
        if callable(data) and hasattr(predictor, "iloc"):
            warnings.warn(
                "Arguments look reversed (DataFrame, callable). "
                "Use Tester(predictor, data). Swapping for you.",
                UserWarning,
                stacklevel=2,
            )
            predictor, data = data, predictor
        if not callable(predictor):
            raise TypeError(
                f"predictor must be callable, got {type(predictor).__name__!r}"
            )
        if not hasattr(data, "iloc"):
            raise TypeError(
                "data must support .iloc (e.g. pandas.DataFrame), "
                f"got {type(data).__name__!r}"
            )
        self.predictor = predictor
        self.data = data
        self.title = title or predictor.__name__.replace("_", " ").title()
        self.size = size

        # Use numpy arrays for numerical data
        self.guesses = np.zeros(size)
        self.truths = np.zeros(size)
        self.errors = np.zeros(size)
        self.lche = np.zeros(size)
        self.sles = np.zeros(size)
        self.colors = []  # Keep as list for strings

        # Counters
        self.green_count = 0
        self.orange_count = 0
        self.red_count = 0

        # Cache for computed metrics
        self._average_error = None
        self._rmsle = None

    def run_datapoint(self, i):
        try:
            datapoint = self.data.iloc[i]
            guess = float(self.predictor(datapoint))
            truth = float(datapoint["price"])

            error = abs(truth - guess)
            log_error = math.log(truth + 1) - math.log(guess + 1)
            sle = log_error**2
            log_cosh_error = self.safe_log_cosh(error)

            color = self.color_for(error, truth)

            # Update counters
            if color == "green":
                self.green_count += 1
            elif color == "orange":
                self.orange_count += 1
            else:  # red
                self.red_count += 1

            # Store results in arrays
            self.guesses[i] = guess
            self.truths[i] = truth
            self.errors[i] = error
            self.lche[i] = log_cosh_error
            self.sles[i] = sle
            self.colors.append(color)

            return color

        except (ValueError, TypeError, AttributeError) as e:
            print(f"Error processing datapoint {i}: {e}")
            self.colors.append("red")
            self.red_count += 1
            return "red"

    def safe_log_cosh(self, x):
        """Avoids overflow in log cosh calculation"""
        x = max(min(x, 500), -500)  # Cap between -500 and 500
        return math.log(math.cosh(x))

    def color_for(self, error, truth):
        if truth <= 0:
            return "orange"
        if error < 40 or error / truth < 0.2:
            return "green"
        elif error < 80 or error / truth < 0.4:
            return "orange"
        else:
            return "red"

    @property
    def average_error(self):
        if self._average_error is None:
            self._average_error = np.mean(self.errors)
        return self._average_error

    @property
    def rmsle(self):
        if self._rmsle is None:
            self._rmsle = math.sqrt(np.mean(self.sles))
        return self._rmsle

    def chart(self, title):
        truths = self.truths[: self.size]
        guesses = self.guesses[: self.size]
        max_val = float(max(np.max(truths), np.max(guesses), 1.0))

        x = np.linspace(0, max_val, 80)
        y_low_40, y_high_40 = x * 0.6, x * 1.4
        y_low_20, y_high_20 = x * 0.8, x * 1.2

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([x, x[::-1]]),
                y=np.concatenate([y_high_40, y_low_40[::-1]]),
                fill="toself",
                fillcolor="rgba(255,165,0,0.12)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                name="±40% Range",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([x, x[::-1]]),
                y=np.concatenate([y_high_20, y_low_20[::-1]]),
                fill="toself",
                fillcolor="rgba(0,128,0,0.12)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                name="±20% Range",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode="lines",
                line=dict(color="deepskyblue", width=2, dash="dash"),
                name="Perfect prediction",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=truths,
                y=guesses,
                mode="markers",
                marker=dict(size=7, color=self.colors, opacity=0.65),
                name="Predictions",
            )
        )

        green_pct = self.green_count / self.size * 100
        orange_pct = self.orange_count / self.size * 100
        red_pct = self.red_count / self.size * 100
        stats_text = (
            f"<b>Accuracy</b><br>"
            f"Green: {green_pct:.1f}%<br>"
            f"Orange: {orange_pct:.1f}%<br>"
            f"Red: {red_pct:.1f}%<br>"
            f"Avg error: ${self.average_error:,.2f}<br>"
            f"RMSLE: {self.rmsle:.2f}"
        )
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.02,
            y=0.98,
            xanchor="left",
            yanchor="top",
            text=stats_text,
            showarrow=False,
            align="left",
            bgcolor="rgba(255,255,255,0.9)",
            borderpad=6,
        )

        fig.update_layout(
            title=dict(text=title),
            xaxis_title="True price ($)",
            yaxis_title="Predicted price ($)",
            width=1000,
            height=800,
            template="plotly_white",
            xaxis=dict(range=[0, max_val]),
            yaxis=dict(range=[0, max_val]),
            legend=dict(yanchor="bottom", y=0.02, xanchor="right", x=0.99),
        )
        try:
            from IPython import get_ipython  # type: ignore[import-untyped]

            if get_ipython() is not None:
                fig.show()
        except Exception:
            pass

    def report(self):
        # Print summary statistics with color
        print("\nTest Results Summary:")
        print(f"Total Predictions: {self.size}")
        print(
            f"{GREEN}Correct (Green): {self.green_count} ({self.green_count / self.size * 100:.1f}%){RESET}"
        )
        print(
            f"{ORANGE}Close (Orange): {self.orange_count} ({self.orange_count / self.size * 100:.1f}%){RESET}"
        )
        print(
            f"{RED}Wrong (Red): {self.red_count} ({self.red_count / self.size * 100:.1f}%){RESET}"
        )
        print(f"Average Error: ${self.average_error:,.2f}")
        print(f"RMSLE: {self.rmsle:.2f}")

        title = f"{self.title} Error=${self.average_error:,.2f}  RMSLE={self.rmsle:.2f}  HIT={self.green_count / self.size * 100:.1f}%"
        self.chart(title)

    def run(self):
        # Progress bar for overall testing
        with tqdm(
            total=self.size,
            desc="Testing Progress",
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        ) as pbar:
            # Progress bar for correct predictions
            expected_correct = int(self.size * 0.7)  # Set expected correct to 70%
            with tqdm(
                total=expected_correct,
                desc="Correct Predictions",
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}",
            ) as correct_pbar:
                for i in range(self.size):
                    color = self.run_datapoint(i)
                    pbar.update(1)

                    if color == "green" and correct_pbar.n < expected_correct:
                        correct_pbar.update(1)

        self.report()

    @classmethod
    def test(cls, function, data, title=None, size=250):
        cls(function, data, title=title, size=size).run()


def evaluate(function, data, *, title=None, size=250):
    """Run :class:`Tester`; same as :meth:`Tester.test`."""
    Tester(function, data, title=title, size=size).run()
