import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
matplotlib.use("TkAgg")


class ScatterVisualizer:
    def __init__(self, interactive=False, xlim=0, ylim=0, offset=1, log_scale=False):
        self.interactive = interactive
        self.log_scale = log_scale
        self.offset = offset
        self.xlim = xlim
        self.ylim = ylim

        matplotlib.interactive(self.interactive)

        self.fig, self.ax = plt.subplots()
        self.my_plot = self.ax.scatter([], [])
        self.my_plot.set_edgecolor("white")
        self.ax.set_xlim(0, xlim)
        if self.log_scale:
            self.ax.set_yscale("log")

    def set_title(self, title: str = ""):
        self.ax.set_title(title)

    def set_labels(self, xlabel: str = "x", ylabel: str = "y"):
        self.ax.set_xlabel(xlabel=xlabel)
        self.ax.set_ylabel(ylabel=ylabel)

    def set_limits(self, n: int):
        self.ax.set_xlim(-n, n)
        self.ax.set_ylim(-n, n)

    def plot_data(self, x, y):
        matplotlib.interactive(False)
        self.fig, self.ax = plt.subplots()
        self.my_plot = self.ax.scatter([], [])
        self.my_plot.set_edgecolor("white")
        self.ax.set_xlim(0, self.xlim)
        try:
            self.ax.set_ylim(
                [np.min(y, axis=0)[0] - self.offset, np.max(y, axis=0)[0] + self.offset]
            )
        except IndexError:
            self.ax.set_ylim(
                [np.min(y, axis=0) - self.offset, np.max(y, axis=0) + self.offset]
            )
        self.set_labels(xlabel="(time)", ylabel="(value)")
        if self.log_scale:
            self.ax.set_yscale("log")
        self.ax.plot(x, y, "o-")
        plt.show()
