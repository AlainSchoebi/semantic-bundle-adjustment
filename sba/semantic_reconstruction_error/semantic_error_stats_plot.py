#!/usr/bin/env python3

# Numpy
import numpy as np

# Python
from pathlib import Path
import json

# Matplotlib
import matplotlib.pyplot as plt

# Src
from sba.tree_parser import TreeParser

# Logging
from sba.utils.loggers import get_logger
logger = get_logger(__name__)

def main(args):

    if not Path(args.error_file).is_file():
        logger.error(f"The provided error file '{args.error_file}' " +
                     f"does not exist.")
        raise FileExistsError(f"The provided error file '{args.error_file}' " +
                              f"does not exist.")

    # Read file
    with open(args.error_file, 'r') as json_file:
        errors = json.load(json_file)

    new_errors = {}
    for key, values in errors.items():
        new_errors[key] = []
        for value in values:
            new_errors[key].append([int(value[0]), int(value[1])])

    # Extract values
    x, y = [], []
    all_x, all_y = [], []

    for key, values in errors.items():
        x.append(float(key))
        y.append([])
        for value in values:
            y[-1].append(value[0] / value[1])
            all_x.append(float(key))
            all_y.append(value[0] / value[1])

    # Figure
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(111)

    # Boxplot
    ax.boxplot(y,
               patch_artist=True,
               positions=x,
               boxprops=dict(facecolor="lightblue", color="blue"),
               whiskerprops=dict(color="blue"),
               flierprops=dict(visible=False),
               capprops=dict(color="blue"),
               medianprops=dict(visible=False),
               zorder=1)

    # Scatter points
    ax.scatter(all_x, all_y, edgecolors='black', facecolors='none', marker='o')

    # Display the plot
    plt.show(block=True)


if __name__ == "__main__":
    parser = TreeParser(description="Semantic error statistics plotting.")

    parser.add_argument("--error_file", type=str, required=False,
                        default="out/last_semantic_error_stats/errors.json",
                        help= "The path to the .json file containg the errors.")

    args = parser.parse_args()

    main(args)