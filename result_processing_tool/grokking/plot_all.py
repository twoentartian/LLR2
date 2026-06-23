import argparse

from plot_grokking_logs import main as plot_grokking_logs_main
from plot_correct_position import main as plot_correct_position_main
from visualizing_arithemetic_dataset import main as visualizing_arithemetic_dataset_main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=("Do all post processing steps"))
    parser.add_argument("folder",help="Root folder to process (directly or recursively)",)
    parser.add_argument("-o","--override",help="Override existing plots",action="store_true")
    args = parser.parse_args()
    plot_grokking_logs_main(args.folder, override_existing=args.override)
    plot_correct_position_main(args.folder, override_existing=args.override)
    visualizing_arithemetic_dataset_main(args.folder, override_existing=args.override)