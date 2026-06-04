import argparse

import _init_paths

from lib.test.analysis.plot_results import plot_results, print_results
from lib.test.evaluation import get_dataset, trackerlist


def main():
    parser = argparse.ArgumentParser(description="Analyze TEC stage1 tracking results.")
    parser.add_argument("--dataset_name", type=str, default="otb_lang")
    parser.add_argument("--report_name", type=str, default=None)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    dataset_name = args.dataset_name
    report_name = args.report_name or "{}_tec_stage1".format(dataset_name)

    trackers = []
    trackers.extend(trackerlist(
        name="dutrack",
        parameter_name="dutrack_384_full",
        dataset_name=dataset_name,
        display_name="A0_baseline_ep47",
    ))
    trackers.extend(trackerlist(
        name="dutrack",
        parameter_name="dutrack_384_full_tec_stage1",
        dataset_name=dataset_name,
        display_name="A1_TEC_normal_ep5",
    ))
    trackers.extend(trackerlist(
        name="dutrack",
        parameter_name="dutrack_384_full_tec_stage1_wrong",
        dataset_name=dataset_name,
        display_name="A1_TEC_wrong_ep5",
    ))
    trackers.extend(trackerlist(
        name="dutrack",
        parameter_name="dutrack_384_full_tec_stage1_generic",
        dataset_name=dataset_name,
        display_name="A1_TEC_generic_ep5",
    ))

    dataset = get_dataset(dataset_name)
    plot_types = ("success", "norm_prec", "prec")

    if args.plot:
        plot_results(
            trackers,
            dataset,
            report_name,
            merge_results=False,
            plot_types=plot_types,
            skip_missing_seq=False,
            force_evaluation=True,
            plot_bin_gap=0.05,
        )
    print_results(
        trackers,
        dataset,
        report_name,
        merge_results=False,
        plot_types=plot_types,
    )


if __name__ == "__main__":
    main()
