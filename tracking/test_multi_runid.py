import os
import sys
import argparse

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.evaluation import get_dataset
from lib.test.evaluation.running import run_dataset
from lib.test.evaluation.tracker import Tracker


def parse_runids(runids_text: str):
    runids = []
    for token in runids_text.split(','):
        token = token.strip()
        if token:
            runids.append(int(token))
    if not runids:
        raise ValueError("runids is empty")
    return runids


def run_tracker_multi(tracker_name, tracker_param, runids, dataset_name='otb', sequence=None, debug=0, threads=0, num_gpus=1):
    dataset = get_dataset(dataset_name)

    if sequence is not None:
        dataset = [dataset[sequence]]

    for run_id in runids:
        print(f"\n===== Evaluating run_id={run_id} on {dataset_name} =====")
        trackers = [Tracker(tracker_name, tracker_param, dataset_name, run_id)]
        run_dataset(dataset, trackers, debug, threads, num_gpus=num_gpus)


def main():
    parser = argparse.ArgumentParser(description='Run tracker on multiple runids sequentially.')
    parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
    parser.add_argument('tracker_param', type=str, help='Name of config file.')
    parser.add_argument('--runids', type=str, required=True, help='Comma-separated run ids, e.g. "1,8,15".')
    parser.add_argument('--dataset_name', type=str, default='lasot', help='Dataset name.')
    parser.add_argument('--sequence', type=str, default=None, help='Sequence index or sequence name.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--threads', type=int, default=0, help='Number of threads.')
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of gpus for eval workers.')

    args = parser.parse_args()

    try:
        seq_name = int(args.sequence)
    except Exception:
        seq_name = args.sequence

    runids = parse_runids(args.runids)
    run_tracker_multi(args.tracker_name, args.tracker_param, runids, args.dataset_name, seq_name, args.debug, args.threads, args.num_gpus)


if __name__ == '__main__':
    main()

