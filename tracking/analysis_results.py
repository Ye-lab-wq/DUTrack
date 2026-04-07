import _init_paths
import os
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]

from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist


def _tracker_label(tracker):
    if tracker.display_name is not None:
        return tracker.display_name
    if tracker.run_id is None:
        return f'{tracker.name}/{tracker.parameter_name}'
    return f'{tracker.name}/{tracker.parameter_name}_{tracker.run_id:03d}'


def filter_trackers_with_results(trackers, dataset_name):
    available = []
    missing = []

    for tracker in trackers:
        dataset_result_dir = os.path.join(tracker.results_dir, dataset_name)
        has_result_dir = os.path.isdir(dataset_result_dir)
        has_txt = has_result_dir and any(name.endswith('.txt') for name in os.listdir(dataset_result_dir))

        if has_txt:
            available.append(tracker)
        else:
            missing.append((_tracker_label(tracker), dataset_result_dir))

    if missing:
        print('Missing tracker results for dataset {}:'.format(dataset_name))
        for label, result_dir in missing:
            print('  - {} -> {}'.format(label, result_dir))

    return available


trackers = []
dataset_name = 'tnl2k'  # choices: 'tnl2k', 'otb_lang', 'lasot'

if dataset_name == 'otb_lang':
    report_name = 'otb_lang_reinject_ft_compare'
    skip_missing_seq = False

    trackers.extend(trackerlist(name='dutrack',
                                parameter_name='dutrack_384_full_047_updatekey',
                                dataset_name=dataset_name,
                                display_name='dutrack_384_full_047_updatekey'))
    trackers.extend(trackerlist(name='dutrack',
                                parameter_name='dutrack_384_lasot_dynamic_token',
                                dataset_name=dataset_name,
                                run_ids=1,
                                display_name='dutrack_dynamic_token_ep0001'))
    trackers.extend(trackerlist(name='dutrack',
                                parameter_name='dutrack_384_lasot_dynamic_token',
                                dataset_name=dataset_name,
                                run_ids=8,
                                display_name='dutrack_dynamic_token_ep0008'))
    trackers.extend(trackerlist(name='dutrack',
                                parameter_name='dutrack_384_lasot_dynamic_token',
                                dataset_name=dataset_name,
                                run_ids=15,
                                display_name='dutrack_dynamic_token_ep0015'))
    trackers.extend(trackerlist(name='dutrack',
                                parameter_name='dutrack_384_lasot_dynamic_token_reinject_ft',
                                dataset_name=dataset_name,
                                run_ids=15,
                                display_name='reinject_ft_ep0015'))
    trackers.extend(trackerlist(name='dutrack',
                                parameter_name='dutrack_384_lasot_dynamic_token_reinject_ft',
                                dataset_name=dataset_name,
                                run_ids=20,
                                display_name='reinject_ft_ep0020'))
elif dataset_name == 'lasot':
    report_name = 'lasot_reinject_ft_compare'
    skip_missing_seq = False

    trackers.extend(trackerlist(name='dutrack',
                                parameter_name='dutrack_384_full_047_updatekey',
                                dataset_name=dataset_name,
                                display_name='dutrack_384_full_047_updatekey'))
    trackers.extend(trackerlist(name='dutrack',
                                parameter_name='dutrack_384_lasot_dynamic_token',
                                dataset_name=dataset_name,
                                run_ids=1,
                                display_name='dutrack_dynamic_token_ep0001'))
    trackers.extend(trackerlist(name='dutrack',
                                parameter_name='dutrack_384_lasot_dynamic_token',
                                dataset_name=dataset_name,
                                run_ids=8,
                                display_name='dutrack_dynamic_token_ep0008'))
    trackers.extend(trackerlist(name='dutrack',
                                parameter_name='dutrack_384_lasot_dynamic_token',
                                dataset_name=dataset_name,
                                run_ids=15,
                                display_name='dutrack_dynamic_token_ep0015'))

    trackers.extend(trackerlist(name='dutrack',
                                parameter_name='dutrack_384_lasot_dynamic_token_reinject_ft',
                                dataset_name=dataset_name,
                                run_ids=15,
                                display_name='reinject_ft_ep0015'))
    trackers.extend(trackerlist(name='dutrack',
                                parameter_name='dutrack_384_lasot_dynamic_token_reinject_ft',
                                dataset_name=dataset_name,
                                run_ids=20,
                                display_name='reinject_ft_ep0020'))
elif dataset_name == 'tnl2k':
    report_name = f'{dataset_name}_updatekey_compare'
    # TNL2K currently has a few damaged sequences on disk. Enable skip_missing_seq so
    # analysis can proceed over the finished subset without forcing a full rerun first.
    skip_missing_seq = True

    trackers.extend(trackerlist(name='dutrack',
                                parameter_name='dutrack_384_full_updatekey',
                                dataset_name=dataset_name,
                                display_name='dutrack_384_full_047_updatekey'))
    # Add more TNL2K runs here after their tracking results are available.
    # trackers.extend(trackerlist(name='dutrack',
    #                             parameter_name='dutrack_384_lasot_dynamic_token_reinject_ft',
    #                             dataset_name=dataset_name,
    #                             run_ids=15,
    #                             display_name='reinject_ft_ep0015'))
else:
    raise ValueError(f'Unsupported dataset_name: {dataset_name}')

trackers = filter_trackers_with_results(trackers, dataset_name)
if not trackers:
    raise RuntimeError(f'No tracker results available for dataset {dataset_name}')

dataset = get_dataset(dataset_name)
plot_results(trackers, dataset, report_name, merge_results=False, plot_types=('success', 'norm_prec','prec'),
	         skip_missing_seq=skip_missing_seq, force_evaluation=True, plot_bin_gap=0.05)
print_results(trackers, dataset, report_name, merge_results=False, plot_types=('success', 'norm_prec', 'prec'))
