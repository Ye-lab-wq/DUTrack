import _init_paths
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]

from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist


trackers = []
dataset_name = 'hoot_all'

report_name = f'{dataset_name}_compare'

trackers.extend(trackerlist(name='dutrack', parameter_name='dutrack_384_full_hoot_all_lang_normal', dataset_name=dataset_name,
                            display_name='dutrack_hoot_all_lang_normal'))
trackers.extend(trackerlist(name='dutrack', parameter_name='dutrack_384_full_hoot_all_lang_shuffle', dataset_name=dataset_name,
                            display_name='dutrack_hoot_all_lang_shuffle'))
trackers.extend(trackerlist(name='dutrack', parameter_name='dutrack_384_full_hoot_all_lang_wrong', dataset_name=dataset_name,
                            display_name='dutrack_hoot_all_lang_wrong'))
trackers.extend(trackerlist(name='dutrack', parameter_name='dutrack_384_full_hoot_all_lang_generic', dataset_name=dataset_name,
                            display_name='dutrack_hoot_all_lang_generic'))
trackers.extend(trackerlist(name='dutrack', parameter_name='dutrack_384_full_hoot_all_lang_no_update', dataset_name=dataset_name,
                            display_name='dutrack_hoot_all_lang_no_update'))


dataset = get_dataset(dataset_name)
plot_results(trackers, dataset, report_name, merge_results=False, plot_types=('success', 'norm_prec', 'prec'),
             skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05)
print_results(trackers, dataset, report_name, merge_results=False, plot_types=('success', 'norm_prec', 'prec'))
