import _init_paths
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]

from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist


trackers = []
dataset_name = 'otb_lang'
# dataset_name = 'lasot' # lasot_extension_subset

report_name = f'{dataset_name}_compare'

trackers.extend(trackerlist(name='dutrack', parameter_name='dutrack_384_full', dataset_name=dataset_name,
	                    display_name='DUTrack'))
trackers.extend(trackerlist(name='dutrack', parameter_name='dutrack_384_full_047_DTCM', dataset_name=dataset_name,
                            display_name='dutrack_dutrack_384_full_047_DTCM'))
trackers.extend(trackerlist(name='dutrack', parameter_name='dutrack_384_full_047_updatekey', dataset_name=dataset_name,
                            display_name='dutrack_dutrack_384_full_047_updatekey'))


# For VOT evaluate
dataset = get_dataset(dataset_name)
# dataset = get_dataset('otb', 'nfs', 'uav', 'tc128ce')
plot_results(trackers, dataset, dataset_name, merge_results=False, plot_types=('success', 'norm_prec','prec'),
	         skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05)
print_results(trackers, dataset, report_name, merge_results=False, plot_types=('success', 'norm_prec', 'prec'))

# print_results(trackers, dataset, 'UNO', merge_results=True, plot_types=('success', 'prec'))

