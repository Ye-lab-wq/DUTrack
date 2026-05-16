import _init_paths
import argparse
import matplotlib.pyplot as plt

from lib.test.analysis.plot_results import plot_results, print_results
from lib.test.evaluation import get_dataset, trackerlist


plt.rcParams['figure.figsize'] = [8, 8]


def _add(trackers, dataset_name, parameter_name, display_name, run_ids=None):
    trackers.extend(trackerlist(name='dutrack',
                                parameter_name=parameter_name,
                                dataset_name=dataset_name,
                                run_ids=run_ids,
                                display_name=display_name))


def build_otb_lang_trackers(dataset_name):
    trackers = []
    report_name = 'otb_lang_compare'
    skip_missing_seq = False

    # _add(trackers, dataset_name, 'dutrack_384_full', 'DUTrack')
    _add(trackers, dataset_name, 'dutrack_384_full_047_updatekey', 'dutrack_384_full_047_updatekey')
    _add(trackers, dataset_name, 'dutrack_384_full_047_ori', 'dutrack_384_full_047_ori')
    _add(trackers, dataset_name, 'dutrack_384_full_047', 'dutrack_384_full_047')
    # _add(trackers, dataset_name, 'dutrack_384_full_vlte', 'vlte_score_only_ep0047', run_ids=47)

    _add(trackers, dataset_name, 'dutrack_384_full_e5_control', 'baseline_e5_control', run_ids=5)
    # _add(trackers, dataset_name, 'dutrack_384_full_vlte_e5', 'vlte_score_loss_ep0005', run_ids=5)
    # _add(trackers, dataset_name, 'dutrack_384_full_vlte_reweight_e5', 'vlte_score_reweight_ep0005', run_ids=5)
    # _add(trackers, dataset_name, 'dutrack_384_full_vlte_w0001_e5', 'vlte_score_w0001_ep0005', run_ids=5)
    _add(trackers, dataset_name, 'dutrack_384_full_vlte_detach_e5', 'vlte_score_detach_ep0005', run_ids=5)

    _add(trackers, dataset_name, 'dutrack_384_full_vlte_tepolicy_a002_e5',
         'vlte_tepolicy_a002_ep0005', run_ids=5)
    # _add(trackers, dataset_name, 'dutrack_384_full_vlte_tepolicy_a005_e5',
    #      'vlte_tepolicy_a005_ep0005', run_ids=5)
    # _add(trackers, dataset_name, 'dutrack_384_full_vlte_tepolicy_a001_e5',
    #      'vlte_tepolicy_a001_ep0005', run_ids=5)
    # _add(trackers, dataset_name, 'dutrack_384_full_vlte_tepolicy_a0015_e5',
    #      'vlte_tepolicy_a0015_ep0005', run_ids=5)
    # _add(trackers, dataset_name, 'dutrack_384_full_vlte_tepolicy_a002_last2_e5',
    #      'vlte_tepolicy_a002_last2_ep0005', run_ids=5)
    # _add(trackers, dataset_name, 'dutrack_384_full_vlte_tepolicy_a002_norm_e5',
    #      'vlte_tepolicy_a002_norm_ep0005', run_ids=5)
    # _add(trackers, dataset_name, 'dutrack_384_full_vlte_tepolicy_a002_langrefine05_e5',
    #      'vlte_tepolicy_a002_langrefine05_ep0005', run_ids=5)
    # _add(trackers, dataset_name, 'dutrack_384_full_vlte_tepolicy_a002_gumbelres01_e5',
    #      'vlte_tepolicy_a002_gumbelres01_ep0005', run_ids=5)
    # _add(trackers, dataset_name, 'dutrack_384_full_vlte_tepolicy_a002_toppolicy20_e5',
    #      'vlte_tepolicy_a002_toppolicy20_ep0005', run_ids=5)
    _add(trackers, dataset_name, 'dutrack_384_full_vlte_tepolicy_a002_toppolicy30_e5',
         'vlte_tepolicy_a002_toppolicy30_ep0005', run_ids=5)
    _add(trackers, dataset_name, 'dutrack_384_full_vlte_tepolicy_a002_toppolicy30_neg010_e5',
         'vlte_tepolicy_a002_toppolicy30_neg010_ep0005', run_ids=5)
    # _add(trackers, dataset_name, 'dutrack_384_full_vlte_tepolicy_a002_toppolicy25_e5',
    #      'vlte_tepolicy_a002_toppolicy25_ep0005', run_ids=5)
    # _add(trackers, dataset_name, 'dutrack_384_full_vlte_tepolicy_a002_toppolicy35_e5',
    #      'vlte_tepolicy_a002_toppolicy35_ep0005', run_ids=5)
    # _add(trackers, dataset_name, 'dutrack_384_full_vlte_tepolicy_a002_toppolicy20_weakneg_e5',
    #      'vlte_tepolicy_a002_toppolicy20_weakneg_ep0005', run_ids=5)
    _add(trackers, dataset_name,
         'dutrack_384_full_vlte_tepolicy_a002_toppolicy30_neg010_fulltrain_e20_bs2_accum4',
         'vlte_toppolicy30_neg010_fulltrain_ep0019', run_ids=19)
    _add(trackers, dataset_name,
         'dutrack_384_full_vlte_tepolicy_a002_toppolicy30_neg010_fulltrain_e20_bs2_accum4',
         'vlte_toppolicy30_neg010_fulltrain_ep0015', run_ids=15)

    # _add(trackers, dataset_name, 'dutrack_384_full_vlte_tepolicy_a002_gauss_e5',
    #      'vlte_tepolicy_a002_gauss_ep0005', run_ids=5)
    # _add(trackers, dataset_name, 'dutrack_384_full_vlte_tepolicy_a002_boxgauss_e5',
    #      'vlte_tepolicy_a002_boxgauss_ep0005', run_ids=5)

    return trackers, report_name, skip_missing_seq


def build_lasot_trackers(dataset_name):
    trackers = []
    report_name = 'lasot_compare'
    skip_missing_seq = False

    _add(trackers, dataset_name, 'dutrack_384_full_047_updatekey',
         'dutrack_384_full_047_updatekey')
    _add(trackers, dataset_name, 'dutrack_384_full_047',
         'dutrack_384_full_047')
    _add(trackers, dataset_name, 'dutrack_384_full_e5_control',
         'baseline_e5_control', run_ids=5)
    _add(trackers, dataset_name, 'dutrack_384_full_vlte_detach_e5',
         'vlte_score_detach_ep0005', run_ids=5)
    _add(trackers, dataset_name, 'dutrack_384_full_vlte_tepolicy_a002_toppolicy30_neg010_e5',
         'vlte_tepolicy_a002_toppolicy30_neg010_ep0005', run_ids=5)
    _add(trackers, dataset_name,
         'dutrack_384_full_vlte_tepolicy_a002_toppolicy30_neg010_fulltrain_e20_bs2_accum4',
         'vlte_toppolicy30_neg010_fulltrain_ep0019', run_ids=19)

    return trackers, report_name, skip_missing_seq


def build_tnl2k_trackers(dataset_name):
    trackers = []
    report_name = 'tnl2k_compare'
    # Some local TNL2K result directories were produced before the full set
    # was available, so allow explicit partial-result analysis.
    skip_missing_seq = True

    _add(trackers, dataset_name, 'dutrack_384_full_updatekey',
         'dutrack_384_full_updatekey')
    _add(trackers, dataset_name, 'dutrack_384_full_vlte_tepolicy_a002_toppolicy30_neg010_e5',
         'vlte_tepolicy_a002_toppolicy30_neg010_ep0005', run_ids=5)

    return trackers, report_name, skip_missing_seq


def build_hoot_trackers(dataset_name):
    trackers = []
    report_name = '{}_compare'.format(dataset_name)
    skip_missing_seq = False

    _add(trackers, dataset_name, 'dutrack_384_full_047_updatekey',
         'dutrack_384_full_047_updatekey')
    _add(trackers, dataset_name, 'dutrack_384_full_vlte_tepolicy_a002_toppolicy30_neg010_e5',
         'vlte_tepolicy_a002_toppolicy30_neg010_ep0005', run_ids=5)
    # _add(trackers, dataset_name, 'dutrack_384_full_vlte_tepolicy_a002_toppolicy30_neg010_langupd_e5',
    #      'vlte_toppolicy30_neg010_langupd')
    # _add(trackers, dataset_name, 'dutrack_384_full_vlte_tepolicy_a002_toppolicy30_neg010_langupd_adapt_e5',
    #      'vlte_toppolicy30_neg010_langupd_adapt')
    _add(trackers, dataset_name, 'dutrack_384_full_vlte_tepolicy_a002_toppolicy30_neg010_langupd_qgate_e5',
         'vlte_toppolicy30_neg010_langupd_qgate')
    # _add(trackers, dataset_name, 'dutrack_384_full_vlte_tepolicy_a002_toppolicy30_neg010_langupd_qgate_crop125_e5',
    #      'vlte_toppolicy30_neg010_langupd_qgate_crop125')
    # _add(trackers, dataset_name, 'dutrack_384_full_vlte_tepolicy_a002_toppolicy30_neg010_langupd_qgate_crop150_e5',
        #  'vlte_toppolicy30_neg010_langupd_qgate_crop150')
    # _add(trackers, dataset_name,
    #      'dutrack_384_full_vlte_tepolicy_a002_toppolicy30_neg010_fulltrain_e20_bs2_accum4',
    #      'vlte_toppolicy30_neg010_fulltrain_ep0019', run_ids=19)

    return trackers, report_name, skip_missing_seq


def build_got10k_trackers(dataset_name):
    trackers = []
    report_name = 'got10k_compare'
    skip_missing_seq = False

    _add(trackers, dataset_name, 'dutrack_384_lasot_dynamic_token',
         'dynamic_token_ep0001', run_ids=1)
    _add(trackers, dataset_name, 'dutrack_384_lasot_dynamic_token',
         'dynamic_token_ep0008', run_ids=8)
    _add(trackers, dataset_name, 'dutrack_384_lasot_dynamic_token',
         'dynamic_token_ep0015', run_ids=15)
    _add(trackers, dataset_name, 'dutrack_384_lasot_dynamic_token_reinject_ft',
         'dynamic_token_reinject_ft_ep0015', run_ids=15)
    _add(trackers, dataset_name, 'dutrack_384_lasot_dynamic_token_reinject_ft',
         'dynamic_token_reinject_ft_ep0020', run_ids=20)

    return trackers, report_name, skip_missing_seq


def build_default_trackers(dataset_name, config=None, run_id=None):
    trackers = []
    report_name = f'{dataset_name}_compare'
    skip_missing_seq = dataset_name in ('tnl2k',)

    if config is None:
        _add(trackers, dataset_name, 'dutrack_384_full', 'DUTrack')
    else:
        _add(trackers, dataset_name, config, config, run_ids=run_id)

    return trackers, report_name, skip_missing_seq


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze DUTrack results by dataset.')
    parser.add_argument('--dataset_name', type=str, default='otb_lang')
    parser.add_argument('--config', type=str, default=None,
                        help='Optional single tracker config/parameter name to evaluate.')
    parser.add_argument('--runid', type=int, default=None,
                        help='Optional run id for --config.')
    parser.add_argument('--no_plot', action='store_true')
    parser.add_argument('--skip_missing_seq', action='store_true')
    parser.add_argument('--use_cache', action='store_true',
                        help='Reuse cached eval_data.pkl instead of recomputing.')
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_name = args.dataset_name

    if args.config is not None:
        trackers, report_name, skip_missing_seq = build_default_trackers(dataset_name, args.config, args.runid)
    elif dataset_name == 'otb_lang':
        trackers, report_name, skip_missing_seq = build_otb_lang_trackers(dataset_name)
    elif dataset_name == 'lasot':
        trackers, report_name, skip_missing_seq = build_lasot_trackers(dataset_name)
    elif dataset_name == 'tnl2k':
        trackers, report_name, skip_missing_seq = build_tnl2k_trackers(dataset_name)
    elif dataset_name in ('hoot', 'hoot_all', 'hoot_balanced20', 'hoot_balanced40'):
        trackers, report_name, skip_missing_seq = build_hoot_trackers(dataset_name)
    elif dataset_name in ('got10k_val', 'got10k_ltrval', 'got10k_test'):
        trackers, report_name, skip_missing_seq = build_got10k_trackers(dataset_name)
    else:
        trackers, report_name, skip_missing_seq = build_default_trackers(dataset_name)

    if args.skip_missing_seq:
        skip_missing_seq = True

    dataset = get_dataset(dataset_name)
    common_kwargs = dict(skip_missing_seq=skip_missing_seq, plot_bin_gap=0.05)

    if not args.no_plot:
        plot_results(trackers, dataset, report_name,
                     merge_results=False,
                     plot_types=('success', 'norm_prec', 'prec'),
                     force_evaluation=not args.use_cache,
                     **common_kwargs)

    print_results(trackers, dataset, report_name,
                  merge_results=False,
                  plot_types=('success', 'norm_prec', 'prec'),
                  force_evaluation=args.no_plot and not args.use_cache,
                  **common_kwargs)


if __name__ == '__main__':
    main()
