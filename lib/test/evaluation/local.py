from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/home/b520/Downloads/yelin/data/got10k_lmdb'
    settings.got10k_path = '/home/b520/Downloads/yelin/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/home/b520/Downloads/yelin/data/itb'
    settings.lasot_extension_subset_path = '/home/b520/Downloads/yelin/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/home/b520/Downloads/yelin/data/lasot_lmdb'
    settings.lasot_path = '/home/b520/Downloads/yelin/data/lasot'
    settings.mgit_path = '/home/b520/Downloads/yelin/data/MGIT'
    settings.network_path = '/home/b520/Downloads/yelin/DUTrack/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/home/b520/Downloads/yelin/data/nfs'
    settings.otb_lang_path = '/home/b520/Downloads/yelin/data/OTB_sentences'
    settings.otb_path = '/home/b520/Downloads/yelin/data/otb'
    settings.prj_dir = '/home/b520/Downloads/yelin/DUTrack'
    settings.result_plot_path = '/home/b520/Downloads/yelin/DUTrack/output/test/result_plots'
    settings.results_path = '/home/b520/Downloads/yelin/DUTrack/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/home/b520/Downloads/yelin/DUTrack/output'
    settings.segmentation_path = '/home/b520/Downloads/yelin/DUTrack/output/test/segmentation_results'
    settings.tc128_path = '/home/b520/Downloads/yelin/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/home/b520/Downloads/yelin/data/TNL2K'
    settings.tpl_path = ''
    settings.trackingnet_path = '/home/b520/Downloads/yelin/data/trackingnet'
    settings.uav_path = '/home/b520/Downloads/yelin/data/uav'
    settings.vot18_path = '/home/b520/Downloads/yelin/data/vot2018'
    settings.vot22_path = '/home/b520/Downloads/yelin/data/vot2022'
    settings.vot_path = '/home/b520/Downloads/yelin/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

