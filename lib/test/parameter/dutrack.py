from lib.test.utils import TrackerParams
import os
from lib.test.evaluation.environment import env_settings
from lib.config.dutrack.config import cfg, update_config_from_file


def parameters(yaml_name: str, run_id=None):
    params = TrackerParams()
    prj_dir = env_settings().prj_dir
    save_dir = env_settings().save_dir
    # update default config from yaml file
    yaml_file = os.path.join(prj_dir, 'experiments/dutrack/%s.yaml' % yaml_name)
    update_config_from_file(yaml_file)
    params.cfg = cfg
    params.param_name = yaml_name
    # print("test config: ", cfg)

    # template and search region
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE

    # Network checkpoint path
    checkpoint_config = getattr(cfg.TEST, "CHECKPOINT_CONFIG", "")
    checkpoint_config = checkpoint_config if checkpoint_config else yaml_name
    if run_id is None:
        params.checkpoint = os.path.join(save_dir, "checkpoints/train/dutrack/%s/DUTrack_ep%04d.pth.tar" %
                                        (checkpoint_config, cfg.TEST.EPOCH))
    else:
        params.checkpoint = os.path.join(save_dir, "checkpoints/train/dutrack/%s/DUTrack_ep%04d.pth.tar" %
                                        (checkpoint_config, run_id))
    
    # whether to save boxes from all queries
    params.save_all_boxes = False

    return params
