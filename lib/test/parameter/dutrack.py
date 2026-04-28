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
    # print("test config: ", cfg)

    # template and search region
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE

    # Network checkpoint path
    checkpoint_dir = os.path.join(save_dir, "checkpoints/train/dutrack/%s" % yaml_name)
    if run_id is None:
        checkpoint_name = "DUTrack_ep%04d.pth.tar" % cfg.TEST.EPOCH
    elif isinstance(run_id, str):
        run_id_key = run_id.lower()
        if run_id_key in ("best", "latest"):
            checkpoint_name = "DUTrack_%s.pth.tar" % run_id_key
        else:
            run_id = int(run_id)
            checkpoint_name = "DUTrack_ep%04d.pth.tar" % run_id
    else:
        checkpoint_name = "DUTrack_ep%04d.pth.tar" % run_id

    params.checkpoint = os.path.join(checkpoint_dir, checkpoint_name)

    # whether to save boxes from all queries
    params.save_all_boxes = False

    return params
