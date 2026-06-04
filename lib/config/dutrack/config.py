from easydict import EasyDict as edict
import yaml

"""
Add default config for OSTrack.
"""
cfg = edict()

# MODEL
cfg.MODEL = edict()
cfg.MODEL.PRETRAIN_FILE = "mae_pretrain_vit_base.pth"
cfg.MODEL.EXTRA_MERGER = False

cfg.MODEL.RETURN_INTER = False
cfg.MODEL.RETURN_STAGES = []

# MODEL.BACKBONE
cfg.MODEL.BACKBONE = edict()
cfg.MODEL.BACKBONE.TYPE = "vit_base_patch16_224"
cfg.MODEL.BACKBONE.STRIDE = 16
cfg.MODEL.BACKBONE.MID_PE = False
cfg.MODEL.BACKBONE.SEP_SEG = False
cfg.MODEL.BACKBONE.CAT_MODE = 'direct'
cfg.MODEL.BACKBONE.MERGE_LAYER = 0
cfg.MODEL.BACKBONE.ADD_CLS_TOKEN = False
cfg.MODEL.BACKBONE.TOKEN_LEN = 1
cfg.MODEL.BACKBONE.TOP_K = 3
cfg.MODEL.BACKBONE.CLS_TOKEN_USE_MODE = 'ignore'
cfg.MODEL.BACKBONE.ATTN_TYPE = 'concat'

cfg.MODEL.BACKBONE.CE_LOC = []
cfg.MODEL.BACKBONE.CE_KEEP_RATIO = []
cfg.MODEL.BACKBONE.CE_TEMPLATE_RANGE = 'ALL'  # choose between ALL, CTR_POINT, CTR_REC, GT_BOX
cfg.MODEL.BACKBONE.BERT_DIR = 'ALL'
cfg.MODEL.BACKBONE.BLIP_DIR = 'ALL'

# MODEL.HEAD
cfg.MODEL.HEAD = edict()
cfg.MODEL.HEAD.TYPE = "CENTER"
cfg.MODEL.HEAD.NUM_CHANNELS = 256

# MODEL.TEC
# Tracking Evidence Calibration is disabled by default to preserve baseline behavior.
cfg.MODEL.TEC = edict()
cfg.MODEL.TEC.ENABLE = False
cfg.MODEL.TEC.EVIDENCE_DIM = 128
cfg.MODEL.TEC.GAMMA_INIT = 0.01
cfg.MODEL.TEC.LANG_SOURCE = 'raw'  # raw or fuse
cfg.MODEL.TEC.TARGET_POOL = 'center'
cfg.MODEL.TEC.CENTER_RATIO = 0.5
cfg.MODEL.TEC.MIN_VALID_TOKENS = 3
cfg.MODEL.TEC.DROPOUT = 0.0
cfg.MODEL.TEC.FREEZE_BACKBONE = False
cfg.MODEL.TEC.FREEZE_HEAD = False

# MODEL.EVIDENCE_LAYER
# Stage-2 explicit evidence layer. Keep it separate from Stage-1 TEC to avoid
# mixing residual calibration with evidence-layer experiments.
cfg.MODEL.EVIDENCE_LAYER = edict()
cfg.MODEL.EVIDENCE_LAYER.ENABLE = False
cfg.MODEL.EVIDENCE_LAYER.EVIDENCE_DIM = 128
cfg.MODEL.EVIDENCE_LAYER.GAMMA_INIT = 0.01
cfg.MODEL.EVIDENCE_LAYER.BETA = 0.25
cfg.MODEL.EVIDENCE_LAYER.D_MAG_MAX = 1.0
cfg.MODEL.EVIDENCE_LAYER.D_NORM_EPS = 1e-4
cfg.MODEL.EVIDENCE_LAYER.RESIDUAL_INIT_SCALE = 1e-3
cfg.MODEL.EVIDENCE_LAYER.LANG_SOURCE = 'raw'  # raw or fuse
cfg.MODEL.EVIDENCE_LAYER.TARGET_POOL = 'center'
cfg.MODEL.EVIDENCE_LAYER.CENTER_RATIO = 0.5
cfg.MODEL.EVIDENCE_LAYER.MIN_VALID_TOKENS = 3
cfg.MODEL.EVIDENCE_LAYER.NUM_EVIDENCE_SLOTS = 4
cfg.MODEL.EVIDENCE_LAYER.ATTENTION_UNIFORM_MIX = 0.05
cfg.MODEL.EVIDENCE_LAYER.DROPOUT = 0.0
cfg.MODEL.EVIDENCE_LAYER.FREEZE_BACKBONE = False
cfg.MODEL.EVIDENCE_LAYER.FREEZE_HEAD = False

# MODEL.EVIDENCE_UNIT_LAYER
# Stage-2R phrase-aware evidence-unit layer. This is independent from the
# Stage-2 token-level EvidenceLayer: language is first lifted into local
# phrase/evidence units, then search regions read those units in a
# target-conditioned evidence space.
cfg.MODEL.EVIDENCE_UNIT_LAYER = edict()
cfg.MODEL.EVIDENCE_UNIT_LAYER.ENABLE = False
cfg.MODEL.EVIDENCE_UNIT_LAYER.EVIDENCE_DIM = 128
cfg.MODEL.EVIDENCE_UNIT_LAYER.GAMMA_INIT = 0.01
cfg.MODEL.EVIDENCE_UNIT_LAYER.BETA = 0.25
cfg.MODEL.EVIDENCE_UNIT_LAYER.D_MAG_MAX = 1.0
cfg.MODEL.EVIDENCE_UNIT_LAYER.D_NORM_EPS = 1e-4
cfg.MODEL.EVIDENCE_UNIT_LAYER.RESIDUAL_INIT_SCALE = 1e-3
cfg.MODEL.EVIDENCE_UNIT_LAYER.LANG_SOURCE = 'raw'  # raw or fuse
cfg.MODEL.EVIDENCE_UNIT_LAYER.TARGET_POOL = 'center'
cfg.MODEL.EVIDENCE_UNIT_LAYER.CENTER_RATIO = 0.5
cfg.MODEL.EVIDENCE_UNIT_LAYER.MIN_EVIDENCE_UNITS = 2
cfg.MODEL.EVIDENCE_UNIT_LAYER.PHRASE_WINDOW = 3
cfg.MODEL.EVIDENCE_UNIT_LAYER.DROPOUT = 0.0
cfg.MODEL.EVIDENCE_UNIT_LAYER.FREEZE_BACKBONE = False
cfg.MODEL.EVIDENCE_UNIT_LAYER.FREEZE_HEAD = False


# TRAIN
cfg.TRAIN = edict()
cfg.TRAIN.LR = 0.0001
cfg.TRAIN.WEIGHT_DECAY = 0.0001
cfg.TRAIN.EPOCH = 500
cfg.TRAIN.LR_DROP_EPOCH = 400
cfg.TRAIN.BATCH_SIZE = 16
cfg.TRAIN.NUM_WORKER = 8
cfg.TRAIN.OPTIMIZER = "ADAMW"
cfg.TRAIN.BACKBONE_MULTIPLIER = 0.1
cfg.TRAIN.GIOU_WEIGHT = 2.0
cfg.TRAIN.L1_WEIGHT = 5.0
cfg.TRAIN.FREEZE_LAYERS = [0, ]
cfg.TRAIN.PRINT_INTERVAL = 50
cfg.TRAIN.VAL_EPOCH_INTERVAL = 20
cfg.TRAIN.KEEP_LAST_CHECKPOINTS = 0
cfg.TRAIN.KEEP_CHECKPOINT_EPOCHS = []
cfg.TRAIN.GRAD_CLIP_NORM = 0.1
cfg.TRAIN.AMP = False
cfg.TRAIN.BBOX_TASK = False

cfg.TRAIN.CE_START_EPOCH = 20  # candidate elimination start epoch
cfg.TRAIN.CE_WARM_EPOCH = 80  # candidate elimination warm up epoch
cfg.TRAIN.DROP_PATH_RATE = 0.1  # drop path rate for ViT backbone

# TRAIN.SCHEDULER
cfg.TRAIN.SCHEDULER = edict()
cfg.TRAIN.SCHEDULER.TYPE = "step"
cfg.TRAIN.SCHEDULER.DECAY_RATE = 0.1

# DATA
cfg.DATA = edict()
cfg.DATA.SAMPLER_MODE = "causal"  # sampling methods
cfg.DATA.MEAN = [0.485, 0.456, 0.406]
cfg.DATA.STD = [0.229, 0.224, 0.225]
cfg.DATA.MAX_SAMPLE_INTERVAL = 200
# DATA.TRAIN
cfg.DATA.TRAIN = edict()
cfg.DATA.TRAIN.DATASETS_NAME = ["LASOT", "GOT10K_vottrain"]
cfg.DATA.TRAIN.DATASETS_RATIO = [1, 1]
cfg.DATA.TRAIN.SAMPLE_PER_EPOCH = 60000
# DATA.VAL
cfg.DATA.VAL = edict()
cfg.DATA.VAL.DATASETS_NAME = ["GOT10K_votval"]
cfg.DATA.VAL.DATASETS_RATIO = [1]
cfg.DATA.VAL.SAMPLE_PER_EPOCH = 10000
# DATA.SEARCH
cfg.DATA.SEARCH = edict()
cfg.DATA.SEARCH.SIZE = 320
cfg.DATA.SEARCH.FACTOR = 5.0
cfg.DATA.SEARCH.CENTER_JITTER = 4.5
cfg.DATA.SEARCH.SCALE_JITTER = 0.5
cfg.DATA.SEARCH.NUMBER = 1
# DATA.TEMPLATE
cfg.DATA.TEMPLATE = edict()
cfg.DATA.TEMPLATE.NUMBER = 1
cfg.DATA.TEMPLATE.SIZE = 128
cfg.DATA.TEMPLATE.FACTOR = 2.0
cfg.DATA.TEMPLATE.CENTER_JITTER = 0
cfg.DATA.TEMPLATE.SCALE_JITTER = 0

# TEST
cfg.TEST = edict()
cfg.TEST.TEMPLATE_FACTOR = 2.0
cfg.TEST.TEMPLATE_SIZE = 128
cfg.TEST.TEMPLATE_NUMBER = 1
cfg.TEST.MEMORY_THRESHOLD = 1000
cfg.TEST.SEARCH_FACTOR = 5.0
cfg.TEST.SEARCH_SIZE = 320
cfg.TEST.EPOCH = 500
cfg.TEST.CHECKPOINT_NAME = ''

# Language sensitivity test: normal, shuffle, wrong, generic, no_update
cfg.TEST.LANG_MODE = 'normal'


def _edict2dict(dest_dict, src_edict):
    if isinstance(dest_dict, dict) and isinstance(src_edict, dict):
        for k, v in src_edict.items():
            if not isinstance(v, edict):
                dest_dict[k] = v
            else:
                dest_dict[k] = {}
                _edict2dict(dest_dict[k], v)
    else:
        return


def gen_config(config_file):
    cfg_dict = {}
    _edict2dict(cfg_dict, cfg)
    with open(config_file, 'w') as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)


def _update_config(base_cfg, exp_cfg):
    if isinstance(base_cfg, dict) and isinstance(exp_cfg, edict):
        for k, v in exp_cfg.items():
            if k in base_cfg:
                if not isinstance(v, dict):
                    base_cfg[k] = v
                else:
                    _update_config(base_cfg[k], v)
            else:
                raise ValueError("{} not exist in config.py".format(k))
    else:
        return


def update_config_from_file(filename, base_cfg=None):
    exp_config = None
    with open(filename) as f:
        exp_config = edict(yaml.safe_load(f))
        if base_cfg is not None:
            _update_config(base_cfg, exp_config)
        else:
            _update_config(cfg, exp_config)
