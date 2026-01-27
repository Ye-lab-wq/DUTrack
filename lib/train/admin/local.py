class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/home/b520/Downloads/yelin/DUTrack'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/home/b520/Downloads/yelin/DUTrack/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/home/b520/Downloads/yelin/DUTrack/pretrained_networks'
        self.lasot_dir = '/home/b520/Downloads/yelin/data/lasot'
        self.got10k_dir = '/home/b520/Downloads/yelin/data/got10k/train'
        self.got10k_val_dir = '/home/b520/Downloads/yelin/data/got10k/val'
        self.lasot_lmdb_dir = '/home/b520/Downloads/yelin/data/lasot_lmdb'
        self.got10k_lmdb_dir = '/home/b520/Downloads/yelin/data/got10k_lmdb'
        self.trackingnet_dir = '/home/b520/Downloads/yelin/data/trackingnet'
        self.trackingnet_lmdb_dir = '/home/b520/Downloads/yelin/data/trackingnet_lmdb'
        self.coco_dir = '/home/b520/Downloads/yelin/data/coco'
        self.coco_lmdb_dir = '/home/b520/Downloads/yelin/data/coco_lmdb'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = '/home/b520/Downloads/yelin/data/vid'
        self.imagenet_lmdb_dir = '/home/b520/Downloads/yelin/data/vid_lmdb'
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.tnl2k_dir = '/home/b520/Downloads/yelin/data/TNL2K'
        self.mgit_dir = '/home/b520/Downloads/yelin/data/MGIT'
