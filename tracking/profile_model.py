import os
import sys

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

import argparse
import importlib
import time

import numpy as np
import torch
from lib.utils.misc import NestedTensor
from thop import profile
from thop.utils import clever_format


def parse_args():
    parser = argparse.ArgumentParser(description='Profile FLOPs/Params and speed')
    parser.add_argument('--script', type=str, default='dutrack', choices=['dutrack'],
                        help='training script name')
    parser.add_argument('--config', type=str, default='dutrack_384_full', help='yaml configure file name')

    parser.add_argument('--warmup_iters', type=int, default=500)
    parser.add_argument('--test_iters', type=int, default=1000)

    parser.add_argument('--dynamic_descript', type=int, default=0,
                        help='Include BLIP->text description generation in the timing (0/1).')
    parser.add_argument('--descript_update_interval', type=int, default=1,
                        help='Update description every N iterations when --dynamic_descript=1.')
    parser.add_argument('--descript_cls', type=str, default='object',
                        help='Text prompt/class passed to BLIP; use "none" to disable.')
    parser.add_argument('--descript_image_size', type=int, default=384,
                        help='Dummy image size for BLIP description generation.')

    return parser.parse_args()


def evaluate_dutrack(model, template, search, descript, warmup_iters=500, test_iters=1000):
    macs1, params1 = profile(model, inputs=(template, search, descript), custom_ops=None, verbose=False)
    macs, params = clever_format([macs1, params1], "%.3f")
    print('overall macs is ', macs)
    print('overall params is ', params)

    print("testing speed (network only) ...")
    torch.cuda.synchronize()
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(template, search, descript)
        start = time.time()
        for _ in range(test_iters):
            _ = model(template, search, descript)
        torch.cuda.synchronize()
        end = time.time()

    avg_lat = (end - start) / test_iters
    print("The average overall latency is %.2f ms" % (avg_lat * 1000))
    print("FPS is %.2f fps" % (1. / avg_lat))


def evaluate_dutrack_with_dynamic_descript(
    model,
    template,
    search,
    descript_refiner,
    refiner_image,
    refiner_cls,
    warmup_iters=500,
    test_iters=1000,
    update_interval=1,
):
    print(f"testing speed (end-to-end, update descript every {update_interval} iters) ...")

    def maybe_update_descript(iter_idx, current_descript):
        if iter_idx % update_interval != 0:
            return current_descript, 0.0
        torch.cuda.synchronize()
        t0 = time.time()
        new_desc = descript_refiner(refiner_image, cls=refiner_cls)
        torch.cuda.synchronize()
        t1 = time.time()
        return new_desc, (t1 - t0)

    total_update_time = 0.0
    total_forward_time = 0.0

    with torch.no_grad():
        cur_desc = descript_refiner(refiner_image, cls=refiner_cls)

        # warmup
        for i in range(warmup_iters):
            cur_desc, _ = maybe_update_descript(i, cur_desc)
            descript = [[cur_desc] for _ in range(len(search))]
            _ = model(template, search, descript)

        # timed
        torch.cuda.synchronize()
        start = time.time()
        for i in range(test_iters):
            cur_desc, dt_update = maybe_update_descript(i, cur_desc)
            total_update_time += dt_update

            descript = [[cur_desc] for _ in range(len(search))]

            torch.cuda.synchronize()
            t0 = time.time()
            _ = model(template, search, descript)
            torch.cuda.synchronize()
            t1 = time.time()
            total_forward_time += (t1 - t0)

        torch.cuda.synchronize()
        end = time.time()

    avg_lat = (end - start) / test_iters
    print("The average end-to-end latency is %.2f ms" % (avg_lat * 1000))
    print("End-to-end FPS is %.2f fps" % (1. / avg_lat))

    if update_interval > 0:
        num_updates = (test_iters + update_interval - 1) // update_interval
        if num_updates > 0:
            print("Avg descript update latency: %.2f ms" % ((total_update_time / num_updates) * 1000))
    print("Avg model forward latency (measured with sync): %.2f ms" % ((total_forward_time / test_iters) * 1000))


def evaluate_vit_separate(model, template, search):
    raise NotImplementedError(
        "DUTrack model does not implement forward_backbone/forward_cat; use evaluate_dutrack instead."
    )


def get_data(bs, sz):
    img_patch = torch.randn(bs, 3, sz, sz)
    att_mask = torch.rand(bs, sz, sz) > 0.5
    return NestedTensor(img_patch, att_mask)


if __name__ == "__main__":
    device = "cuda:0"
    torch.cuda.set_device(device)

    args = parse_args()

    yaml_fname = 'experiments/%s/%s.yaml' % (args.script, args.config)
    config_module = importlib.import_module('lib.config.%s.config' % args.script)
    cfg = config_module.cfg
    config_module.update_config_from_file(yaml_fname)

    bs = 1
    z_sz = cfg.TEST.TEMPLATE_SIZE
    x_sz = cfg.TEST.SEARCH_SIZE

    if args.script == "dutrack":
        model_module = importlib.import_module('lib.models')
        model_constructor = model_module.build_dutrack
        model = model_constructor(cfg, training=False)

        template = torch.randn(bs, 3, z_sz, z_sz)
        search = torch.randn(bs, 3, x_sz, x_sz)

        model = model.to(device)
        template = template.to(device)
        search = search.to(device)

        template_len = 3
        template_list = [template for _ in range(template_len)]
        search_list = [search]

        descript = [["a target object"] for _ in range(len(search_list))]
        evaluate_dutrack(
            model,
            template_list,
            search_list,
            descript,
            warmup_iters=args.warmup_iters,
            test_iters=args.test_iters,
        )

        if args.dynamic_descript:
            from lib.models.dutrack.i2d import descriptgenRefiner

            cls = None if args.descript_cls.lower() == 'none' else args.descript_cls
            dummy_img = np.random.randint(
                0,
                256,
                size=(args.descript_image_size, args.descript_image_size, 3),
                dtype=np.uint8,
            )

            refiner = descriptgenRefiner(cfg.MODEL.BACKBONE.BLIP_DIR, cfg.MODEL.BACKBONE.BERT_DIR)
            evaluate_dutrack_with_dynamic_descript(
                model,
                template_list,
                search_list,
                refiner,
                dummy_img,
                cls,
                warmup_iters=args.warmup_iters,
                test_iters=args.test_iters,
                update_interval=max(1, args.descript_update_interval),
            )
    else:
        raise NotImplementedError
