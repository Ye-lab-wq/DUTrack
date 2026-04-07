import cv2
import numpy as np
import sys
import os
import textwrap

import torch


def _draw_box_with_label(img, box, color, label, label_pos='top', font_scale=0.45, thickness=1):
    x, y, w, h = [int(round(v)) for v in box]
    cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
    if label_pos == 'bottom':
        text_xy = (x, min(img.shape[0] - 8, y + h + 18))
    else:
        text_xy = (x, max(18, y - 6))
    cv2.putText(img, label, text_xy, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


def _map_box_to_view(box, view_box, out_w, out_h):
    vx, vy, vw, vh = view_box
    if vw <= 1e-6 or vh <= 1e-6:
        return None
    sx = out_w / float(vw)
    sy = out_h / float(vh)
    bx, by, bw, bh = box
    return [(bx - vx) * sx, (by - vy) * sy, bw * sx, bh * sy]


def _draw_patch_grid(img, feat_sz, color=(235, 235, 235), thickness=1):
    if feat_sz <= 1:
        return
    h, w = img.shape[:2]
    for i in range(1, feat_sz):
        x = int(round(i * w / float(feat_sz)))
        y = int(round(i * h / float(feat_sz)))
        cv2.line(img, (x, 0), (x, h - 1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (0, y), (w - 1, y), color, thickness, cv2.LINE_AA)


def _draw_topk_neighborhoods(img, top_indices, feat_sz, radius=1, border_color=(255, 255, 255), border_thickness=2):
    if top_indices is None:
        return img
    if isinstance(top_indices, torch.Tensor):
        top_indices = top_indices.detach().view(-1).cpu().tolist()
    if len(top_indices) == 0:
        return img

    h, w = img.shape[:2]
    for idx in top_indices:
        row = int(idx) // feat_sz
        col = int(idx) % feat_sz
        r0 = max(0, row - radius)
        r1 = min(feat_sz - 1, row + radius)
        c0 = max(0, col - radius)
        c1 = min(feat_sz - 1, col + radius)

        x0 = int(round(c0 * w / float(feat_sz)))
        y0 = int(round(r0 * h / float(feat_sz)))
        x1 = int(round((c1 + 1) * w / float(feat_sz))) - 1
        y1 = int(round((r1 + 1) * h / float(feat_sz))) - 1
        cv2.rectangle(img, (x0, y0), (x1, y1), border_color, border_thickness)
    return img


def vis_mask_token(heat_data, img=None, show_size=(150, 150), factor=0.4, window="feature"):
    """ 可视化特征

    Args:
        heat_data (_type_): (H, W)
        img (_type_, optional): _description_. Defaults to None.
        show_size (tuple, optional): _description_. Defaults to (150, 150).
        factor (float, optional): _description_. Defaults to 0.4.
        window (str, optional): _description_. Defaults to "feature".

    Returns:
        _type_: _description_
    """
    heat_data = heat_data.cpu().numpy()
    heat_data = cv2.resize(heat_data, show_size)

    heat_data_x = heat_data
    Min = np.min(heat_data_x)
    Max = np.max(heat_data_x)
    Sum = np.mean(heat_data_x)

    # sys.float_info.epsilon：是一个极小的数，用于避免除数为0的情况，即 heat_data矩阵为0的情况
    # heat_data_max = (heat_data_x - Min) / (Max - Min + sys.float_info.epsilon)
    if (Max - Min) != 0 and not np.isnan(Max - Min):
        heat_data_max = (heat_data_x - Min) / (Max - Min)
    else:
        heat_data_max = (heat_data_x - Min) / (Max - Min + sys.float_info.epsilon)

    heat_data = heat_data_max

    heat_data = np.uint8(255 * heat_data)
    heat_data = cv2.applyColorMap(heat_data, cv2.COLORMAP_JET)

    if img is not None:
        img = cv2.resize(img, show_size)
        heat_map_data = np.uint8(img * (1 - factor) + heat_data * factor)
    else:
        heat_map_data = heat_data

    font = cv2.FONT_HERSHEY_SIMPLEX
    heat_map_data = cv2.putText(heat_map_data, window, (0, 0), color=(255, 0, 0), fontFace=font, fontScale=1.2)
    return heat_map_data

def visualize_attn(attn, img,dataset_name,frame_num):
    '''
    img: (3,256,256)
    attn: (1,8,1,256)
    '''
    # print(attn)
    """
    将attn拆分为8个部分，每个部分生成一张彩色图片并保存到本地文件。
    img: 3x256x256的张量，表示原始图片。
    attn: 8x1x256的张量，表示注意力图。
    """
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    attn = attn[0].squeeze(0)
    attn = torch.mean(attn,dim=0)
    #print(attn.shape, img.shape, 'decoder_visualize')
    attn = attn[-256:,]
    heatmap_data =  vis_mask_token(attn.reshape(16,16), img)

    path = '/home/local_data/lxh/data/VLresult/Visual_attn/%s/%s.jpg'%(dataset_name,frame_num)
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    cv2.imwrite(path, heatmap_data)#如果路径不存在，保存失败，但是不会报错


def visualize_cls_l2s(attn_l2s, img, save_path, top_indices=None, title="CLS->Search"):
    if isinstance(attn_l2s, torch.Tensor):
        attn_l2s = attn_l2s.detach().float().cpu()
    else:
        attn_l2s = torch.tensor(attn_l2s, dtype=torch.float32)

    num_tokens = int(attn_l2s.numel())
    feat_sz = int(round(num_tokens ** 0.5))
    if feat_sz * feat_sz != num_tokens:
        raise ValueError("Search token count is not a square number: {}".format(num_tokens))

    heat = attn_l2s.reshape(feat_sz, feat_sz)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    overlay = vis_mask_token(heat, img_bgr, show_size=(img.shape[1], img.shape[0]), factor=0.45, window=title)
    _draw_patch_grid(overlay, feat_sz)
    _draw_topk_neighborhoods(overlay, top_indices, feat_sz)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, overlay)


def visualize_cls_l2s_with_context(attn_l2s, search_img, orig_img, save_path, top_indices=None,
                                   search_crop_box=None, ref_box=None, pred_box=None,
                                   description=None, status_lines=None,
                                   title="CLS->Search"):
    if isinstance(attn_l2s, torch.Tensor):
        attn_l2s = attn_l2s.detach().float().cpu()
    else:
        attn_l2s = torch.tensor(attn_l2s, dtype=torch.float32)

    num_tokens = int(attn_l2s.numel())
    feat_sz = int(round(num_tokens ** 0.5))
    if feat_sz * feat_sz != num_tokens:
        raise ValueError("Search token count is not a square number: {}".format(num_tokens))

    heat = attn_l2s.reshape(feat_sz, feat_sz)
    search_bgr = cv2.cvtColor(search_img, cv2.COLOR_RGB2BGR)
    overlay = vis_mask_token(heat, search_bgr, show_size=(search_img.shape[1], search_img.shape[0]), factor=0.45, window=title)
    _draw_patch_grid(overlay, feat_sz)
    _draw_topk_neighborhoods(overlay, top_indices, feat_sz)

    orig_bgr = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR).copy()
    if search_crop_box is not None:
        _draw_box_with_label(orig_bgr, search_crop_box, (0, 255, 255), 'search crop', font_scale=0.48, thickness=1)

    if ref_box is not None:
        _draw_box_with_label(orig_bgr, ref_box, (0, 255, 0), 'prev box', 'bottom', font_scale=0.48, thickness=1)

    if pred_box is not None:
        _draw_box_with_label(orig_bgr, pred_box, (0, 0, 255), 'pred box', font_scale=0.48, thickness=1)

    search_panel = cv2.cvtColor(search_img, cv2.COLOR_RGB2BGR)
    # Build a slightly expanded context panel around the original search crop.
    if search_crop_box is not None:
        sx, sy, sw, sh = [float(v) for v in search_crop_box]
        expand_ratio = 0.10
        ex = sw * expand_ratio
        ey = sh * expand_ratio
        vx1 = max(0.0, sx - ex)
        vy1 = max(0.0, sy - ey)
        vx2 = min(float(orig_bgr.shape[1]), sx + sw + ex)
        vy2 = min(float(orig_bgr.shape[0]), sy + sh + ey)
        vw = max(1.0, vx2 - vx1)
        vh = max(1.0, vy2 - vy1)
        view_box = [vx1, vy1, vw, vh]

        crop = orig_bgr[int(round(vy1)):int(round(vy2)), int(round(vx1)):int(round(vx2))].copy()
        if crop.size > 0:
            search_panel = cv2.resize(crop, (search_img.shape[1], search_img.shape[0]))
        if ref_box is not None:
            mapped_ref = _map_box_to_view(ref_box, view_box, search_panel.shape[1], search_panel.shape[0])
            if mapped_ref is not None:
                _draw_box_with_label(search_panel, mapped_ref, (0, 255, 0), '', 'bottom', font_scale=0.48, thickness=1)
        if pred_box is not None:
            mapped_pred = _map_box_to_view(pred_box, view_box, search_panel.shape[1], search_panel.shape[0])
            if mapped_pred is not None:
                _draw_box_with_label(search_panel, mapped_pred, (0, 0, 255), '', font_scale=0.48, thickness=1)

    cv2.putText(overlay, 'CLS->search', (12, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.58, (20, 20, 20), 1, cv2.LINE_AA)

    top_h = max(search_panel.shape[0], overlay.shape[0])
    top_w = search_panel.shape[1] + overlay.shape[1]
    canvas_w = max(top_w, orig_bgr.shape[1])
    canvas_h = top_h + orig_bgr.shape[0]
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    top_x = (canvas_w - top_w) // 2
    search_y = (top_h - search_panel.shape[0]) // 2
    overlay_y = (top_h - overlay.shape[0]) // 2
    canvas[search_y:search_y + search_panel.shape[0], top_x:top_x + search_panel.shape[1]] = search_panel
    overlay_x = top_x + search_panel.shape[1]
    canvas[overlay_y:overlay_y + overlay.shape[0], overlay_x:overlay_x + overlay.shape[1]] = overlay

    bottom_x = (canvas_w - orig_bgr.shape[1]) // 2
    canvas[top_h:top_h + orig_bgr.shape[0], bottom_x:bottom_x + orig_bgr.shape[1]] = orig_bgr

    header_lines = []
    if description:
        description = 'Description: {}'.format(str(description).strip())
        header_lines.extend(textwrap.wrap(description, width=90) or [description])
    if status_lines:
        for line in status_lines:
            header_lines.extend(textwrap.wrap(str(line).strip(), width=90) or [str(line).strip()])

    if header_lines:
        line_h = 26
        pad_top = 12 + line_h * len(header_lines)
        text_canvas = np.full((pad_top, canvas.shape[1], 3), 255, dtype=np.uint8)
        for i, line in enumerate(header_lines):
            y = 26 + i * line_h
            cv2.putText(text_canvas, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (20, 20, 20), 2, cv2.LINE_AA)
        canvas = np.concatenate([text_canvas, canvas], axis=0)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, canvas)
