import csv
import math
import os
import re
from pathlib import Path
from matplotlib.path import Path as MplPath
from statistics import mean, pstdev

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import gaussian_filter, maximum_filter, binary_fill_holes
from scipy.spatial import ConvexHull, QhullError
from skimage.measure import regionprops
from skimage.segmentation import watershed


# =========================
# 基础工具函数
# =========================
def sanitize_filename(name):
    return re.sub(r'[<>:"/\\|?*]+', '_', str(name))


def safe_output_dir(folder):
    folder = Path(folder).expanduser().resolve()
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def safe_savefig(fig, folder, filename, dpi=400, **kwargs):
    out_dir = safe_output_dir(folder)
    out_path = out_dir / sanitize_filename(filename)
    fig.savefig(str(out_path), dpi=dpi, **kwargs)
    return str(out_path)


def set_publication_rcparams():
    """统一图件基础风格，偏向期刊投稿版式。"""
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 8.5,
        'ytick.labelsize': 8.5,
        'legend.fontsize': 9,
        'figure.titlesize': 13,
        'savefig.dpi': 600,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'axes.linewidth': 0.8
    })


# =========================
# 几何计算函数
# =========================
def calculate_crown_angle(tree_top, point1, point2):
    """计算树顶到两个端点构成的真实 3D 夹角（0-180°）"""
    tree_top = np.asarray(tree_top, dtype=float)
    point1 = np.asarray(point1, dtype=float)
    point2 = np.asarray(point2, dtype=float)

    v1 = point1 - tree_top
    v2 = point2 - tree_top

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0

    cosine = np.dot(v1, v2) / (n1 * n2)
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def draw_3d_angle_sector(ax, tree_top, point1, point2,
                         facecolor='#74A57F', edgecolor='#FF7F0E',
                         alpha=0.26, linewidth=2.2):
    """绘制树顶到两个端点之间的 3D 扇形"""
    tree_top = np.asarray(tree_top, dtype=float)
    point1 = np.asarray(point1, dtype=float)
    point2 = np.asarray(point2, dtype=float)

    v1 = point1 - tree_top
    v2 = point2 - tree_top

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return None

    u1 = v1 / n1
    u2 = v2 / n2

    cosang = np.clip(np.dot(u1, u2), -1.0, 1.0)
    theta = np.arccos(cosang)
    if theta < 1e-9:
        return None

    radius = 0.30 * min(n1, n2)
    radius = max(radius, 0.45)

    ts = np.linspace(0.0, 1.0, 40)
    sin_theta = np.sin(theta)
    arc_pts = []
    for t in ts:
        a = np.sin((1.0 - t) * theta) / max(sin_theta, 1e-12)
        b = np.sin(t * theta) / max(sin_theta, 1e-12)
        direction = a * u1 + b * u2
        direction /= max(np.linalg.norm(direction), 1e-12)
        arc_pts.append(tree_top + radius * direction)
    arc_pts = np.asarray(arc_pts)

    verts = np.vstack([tree_top, arc_pts, tree_top])
    patch = Poly3DCollection([verts], facecolors=facecolor, edgecolors='none', alpha=alpha)
    ax.add_collection3d(patch)
    ax.plot(arc_pts[:, 0], arc_pts[:, 1], arc_pts[:, 2],
            color=edgecolor, linewidth=linewidth, solid_capstyle='round')
    return arc_pts


def _scanline_extreme_pair(points_3d, width_axis=0, bin_width=0.10,
                            min_scanline_points=2, max_height_diff=None):

    if points_3d is None or points_3d.shape[0] < 2:
        return None, None, 0.0, 0.0, False, None

    points_3d = np.asarray(points_3d, dtype=float)
    if width_axis not in (0, 1):
        raise ValueError('width_axis must be 0 for EW or 1 for NS.')

    fixed_axis = 1 - width_axis
    bin_width = float(bin_width)
    if bin_width <= 0:
        raise ValueError('bin_width must be positive.')

    fixed_vals = points_3d[:, fixed_axis]
    fixed_min = float(np.nanmin(fixed_vals))
    bin_ids = np.floor((fixed_vals - fixed_min) / bin_width).astype(int)

    best_p1 = None
    best_p2 = None
    best_width = 0.0
    best_zdiff = 0.0
    best_fixed_value = None

    for bid in np.unique(bin_ids):
        pts = points_3d[bin_ids == bid]
        if pts.shape[0] < min_scanline_points:
            continue

        fixed_value = float(np.nanmedian(pts[:, fixed_axis]))

        if max_height_diff is None:
            vals = pts[:, width_axis]
            i_min = int(np.nanargmin(vals))
            i_max = int(np.nanargmax(vals))
            raw_p1 = pts[i_min].copy()
            raw_p2 = pts[i_max].copy()
            width = float(abs(raw_p2[width_axis] - raw_p1[width_axis]))
            z_diff = float(abs(raw_p2[2] - raw_p1[2]))
        else:
            raw_p1 = None
            raw_p2 = None
            width = 0.0
            z_diff = 0.0
            n = pts.shape[0]
            for i in range(n):
                for j in range(i + 1, n):
                    zi = float(pts[i, 2])
                    zj = float(pts[j, 2])
                    zd = abs(zi - zj)
                    if zd > max_height_diff:
                        continue
                    w = float(abs(pts[j, width_axis] - pts[i, width_axis]))
                    if w > width:
                        width = w
                        z_diff = zd
                        raw_p1 = pts[i].copy()
                        raw_p2 = pts[j].copy()
            if raw_p1 is None or raw_p2 is None:
                continue

        if width > best_width:
            # 构造“同一扫描线”端点：固定轴坐标统一为该扫描带中位数。
            p1 = raw_p1.copy()
            p2 = raw_p2.copy()
            p1[fixed_axis] = fixed_value
            p2[fixed_axis] = fixed_value

            # 确保 p1 是较小端，p2 是较大端，便于可视化和检查。
            if p1[width_axis] > p2[width_axis]:
                p1, p2 = p2, p1

            best_p1 = p1
            best_p2 = p2
            best_width = width
            best_zdiff = z_diff
            best_fixed_value = fixed_value

    return best_p1, best_p2, float(best_width), float(best_zdiff), best_p1 is not None, best_fixed_value


def find_ew_ns_mean_crown_width(points_3d, max_height_diff=None,
                                scan_bin_width=0.10, min_scanline_points=2):

    if points_3d is None or points_3d.shape[0] < 2:
        return {
            'p_ew1': None, 'p_ew2': None, 'p_ns1': None, 'p_ns2': None,
            'd_ew': 0.0, 'd_ns': 0.0, 'dmean': 0.0,
            'z_diff_ew': 0.0, 'z_diff_ns': 0.0,
            'ew_fixed_y': None, 'ns_fixed_x': None,
            'found_ew': False, 'found_ns': False, 'found': False
        }

    p_ew1, p_ew2, d_ew, zd_ew, found_ew, ew_fixed_y = _scanline_extreme_pair(
        points_3d,
        width_axis=0,
        bin_width=scan_bin_width,
        min_scanline_points=min_scanline_points,
        max_height_diff=max_height_diff
    )
    p_ns1, p_ns2, d_ns, zd_ns, found_ns, ns_fixed_x = _scanline_extreme_pair(
        points_3d,
        width_axis=1,
        bin_width=scan_bin_width,
        min_scanline_points=min_scanline_points,
        max_height_diff=max_height_diff
    )

    found = bool(found_ew and found_ns)
    dmean = float((d_ew + d_ns) / 2.0) if found else 0.0

    return {
        'p_ew1': p_ew1, 'p_ew2': p_ew2,
        'p_ns1': p_ns1, 'p_ns2': p_ns2,
        'd_ew': float(d_ew), 'd_ns': float(d_ns), 'dmean': dmean,
        'z_diff_ew': float(zd_ew), 'z_diff_ns': float(zd_ns),
        'ew_fixed_y': ew_fixed_y, 'ns_fixed_x': ns_fixed_x,
        'found_ew': bool(found_ew), 'found_ns': bool(found_ns), 'found': found
    }


# =========================
# CHM / treetop / watershed
# =========================
def build_chm(points, res=0.2):
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    x_min, y_min = x.min(), y.min()
    x_max, y_max = x.max(), y.max()
    nx = int(np.ceil((x_max - x_min) / res)) + 1
    ny = int(np.ceil((y_max - y_min) / res)) + 1

    ix = np.floor((x - x_min) / res).astype(int)
    iy = np.floor((y - y_min) / res).astype(int)
    flat_idx = iy * nx + ix
    z_fill = np.full(ny * nx, -np.inf, dtype=float)
    np.maximum.at(z_fill, flat_idx, z)
    chm = z_fill.reshape(ny, nx)
    chm[chm == -np.inf] = np.nan
    return chm, x_min, y_min, res


def pixel_centers(start_idx, end_idx, origin_min, res):
    idx = np.arange(start_idx, end_idx)
    return origin_min + (idx + 0.5) * res


def get_tree_bbox_pixels(labels, label_id, margin_px=8):
    yy, xx = np.where(labels == label_id)
    if yy.size == 0:
        return None
    y1 = max(0, int(yy.min()) - margin_px)
    y2 = min(labels.shape[0], int(yy.max()) + margin_px + 1)
    x1 = max(0, int(xx.min()) - margin_px)
    x2 = min(labels.shape[1], int(xx.max()) + margin_px + 1)
    return x1, x2, y1, y2


def window_size_from_height(z, a=0.12, b=0.8, ws_min=1.0, ws_max=5.0):
    ws = a * float(z) + b
    return float(np.clip(ws, ws_min, ws_max))


def adaptive_prominence_ok(chm, y, x, r_px, thr=0.5):
    rin = max(1, int(round(0.7 * r_px)))
    rout = max(rin + 2, int(round(1.8 * r_px)))

    y1 = max(0, y - rout)
    y2 = min(chm.shape[0], y + rout + 1)
    x1 = max(0, x - rout)
    x2 = min(chm.shape[1], x + rout + 1)
    win = chm[y1:y2, x1:x2]
    if win.size == 0 or np.isnan(chm[y, x]):
        return False

    yy, xx = np.ogrid[y1:y2, x1:x2]
    dist2 = (yy - y) ** 2 + (xx - x) ** 2
    ring = (dist2 <= rout ** 2) & (dist2 >= rin ** 2)
    vals = win[ring & ~np.isnan(win)]
    if vals.size < 12:
        return True
    return (chm[y, x] - np.median(vals)) >= thr


def find_intertree_boundaries(labels):
    labels = np.asarray(labels)
    bounds = np.zeros_like(labels, dtype=bool)

    diff_ud = ((labels[:-1, :] != labels[1:, :]) & (labels[:-1, :] > 0) & (labels[1:, :] > 0))
    bounds[:-1, :] |= diff_ud
    bounds[1:, :] |= diff_ud

    diff_lr = ((labels[:, :-1] != labels[:, 1:]) & (labels[:, :-1] > 0) & (labels[:, 1:] > 0))
    bounds[:, :-1] |= diff_lr
    bounds[:, 1:] |= diff_lr
    return bounds


def detect_treetops_vwf(
        points,
        min_height=2.0,
        chm_res=0.15,
        smooth_sigma=0.5,
        vwf_a=0.08,
        vwf_b=0.35,
        ws_min=0.45,
        ws_max=2.2,
        prominence_min=0.15,
        min_crown_area_m2=0.08,
        debug=True
):
    pts = points[points[:, 2] >= min_height]
    if pts.shape[0] < 10:
        return np.empty((0, 3)), None

    chm, x_min, y_min, res = build_chm(pts, res=chm_res)
    valid = ~np.isnan(chm)
    if not np.any(valid):
        return np.empty((0, 3)), None

    fill_val = float(np.nanmin(chm) - 5.0)
    chm_fill = np.nan_to_num(chm, nan=fill_val)
    chm_s = gaussian_filter(chm_fill, sigma=smooth_sigma)
    chm_s[np.isnan(chm)] = np.nan
    mask = ~np.isnan(chm_s)

    local_max = maximum_filter(np.nan_to_num(chm_s, nan=-1e9), size=3, mode='nearest')
    cand_mask = mask & np.isfinite(chm_s) & (chm_s >= local_max - 1e-9)
    cand_coords = np.argwhere(cand_mask)
    if debug:
        print(f"[DEBUG] candidate local maxima: {len(cand_coords)}")

    if cand_coords.size == 0:
        vis = {
            "chm": chm,
            "mask": mask,
            "peaks_px": np.empty((0, 2), int),
            "labels": np.zeros_like(chm, int),
            "bounds": np.zeros_like(chm, bool),
            "x_min": x_min,
            "y_min": y_min,
            "res": res,
            "treetops_px": np.empty((0, 2), int),
            "treetop_records": []
        }
        return np.empty((0, 3)), vis

    cand_vals = chm_s[cand_coords[:, 0], cand_coords[:, 1]]
    order = np.argsort(-cand_vals)
    cand_coords = cand_coords[order]
    cand_vals = cand_vals[order]

    kept_coords = []
    kept_radii = []
    for (y, x), zc in zip(cand_coords, cand_vals):
        ws_m = window_size_from_height(zc, a=vwf_a, b=vwf_b, ws_min=ws_min, ws_max=ws_max)
        r_px = max(1, int(round((ws_m / res) / 2.0)))

        suppressed = False
        for (yk, xk), rk in zip(kept_coords, kept_radii):
            if (x - xk) ** 2 + (y - yk) ** 2 <= max(r_px, rk) ** 2:
                suppressed = True
                break
        if suppressed:
            continue

        y1 = max(0, y - r_px)
        y2 = min(chm_s.shape[0], y + r_px + 1)
        x1 = max(0, x - r_px)
        x2 = min(chm_s.shape[1], x + r_px + 1)
        win = chm_s[y1:y2, x1:x2]
        yy, xx = np.ogrid[y1:y2, x1:x2]
        circle = (yy - y) ** 2 + (xx - x) ** 2 <= r_px ** 2
        local_vals = win[circle & ~np.isnan(win)]
        if local_vals.size == 0:
            continue
        if chm_s[y, x] + 1e-9 < np.nanmax(local_vals):
            continue

        if not adaptive_prominence_ok(chm_s, y, x, r_px, thr=prominence_min):
            continue

        kept_coords.append((int(y), int(x)))
        kept_radii.append(int(r_px))

    peaks = np.array(kept_coords, dtype=int) if kept_coords else np.empty((0, 2), dtype=int)
    if debug:
        print(f"[DEBUG] peaks after VWF + NMS + prominence: {len(peaks)}")

    if peaks.shape[0] == 0:
        vis = {
            "chm": chm,
            "mask": mask,
            "peaks_px": np.empty((0, 2), int),
            "labels": np.zeros_like(chm, int),
            "bounds": np.zeros_like(chm, bool),
            "x_min": x_min,
            "y_min": y_min,
            "res": res,
            "treetops_px": np.empty((0, 2), int),
            "treetop_records": []
        }
        return np.empty((0, 3)), vis

    markers = np.zeros_like(chm, dtype=int)
    for i, (yy0, xx0) in enumerate(peaks, start=1):
        markers[yy0, xx0] = i

    labels = watershed(
        -np.nan_to_num(chm_s, nan=fill_val),
        markers=markers,
        mask=mask,
        watershed_line=False
    )

    treetops = []
    treetops_px = []
    treetop_records = []

    for region in regionprops(labels, intensity_image=np.nan_to_num(chm, nan=fill_val)):
        if region.label == 0:
            continue

        area_m2 = float(region.area * res * res)
        if area_m2 < min_crown_area_m2:
            continue

        coords = region.coords
        vals = chm[coords[:, 0], coords[:, 1]]
        if np.all(np.isnan(vals)):
            continue
        arg = int(np.nanargmax(vals))
        y, x = coords[arg]
        z_top = float(chm[y, x])
        if np.isnan(z_top):
            continue

        xw = x_min + (x + 0.5) * res
        yw = y_min + (y + 0.5) * res
        crown_radius_m = float(np.sqrt(area_m2 / np.pi))
        crown_diameter_m = 2.0 * crown_radius_m

        record = {
            "label_id": int(region.label),
            "position": [float(xw), float(yw), float(z_top)],
            "crown_area_m2": area_m2,
            "crown_radius_m": crown_radius_m,
            "crown_diameter_m": crown_diameter_m
        }
        treetops.append(record["position"])
        treetops_px.append([y, x])
        treetop_records.append(record)

    if debug:
        print(f"[DEBUG] final treetops after watershed area filter: {len(treetops)}")

    vis = {
        "chm": chm,
        "mask": mask,
        "peaks_px": peaks,
        "labels": labels,
        "bounds": find_intertree_boundaries(labels),
        "x_min": x_min,
        "y_min": y_min,
        "res": res,
        "treetops_px": np.array(treetops_px, dtype=int) if treetops_px else np.empty((0, 2), int),
        "treetop_records": treetop_records
    }
    return np.array(treetops, dtype=float), vis


# =========================
# 候选点提取
# =========================
def harvest_points_in_zone(
        points, x0, y0,
        z_low, z_high,
        xy_radius,
        labels, x_min_chm, y_min_chm, res,
        min_points=5,
        filter_by_label=True
):
    mask_xyz = (
        (points[:, 0] >= x0 - xy_radius) & (points[:, 0] <= x0 + xy_radius) &
        (points[:, 1] >= y0 - xy_radius) & (points[:, 1] <= y0 + xy_radius) &
        (points[:, 2] >= z_low) & (points[:, 2] <= z_high)
    )
    cand = points[mask_xyz]

    if cand.shape[0] < min_points:
        return cand

    if filter_by_label:
        ix_top = int(np.floor((x0 - x_min_chm) / res))
        iy_top = int(np.floor((y0 - y_min_chm) / res))
        if 0 <= iy_top < labels.shape[0] and 0 <= ix_top < labels.shape[1]:
            label_id = labels[iy_top, ix_top]
            if label_id != 0:
                ix = np.floor((cand[:, 0] - x_min_chm) / res).astype(int)
                iy = np.floor((cand[:, 1] - y_min_chm) / res).astype(int)
                in_bounds = (iy >= 0) & (iy < labels.shape[0]) & (ix >= 0) & (ix < labels.shape[1])
                cand = cand[in_bounds]
                iy = iy[in_bounds]
                ix = ix[in_bounds]
                same_tree = labels[iy, ix] == label_id
                cand = cand[same_tree]

    return cand


def harvest_points_fixed_section(
        points, x0, y0, z_top, xy_radius,
        labels, x_min_chm, y_min_chm, res,
        fixed_section_drop=1.0, section_thickness=0.30,
        min_points=5, filter_by_label=True
):
    z_center = z_top - fixed_section_drop
    z_low = z_center - section_thickness / 2.0
    z_high = z_center + section_thickness / 2.0
    if z_high <= 0:
        return np.empty((0, 3)), (z_low, z_high)

    cand = harvest_points_in_zone(
        points, x0, y0, z_low, z_high, xy_radius,
        labels, x_min_chm, y_min_chm, res,
        min_points=min_points,
        filter_by_label=filter_by_label
    )

    # 一次轻微扩带兜底
    if cand.shape[0] < min_points:
        section_thickness2 = section_thickness * 2.0
        z_low = z_center - section_thickness2 / 2.0
        z_high = z_center + section_thickness2 / 2.0
        cand = harvest_points_in_zone(
            points, x0, y0, z_low, z_high, xy_radius,
            labels, x_min_chm, y_min_chm, res,
            min_points=min_points,
            filter_by_label=filter_by_label
        )

    return cand, (z_low, z_high)


def harvest_points_upper_max(
        points, x0, y0, z_top, xy_radius,
        labels, x_min_chm, y_min_chm, res,
        upper_depth=1.0,
        min_points=5, filter_by_label=True
):
    z_low = z_top - upper_depth
    z_high = z_top
    cand = harvest_points_in_zone(
        points, x0, y0, z_low, z_high, xy_radius,
        labels, x_min_chm, y_min_chm, res,
        min_points=min_points,
        filter_by_label=filter_by_label
    )
    return cand, (z_low, z_high)


def harvest_points_constrained_apex_excluded(
        points, x0, y0, z_top, xy_radius,
        labels, x_min_chm, y_min_chm, res,
        upper_depth=1.0, apex_exclusion=0.25,
        min_points=5, filter_by_label=True
):
    z_low = z_top - upper_depth
    z_high = z_top - apex_exclusion
    if z_high <= z_low:
        z_high = z_top - max(0.10, apex_exclusion * 0.5)
    cand = harvest_points_in_zone(
        points, x0, y0, z_low, z_high, xy_radius,
        labels, x_min_chm, y_min_chm, res,
        min_points=min_points,
        filter_by_label=filter_by_label
    )
    return cand, (z_low, z_high)


CONVENTIONAL_METRIC_KEYS = [
    'Hmax_m',
    'Hmean_m',
    'CW_EW_m',
    'CW_NS_m',
    'CW_mean_m',
    'CPA_m2',
    'CV_hull_m3',
    'CV_voxel_m3',
    'PD_area_pts_m2',
    'PD_volume_pts_m3',
    'CanopyCover',
    'LAI_proxy'
]


def extract_segmented_upper_crown_points(
        points,
        label_id,
        labels,
        x_min_chm,
        y_min_chm,
        res,
        min_height=None
):
    """
    提取某一 watershed 单木对应的全部 upper-crown 点。

    当前输入 TXT 本身已经是树冠上半部，因此本函数不再按 CAA 的 1 m 分析深度
    做二次垂直截取，只按 watershed label 提取同一棵树的全部输入点。
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 3:
        return np.empty((0, 3), dtype=float)

    pts = pts[:, :3]
    pts = pts[np.all(np.isfinite(pts), axis=1)]

    if min_height is not None:
        pts = pts[pts[:, 2] >= float(min_height)]

    if pts.shape[0] == 0:
        return np.empty((0, 3), dtype=float)

    ix = np.floor((pts[:, 0] - x_min_chm) / res).astype(int)
    iy = np.floor((pts[:, 1] - y_min_chm) / res).astype(int)

    valid = (
        (ix >= 0) & (ix < labels.shape[1]) &
        (iy >= 0) & (iy < labels.shape[0])
    )

    pts = pts[valid]
    ix = ix[valid]
    iy = iy[valid]

    if pts.shape[0] == 0:
        return np.empty((0, 3), dtype=float)

    same_tree = labels[iy, ix] == int(label_id)
    return pts[same_tree]


def calculate_canopy_cover_and_lai_proxy(
        xy,
        grid_res=0.20,
        extinction_coefficient=0.5
):

    xy = np.asarray(xy, dtype=float)
    xy = xy[np.all(np.isfinite(xy), axis=1)]
    xy = np.unique(xy, axis=0)

    if xy.shape[0] < 3:
        return np.nan, np.nan

    grid_res = float(grid_res)
    k = float(extinction_coefficient)
    if grid_res <= 0:
        raise ValueError('grid_res must be > 0.')
    if k <= 0:
        raise ValueError('extinction_coefficient must be > 0.')

    try:
        hull = ConvexHull(xy, qhull_options='QJ')
    except QhullError:
        return np.nan, np.nan

    polygon = xy[hull.vertices]

    x_min = float(np.min(polygon[:, 0]))
    x_max = float(np.max(polygon[:, 0]))
    y_min = float(np.min(polygon[:, 1]))
    y_max = float(np.max(polygon[:, 1]))

    # 以固定网格边界覆盖整个 crown hull
    x0 = math.floor(x_min / grid_res) * grid_res
    y0 = math.floor(y_min / grid_res) * grid_res
    nx = max(1, int(math.ceil((x_max - x0) / grid_res)))
    ny = max(1, int(math.ceil((y_max - y0) / grid_res)))

    x_centers = x0 + (np.arange(nx) + 0.5) * grid_res
    y_centers = y0 + (np.arange(ny) + 0.5) * grid_res
    xx, yy = np.meshgrid(x_centers, y_centers)
    centers = np.column_stack([xx.ravel(), yy.ravel()])

    path = MplPath(polygon, closed=True)
    inside = path.contains_points(centers, radius=1e-10).reshape(ny, nx)
    n_inside = int(np.count_nonzero(inside))

    if n_inside == 0:
        return np.nan, np.nan

    # 将原始点映射到相同的 XY 网格
    ix = np.floor((xy[:, 0] - x0) / grid_res).astype(int)
    iy = np.floor((xy[:, 1] - y0) / grid_res).astype(int)
    valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)

    occupied = np.zeros((ny, nx), dtype=bool)
    occupied[iy[valid], ix[valid]] = True

    n_occupied_inside = int(np.count_nonzero(occupied & inside))
    canopy_cover = float(n_occupied_inside / n_inside)
    canopy_cover = float(np.clip(canopy_cover, 0.0, 1.0))

    gap_fraction = max(1.0 - canopy_cover, 1e-6)
    lai_proxy = float(-np.log(gap_fraction) / k)

    return canopy_cover, lai_proxy


def compute_conventional_upper_crown_metrics(
        tree_points,
        scan_bin_width=0.10,
        min_scanline_points=2,
        voxel_size=0.20,
        canopy_grid_res=0.20,
        lai_extinction_coefficient=0.5
):

    result = {key: np.nan for key in CONVENTIONAL_METRIC_KEYS}

    pts = np.asarray(tree_points, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] < 3:
        return result

    xyz = pts[:, :3]
    xyz = xyz[np.all(np.isfinite(xyz), axis=1)]
    if xyz.shape[0] < 3:
        return result

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    xy = xyz[:, :2]
    n_points = int(xyz.shape[0])

    # 1-2. 常规高度指标
    result['Hmax_m'] = float(np.max(z))
    result['Hmean_m'] = float(np.mean(z))

    # 3-5. EW / NS / mean crown width
    # 与 CAA 主程序保持同一套“扫描带”逻辑，但这里使用整棵树的 upper-crown points。
    width_geom = find_ew_ns_mean_crown_width(
        xyz,
        max_height_diff=None,
        scan_bin_width=scan_bin_width,
        min_scanline_points=min_scanline_points
    )
    result['CW_EW_m'] = float(width_geom.get('d_ew', 0.0))
    result['CW_NS_m'] = float(width_geom.get('d_ns', 0.0))
    if width_geom.get('found', False):
        result['CW_mean_m'] = float(width_geom.get('dmean', 0.0))
    else:
        # 即使只有一个方向成功，也保留可用方向；mean 需要两个方向均有效。
        result['CW_mean_m'] = np.nan

    # 6. Crown projected area (2D convex hull)
    cpa = np.nan
    unique_xy = np.unique(xy, axis=0)
    if unique_xy.shape[0] >= 3:
        try:
            hull2d = ConvexHull(unique_xy, qhull_options='QJ')
            # scipy 2D ConvexHull: volume = polygon area
            cpa = float(hull2d.volume)
        except QhullError:
            cpa = np.nan
    result['CPA_m2'] = cpa

    # 7. 3D convex-hull crown volume
    cv_hull = np.nan
    unique_xyz = np.unique(xyz, axis=0)
    if unique_xyz.shape[0] >= 4:
        centered = unique_xyz - np.mean(unique_xyz, axis=0)
        if np.linalg.matrix_rank(centered) >= 3:
            try:
                hull3d = ConvexHull(unique_xyz, qhull_options='QJ')
                cv_hull = float(hull3d.volume)
            except QhullError:
                cv_hull = np.nan
    result['CV_hull_m3'] = cv_hull

    # 8. Voxel crown volume
    voxel_size = float(voxel_size)
    if voxel_size <= 0:
        raise ValueError('voxel_size must be > 0.')

    xyz_local = xyz - np.min(xyz, axis=0)
    voxel_idx = np.floor(xyz_local / voxel_size).astype(np.int64)
    occupied_voxels = np.unique(voxel_idx, axis=0)
    cv_voxel = float(occupied_voxels.shape[0] * voxel_size ** 3)
    result['CV_voxel_m3'] = cv_voxel

    # 9. Point density normalized by projected area
    if np.isfinite(cpa) and cpa > 1e-12:
        result['PD_area_pts_m2'] = float(n_points / cpa)

    # 10. Point density normalized by convex-hull volume
    if np.isfinite(cv_hull) and cv_hull > 1e-12:
        result['PD_volume_pts_m3'] = float(n_points / cv_hull)

    # 11-12. Canopy cover / LAI proxy
    canopy_cover, lai_proxy = calculate_canopy_cover_and_lai_proxy(
        xy,
        grid_res=canopy_grid_res,
        extinction_coefficient=lai_extinction_coefficient
    )
    result['CanopyCover'] = canopy_cover
    result['LAI_proxy'] = lai_proxy

    return result

# =========================
# 3D 视图美化
# =========================
def set_equal_3d_axes(ax, xlim, ylim, zlim, pad_ratio=0.06, use_ortho=True):
    x0, x1 = map(float, xlim)
    y0, y1 = map(float, ylim)
    z0, z1 = map(float, zlim)

    rx = max(x1 - x0, 1e-6)
    ry = max(y1 - y0, 1e-6)
    rz = max(z1 - z0, 1e-6)
    rmax = max(rx, ry, rz) * (1.0 + pad_ratio)
    half = rmax / 2.0

    xc = (x0 + x1) / 2.0
    yc = (y0 + y1) / 2.0
    zc = (z0 + z1) / 2.0

    ax.set_xlim(xc - half, xc + half)
    ax.set_ylim(yc - half, yc + half)
    ax.set_zlim(zc - half, zc + half)
    ax.set_box_aspect((1.0, 1.0, 1.0))

    if use_ortho:
        try:
            ax.set_proj_type('ortho')
        except Exception:
            pass


def style_3d_axis(ax, elev=24, azim=-58, ticksize=8.5):
    """期刊风格 3D 坐标轴：白底、浅网格、低视觉干扰。"""
    ax.view_init(elev=elev, azim=azim)
    ax.tick_params(axis='both', which='major', pad=2, labelsize=ticksize, width=0.7, length=3)
    ax.tick_params(axis='z', which='major', pad=4, labelsize=ticksize, width=0.7, length=3)

    try:
        ax.xaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
        ax.yaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
        ax.zaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
        ax.xaxis.pane.set_edgecolor((0.88, 0.88, 0.88, 1.0))
        ax.yaxis.pane.set_edgecolor((0.88, 0.88, 0.88, 1.0))
        ax.zaxis.pane.set_edgecolor((0.88, 0.88, 0.88, 1.0))
    except Exception:
        pass

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis._axinfo['grid']['linewidth'] = 0.45
            axis._axinfo['grid']['linestyle'] = ':'
            axis._axinfo['grid']['color'] = (0.72, 0.72, 0.72, 0.55)
            axis._axinfo['axisline']['linewidth'] = 0.75
            axis._axinfo['axisline']['color'] = (0.25, 0.25, 0.25, 1.0)
            axis._axinfo['tick']['inward_factor'] = 0.0
            axis._axinfo['tick']['outward_factor'] = 0.22
        except Exception:
            pass


def set_panel_title(ax, panel_label, title, fontsize=12):
    """统一三联图 panel 标题格式。"""
    ax.set_title(f"{panel_label} {title}", fontsize=fontsize, fontweight='bold', loc='left', pad=8)


def compact_legend(ax, loc='upper right', bbox_to_anchor=None, fontsize=9, ncol=1):
    """统一图例风格，减少遮挡。"""
    kwargs = dict(
        loc=loc,
        frameon=True,
        framealpha=0.94,
        facecolor='white',
        edgecolor='0.72',
        fontsize=fontsize,
        borderpad=0.38,
        handlelength=1.35,
        handletextpad=0.45,
        labelspacing=0.32,
        markerscale=1.05,
        ncol=ncol
    )
    if bbox_to_anchor is not None:
        kwargs['bbox_to_anchor'] = bbox_to_anchor
    return ax.legend(**kwargs)


# =========================
# 可视化：总分割图
# =========================
def plot_outer_contours(ax, labels, linewidth=0.7, alpha=0.9):
    label_ids = np.unique(labels)
    for label_id in label_ids:
        if label_id == 0:
            continue
        bbox = get_tree_bbox_pixels(labels, int(label_id), margin_px=2)
        if bbox is None:
            continue
        x1, x2, y1, y2 = bbox
        crown = binary_fill_holes(labels[y1:y2, x1:x2] == label_id).astype(float)
        if np.max(crown) <= 0:
            continue
        yy = np.arange(y1, y2)
        xx = np.arange(x1, x2)
        xx_grid, yy_grid = np.meshgrid(xx, yy)
        ax.contour(xx_grid, yy_grid, crown,
                   levels=[0.5], colors='black', linewidths=linewidth, alpha=alpha)


def visualize_treetop_detection(vis, output_dir, boundary_mode="outer_contour"):
    if vis is None:
        return

    chm = vis["chm"]
    labels = vis["labels"]
    bounds = vis["bounds"]
    treetops_px = vis.get("treetops_px", np.empty((0, 2), int))

    yy = np.arange(chm.shape[0])
    xx = np.arange(chm.shape[1])
    xx_grid, yy_grid = np.meshgrid(xx, yy)

    fig, ax = plt.subplots(figsize=(10.8, 9.2))
    im = ax.imshow(chm, cmap='viridis', origin='lower', interpolation='nearest')

    if boundary_mode == 'outer_contour':
        plot_outer_contours(ax, labels, linewidth=0.7, alpha=0.9)
        title = 'CHM with crown outer contours and treetops'
    else:
        if np.any(bounds):
            ax.contour(xx_grid, yy_grid, bounds.astype(float), levels=[0.5],
                       colors='black', linewidths=0.7, linestyles='--')
        title = 'CHM with inter-tree watershed boundaries and treetops'

    if treetops_px.size > 0:
        ax.scatter(treetops_px[:, 1], treetops_px[:, 0],
                   marker='*', s=105, c='red', edgecolors='black', linewidths=0.5, zorder=5)

    ax.set_title(title, fontsize=15, fontweight='bold', pad=12)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Height (m)')
    plt.tight_layout()
    safe_savefig(fig, output_dir, '00_CHM_segmentation_overview.png', dpi=250, bbox_inches='tight')
    plt.close(fig)


# =========================
# 结果计算
# =========================

def build_ew_ns_method_result(method_name, candidate_points, z_range, tree_top,
                              angle_threshold=70.0, max_height_diff=None,
                              scan_bin_width=0.10, min_scanline_points=2):
    """基于 EW/NS 双方向平均冠幅，构建单个 CAA 方法的结果字典。"""
    if candidate_points.shape[0] >= 2:
        geom = find_ew_ns_mean_crown_width(
            candidate_points,
            max_height_diff=max_height_diff,
            scan_bin_width=scan_bin_width,
            min_scanline_points=min_scanline_points
        )
    else:
        geom = find_ew_ns_mean_crown_width(
            np.empty((0, 3)),
            max_height_diff=max_height_diff,
            scan_bin_width=scan_bin_width,
            min_scanline_points=min_scanline_points
        )

    found = bool(geom['found'])
    if found:
        angle_ew = calculate_crown_angle(tree_top, geom['p_ew1'], geom['p_ew2'])
        angle_ns = calculate_crown_angle(tree_top, geom['p_ns1'], geom['p_ns2'])
        angle = float((angle_ew + angle_ns) / 2.0)
        # 为兼容原有可视化/输出字段，p1/p2 使用宽度更大的方向端点。
        if geom['d_ew'] >= geom['d_ns']:
            p1, p2 = geom['p_ew1'], geom['p_ew2']
            z_diff = geom['z_diff_ew']
        else:
            p1, p2 = geom['p_ns1'], geom['p_ns2']
            z_diff = geom['z_diff_ns']
    else:
        angle_ew = 0.0
        angle_ns = 0.0
        angle = 0.0
        p1 = p2 = None
        z_diff = 0.0

    return {
        'method': method_name,
        'points': candidate_points,
        'z_range': z_range,
        'p1': p1,
        'p2': p2,
        'p_ew1': geom['p_ew1'],
        'p_ew2': geom['p_ew2'],
        'p_ns1': geom['p_ns1'],
        'p_ns2': geom['p_ns2'],
        'dmax': float(geom['dmean']),
        'dmean': float(geom['dmean']),
        'd_ew': float(geom['d_ew']),
        'd_ns': float(geom['d_ns']),
        'z_diff': float(z_diff),
        'z_diff_ew': float(geom['z_diff_ew']),
        'z_diff_ns': float(geom['z_diff_ns']),
        'ew_fixed_y': geom.get('ew_fixed_y'),
        'ns_fixed_x': geom.get('ns_fixed_x'),
        'scan_bin_width': float(scan_bin_width),
        'angle': float(angle),
        'angle_ew': float(angle_ew),
        'angle_ns': float(angle_ns),
        'found': found,
        'found_ew': bool(geom['found_ew']),
        'found_ns': bool(geom['found_ns']),
        'status': 'Healthy' if angle > angle_threshold else 'Damaged'
    }

def analyze_three_methods_for_tree(
        points,
        rec,
        labels, x_min_chm, y_min_chm, res,
        fixed_section_drop=1.0,
        fixed_section_thickness=0.30,
        upper_depth=1.0,
        apex_exclusion=0.25,
        max_height_diff=0.30,
        scan_bin_width=0.10,
        min_scanline_points=2,
        density_threshold=5,
        angle_threshold=70.0,
        filter_by_label=True
):
    p = np.array(rec['position'], dtype=float)
    x0, y0, z_top = p

    crown_radius = float(rec['crown_radius_m'])
    xy_radius = float(np.clip(max(1.0, 1.15 * crown_radius), 1.0, 4.0))
    grid_size = float(max(2.0, np.ceil((2.0 * xy_radius) / 0.5) * 0.5))
    grid_x = np.floor(x0 / grid_size) * grid_size + grid_size / 2.0
    grid_y = np.floor(y0 / grid_size) * grid_size + grid_size / 2.0

    # 上下文所用的 analysis zone（对应上冠层法，用于 top-row 的 analysis area）
    context_points, context_range = harvest_points_upper_max(
        points, x0, y0, z_top, xy_radius,
        labels, x_min_chm, y_min_chm, res,
        upper_depth=upper_depth,
        min_points=density_threshold,
        filter_by_label=filter_by_label
    )

    method_results = {}

    # ---------- Method 1: fixed section ----------
    m1_points, m1_range = harvest_points_fixed_section(
        points, x0, y0, z_top, xy_radius,
        labels, x_min_chm, y_min_chm, res,
        fixed_section_drop=fixed_section_drop,
        section_thickness=fixed_section_thickness,
        min_points=density_threshold,
        filter_by_label=filter_by_label
    )
    method_results['CAA_fixed_section'] = build_ew_ns_method_result(
        'CAA_fixed_section', m1_points, m1_range, p,
        angle_threshold=angle_threshold,
        max_height_diff=None,
        scan_bin_width=scan_bin_width,
        min_scanline_points=min_scanline_points
    )

    # ---------- Method 2: upper max ----------
    m2_points, m2_range = harvest_points_upper_max(
        points, x0, y0, z_top, xy_radius,
        labels, x_min_chm, y_min_chm, res,
        upper_depth=upper_depth,
        min_points=density_threshold,
        filter_by_label=filter_by_label
    )
    method_results['CAA_upper_max'] = build_ew_ns_method_result(
        'CAA_upper_max', m2_points, m2_range, p,
        angle_threshold=angle_threshold,
        max_height_diff=None,
        scan_bin_width=scan_bin_width,
        min_scanline_points=min_scanline_points
    )

    # ---------- Method 3: constrained + apex excluded ----------
    m3_points, m3_range = harvest_points_constrained_apex_excluded(
        points, x0, y0, z_top, xy_radius,
        labels, x_min_chm, y_min_chm, res,
        upper_depth=upper_depth,
        apex_exclusion=apex_exclusion,
        min_points=density_threshold,
        filter_by_label=filter_by_label
    )
    method_results['CAA_constrained_apex_excluded'] = build_ew_ns_method_result(
        'CAA_constrained_apex_excluded', m3_points, m3_range, p,
        angle_threshold=angle_threshold,
        max_height_diff=max_height_diff,
        scan_bin_width=scan_bin_width,
        min_scanline_points=min_scanline_points
    )

    return {
        'position': p.tolist(),
        'label_id': int(rec['label_id']),
        'grid_x': float(grid_x),
        'grid_y': float(grid_y),
        'grid_size': float(grid_size),
        'crown_area_m2': float(rec['crown_area_m2']),
        'crown_radius_m': float(rec['crown_radius_m']),
        'xy_radius': float(xy_radius),
        'context_points': context_points,
        'context_range': context_range,
        'methods': method_results
    }


# =========================
# 可视化：2×3 comparison 图
# =========================
def add_text_box(ax, text, fontsize=11, loc=(0.03, 0.97)):

    text_kwargs = dict(
        transform=ax.transAxes,
        ha='left',
        va='top',
        fontsize=fontsize,
        bbox=dict(
            boxstyle='round,pad=0.35',
            facecolor='white',
            edgecolor='0.4',
            alpha=0.92
        )
    )

    if hasattr(ax, 'text2D'):
        ax.text2D(loc[0], loc[1], text, **text_kwargs)
    else:
        ax.text(loc[0], loc[1], text, **text_kwargs)


def add_angle_summary_box(ax, angle_mean, angle_ew, angle_ns, loc=(0.045, 0.90)):

    x0, y0 = loc

    # 真实换行：每个角度独立一行，避免显示成 nθ_EW / nθ_NS
    text = (
        rf'$\theta_{{mean}}$ = {angle_mean:.1f}$^\circ$' + '\n' +
        rf'$\theta_{{EW}}$ = {angle_ew:.1f}$^\circ$' + '\n' +
        rf'$\theta_{{NS}}$ = {angle_ns:.1f}$^\circ$'
    )

    box_kwargs = dict(
        transform=ax.transAxes,
        ha='left',
        va='top',
        fontsize=10.2,
        color='black',
        linespacing=1.35,
        bbox=dict(
            boxstyle='round,pad=0.34',
            facecolor='white',
            edgecolor='0.62',
            linewidth=0.75,
            alpha=0.96
        )
    )

    if hasattr(ax, 'text2D'):
        ax.text2D(x0, y0, text, **box_kwargs)
    else:
        ax.text(x0, y0, text, **box_kwargs)

def draw_full_point_cloud_panel(ax, tree_id, tree_data, all_points):
    clr_grid = '#7A869A'
    clr_anal = '#1f77b4'
    clr_top = '#D62728'
    star_pe = [pe.Stroke(linewidth=1.9, foreground='k'), pe.Normal()]

    x0, y0, z_top = tree_data['position']
    grid_x, grid_y = tree_data['grid_x'], tree_data['grid_y']
    grid_size = tree_data['grid_size']
    xy_radius = tree_data['xy_radius']

    gx_min, gx_max = grid_x - grid_size / 2.0, grid_x + grid_size / 2.0
    gy_min, gy_max = grid_y - grid_size / 2.0, grid_y + grid_size / 2.0

    grid_mask = (
        (all_points[:, 0] >= gx_min) & (all_points[:, 0] <= gx_max) &
        (all_points[:, 1] >= gy_min) & (all_points[:, 1] <= gy_max)
    )
    grid_points = all_points[grid_mask]

    context_points = tree_data['context_points']

    if grid_points.size:
        ax.scatter(grid_points[:, 0], grid_points[:, 1], grid_points[:, 2],
                   c=clr_grid, s=12, alpha=0.72, linewidths=0.15,
                   edgecolors='white', depthshade=False)

    if context_points.size:
        ax.scatter(context_points[:, 0], context_points[:, 1], context_points[:, 2],
                   c=clr_anal, s=26, alpha=0.95, linewidths=0.18,
                   edgecolors='white', depthshade=False)

    h = ax.scatter(x0, y0, z_top, c=clr_top, s=210, marker='*',
                   linewidths=0.9, edgecolors='black', depthshade=False)
    h.set_path_effects(star_pe)

    if grid_points.size:
        zmin, zmax = grid_points[:, 2].min() - 0.5, grid_points[:, 2].max() + 0.5
    else:
        zmin, zmax = z_top - 1.0, z_top + 1.0

    set_equal_3d_axes(ax, (gx_min, gx_max), (gy_min, gy_max), (zmin, zmax), pad_ratio=0.06, use_ortho=True)
    style_3d_axis(ax, elev=24, azim=-58, ticksize=9)
    ax.set_xlabel('X (m)', labelpad=6)
    ax.set_ylabel('Y (m)', labelpad=6)
    ax.set_zlabel('Z (m)', labelpad=8)
    ax.set_title(f'A. Full point cloud (Tree {tree_id})', fontsize=13, fontweight='bold', pad=10)

    txt = f"Context points = {context_points.shape[0]}\nXY radius = {xy_radius:.2f} m"
    add_text_box(ax, txt, fontsize=10)


def draw_analysis_area_panel(ax, tree_id, tree_data):
    clr_anal = '#1f77b4'
    clr_top = '#D62728'
    star_pe = [pe.Stroke(linewidth=1.9, foreground='k'), pe.Normal()]

    x0, y0, z_top = tree_data['position']
    pts = tree_data['context_points']
    z_low, z_high = tree_data['context_range']
    xy_radius = tree_data['xy_radius']

    if pts.size:
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   c=clr_anal, s=28, alpha=0.90, linewidths=0.18,
                   edgecolors='white', depthshade=False)
        zmin, zmax = pts[:, 2].min() - 0.3, pts[:, 2].max() + 0.3
    else:
        zmin, zmax = z_low - 0.2, z_high + 0.2

    h = ax.scatter(x0, y0, z_top, c=clr_top, s=220, marker='*',
                   linewidths=0.9, edgecolors='black', depthshade=False)
    h.set_path_effects(star_pe)

    set_equal_3d_axes(ax,
                      (x0 - xy_radius, x0 + xy_radius),
                      (y0 - xy_radius, y0 + xy_radius),
                      (zmin, max(z_top + 0.2, zmax)),
                      pad_ratio=0.06, use_ortho=True)
    style_3d_axis(ax, elev=22, azim=-48, ticksize=9)
    ax.set_xlabel('X (m)', labelpad=6)
    ax.set_ylabel('Y (m)', labelpad=6)
    ax.set_zlabel('Z (m)', labelpad=8)
    ax.set_title('B. Analysis area', fontsize=13, fontweight='bold', pad=10)

    txt = f"Upper zone\nZ range = [{z_low:.2f}, {z_high:.2f}] m\nN = {pts.shape[0]}"
    add_text_box(ax, txt, fontsize=10)


def draw_summary_panel(ax, tree_id, tree_data):
    ax.axis('off')
    x0, y0, z_top = tree_data['position']

    summary = (
        f"C. Tree summary\n\n"
        f"Tree ID: {tree_id}\n"
        f"Label ID: {tree_data['label_id']}\n"
        f"Treetop X: {x0:.2f} m\n"
        f"Treetop Y: {y0:.2f} m\n"
        f"Treetop Z: {z_top:.2f} m\n\n"
        f"Crown area: {tree_data['crown_area_m2']:.2f} m²\n"
        f"Crown radius: {tree_data['crown_radius_m']:.2f} m\n"
        f"XY search radius: {tree_data['xy_radius']:.2f} m"
    )
    ax.text(0.05, 0.95, summary, ha='left', va='top', fontsize=12,
            bbox=dict(boxstyle='round,pad=0.45', facecolor='whitesmoke', edgecolor='0.55', alpha=0.98))


def _draw_width_pair(ax, tree_top, p1, p2, line_color, end_color, label, show_vectors=True, show_sector=True):
    """在 3D 子图中绘制一个方向的冠幅端点、连线及对应角度扇形。"""
    if p1 is None or p2 is None:
        return
    tree_top = np.asarray(tree_top, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    x0, y0, z0 = tree_top

    if show_vectors:
        ax.plot([x0, p1[0]], [y0, p1[1]], [z0, p1[2]], c='gray', linewidth=1.2, linestyle='--', alpha=0.45)
        ax.plot([x0, p2[0]], [y0, p2[1]], [z0, p2[2]], c='gray', linewidth=1.2, linestyle='--', alpha=0.45)

    ax.scatter([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
               c=end_color, s=95, marker='s', linewidths=0.65,
               edgecolors='black', depthshade=False, label=f'{label} endpoints')
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
            c=line_color, linewidth=3.2, solid_capstyle='round', label=f'{label} width')

    if show_sector:
        draw_3d_angle_sector(ax, tree_top, p1, p2,
                             facecolor=line_color, edgecolor=line_color,
                             alpha=0.12, linewidth=1.4)


def draw_method_panel(ax, title_prefix, tree_top, method_result, xy_radius):
    clr_points = '#1f77b4'
    clr_top = '#D62728'
    clr_ew = '#E41A1C'
    clr_ns = '#6A3D9A'
    clr_ew_end = '#FF9F1C'
    clr_ns_end = '#8E7CC3'
    star_pe = [pe.Stroke(linewidth=1.8, foreground='k'), pe.Normal()]

    pts = method_result['points']
    angle = method_result['angle']
    dmean = method_result.get('dmean', method_result.get('dmax', 0.0))
    d_ew = method_result.get('d_ew', 0.0)
    d_ns = method_result.get('d_ns', 0.0)

    x0, y0, z_top = tree_top

    if pts.size:
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   c=clr_points, s=26, alpha=0.76, linewidths=0.14,
                   edgecolors='white', depthshade=False)
        zmin, zmax = pts[:, 2].min() - 0.25, pts[:, 2].max() + 0.25
    else:
        z_low, z_high = method_result['z_range']
        zmin, zmax = z_low - 0.2, z_high + 0.2

    h = ax.scatter(x0, y0, z_top, c=clr_top, s=230, marker='*',
                   linewidths=0.9, edgecolors='black', depthshade=False, label='Treetop')
    h.set_path_effects(star_pe)

    extent_pts = [np.asarray(tree_top, dtype=float)]
    if method_result.get('found', False):
        p_ew1 = method_result.get('p_ew1')
        p_ew2 = method_result.get('p_ew2')
        p_ns1 = method_result.get('p_ns1')
        p_ns2 = method_result.get('p_ns2')

        _draw_width_pair(ax, tree_top, p_ew1, p_ew2, clr_ew, clr_ew_end, 'EW')
        _draw_width_pair(ax, tree_top, p_ns1, p_ns2, clr_ns, clr_ns_end, 'NS')

        for pp in (p_ew1, p_ew2, p_ns1, p_ns2):
            if pp is not None:
                extent_pts.append(np.asarray(pp, dtype=float))

        extent_pts = np.vstack(extent_pts)
        dx = np.max(np.abs(extent_pts[:, 0] - x0)) + 0.50
        dy = np.max(np.abs(extent_pts[:, 1] - y0)) + 0.50
        dz = np.max(np.abs(extent_pts[:, 2] - z_top)) + 0.50
        set_equal_3d_axes(ax,
                          (x0 - dx, x0 + dx),
                          (y0 - dy, y0 + dy),
                          (z_top - dz, z_top + dz),
                          pad_ratio=0.05, use_ortho=True)
    else:
        set_equal_3d_axes(ax,
                          (x0 - xy_radius, x0 + xy_radius),
                          (y0 - xy_radius, y0 + xy_radius),
                          (zmin, max(zmax, z_top + 0.3)),
                          pad_ratio=0.05, use_ortho=True)

    style_3d_axis(ax, elev=22, azim=-48, ticksize=9)
    ax.set_xlabel('X (m)', labelpad=6)
    ax.set_ylabel('Y (m)', labelpad=6)
    ax.set_zlabel('Z (m)', labelpad=8)
    ax.set_title(title_prefix, fontsize=13, fontweight='bold', pad=10)

    txt = (
        f"CAAmean = {angle:.1f}°\n"
        f"Dmean = {dmean:.2f} m\n"
        f"EW = {d_ew:.2f} m | NS = {d_ns:.2f} m"
    )
    add_text_box(ax, txt, fontsize=10.5)

def visualize_comparison_figure(tree_id, tree_data, all_points, output_dir):
    fig = plt.figure(figsize=(18.5, 12.5), constrained_layout=False)

    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    ax3 = fig.add_subplot(2, 3, 3)
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    ax5 = fig.add_subplot(2, 3, 5, projection='3d')
    ax6 = fig.add_subplot(2, 3, 6, projection='3d')

    draw_full_point_cloud_panel(ax1, tree_id, tree_data, all_points)
    draw_analysis_area_panel(ax2, tree_id, tree_data)
    draw_summary_panel(ax3, tree_id, tree_data)

    tree_top = np.asarray(tree_data['position'], dtype=float)
    xy_radius = tree_data['xy_radius']

    draw_method_panel(ax4, 'D. Fixed section', tree_top,
                      tree_data['methods']['CAA_fixed_section'], xy_radius)
    draw_method_panel(ax5, 'E. Upper max', tree_top,
                      tree_data['methods']['CAA_upper_max'], xy_radius)
    draw_method_panel(ax6, 'F. Constrained + apex excluded', tree_top,
                      tree_data['methods']['CAA_constrained_apex_excluded'], xy_radius)

    plt.tight_layout(pad=1.6, w_pad=1.8, h_pad=2.0)
    safe_savefig(fig, output_dir, f"tree_{tree_id:03d}_CAA_comparison.png", dpi=350, bbox_inches='tight')
    plt.close(fig)


# =========================
# 可选：单方法三联图（保持原始三联图结构，默认关闭）
# =========================
def visualize_single_method_figure(tree_id, tree_data, method_key, all_points, output_dir):
    method_result = tree_data['methods'][method_key]

    clr_grid = '#7A869A'
    clr_anal = '#1f77b4'
    clr_top = '#D62728'
    clr_ends = '#FF7F0E'
    clr_line = '#E41A1C'
    clr_surf = '#74A57F'
    star_pe = [pe.Stroke(linewidth=1.9, foreground='k'), pe.Normal()]
    txt_pe = [pe.Stroke(linewidth=3.2, foreground='white'), pe.Normal()]

    fig = plt.figure(figsize=(18.6, 6.8), constrained_layout=False)

    x0, y0, z_top = tree_data['position']
    grid_size = tree_data['grid_size']
    xy_radius = tree_data['xy_radius']
    z_low, z_high = method_result['z_range']
    analysis_points = method_result['points']

    # 为第一幅图准备 grid points
    grid_x = tree_data['grid_x']
    grid_y = tree_data['grid_y']
    gx_min, gx_max = grid_x - grid_size / 2, grid_x + grid_size / 2
    gy_min, gy_max = grid_y - grid_size / 2, grid_y + grid_size / 2

    grid_mask = (
        (all_points[:, 0] >= gx_min) & (all_points[:, 0] <= gx_max) &
        (all_points[:, 1] >= gy_min) & (all_points[:, 1] <= gy_max)
    )
    grid_points = all_points[grid_mask]

    labelpad_xy = 6
    labelpad_z = 7

    # -------------------------
    # Panel 1: Grid View
    # -------------------------
    ax1 = fig.add_subplot(131, projection='3d')

    if grid_points.size:
        ax1.scatter(
            grid_points[:, 0], grid_points[:, 1], grid_points[:, 2],
            c=clr_grid, s=9, alpha=0.58, linewidths=0.18,
            edgecolors='white', label='Grid points', depthshade=False
        )

    if analysis_points.size:
        ax1.scatter(
            analysis_points[:, 0], analysis_points[:, 1], analysis_points[:, 2],
            c=clr_anal, s=18, alpha=0.92, linewidths=0.22,
            edgecolors='white', label='Analysis area', depthshade=False
        )

    h_top1 = ax1.scatter(
        x0, y0, z_top, c=clr_top, s=180, marker='*',
        linewidths=0.9, edgecolors='black', label='Treetop', depthshade=False
    )
    h_top1.set_path_effects(star_pe)

    if grid_points.size:
        z1_min, z1_max = grid_points[:, 2].min() - 0.5, grid_points[:, 2].max() + 0.5
    else:
        z1_min, z1_max = z_top - 1.0, z_top + 1.0

    ax1.set(xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)')
    ax1.xaxis.labelpad = labelpad_xy
    ax1.yaxis.labelpad = labelpad_xy
    ax1.zaxis.labelpad = labelpad_z
    set_equal_3d_axes(ax1, (gx_min, gx_max), (gy_min, gy_max), (z1_min, z1_max), pad_ratio=0.06, use_ortho=True)
    style_3d_axis(ax1)
    compact_legend(ax1, loc='upper right', fontsize=8.8)
    set_panel_title(ax1, '(a)', f'Full point cloud - Tree {tree_id}')

    # -------------------------
    # Panel 2: 3D Analysis
    # -------------------------
    ax2 = fig.add_subplot(132, projection='3d')

    if analysis_points.size:
        ax2.scatter(
            analysis_points[:, 0], analysis_points[:, 1], analysis_points[:, 2],
            c=clr_anal, s=20, alpha=0.88, linewidths=0.18,
            edgecolors='white', label='Analysis points', depthshade=False
        )
        zmin, zmax = analysis_points[:, 2].min(), analysis_points[:, 2].max()
    else:
        zmin, zmax = z_low, z_high

    h_top2 = ax2.scatter(
        x0, y0, z_top, c=clr_top, s=180, marker='*',
        linewidths=0.9, edgecolors='black', label='Treetop', depthshade=False
    )
    h_top2.set_path_effects(star_pe)

    p1 = method_result['p1']
    p2 = method_result['p2']
    p_ew1 = method_result.get('p_ew1')
    p_ew2 = method_result.get('p_ew2')
    p_ns1 = method_result.get('p_ns1')
    p_ns2 = method_result.get('p_ns2')

    if method_result['found']:
        _draw_width_pair(ax2, np.array(tree_data['position']), p_ew1, p_ew2,
                         '#E41A1C', '#FF9F1C', 'EW', show_vectors=True, show_sector=False)
        _draw_width_pair(ax2, np.array(tree_data['position']), p_ns1, p_ns2,
                         '#6A3D9A', '#8E7CC3', 'NS', show_vectors=True, show_sector=False)

        txt2 = (
            f"D_EW={method_result.get('d_ew', 0.0):.2f} m\n"
            f"D_NS={method_result.get('d_ns', 0.0):.2f} m\n"
            f"Dmean={method_result.get('dmean', 0.0):.2f} m"
        )
        add_text_box(ax2, txt2, fontsize=11, loc=(0.03, 0.97))

    ax2.set(xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)')
    ax2.xaxis.labelpad = labelpad_xy
    ax2.yaxis.labelpad = labelpad_xy
    ax2.zaxis.labelpad = labelpad_z
    set_equal_3d_axes(
        ax2,
        (x0 - xy_radius, x0 + xy_radius),
        (y0 - xy_radius, y0 + xy_radius),
        (min(zmin, z_top - 0.2), max(zmax, z_top + 0.2)),
        pad_ratio=0.06,
        use_ortho=True
    )
    style_3d_axis(ax2)
    compact_legend(ax2, loc='upper right', fontsize=8.8)
    set_panel_title(ax2, '(b)', 'EW/NS diameter extraction')

    # -------------------------
    # Panel 3: 3D Crown Angle View
    # -------------------------
    ax3 = fig.add_subplot(133, projection='3d')
    angle = float(method_result.get('angle', 0.0))

    if method_result['found']:
        tree_top = np.array(tree_data['position'])
        arc_pts_all = []

        for pair_name, pa, pb, clr in [
            ('EW', p_ew1, p_ew2, '#E41A1C'),
            ('NS', p_ns1, p_ns2, '#6A3D9A')
        ]:
            if pa is None or pb is None:
                continue
            arc_pts = draw_3d_angle_sector(
                ax3, tree_top, pa, pb,
                facecolor=clr, edgecolor=clr, alpha=0.18, linewidth=2.0
            )
            _draw_width_pair(ax3, tree_top, pa, pb, clr, clr, pair_name,
                             show_vectors=True, show_sector=False)
            if arc_pts is not None and np.size(arc_pts) > 0:
                arc_pts_all.append(arc_pts)

        h_top3 = ax3.scatter(
            tree_top[0], tree_top[1], tree_top[2], c=clr_top,
            s=185, marker='*', linewidths=0.9, edgecolors='black', depthshade=False
        )
        h_top3.set_path_effects(star_pe)

        all_pts = [tree_top]
        for pp in (p_ew1, p_ew2, p_ns1, p_ns2):
            if pp is not None:
                all_pts.append(np.asarray(pp, dtype=float))
        all_pts = np.vstack(all_pts)
        if arc_pts_all:
            all_pts = np.vstack([all_pts] + arc_pts_all)

        dx = np.max(np.abs(all_pts[:, 0] - x0)) + 0.55
        dy = np.max(np.abs(all_pts[:, 1] - y0)) + 0.55
        dz = np.max(np.abs(all_pts[:, 2] - z_top)) + 0.55
        set_equal_3d_axes(
            ax3,
            (x0 - dx, x0 + dx),
            (y0 - dy, y0 + dy),
            (z_top - dz, z_top + dz),
            pad_ratio=0.04,
            use_ortho=True
        )

        add_angle_summary_box(
            ax3,
            angle_mean=angle,
            angle_ew=method_result.get('angle_ew', 0.0),
            angle_ns=method_result.get('angle_ns', 0.0),
            loc=(0.045, 0.90)
        )
    else:
        if analysis_points.size:
            ax3.scatter(
                analysis_points[:, 0], analysis_points[:, 1], analysis_points[:, 2],
                c=clr_anal, s=24, alpha=0.65, linewidths=0.16,
                edgecolors='white', depthshade=False
            )

        h_top3 = ax3.scatter(
            x0, y0, z_top, color=clr_top, s=180, marker='*',
            linewidths=0.9, edgecolors='black', depthshade=False
        )
        h_top3.set_path_effects(star_pe)

        add_text_box(
            ax3,
            "Insufficient analysis points\nNo valid EW/NS diameter computed",
            fontsize=10.5,
            loc=(0.045, 0.90)
        )

        set_equal_3d_axes(
            ax3,
            (x0 - xy_radius, x0 + xy_radius),
            (y0 - xy_radius, y0 + xy_radius),
            (min(z_low, z_top - 0.8), max(z_high, z_top + 0.8)),
            pad_ratio=0.04,
            use_ortho=True
        )

    ax3.set(xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)')
    ax3.xaxis.labelpad = labelpad_xy
    ax3.yaxis.labelpad = labelpad_xy
    ax3.zaxis.labelpad = labelpad_z
    style_3d_axis(ax3, elev=22, azim=-48)
    set_panel_title(ax3, '(c)', '3D crown angle')

    plt.tight_layout(pad=1.8, w_pad=2.0)
    safe_savefig(fig, output_dir, f"tree_{tree_id:03d}_{method_key}_analysis.png", dpi=600, bbox_inches='tight')
    plt.close(fig)


# =========================
# CSV 输出：仅输出三种方法的 CAA 角度
# =========================
def write_csv(path, header, rows):
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def write_segmentation_parameters_txt(path, params):

    with open(path, 'w', encoding='utf-8') as f:
        f.write('# treetop / segmentation 参数\n')
        for key in [
            'min_height',
            'chm_res',
            'smooth_sigma',
            'vwf_a',
            'vwf_b',
            'ws_min',
            'ws_max',
            'prominence_min',
            'min_crown_area_m2'
        ]:
            f.write(f'{key} = {params[key]}\n')


def build_angle_only_rows(results):

    rows = []

    for tree_id, tree_data in enumerate(results, start=1):
        x, y, z = tree_data['position']

        fixed_angle = tree_data['methods']['CAA_fixed_section']['angle']
        upper_angle = tree_data['methods']['CAA_upper_max']['angle']
        constrained_angle = tree_data['methods']['CAA_constrained_apex_excluded']['angle']

        rows.append([
            tree_id,
            tree_data['label_id'],
            f"{x:.3f}",
            f"{y:.3f}",
            f"{z:.3f}",
            f"{fixed_angle:.3f}",
            f"{upper_angle:.3f}",
            f"{constrained_angle:.3f}"
        ])

    return rows


def build_diameter_detail_rows(results):
    rows = []
    method_keys = [
        'CAA_fixed_section',
        'CAA_upper_max',
        'CAA_constrained_apex_excluded'
    ]

    for tree_id, tree_data in enumerate(results, start=1):
        x, y, z = tree_data['position']
        for method_key in method_keys:
            m = tree_data['methods'][method_key]
            rows.append([
                tree_id,
                tree_data['label_id'],
                method_key,
                f"{x:.3f}",
                f"{y:.3f}",
                f"{z:.3f}",
                int(m.get('points', np.empty((0, 3))).shape[0]),
                f"{m.get('d_ew', 0.0):.3f}",
                f"{m.get('d_ns', 0.0):.3f}",
                f"{m.get('dmean', 0.0):.3f}",
                '' if m.get('ew_fixed_y') is None else f"{m.get('ew_fixed_y'):.3f}",
                '' if m.get('ns_fixed_x') is None else f"{m.get('ns_fixed_x'):.3f}",
                f"{m.get('scan_bin_width', 0.0):.3f}",
                f"{m.get('z_diff_ew', 0.0):.3f}",
                f"{m.get('z_diff_ns', 0.0):.3f}",
                f"{m.get('angle_ew', 0.0):.3f}",
                f"{m.get('angle_ns', 0.0):.3f}",
                f"{m.get('angle', 0.0):.3f}",
                m.get('status', 'NA'),
                int(bool(m.get('found', False)))
            ])
    return rows



def _format_metric_value(value, digits=6):
    """CSV 数值格式化：NaN/Inf 输出为空白。"""
    try:
        value = float(value)
    except Exception:
        return ''
    if not np.isfinite(value):
        return ''
    return f'{value:.{digits}f}'


def build_caa_and_conventional_rows(results):
    rows = []

    for tree_id, tree_data in enumerate(results, start=1):
        x, y, z = tree_data['position']
        methods = tree_data['methods']
        metrics = tree_data.get('conventional_metrics', {})

        row = [
            tree_id,
            tree_data['label_id'],
            f'{x:.3f}',
            f'{y:.3f}',
            f'{z:.3f}',
            int(tree_data.get('upper_crown_npoints', 0)),
            f"{methods['CAA_fixed_section']['angle']:.3f}",
            f"{methods['CAA_upper_max']['angle']:.3f}",
            f"{methods['CAA_constrained_apex_excluded']['angle']:.3f}"
        ]

        for key in CONVENTIONAL_METRIC_KEYS:
            row.append(_format_metric_value(metrics.get(key, np.nan), digits=6))

        rows.append(row)

    return rows


# =========================
# 主程序
# =========================
if __name__ == '__main__':
    set_publication_rcparams()

    # -------------------------
    # 输入 / 输出设置
    # -------------------------
    input_txt_path = 'sampledata.txt'
    output_root = safe_output_dir('CombinedResults_sampledata')

    seg_dir = safe_output_dir(output_root / '00_SegmentationOverview')
    table_dir = safe_output_dir(output_root / '01_Tables')
    comparison_dir = safe_output_dir(output_root / '02_CAA_ComparisonFigures')

    # 可选：若需要分别导出三方法图，将此开关改为 True
    export_single_method_figures = False
    single_root = safe_output_dir(output_root / '03_SingleMethodFigures_Optional')
    single_fixed_dir = safe_output_dir(single_root / 'CAA_fixed_section')
    single_upper_dir = safe_output_dir(single_root / 'CAA_upper_max')
    single_cons_dir = safe_output_dir(single_root / 'CAA_constrained_apex_excluded')

    # -------------------------
    # 参数设置
    # -------------------------
    angle_threshold = 70.0
    density_threshold = 5
    filter_by_label = True

    # treetop / segmentation 参数
    min_height = 2.0
    chm_res = 0.2
    smooth_sigma = 0.9
    vwf_a = 0.12
    vwf_b = 0.35
    ws_min = 0.4
    ws_max = 8
    prominence_min = 0.3
    min_crown_area_m2 = 0.4

    segmentation_params = {
        'min_height': min_height,
        'chm_res': chm_res,
        'smooth_sigma': smooth_sigma,
        'vwf_a': vwf_a,
        'vwf_b': vwf_b,
        'ws_min': ws_min,
        'ws_max': ws_max,
        'prominence_min': prominence_min,
        'min_crown_area_m2': min_crown_area_m2
    }
    write_segmentation_parameters_txt(table_dir / 'treetop_segmentation_parameters.txt', segmentation_params)

    # 三种 CAA 参数
    fixed_section_drop = 1    #方法1：树顶下 1 m 左右
    fixed_section_thickness = 0.10 # 方法1：截面厚度
    upper_depth = 1             # 方法2/3：上冠层分析深度
    apex_exclusion = 0.2         # 方法3：树顶尖端排除带
    max_height_diff = 0.30         # 方法3：端点 Z 高差约束
    scan_bin_width = 0.10       # EW/NS 扫描带宽度：D_EW 同一 Y 带内计算，D_NS 同一 X 带内计算
    min_scanline_points = 2     # 每个扫描带内至少需要的点数

    # 12 个 conventional upper-crown metrics 参数
    # voxel/grid 分辨率与 CHM 分辨率保持一致，避免额外引入不必要的空间尺度。
    voxel_size = chm_res
    canopy_grid_res = chm_res
    lai_extinction_coefficient = 0.5  # 仅用于 LAI_proxy，不应解释为实测 LAI

    # -------------------------
    # 读取点云
    # -------------------------
    if not os.path.exists(input_txt_path):
        raise FileNotFoundError(
            f"未找到输入点云文件：{input_txt_path}\n"
            f"请将点云文本文件放在脚本同目录下，并命名为 {input_txt_path}。"
        )

    points = np.loadtxt(input_txt_path, usecols=(0, 1, 2))
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('输入点云必须至少包含 X,Y,Z 三列。')

    # -------------------------
    # treetop detection + segmentation
    # -------------------------
    treetops, vis = detect_treetops_vwf(
        points,
        min_height=min_height,
        chm_res=chm_res,
        smooth_sigma=smooth_sigma,
        vwf_a=vwf_a,
        vwf_b=vwf_b,
        ws_min=ws_min,
        ws_max=ws_max,
        prominence_min=prominence_min,
        min_crown_area_m2=min_crown_area_m2,
        debug=True
    )

    if vis is None or len(vis.get('treetop_records', [])) == 0:
        raise RuntimeError('未检测到有效 treetops，请检查输入点云或适当放宽参数。')

    visualize_treetop_detection(vis, seg_dir, boundary_mode='outer_contour')

    labels = vis['labels']
    x_min_chm = vis['x_min']
    y_min_chm = vis['y_min']
    res = vis['res']
    treetop_records = vis['treetop_records']

    # -------------------------
    # 逐树分析三种方法
    # -------------------------
    results = []
    print(f'开始逐树分析，总 treetops: {len(treetop_records)}')
    for idx, rec in enumerate(treetop_records, start=1):
        tree_result = analyze_three_methods_for_tree(
            points,
            rec,
            labels, x_min_chm, y_min_chm, res,
            fixed_section_drop=fixed_section_drop,
            fixed_section_thickness=fixed_section_thickness,
            upper_depth=upper_depth,
            apex_exclusion=apex_exclusion,
            max_height_diff=max_height_diff,
            scan_bin_width=scan_bin_width,
            min_scanline_points=min_scanline_points,
            density_threshold=density_threshold,
            angle_threshold=angle_threshold,
            filter_by_label=filter_by_label
        )

        # -------------------------------------------------
        # 同一棵树同步提取 12 个 conventional upper-crown metrics
        # -------------------------------------------------
        upper_crown_points = extract_segmented_upper_crown_points(
            points=points,
            label_id=rec['label_id'],
            labels=labels,
            x_min_chm=x_min_chm,
            y_min_chm=y_min_chm,
            res=res,
            min_height=min_height
        )

        conventional_metrics = compute_conventional_upper_crown_metrics(
            upper_crown_points,
            scan_bin_width=scan_bin_width,
            min_scanline_points=min_scanline_points,
            voxel_size=voxel_size,
            canopy_grid_res=canopy_grid_res,
            lai_extinction_coefficient=lai_extinction_coefficient
        )

        tree_result['upper_crown_npoints'] = int(upper_crown_points.shape[0])
        tree_result['conventional_metrics'] = conventional_metrics

        # CAA 与 conventional metrics 共用同一个 tree_result，保证严格一一对应
        results.append(tree_result)

        # 2×3 comparison 图
        visualize_comparison_figure(idx, tree_result, points, comparison_dir)

        # 可选单方法图
        if export_single_method_figures:
            visualize_single_method_figure(idx, tree_result, 'CAA_fixed_section', points, single_fixed_dir)
            visualize_single_method_figure(idx, tree_result, 'CAA_upper_max', points, single_upper_dir)
            visualize_single_method_figure(idx, tree_result, 'CAA_constrained_apex_excluded', points, single_cons_dir)

        if idx % 25 == 0 or idx == len(treetop_records):
            print(f'分析进度: {idx}/{len(treetop_records)}')

    # -------------------------
    # 输出表格：仅输出每棵树三种方法计算得到的 CAA 角度
    # -------------------------
    angle_rows = build_angle_only_rows(results)

    angle_header = [
        'TreeID',
        'LabelID',
        'X',
        'Y',
        'Z',
        'CAA_fixed_section_deg',
        'CAA_upper_max_deg',
        'CAA_constrained_apex_excluded_deg'
    ]

    write_csv(table_dir / 'CAA_angles_only.csv', angle_header, angle_rows)

    detail_rows = build_diameter_detail_rows(results)
    detail_header = [
        'TreeID', 'LabelID', 'Method', 'X', 'Y', 'Z', 'NPoints',
        'D_EW_m', 'D_NS_m', 'Dmean_m',
        'EW_fixed_Y_m', 'NS_fixed_X_m', 'Scan_bin_width_m',
        'ZDiff_EW_m', 'ZDiff_NS_m',
        'CAA_EW_deg', 'CAA_NS_deg', 'CAAmean_deg',
        'Status', 'Found'
    ]
    write_csv(table_dir / 'CAA_diameter_details.csv', detail_header, detail_rows)

    # -------------------------
    # 输出：三种 CAA + 12 个传统 upper-crown 指标（同树同一行）
    # -------------------------
    combined_rows = build_caa_and_conventional_rows(results)
    combined_header = [
        'TreeID',
        'LabelID',
        'X',
        'Y',
        'Z',
        'UpperCrown_NPoints',
        'CAA_fixed_section_deg',
        'CAA_upper_max_deg',
        'CAA_constrained_apex_excluded_deg'
    ] + CONVENTIONAL_METRIC_KEYS

    write_csv(
        table_dir / 'CAA_and_12_conventional_upper_crown_metrics.csv',
        combined_header,
        combined_rows
    )

    print('\n完成！输出目录如下：')
    print(output_root)
    print('├─ 00_SegmentationOverview : 总单木分割结果图')
    print('├─ 01_Tables               : CAA_angles_only.csv + CAA_diameter_details.csv + CAA_and_12_conventional_upper_crown_metrics.csv')
    print('├─ 02_CAA_ComparisonFigures: 每棵树 2×3 comparison 图')
    if export_single_method_figures:
        print('└─ 03_SingleMethodFigures_Optional: 可选单方法图')
    else:
        print('└─ 03_SingleMethodFigures_Optional: 默认未导出（export_single_method_figures=False）')
