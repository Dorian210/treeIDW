# %%
import numpy as np
import matplotlib.pyplot as plt
import re
from treeIDW import treeIDW
from bsplyne import BSpline

def parse_svg_bezier(path_str: str) -> np.ndarray:
    line_to_cubic = lambda p0, p1: np.array([p0, (2*p0 + p1)/3, (p0 + 2*p1)/3, p1])
    tokens = re.findall(r'[A-Za-z]|[-+]?\d*\.?\d+(?:e[-+]?\d+)?', path_str)
    segments, current, start, last_cp = [], np.zeros(2), None, None
    i = 0
    while i < len(tokens):
        cmd = tokens[i] if tokens[i].isalpha() else cmd; i += cmd.isalpha()
        rel = cmd.islower(); cmd = cmd.upper()
        get_pts = lambda n: np.array([float(tokens[j]) for j in range(i, i+n)]).reshape(-1,2)
        if cmd == 'M':
            current = get_pts(2)[0]; start = current; i += 2
            while i < len(tokens) and not tokens[i].isalpha():
                new = current + get_pts(2)[0] if rel else get_pts(2)[0]; i += 2
                segments.append(line_to_cubic(current, new)); current = new
        elif cmd in 'LHV':
            while i < len(tokens) and not tokens[i].isalpha():
                new = current.copy()
                if cmd == 'L': new = current + get_pts(2)[0] if rel else get_pts(2)[0]; i += 2
                elif cmd == 'H': new[0] = current[0] + float(tokens[i]) if rel else float(tokens[i]); i += 1
                elif cmd == 'V': new[1] = current[1] + float(tokens[i]) if rel else float(tokens[i]); i += 1
                segments.append(line_to_cubic(current, new)); current = new
        elif cmd in 'CS':
            while i < len(tokens) and not tokens[i].isalpha():
                cp1 = (2*current - last_cp) if cmd == 'S' and last_cp is not None else get_pts(2)[0]; i += (cmd == 'S') * -2 + 2
                cp2, end = get_pts(4) if cmd == 'C' else get_pts(2); i += 4 if cmd == 'C' else 2
                if rel: cp1, cp2, end = current + cp1, current + cp2, current + end
                segments.append(np.array([current, cp1, cp2, end])); current, last_cp = end, cp2
        elif cmd in 'QT':
            while i < len(tokens) and not tokens[i].isalpha():
                ctrl = (2*current - last_cp) if cmd == 'T' and last_cp is not None else get_pts(2)[0]; i += (cmd == 'T') * -2 + 2
                end = get_pts(2)[0]; i += 2
                if rel: ctrl, end = current + ctrl, current + end
                cp1, cp2 = current + 2/3*(ctrl - current), end + 2/3*(ctrl - end)
                segments.append(np.array([current, cp1, cp2, end])); current, last_cp = end, ctrl
        elif cmd == 'Z' and start is not None:
            segments.append(line_to_cubic(current, start)); current = start
    segments = np.array(segments)
    segments[:, :, 1] *= -1
    return segments

def bezier_to_bspline(segments: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    p = 3
    L_approx = 0.5*(np.linalg.norm(segments[:, 3] - segments[:, 0], axis=-1) 
                    + (np.linalg.norm(segments[:, 1] - segments[:, 0], axis=-1) 
                       + np.linalg.norm(segments[:, 2] - segments[:, 1], axis=-1) 
                       + np.linalg.norm(segments[:, 3] - segments[:, 2], axis=-1)))
    elem_right_borders = np.cumsum(L_approx)
    knots = np.concatenate(([0]*(p + 1), np.repeat(elem_right_borders, p), [elem_right_borders[-1]]))
    ctrl_pts = np.hstack((segments[:, :-1].reshape((-1, 2)).T, segments[-1, -1].reshape((2, 1))))
    return p, knots, ctrl_pts

def bspline_from_svg_code(svg_code: str) -> tuple[np.ndarray, BSpline]:
    segments = parse_svg_bezier(svg_code)
    p, knots, ctrl_pts = bezier_to_bspline(segments)
    spline = BSpline([p], [knots])
    return ctrl_pts, spline

def plot_arrows(spline, ctrl_pts, fact, ax, loop=False):
    (inf, sup), = spline.getSpans()
    nb = int(fact*(sup - inf))
    XI = [np.linspace(inf, sup, nb, endpoint=(not loop))]
    origins = spline(ctrl_pts, XI)
    vectors, = spline(ctrl_pts, XI, k=1)
    pivot = 'tail' if loop else 'middle'
    ax.quiver(origins[0], origins[1], vectors[0], vectors[1], color='#d95f02', scale=0.3, scale_units='xy', pivot=pivot, width=0.01)
    return origins, vectors

fig, ax = plt.subplots()
ax.set_aspect(1)
fact = 0.28

svg_code_I = "M 74.9323 99.5629 L 74.9323 113.962"
ctrl_pts_I, spline_I = bspline_from_svg_code(svg_code_I)
origins_I, vectors_I = plot_arrows(spline_I, ctrl_pts_I, fact, ax)

svg_code_D = "M 79.2598 99.5629 L 79.2598 113.962 L 82.2859 113.962 C 84.8408 113.962 86.7094 113.383 87.8918 112.226 C 89.0824 111.068 89.6777 109.241 89.6777 106.744 C 89.6777 104.263 89.0824 102.448 87.8918 101.299 C 86.7094 100.142 84.8408 99.5629 82.2859 99.5629 Z"
ctrl_pts_D, spline_D = bspline_from_svg_code(svg_code_D)
origins_D, vectors_D = plot_arrows(spline_D, ctrl_pts_D, fact, ax, loop=True)

svg_code_W = "M 94.0051 99.5629 L 97.6122 113.962 L 101.219 99.5629 L 104.826 113.962 L 108.433 99.5629"
ctrl_pts_W, spline_W = bspline_from_svg_code(svg_code_W)
origins_W, vectors_W = plot_arrows(spline_W, ctrl_pts_W, fact, ax)

pad = 0.5*(ctrl_pts_D[0].min() - ctrl_pts_I[0].max())
x_min = ctrl_pts_I[0].min() - pad
x_max = ctrl_pts_W[0].max() + pad
y_min = ctrl_pts_I[1].mean() - 0.5*(x_max - x_min)
y_max = ctrl_pts_I[1].mean() + 0.5*(x_max - x_min)
n = 30

boundary_nodes = np.hstack((origins_I, origins_D, origins_W)).T # shape (N_boundary, 2)
boundary_field = np.hstack((vectors_I, vectors_D, vectors_W)).T # shape (N_boundary, 2)
internal_nodes = np.stack(np.meshgrid(np.linspace(x_min, x_max, n), np.linspace(y_min, y_max, n))).reshape((2, -1)).T # shape (N_internal, 2)

internal_field = treeIDW(boundary_nodes, boundary_field, internal_nodes, parallel=True) # shape (N_internal, 2)

ax.quiver(*internal_nodes.T, *internal_field.T, color='#1b9e77', scale=0.4, scale_units='xy', width=0.003, pivot='middle', zorder=-1)

ax.set_xticks([])
ax.set_yticks([])
ax.set_frame_on(False)
fig.savefig("../docs/logo.png", dpi=300, bbox_inches='tight', pad_inches=0)

# %%


