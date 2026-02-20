# %%
import numpy as np
import matplotlib.pyplot as plt
from bsplyne import new_disk, MultiPatchBSplineConnectivity

# B-spline surface initial
spline, ctrl_pts = new_disk([0, 0, 0], [0, 0, 1], 1)
ctrl_pts = ctrl_pts[:-1]
ctrl_pts = spline.orderElevation(ctrl_pts, [0, 1])
ctrl_pts = spline.knotInsertion(ctrl_pts, [0, 3])

# Connectivity making through pairs of control points known to be at the same position
indices = np.arange(spline.getNbFunc()).reshape(ctrl_pts.shape[1:])
pairs_y0 = np.stack((indices[0].ravel(), indices[-1].ravel()))
pairs_xy0 = np.stack((indices[1:, -1].ravel(), np.repeat(indices[0, -1], indices.shape[0] - 1)))
pairs = np.hstack((pairs_y0, pairs_xy0)).T
conn = MultiPatchBSplineConnectivity.from_nodes_couples(pairs, np.array(ctrl_pts.shape[1:])[None])

# Extraction of the exterior boundary's connectivity, B-spline, and control points
conn_border, (spline_border, ), surf_to_border = conn.extract_exterior_borders([spline])
unique_ctrl_pts = conn.pack(conn.agglomerate([ctrl_pts]))
unique_ctrl_pts_border = unique_ctrl_pts[:, surf_to_border]
ctrl_pts_border, = conn_border.separate(conn_border.unpack(unique_ctrl_pts_border))

# Creation of a displacement field for the boundary control points
def displacement(x, y):
    centers = np.array([[0.5, 0.0, +0.2, 0.4],
                        [-0.5, 0.5, -0.1, 0.3],
                        [0.0, -0.6, +0.15, 0.5]])
    dx, dy = x[:, None] - centers[:, 0], y[:, None] - centers[:, 1]
    r2 = dx**2 + dy**2
    G = np.exp(-r2/centers[:, 3]**2)
    A = centers[:, 2]
    ux = 10*(-A*dy*G).sum(axis=1) + 0.1*x**2 - 0.05*y + 0.02*x*y
    uy = 10*( A*dx*G).sum(axis=1) - 0.08*y**2 + 0.03*x + 0.01*x*y
    return ux, uy
unique_displ = np.stack(displacement(*unique_ctrl_pts_border)).reshape((2, -1))
displ, = conn_border.separate(conn_border.unpack(unique_displ))
from treeIDW import treeIDW
mask = np.ones(conn.nb_unique_nodes, dtype='bool')
mask[surf_to_border] = False
internal_ctrl_pts = unique_ctrl_pts[:, mask]
internal_propagated_displ = treeIDW(unique_ctrl_pts_border.T, unique_displ.T, internal_ctrl_pts.T, neglectible_treshold=0, parallel=True).T
unique_propagated_displ = np.empty((unique_displ.shape[0], conn.nb_unique_nodes), dtype=unique_displ.dtype)
unique_propagated_displ[:, surf_to_border] = unique_displ
unique_propagated_displ[:, mask] = internal_propagated_displ
propagated_displ, = conn.separate(conn.unpack(unique_propagated_displ))

fig, axs = plt.subplots(2, 3, figsize=(12, 8))

# Panel A - Maillage initial
spline.plotMPL(ctrl_pts, ax=axs[0, 0], language='français')
axs[0, 0].set_title("A. Maillage B-spline initial")
axs[0, 0].axis('off')
axs[0, 0].set_aspect('equal', 'box')

# Panel B - Bordure extraite
spline_border.plotMPL(ctrl_pts_border, ax=axs[0, 1], language='français')
axs[0, 1].set_title("B. Bordure B-spline extraite")
axs[0, 1].axis('off')
axs[0, 1].set_aspect('equal', 'box')

# Panel C - Vide (ou schéma explicatif)
# Positions des points
B = np.array([[0, 1, 1.5], [0, 0.5, -0.5]])  # Trois points de bordure
I = np.array([[0.8], [0]])                   # Point intérieur
# Déplacements fictifs (flèches sur la bordure)
uB = np.array([[0.2, -0.1, -0.15], [0.1, 0.05, 0.15]])
BI = I - B
length = np.linalg.norm(BI, axis=0)
W = 1/length**2
uI = ((uB*W).sum(axis=1)/W.sum())[:, None]
# Points B_i et I
axs[0, 2].scatter(*B, c='#d95f02', s=50, label=r'$\mathbf{B_i}$ (bordure)')
axs[0, 2].scatter(*I, c='#1b9e77', s=50, label=r'$\mathbf{I}$ (point intérieur)')
# Flèches de déplacement sur les B_i
axs[0, 2].quiver(B[0], B[1], uB[0], uB[1], angles='xy', scale_units='xy',
          scale=1, color='#d95f02', width=0.005)
rot = np.array([[0, -1], [1, 0]])
# Lignes reliant I aux B_i (distances)
for i in range(B.shape[1]):
    axs[0, 2].plot([I[0,0], B[0,i]], [I[1,0], B[1,i]], 'k--', lw=0.8)
    # Annoter la distance
    x, y = B[:, i] + 0.5*BI[:, i] + 0.15*np.abs(np.sin(np.arccos(BI[0, i])))*rot@BI[:, i]/length[i]
    axs[0, 2].text(x, y, r'$||\mathbf{I}-\mathbf{B_{%d}}||$' % (i+1), fontsize=10, ha='center', va='center')
    x, y = B[:, i] + 0.1*rot.T@uB[:, i]/np.linalg.norm(uB[:, i])
    axs[0, 2].text(x, y, r'$\mathbf{u}(\mathbf{B_{%d}})$' % (i+1), color='#d95f02', fontsize=10, ha='center', va='center')
axs[0, 2].quiver(I[0], I[1], uI[0], uI[1], angles='xy', scale_units='xy', scale=1, color='#1b9e77', width=0.005)
# Annotation pour d(I)
axs[0, 2].text(*(I + 1.5*uI), r'$\mathbf{u}(\mathbf{I})$', fontsize=12, color='#1b9e77', ha='center', va='center')
# Equations de la méthode
axs[0, 2].text(
    I[0, 0], B[1, 2],  # position dans le repère des axes (en coordonnées relatives avec transform)
    r"$\mathbf{u}(\mathbf{I}) = "
    r"\dfrac{\sum_{i=1}^{n_{bound}} \frac{\mathbf{u}(\mathbf{B_i})}{\|\mathbf{I}-\mathbf{B_i}\|^2}}"
    r"{\sum_{i=1}^{n_{bound}} \frac{1}{\|\mathbf{I}-\mathbf{B_i}\|^2}}$",
    fontsize=14,
    va='center', ha='right',
    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black", lw=1)
)
axs[0, 2].set_xlim([-0.1, 1.6])
axs[0, 2].set_ylim([-0.85, 0.65])
axs[0, 2].set_title("C. Méthode d'interpolation\npondérée par l'inverse de la distance")
axs[0, 2].set_aspect('equal', 'box')
axs[0, 2].axis('off')
axs[0, 2].legend()

# Panel D - Déplacements sur bordure
axs[1, 0].scatter(*ctrl_pts, s=10, color='lightgray', zorder=1)  # contexte maillage
axs[1, 0].plot(*spline_border(ctrl_pts_border, spline_border.linspace()), 'k--', lw=1)
axs[1, 0].quiver(*ctrl_pts_border, *displ, headwidth=0, headlength=0, headaxislength=0, 
                 color='#d95f02', angles='xy', scale_units='xy', scale=1, width=0.01, zorder=2)
axs[1, 0].scatter(*(ctrl_pts_border + displ), s=20, c='#d95f02', marker='o', zorder=3)
axs[1, 0].set_title("D. Champ de déplacement\nimposé sur la bordure")
axs[1, 0].set_aspect('equal', 'box')
axs[1, 0].axis('off')

# Panel E - Propagation interne
axs[1, 1].scatter(*ctrl_pts, s=10, color='lightgray', zorder=1)  # contexte maillage
axs[1, 1].plot(*spline_border(ctrl_pts_border, spline_border.linspace()), 'k--', lw=1)
axs[1, 1].quiver(*internal_ctrl_pts, *internal_propagated_displ, headwidth=0, headlength=0, headaxislength=0, 
                 color='#1b9e77', angles='xy', scale_units='xy', scale=1, width=0.01, zorder=2)
axs[1, 1].scatter(*(internal_ctrl_pts + internal_propagated_displ), s=20, c='#1b9e77', marker='o', zorder=3)
axs[1, 1].set_title("E. Propagation du champ\nde déplacement à l’intérieur")
axs[1, 1].set_aspect('equal', 'box')
axs[1, 1].axis('off')

# Panel F - Résultat final
spline.plotMPL(ctrl_pts + propagated_displ, ax=axs[1, 2], language='français')
axs[1, 2].set_title("F. Maillage final déformé")
axs[1, 2].axis('off')

plt.tight_layout()
fig.savefig("IDW_method.pdf")
plt.show()
# %%
