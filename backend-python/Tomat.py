#FEM TOMATO - MODELLING THE MESH, SHELL, NODE
#------------------------------------------------------
#FUNCTION TO MODEL
# 1. STATIC STRESS (COMPRESSION) - SQUEEZE IT FROM TOP
# 2. DYNAMICS (IMPACT RESPONSE) - DROP IT
# 3. INTERNAL PRESSURE - PRESSURIZED SKIN BURSTING
# 4. WHEN SQUEEZED
# 5. WHEN TOMATO HEATED
# 6. THE JUICY INSIDE TOMATO

#HOW TO DO IT
# 1. membuat fungsi sin dan cos bentuk tomat 3D (x, y, z) -> bulging, dimple di atas
# 2. membuat shell dalam tomat (sehingga mirip bawang merah), kemudian diberi node dalam shell tomat 3D
# 3. menghubungkan node menjadi tetrahedron supaya bisa dianalisis (kalau 2d dalam bentuk segitiga)
#    FEM needs solid elements (tetrahedra) to analyze:
#    Without interior nodes, the mesh would be hollow (just a shell), which gives bad simulation results. The inner points ensure good quality 3D tetrahedral elements throughout the volume
# 4. mengecek kualitas mesh, kalau hasilnya jelek, maka modelnya jelek, hasil analisis rawan error
# 5. filters out bad tetrahedra that were generated outside the tomato shape (supaya modelnya lebih baik lagi)
#    For each tetrahedron, find its centroid (center point)
#    If the centroid is inside the tomato → keep the tetrahedron
#    If the centroid is outside the tomato→ discard the tetrahedron
# 6. mencari triangel yang ada di permukaan tomat, fungsinya untuk
#    1. saat apply force, pressure, temperature, maka kita aplikasikan pada permukaan
#    2. saat ingin mensimulasikan stress, deformation, temperature
#    3. export to 3D format 
# 7. Export to JSON, so we can bring this function to any other python file
# 8. visualize - make it as 2 table, table 1: 3d surface view, table 2: cross section
#    3D surface: showing the quality of mesh, if green that is good, red is bad
#    2D cross section: How tetrahedra fill the interior, Node distribution inside, Density of elements (how many in each area)
# 9. execute it.


import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import Delaunay

# ══════════════════════════════════════════════════════
# STEP A MESH GENERATION OF TOMATO SHAPE
# ══════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════
# STEP 1: Tomato shape function with making mesh
# ══════════════════════════════════════════════════════
# Define the tomato surface parametrically:
# x = r(φ) · sin(φ) · cos(θ)
# y = r(φ) · sin(φ) · sin(θ)
# z = r(φ) · cos(φ)
# where:
#   φ (phi)   = polar angle,   0 (bottom) → π (top)
#   θ (theta) = azimuth angle, 0 → 2π
#   r(φ)      = radius function that shapes the tomato

def tomato_radius(phi):
    #radius as a function of polar angle phi
    #we stack three modifiers on a base sphere
    # 1. oblate: squash it slightly (wider than tall)
    # 2. dimple: indent the top near the stem
    # 3. bulge: add slight equatorial bulge
    base = 1.0

    #oblate: squash along Z axis (tomatoes are wider than tall)
    oblate = 1.0 - 0.22 *np.cos(phi)**2

    #dimple: indent near the top (phi)
    #only active in the top 20% of the tomato
    dimple_zone = np.clip((phi - 0.78 * np.pi) / (0.22 * np.pi), 0, 1)
    dimple = 1.0 - 0.12 * np.sin(dimple_zone*np.pi) ** 2
    #equatorial bulge (subtle)
    bulge = 1.0 + 0.04 * np.sin(phi) ** 6

    return base * oblate * dimple * bulge

#print(tomato_radius(0))         #bottom pole
#print(tomato_radius(np.pi/2))   #equator/widest point
#print(tomato_radius(np.pi))     #top (stem dimple)

# ══════════════════════════════════════════════════════
# STEP 2: Generate surface + interior nodes
# ══════════════════════════════════════════════════════
#   - Generate points on the surface (outer shell)
#   - Generate points on scaled-down inner shells
#   - semacam memodelkan bawang merah yang berlapis
#   - shell yang berlapis gives interior density for solid FEM

def generate_tomato_nodes(n_rings=10, n_sectors=14, n_inner_shells=3):
    nodes = []

#shell scales: 1.0 = surface, smaller = interior
#we always include the surface (scale=1.0)
    shell_scales = np.linspace(1.0, 0.15, n_inner_shells + 1)
    for scale in shell_scales:
        #bottom pole
        nodes.append([0, 0, -scale * tomato_radius(0)])
        #Rings (skip poles)
        for i in range(1, n_rings):
            phi = np.pi * i / n_rings # 0 < phi < phi symbol

            r = scale * tomato_radius(phi)

            for j in range(n_sectors):
                theta = 2 * np.pi * j / n_sectors
                x = r * np.sin(phi) * np.cos(theta)
                y = r * np.sin(phi) * np.sin(theta)
                z = r * np.cos(phi)
                nodes.append([x, y, z])
        #top pole
        nodes.append([0, 0, scale * tomato_radius(np.pi)])
    
    # add center node - helps interior meshing quality, x=0, y=0, z=0
    nodes.append([0.0, 0.0, 0.0])

    nodes = np.array(nodes)

    print(f"✓ Generated {len(nodes)} nodes")                                                #len(nodes): counts how many points were created
    print(f"  Shells: {n_inner_shells + 1}  ×  (1 + {n_rings-1} × {n_sectors} + 1) each")   #This shows the mathematical formula for how nodes were calculated:
    print(f"  Bounding box: x=[{nodes[:,0].min():.3f}, {nodes[:,0].max():.3f}]")            #Bounding box (the 3D space the tomato occupies)
    print(f"                y=[{nodes[:,1].min():.3f}, {nodes[:,1].max():.3f}]")            #The f before the string means "format this string":
    print(f"                z=[{nodes[:,2].min():.3f}, {nodes[:,2].max():.3f}]")
    
    return nodes

# ══════════════════════════════════════════════════════
# STEP 3: Tetrahedralize with Delaunay
# ══════════════════════════════════════════════════════
# Before it was only nodes. disconnected points - useless for analysis!
# Delaunay will connecting the nodes into tetrahedron (4 points each) so we can analysis

#Nodes = Points where we calculate values (stress, temperature, etc.)
#Tetrahedra = Elements that connect nodes into 3D shapes
#The simulation solves equations on each tetrahedron

# Delaunay property: maximizes the minimum angle of all tetrahedra → avoids skinny/degenerate elements.

# Each element = [i, j, k, l] where i,j,k,l are node indices.
# The 4 nodes define a tetrahedron in 3D space.

def tetrahedralize(nodes):
    print("running Tetrahedralization...")
    tri = Delaunay(nodes)
    elements = tri.simplices

    print(f"✓ Generated {len(elements)} tetrahedra")
    return elements

# ══════════════════════════════════════════════════════
# STEP 4: Mesh quality check
# ══════════════════════════════════════════════════════
# A key FEM concept: element quality matters!
# Poor-quality (flat/skinny) tetrahedra cause:
#   - Numerical errors in K matrix
#   - Ill-conditioned system → bad solution
#
# Quality metric: Aspect Ratio = longest_edge / shortest_edge
# Ideal = 1.0 (perfect equilateral tet)
# Acceptable < 5.0
# Bad > 10.0

def compute_mesh_quality(nodes, elements):

#Compute aspect ratio for each tetrahedron.
#Returns array of aspect ratios.
    aspect_ratios = []

    for tet in elements:
        pts = nodes[tet]  # shape (4, 3)

        # All 6 edges of a tetrahedron
        edges = [
            pts[1] - pts[0], pts[2] - pts[0], pts[3] - pts[0],
            pts[2] - pts[1], pts[3] - pts[1], pts[3] - pts[2],
        ]
        lengths = [np.linalg.norm(e) for e in edges]
        aspect = max(lengths) / (min(lengths) + 1e-12)
        aspect_ratios.append(aspect)

    aspect_ratios = np.array(aspect_ratios)

    print(f"\n📐 Mesh Quality Report:")
    print(f"  Aspect ratio — mean:  {aspect_ratios.mean():.2f}")
    print(f"               median: {np.median(aspect_ratios):.2f}")
    print(f"               max:    {aspect_ratios.max():.2f}")
    print(f"  Good elements (AR<5): {(aspect_ratios < 5).sum()} / {len(elements)}")
    print(f"  Bad  elements (AR>10):{(aspect_ratios > 10).sum()} / {len(elements)}")

    return aspect_ratios

nodes = generate_tomato_nodes()
elements = tetrahedralize(nodes)
aspect_ratios = compute_mesh_quality(nodes, elements)

# ══════════════════════════════════════════════════════
# STEP 5: Filter to tomato interior only
# ══════════════════════════════════════════════════════
#
# Delaunay fills the convex hull of the point cloud.
# But our tomato isn't perfectly convex → some tets
# might sneak outside the tomato surface.
#
# Simple fix: check if the centroid of each tet is
# inside the tomato shape. If not, discard it.

def is_inside_tomato(point):
    """
    Returns True if (x,y,z) is inside the tomato surface.
    Uses the parametric radius at the point's polar angle.
    """
    x, y, z = point
    rxy = np.sqrt(x**2 + y**2)
    phi = np.arctan2(rxy, z) if z != 0 else np.pi / 2
    # clamp phi to [0, π]
    phi = np.clip(phi, 0, np.pi)
    r_surface = tomato_radius(phi)
    r_point = np.sqrt(x**2 + y**2 + z**2)
    return r_point <= r_surface * 1.05  # 5% tolerance


def filter_elements(nodes, elements):
    """Keep only tets whose centroid is inside the tomato."""
    kept = []
    for tet in elements:
        centroid = nodes[tet].mean(axis=0)
        if is_inside_tomato(centroid):
            kept.append(tet)
    kept = np.array(kept)
    print(f"\n🔍 Filtered elements: {len(elements)} → {len(kept)} (removed outside tets)")
    return kept


# ══════════════════════════════════════════════════════
# STEP 6: Identify surface triangles
# ══════════════════════════════════════════════════════
#
# For visualization and boundary conditions, we need
# to know which faces are on the surface.
#
# Rule: a triangular face is a surface face if it belongs
# to exactly ONE tetrahedron (interior faces are shared by 2).

def find_surface_triangles(elements):
    """
    Extract surface triangles from tetrahedral mesh.
    A face is on the surface ↔ it belongs to exactly 1 tet.
    """
    from collections import defaultdict
    face_count = defaultdict(int)

    for tet in elements:
        # 4 faces of a tetrahedron (each face = 3 sorted node indices)
        faces = [
            tuple(sorted([tet[0], tet[1], tet[2]])),
            tuple(sorted([tet[0], tet[1], tet[3]])),
            tuple(sorted([tet[0], tet[2], tet[3]])),
            tuple(sorted([tet[1], tet[2], tet[3]])),
        ]
        for face in faces:
            face_count[face] += 1

    surface = [list(f) for f, count in face_count.items() if count == 1]
    print(f"✓ Found {len(surface)} surface triangles")
    return np.array(surface)

# ══════════════════════════════════════════════════════
# STEP 7: Export to JSON
# ══════════════════════════════════════════════════════
def export_json(nodes, elements, surface_triangles, filename="tomato_mesh.json"):
    """
    Export mesh to JSON for the JSX viewer.
    
    Format:
    {
      "nodes":    [[x,y,z], ...],          N nodes
      "elements": [[i,j,k,l], ...],        M tetrahedra
      "surface":  [[i,j,k], ...],          S surface triangles
      "metadata": { ... }
    }
    """
    data = {
        "metadata": {
            "n_nodes": len(nodes),
            "n_elements": len(elements),
            "n_surface_triangles": len(surface_triangles),
            "session": 1,
            "description": "Tomato FEM mesh — Session 1 (geometry only, no FEM solve yet)"
        },
        "nodes": nodes.tolist(),
        "elements": elements.tolist(),
        "surface": surface_triangles.tolist(),
    }

    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\n💾 Exported to '{filename}'")
    print(f"   File size: ~{len(json.dumps(data)) // 1024} KB")

# ══════════════════════════════════════════════════════
# STEP 8: Visualize
# ══════════════════════════════════════════════════════

def visualize_mesh(nodes, elements, surface_triangles, aspect_ratios):
    """
    Plot the mesh in 3D:
      - Left:  surface mesh colored by aspect ratio
      - Right: cross-section slice (Y ≈ 0) to see interior
    """
    fig = plt.figure(figsize=(14, 6), facecolor="#0f0e0d")

    # ── Left: full surface ──────────────────────────────
    ax1 = fig.add_subplot(121, projection="3d", facecolor="#191714")
    ax1.set_title("🍅 Tomato Surface Mesh", color="#f0ebe3", fontsize=12)

    # Color surface triangles by their aspect ratio
    surf_aspect = []
    for tri in surface_triangles:
        # Find which tet this triangle came from (approx: use node distances)
        lengths = [
            np.linalg.norm(nodes[tri[1]] - nodes[tri[0]]),
            np.linalg.norm(nodes[tri[2]] - nodes[tri[1]]),
            np.linalg.norm(nodes[tri[0]] - nodes[tri[2]]),
        ]
        surf_aspect.append(max(lengths) / (min(lengths) + 1e-9))
    surf_aspect = np.array(surf_aspect)

    # Normalize for color
    norm = plt.Normalize(surf_aspect.min(), min(surf_aspect.max(), 8))
    cmap = plt.cm.RdYlGn_r

    polys = [nodes[tri] for tri in surface_triangles]
    poly_col = Poly3DCollection(polys, alpha=0.7, linewidth=0.2)
    colors = cmap(norm(surf_aspect))
    poly_col.set_facecolor(colors)
    poly_col.set_edgecolor("#00000040")
    ax1.add_collection3d(poly_col)

    ax1.set_xlim(-1.2, 1.2); ax1.set_ylim(-1.2, 1.2); ax1.set_zlim(-1.2, 1.2)
    ax1.set_xlabel("X", color="#7a7268"); ax1.set_ylabel("Y", color="#7a7268"); ax1.set_zlabel("Z", color="#7a7268")
    ax1.tick_params(colors="#7a7268")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax1, label="Aspect Ratio", shrink=0.6)

    # ── Right: cross-section ────────────────────────────
    ax2 = fig.add_subplot(122, facecolor="#191714")
    ax2.set_title("Cross-section (Y ≈ 0 slice)", color="#f0ebe3", fontsize=12)
    ax2.set_facecolor("#191714")

    # Show tets whose centroid has |y| < 0.15
    slice_tets = [tet for tet in elements if abs(nodes[tet].mean(axis=0)[1]) < 0.15]

    for tet in slice_tets:
        pts = nodes[tet]
        # Project onto XZ plane (x horizontal, z vertical)
        xs = pts[:, 0]
        zs = pts[:, 2]
        ax2.fill(xs[[0,1,2]], zs[[0,1,2]], alpha=0.3, color="#e03c2e", linewidth=0.3)
        ax2.fill(xs[[0,1,3]], zs[[0,1,3]], alpha=0.3, color="#e03c2e", linewidth=0.3)
        ax2.fill(xs[[0,2,3]], zs[[0,2,3]], alpha=0.3, color="#e03c2e", linewidth=0.3)
        ax2.plot(xs[[0,1,2,0]], zs[[0,1,2,0]], color="#2ab8a0", linewidth=0.3, alpha=0.6)

    # Overlay tomato outline
    phi_line = np.linspace(0, np.pi, 200)
    r_line = np.array([tomato_radius(p) for p in phi_line])
    x_out = r_line * np.sin(phi_line)
    z_out = r_line * np.cos(phi_line)
    ax2.plot( x_out, z_out, color="#e03c2e", linewidth=2, label="surface")
    ax2.plot(-x_out, z_out, color="#e03c2e", linewidth=2)

    # Scatter interior nodes in slice
    slice_nodes = nodes[np.abs(nodes[:, 1]) < 0.1]
    ax2.scatter(slice_nodes[:, 0], slice_nodes[:, 2], s=6, c="#e8c84a", alpha=0.8, zorder=5)

    ax2.set_xlabel("X", color="#7a7268"); ax2.set_ylabel("Z", color="#7a7268")
    ax2.tick_params(colors="#7a7268")
    ax2.set_aspect("equal")
    ax2.legend(facecolor="#191714", labelcolor="#f0ebe3", fontsize=9)

    plt.tight_layout()
    plt.savefig("tomato_mesh.png", dpi=150, bbox_inches="tight", facecolor="#0f0e0d")
    plt.show()
    print("\n📸 Saved: tomato_mesh.png")

# ══════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 52)
    print("  FEM Tomato — Session 1: Mesh Generation")
    print("=" * 52)

    # ── Step 1–2: Generate nodes
    nodes = generate_tomato_nodes(
        n_rings=10,        # ← increase for finer mesh
        n_sectors=14,      # ← increase for smoother circle
        n_inner_shells=3   # ← increase for more interior nodes
    )

    # ── Step 3: Tetrahedralize
    elements = tetrahedralize(nodes)

    # ── Step 4: Quality check
    aspect_ratios = compute_mesh_quality(nodes, elements)

    # ── Step 5: Filter outside tets
    elements = filter_elements(nodes, elements)

    # ── Step 6: Find surface
    surface_triangles = find_surface_triangles(elements)

    # ── Step 7: Export JSON
    export_json(nodes, elements, surface_triangles, "tomato_mesh.json")

    # ── Step 8: Visualize
    visualize_mesh(nodes, elements, surface_triangles, aspect_ratios)
