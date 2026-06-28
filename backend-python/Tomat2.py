#FEM TOMATO - STIFFNESS MATRIX AND SOLVER
# GOALS
# 1. assemble global K (stiffness)
# 2. apply BC
# 3. solve K x U = f
# 4. compute displacement
# 5. Von Mises stress

# HOW TO DO IT
# 1. Load mesh from our modelling
# 2. menentukan matrix 
# 3. menentukan


#KEY FEM
# B matrix (strain displacement)
# D matrix (elasticity tensor)
# Local stiffness K_e per tetrahedron
# Global assembly
# Boundary conditions (fixed bottom, pushed top)
# Sparse solver (scipy)
# Export results to JSOn for JSX viewer 
import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

# ══════════════════════════════════════════════════════
# STEP 1: Load mesh from Session 1
# ══════════════════════════════════════════════════════
def load_mesh(filename="tomato_mesh.json"):
    with open(filename) as f:
        data = json.load(f)

    nodes    = np.array(data["nodes"])      # (N, 3)
    elements = np.array(data["elements"])   # (M, 4)
    surface  = np.array(data["surface"])    # (S, 3)

    print(f"✓ Loaded mesh: {len(nodes)} nodes, {len(elements)} elements")
    return nodes, elements, surface

# ══════════════════════════════════════════════════════
# STEP 2: Material properties → D matrix
# ══════════════════════════════════════════════════════
# The D matrix (6×6) maps strain → stress via Hooke's law:
#
#   { σ_xx }         { ε_xx }
#   { σ_yy }         { ε_yy }
#   { σ_zz } = D  ·  { ε_zz }
#   { σ_xy }         { γ_xy }
#   { σ_yz }         { γ_yz }
#   { σ_xz }         { γ_xz }
#
# For isotropic linear elastic material:
#   D depends only on E (Young's modulus) and ν (Poisson's ratio)

def make_D_matrix(E=50000, nu=0.45):
    """
    Build 6×6 elasticity matrix D for isotropic material.

    Tomato material properties (approximate):
      E  = 50,000 Pa  (50 kPa) — how stiff it is
      nu = 0.45       — nearly incompressible (like fruit flesh)
                        (water is 0.5, rubber ~0.49)
    """
    c = E / ((1 + nu) * (1 - 2 * nu))

    D = c * np.array([
        [1-nu,   nu,   nu,           0,           0,           0],
        [  nu, 1-nu,   nu,           0,           0,           0],
        [  nu,   nu, 1-nu,           0,           0,           0],
        [   0,    0,    0, (1-2*nu)/2,           0,           0],
        [   0,    0,    0,           0, (1-2*nu)/2,           0],
        [   0,    0,    0,           0,           0, (1-2*nu)/2],
    ])

    print(f"✓ D matrix built  (E={E} Pa, ν={nu})")
    return D


# ══════════════════════════════════════════════════════
# STEP 3: B matrix per tetrahedron
# ══════════════════════════════════════════════════════
#
# For a linear tetrahedron (4 nodes, constant strain):
#
#   ε = B · u_e
#
# where u_e = [u1x,u1y,u1z, u2x,u2y,u2z, u3x,u3y,u3z, u4x,u4y,u4z]
#             = 12 displacement DOFs for this element
#
# B is (6×12) and depends on the shape function derivatives,
# which for a linear tet are just constants (hence "constant strain").
#
# Shape functions for linear tet:
#   N1 = 1 - ξ - η - ζ
#   N2 = ξ
#   N3 = η
#   N4 = ζ
#
# Their derivatives ∂Ni/∂x etc. come from the inverse Jacobian.

def make_B_matrix(tet_nodes):
    """
    Compute B matrix (6×12) for one linear tetrahedron.

    Parameters
    ----------
    tet_nodes : np.ndarray (4, 3)
        x,y,z coordinates of the 4 tet corners

    Returns
    -------
    B   : np.ndarray (6, 12)
    vol : float — volume of tetrahedron (must be > 0)
    """
    # Unpack nodes
    x1,y1,z1 = tet_nodes[0]
    x2,y2,z2 = tet_nodes[1]
    x3,y3,z3 = tet_nodes[2]
    x4,y4,z4 = tet_nodes[3]

    # Jacobian matrix J maps reference coords → real coords
    # J = [ x2-x1  y2-y1  z2-z1 ]
    #     [ x3-x1  y3-y1  z3-z1 ]
    #     [ x4-x1  y4-y1  z4-z1 ]
    J = np.array([
        [x2-x1, y2-y1, z2-z1],
        [x3-x1, y3-y1, z3-z1],
        [x4-x1, y4-y1, z4-z1],
    ])

    detJ = np.linalg.det(J)
    vol  = abs(detJ) / 6.0   # volume of tetrahedron

    if abs(detJ) < 1e-12:
        # Degenerate element — skip
        return None, 0.0

    # Shape function derivatives in real space
    # [dN/dx, dN/dy, dN/dz] = J^{-T} · [dN/dξ, dN/dη, dN/dζ]
    Jinv = np.linalg.inv(J)

    # dN/dξ for the 4 shape functions (in reference space)
    # N1=1-ξ-η-ζ → dN1/dξ=-1, dN1/dη=-1, dN1/dζ=-1
    # N2=ξ       → dN2/dξ= 1, dN2/dη= 0, dN2/dζ= 0
    # N3=η       → dN3/dξ= 0, dN3/dη= 1, dN3/dζ= 0
    # N4=ζ       → dN4/dξ= 0, dN4/dη= 0, dN4/dζ= 1
    dN_ref = np.array([
        [-1, -1, -1],
        [ 1,  0,  0],
        [ 0,  1,  0],
        [ 0,  0,  1],
    ])  # shape (4, 3): row i = [dNi/dξ, dNi/dη, dNi/dζ]

    # Transform to real space: dN/dx = Jinv.T @ dN/dξ
    dN = dN_ref @ Jinv  # shape (4, 3): row i = [dNi/dx, dNi/dy, dNi/dz]

    # Assemble B matrix (6×12)
    # Each node i contributes a (6×3) block to B
    # B = [ dN1/dx    0       0    | dN2/dx  ...  ]
    #     [    0   dN1/dy     0    |   ...        ]
    #     [    0      0    dN1/dz  |   ...        ]
    #     [ dN1/dy dN1/dx     0    |   ...        ]
    #     [    0   dN1/dz  dN1/dy  |   ...        ]
    #     [ dN1/dz    0    dN1/dx  |   ...        ]
    B = np.zeros((6, 12))
    for i in range(4):
        dx, dy, dz = dN[i]
        col = i * 3
        B[0, col  ] = dx
        B[1, col+1] = dy
        B[2, col+2] = dz
        B[3, col  ] = dy;  B[3, col+1] = dx
        B[4, col+1] = dz;  B[4, col+2] = dy
        B[5, col  ] = dz;  B[5, col+2] = dx

    return B, vol


# ══════════════════════════════════════════════════════
# STEP 4: Assemble global stiffness matrix K
# ══════════════════════════════════════════════════════
#
# For each element e:
#   K_e = B_e^T · D · B_e · vol_e    (12×12)
#
# Then scatter K_e into the global K (3N × 3N):
#   Each node has 3 DOFs: u_x, u_y, u_z
#   Node i → global DOFs [3i, 3i+1, 3i+2]
#
# We use a sparse matrix (lil_matrix) because K is huge
# but mostly zeros — only connected nodes interact.

def assemble_K(nodes, elements, D):
    """
    Assemble global stiffness matrix K (sparse, 3N×3N).
    """
    N = len(nodes)
    ndof = 3 * N   # total degrees of freedom

    K = lil_matrix((ndof, ndof))  # sparse: efficient for assembly

    skipped = 0
    for idx, tet in enumerate(elements):
        tet_nodes = nodes[tet]           # (4, 3)
        B, vol = make_B_matrix(tet_nodes)

        if B is None:
            skipped += 1
            continue

        # Local stiffness: K_e = B^T D B * vol  (12×12)
        K_e = B.T @ D @ B * vol

        # Scatter into global K
        # DOF indices for this tet's 4 nodes
        dofs = []
        for node_id in tet:
            dofs += [3*node_id, 3*node_id+1, 3*node_id+2]
        dofs = np.array(dofs)

        # Add K_e to global K at the right positions
        for i in range(12):
            for j in range(12):
                K[dofs[i], dofs[j]] += K_e[i, j]

    print(f"✓ Assembled K matrix: {ndof}×{ndof} ({ndof**2:,} entries)")
    print(f"  Skipped {skipped} degenerate elements")
    return K


# ══════════════════════════════════════════════════════
# STEP 5: Apply boundary conditions
# ══════════════════════════════════════════════════════
#
# Two types of boundary conditions:
#
# 1. Dirichlet (fixed DOFs):
#    Bottom nodes (z < z_min + threshold) → uz = 0 (fixed)
#    This simulates the tomato sitting on a surface.
#
# 2. Neumann (applied load):
#    Top nodes (z > z_max - threshold) → fz = -F (push down)
#    This simulates squeezing from above.
#
# Method: "penalty" or "zeroing row/col"
#   For fixed DOF i: set K[i,i] = large number, f[i] = 0
#   This forces u[i] = 0 in the solution.

def apply_boundary_conditions(K, nodes, load_magnitude=500.0):
    """
    Apply BCs and return load vector f.

    Parameters
    ----------
    K              : sparse stiffness matrix (modified in place)
    nodes          : node coordinates
    load_magnitude : total downward force in Newtons

    Returns
    -------
    f    : load vector (3N,)
    fixed_dofs : list of constrained DOF indices
    """
    N = len(nodes)
    ndof = 3 * N
    f = np.zeros(ndof)

    z_min = nodes[:, 2].min()
    z_max = nodes[:, 2].max()
    z_range = z_max - z_min

    # ── Fixed nodes (bottom 10% of tomato) ──────────────
    fixed_nodes = np.where(nodes[:, 2] < z_min + 0.1 * z_range)[0]
    fixed_dofs  = []
    for n in fixed_nodes:
        fixed_dofs += [3*n, 3*n+1, 3*n+2]  # fix x, y, z

    print(f"  Fixed nodes: {len(fixed_nodes)} (bottom support)")

    # ── Load nodes (top 10% of tomato) ──────────────────
    load_nodes = np.where(nodes[:, 2] > z_max - 0.1 * z_range)[0]
    f_per_node = load_magnitude / max(len(load_nodes), 1)
    for n in load_nodes:
        f[3*n + 2] = -f_per_node   # push down in z direction

    print(f"  Load nodes:  {len(load_nodes)} (top compression, F={load_magnitude} N)")

    # ── Apply Dirichlet BCs via large penalty ────────────
    BIG = 1e20
    for dof in fixed_dofs:
        K[dof, dof] = BIG
        f[dof]      = 0.0

    return f, fixed_dofs


# ══════════════════════════════════════════════════════
# STEP 6: Solve Ku = f
# ══════════════════════════════════════════════════════
#
# Convert sparse K to CSR format (faster for solving),
# then use scipy's sparse direct solver.
#
# u = K^{-1} · f
# (we never actually invert K — too expensive.
#  Instead we solve the linear system directly.)

def solve_system(K, f):
    """Solve K·u = f for displacement vector u."""
    from scipy.sparse import csr_matrix

    print("\n⏳ Solving K·u = f ...")
    K_csr = csr_matrix(K)
    u = spsolve(K_csr, f)

    u_mag = np.sqrt(u[0::3]**2 + u[1::3]**2 + u[2::3]**2)
    print(f"✓ Solved! Max displacement: {u_mag.max():.6f} m")
    print(f"          Avg displacement: {u_mag.mean():.6f} m")
    return u


# ══════════════════════════════════════════════════════
# STEP 7: Compute Von Mises stress
# ══════════════════════════════════════════════════════
#
# For each element, compute:
#   ε = B · u_e          (strain vector, 6×1)
#   σ = D · ε            (stress vector, 6×1)
#   σ_vm = Von Mises scalar stress
#
# Von Mises:
#   σ_vm = √( ½[(σxx-σyy)² + (σyy-σzz)² + (σzz-σxx)²
#               + 6(σxy² + σyz² + σxz²)] )
#
# Then average element stresses to nodes (nodal averaging).

def compute_stress(nodes, elements, u, D):
    """
    Compute Von Mises stress at each node.

    Returns
    -------
    von_mises_nodes : np.ndarray (N,)
    """
    N = len(nodes)
    stress_sum   = np.zeros(N)
    stress_count = np.zeros(N)

    for tet in elements:
        tet_nodes = nodes[tet]
        B, vol = make_B_matrix(tet_nodes)
        if B is None:
            continue

        # Extract element displacements
        dofs = []
        for node_id in tet:
            dofs += [3*node_id, 3*node_id+1, 3*node_id+2]
        u_e = u[dofs]   # (12,)

        # Strain and stress
        eps = B @ u_e       # (6,) strain vector
        sig = D @ eps       # (6,) stress vector [σxx,σyy,σzz,σxy,σyz,σxz]

        # Von Mises scalar
        sxx, syy, szz, sxy, syz, sxz = sig
        vm = np.sqrt(0.5 * ((sxx-syy)**2 + (syy-szz)**2 + (szz-sxx)**2
                            + 6*(sxy**2 + syz**2 + sxz**2)))

        # Accumulate to nodes (simple average)
        for node_id in tet:
            stress_sum[node_id]   += vm
            stress_count[node_id] += 1

    # Avoid division by zero
    stress_count = np.maximum(stress_count, 1)
    von_mises = stress_sum / stress_count

    print(f"\n✓ Von Mises stress computed")
    print(f"  Max σ_vm: {von_mises.max():.1f} Pa  ({von_mises.max()/1000:.2f} kPa)")
    print(f"  Avg σ_vm: {von_mises.mean():.1f} Pa  ({von_mises.mean()/1000:.2f} kPa)")
    return von_mises


# ══════════════════════════════════════════════════════
# STEP 8: Export results to JSON
# ══════════════════════════════════════════════════════

def export_results(nodes, elements, surface, u, von_mises, filename="tomato_results.json"):
    """
    Export mesh + FEM results to JSON for the JSX viewer.
    """
    # Reshape u into (N, 3) displacement vectors
    N = len(nodes)
    disp = u.reshape(-1, 3) if len(u) == 3*N else np.zeros((N, 3))

    # Normalize von mises to [0, 1] for color mapping in JSX
    vm_max = von_mises.max()
    vm_norm = (von_mises / vm_max).tolist() if vm_max > 0 else von_mises.tolist()

    data = {
        "metadata": {
            "n_nodes": N,
            "n_elements": len(elements),
            "session": 2,
            "max_von_mises_Pa": float(von_mises.max()),
            "max_displacement_m": float(np.linalg.norm(disp, axis=1).max()),
            "description": "Tomato FEM — Session 2 (compression solve)"
        },
        "nodes":        nodes.tolist(),
        "elements":     elements.tolist(),
        "surface":      surface.tolist(),
        "displacement": disp.tolist(),      # [[dx,dy,dz], ...]
        "von_mises":    vm_norm,            # [0..1] normalized
        "von_mises_Pa": von_mises.tolist(), # raw Pa values
    }

    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\n💾 Exported results to '{filename}'")


# ══════════════════════════════════════════════════════
# STEP 9: Visualize
# ══════════════════════════════════════════════════════

def visualize_results(nodes, surface, u, von_mises):
    N = len(nodes)
    disp = u[:3*N].reshape(N, 3)

    fig = plt.figure(figsize=(15, 5), facecolor="#0f0e0d")

    # ── Left: original mesh ──────────────────────────────
    ax1 = fig.add_subplot(131, projection="3d", facecolor="#191714")
    ax1.set_title("Original Shape", color="#f0ebe3", fontsize=11)
    polys = [nodes[tri] for tri in surface]
    pc1 = Poly3DCollection(polys, alpha=0.6, linewidth=0.1)
    pc1.set_facecolor("#e03c2e80")
    pc1.set_edgecolor("#00000030")
    ax1.add_collection3d(pc1)
    ax1.set_xlim(-1.2,1.2); ax1.set_ylim(-1.2,1.2); ax1.set_zlim(-1.2,1.2)
    ax1.tick_params(colors="#7a7268")

    # ── Middle: deformed shape (exaggerated) ─────────────
    ax2 = fig.add_subplot(132, projection="3d", facecolor="#191714")
    ax2.set_title("Deformed (50× scale)", color="#f0ebe3", fontsize=11)
    scale = 50
    nodes_def = nodes + disp * scale
    polys_def = [nodes_def[tri] for tri in surface]
    pc2 = Poly3DCollection(polys_def, alpha=0.6, linewidth=0.1)
    pc2.set_facecolor("#2ab8a080")
    pc2.set_edgecolor("#00000030")
    ax2.add_collection3d(pc2)
    ax2.set_xlim(-1.2,1.2); ax2.set_ylim(-1.2,1.2); ax2.set_zlim(-1.2,1.2)
    ax2.tick_params(colors="#7a7268")

    # ── Right: Von Mises stress ───────────────────────────
    ax3 = fig.add_subplot(133, projection="3d", facecolor="#191714")
    ax3.set_title("Von Mises Stress", color="#f0ebe3", fontsize=11)

    vm_max = von_mises.max()
    cmap = plt.cm.RdYlGn_r
    norm = plt.Normalize(0, vm_max*0.3)

    surf_vm = [von_mises[tri].mean() for tri in surface]
    polys_vm = [nodes_def[tri] for tri in surface]
    pc3 = Poly3DCollection(polys_vm, alpha=0.85, linewidth=0.1)
    colors = cmap(norm(surf_vm))
    pc3.set_facecolor(colors)
    pc3.set_edgecolor("#00000020")
    ax3.add_collection3d(pc3)
    ax3.set_xlim(-1.2,1.2); ax3.set_ylim(-1.2,1.2); ax3.set_zlim(-1.2,1.2)
    ax3.tick_params(colors="#7a7268")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax3, label="σ_vm (Pa)", shrink=0.6)

    plt.tight_layout()
    plt.savefig("tomato_results.png", dpi=150, bbox_inches="tight", facecolor="#0f0e0d")
    plt.show()
    print("📸 Saved: tomato_results.png")


# ══════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 52)
    print("  FEM Tomato — Session 2: Stiffness & Solver")
    print("=" * 52)

    # Load mesh from Session 1
    nodes, elements, surface = load_mesh("tomato_mesh.json")

    # Material
    D = make_D_matrix(E=50000, nu=0.45)

    # Assemble K  (this may take a few seconds)
    print("\n⏳ Assembling stiffness matrix...")
    K = assemble_K(nodes, elements, D)

    # Boundary conditions + load
    print("\n📌 Applying boundary conditions...")
    f, fixed_dofs = apply_boundary_conditions(K, nodes, load_magnitude=500.0)

    # Solve
    u = solve_system(K, f)

    # Stress
    von_mises = compute_stress(nodes, elements, u, D)

    # Export
    export_results(nodes, elements, surface, u, von_mises, "tomato_results.json")

    # Visualize
    visualize_results(nodes, surface, u, von_mises)
