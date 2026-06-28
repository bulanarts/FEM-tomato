"""
FEM Tomato — Session 3: Impact, Pressure & Rupture
====================================================
Goal: Simulate 3 scenarios and visualize side by side:
  1. Dynamic impact  — tomato dropped, stress wavpytotravels up
  2. Internal pressure — pressurize from inside like overripe tomato
  3. Rupture map     — which nodes exceed yield stress and burst

Concepts covered:
  - Newmark-β time integration (dynamic FEM)
  - Lumped mass matrix M
  - Internal pressure as nodal loads
  - Yield criterion (Von Mises vs sigma_yield)
  - Rupture zone detection
  - Export final JSON for JSX website

Run:
  python3 fem_tomato_session3.py

Requires:
  tomato_results.json  (from Session 2)
  pip install numpy scipy matplotlib
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve


# ══════════════════════════════════════════════════════
# LOAD SESSION 2 RESULTS
# ══════════════════════════════════════════════════════

def load_results(filename="tomato_results.json"):
    with open(filename) as f:
        data = json.load(f)

    nodes    = np.array(data["nodes"])
    elements = np.array(data["elements"])
    surface  = np.array(data["surface"])

    print(f"✓ Loaded: {len(nodes)} nodes, {len(elements)} elements")
    return nodes, elements, surface


# ══════════════════════════════════════════════════════
# SHARED: D matrix + B matrix + K assembly
# (same as Session 2, copied here so file is standalone)
# ══════════════════════════════════════════════════════

def make_D_matrix(E=80000, nu=0.47):
    c = E / ((1 + nu) * (1 - 2 * nu))
    D = c * np.array([
        [1-nu,   nu,   nu,            0,            0,            0],
        [  nu, 1-nu,   nu,            0,            0,            0],
        [  nu,   nu, 1-nu,            0,            0,            0],
        [   0,    0,    0, (1-2*nu)/2,            0,            0],
        [   0,    0,    0,            0, (1-2*nu)/2,            0],
        [   0,    0,    0,            0,            0, (1-2*nu)/2],
    ])
    return D


def make_B_matrix(tet_nodes):
    x1,y1,z1 = tet_nodes[0]
    x2,y2,z2 = tet_nodes[1]
    x3,y3,z3 = tet_nodes[2]
    x4,y4,z4 = tet_nodes[3]
    J = np.array([
        [x2-x1, y2-y1, z2-z1],
        [x3-x1, y3-y1, z3-z1],
        [x4-x1, y4-y1, z4-z1],
    ])
    detJ = np.linalg.det(J)
    vol  = abs(detJ) / 6.0
    if abs(detJ) < 1e-12:
        return None, 0.0
    Jinv = np.linalg.inv(J)
    dN_ref = np.array([[-1,-1,-1],[1,0,0],[0,1,0],[0,0,1]])
    dN = dN_ref @ Jinv
    B = np.zeros((6, 12))
    for i in range(4):
        dx, dy, dz = dN[i]
        col = i * 3
        B[0,col]=dx; B[1,col+1]=dy; B[2,col+2]=dz
        B[3,col]=dy; B[3,col+1]=dx
        B[4,col+1]=dz; B[4,col+2]=dy
        B[5,col]=dz; B[5,col+2]=dx
    return B, vol


def assemble_K(nodes, elements, D):
    N = len(nodes)
    ndof = 3 * N
    K = lil_matrix((ndof, ndof))
    for tet in elements:
        tet_nodes = nodes[tet]
        B, vol = make_B_matrix(tet_nodes)
        if B is None:
            continue
        K_e = B.T @ D @ B * vol
        dofs = []
        for nid in tet:
            dofs += [3*nid, 3*nid+1, 3*nid+2]
        dofs = np.array(dofs)
        for i in range(12):
            for j in range(12):
                K[dofs[i], dofs[j]] += K_e[i, j]
    return K


def compute_von_mises(nodes, elements, u, D):
    N = len(nodes)
    stress_sum   = np.zeros(N)
    stress_count = np.zeros(N)
    for tet in elements:
        tet_nodes = nodes[tet]
        B, vol = make_B_matrix(tet_nodes)
        if B is None:
            continue
        dofs = []
        for nid in tet:
            dofs += [3*nid, 3*nid+1, 3*nid+2]
        u_e = u[dofs]
        eps = B @ u_e
        sig = D @ eps
        sxx,syy,szz,sxy,syz,sxz = sig
        vm = np.sqrt(0.5*((sxx-syy)**2+(syy-szz)**2+(szz-sxx)**2
                         +6*(sxy**2+syz**2+sxz**2)))
        for nid in tet:
            stress_sum[nid]   += vm
            stress_count[nid] += 1
    stress_count = np.maximum(stress_count, 1)
    return stress_sum / stress_count


# ══════════════════════════════════════════════════════
# SCENARIO 1: DYNAMIC IMPACT
# ══════════════════════════════════════════════════════
#
# Equation of motion:  M·ü + C·u̇ + K·u = f(t)
#
# We use Newmark-β method to integrate through time:
#
#   β = 0.25, γ = 0.5  → unconditionally stable
#
# At each time step:
#   1. Predict:  u_pred = u + dt·v + dt²(0.5-β)·a
#   2. Solve:   (K + c0·M)·Δu = f - K·u_pred
#   3. Update:   a, v, u
#
# The tomato hits the ground at t=0 with velocity v0 downward.
# Impact force is applied at bottom nodes for a short duration.

def simulate_impact(nodes, elements, D, drop_height=0.5):
    """
    Simulate tomato dropped from drop_height meters.
    Returns Von Mises stress field at peak impact.

    Physics:
      v0 = sqrt(2 * g * h)   — velocity at impact
      Impact duration ≈ 5ms
      We run 20 time steps of dt=0.25ms
    """
    print("\n" + "="*50)
    print("  SCENARIO 1: Dynamic Impact")
    print("="*50)

    N    = len(nodes)
    ndof = 3 * N

    # Material + stiffness
    rho  = 1000.0   # kg/m³ — tomato density ≈ water
    E    = 80000
    nu   = 0.47

    # Tomato volume (rough sphere radius ~0.04m)
    R        = 0.04   # meters (4cm radius)
    volume   = (4/3) * np.pi * R**3
    mass     = rho * volume   # ~0.27 kg

    # Impact velocity from free fall
    g  = 9.81
    v0 = np.sqrt(2 * g * drop_height)
    print(f"  Drop height: {drop_height} m")
    print(f"  Impact velocity: {v0:.2f} m/s")
    print(f"  Tomato mass: {mass*1000:.1f} g")

    # Assemble K
    print("  Assembling K...")
    K = assemble_K(nodes, elements, D)
    K_csr = csr_matrix(K)

    # Lumped mass matrix (diagonal)
    # Each node gets equal share of total mass
    # M is stored as a vector (diagonal only)
    mass_per_node = mass / N
    M_diag = np.full(ndof, mass_per_node)

    # Rayleigh damping: C = α·M + β·K  (we use α only for simplicity)
    # α chosen so damping ratio ξ ≈ 0.05 (5% critical)
    omega1 = 50.0   # rad/s — first natural frequency (estimated)
    alpha  = 2 * 0.05 * omega1
    C_diag = alpha * M_diag

    # Newmark-β parameters
    beta_nm = 0.25
    gamma   = 0.50
    dt      = 0.00025   # 0.25 ms
    n_steps = 20

    # Initial conditions
    u = np.zeros(ndof)
    v = np.zeros(ndof)
    # All nodes moving downward at v0 at t=0
    v[2::3] = -v0
    a = np.zeros(ndof)

    # Impact force: large upward reaction at bottom nodes
    z_min   = nodes[:,2].min()
    z_range = nodes[:,2].max() - z_min
    bottom  = np.where(nodes[:,2] < z_min + 0.1*z_range)[0]

    # Newmark constants
    c0 = 1.0 / (beta_nm * dt**2)
    c1 = gamma / (beta_nm * dt)
    c2 = 1.0 / (beta_nm * dt)
    c3 = 1.0 / (2*beta_nm) - 1.0
    c4 = gamma/beta_nm - 1.0
    c5 = dt * (gamma/(2*beta_nm) - 1.0)

    # Effective stiffness: K_eff = K + c0·M + c1·C
    # (diagonal M and C → just add to diagonal of K)
    K_eff = lil_matrix(K_csr)
    for i in range(ndof):
        K_eff[i,i] += c0 * M_diag[i] + c1 * C_diag[i]

    # Apply BCs: fix bottom nodes (zero displacement)
    BIG = 1e20
    for nid in bottom:
        for d in [3*nid, 3*nid+1, 3*nid+2]:
            K_eff[d,d] = BIG

    K_eff_csr = csr_matrix(K_eff)

    # Time integration
    u_peak = np.zeros(ndof)
    vm_peak = np.zeros(N)

    print(f"  Running {n_steps} time steps (dt={dt*1000:.2f}ms)...")
    for step in range(n_steps):
        t = step * dt

        # External force: ground reaction (upward) during impact
        f = np.zeros(ndof)
        impact_duration = 0.003   # 3ms contact
        if t < impact_duration:
            # Ground pushes up with force = mass * deceleration
            decel_force = mass * v0 / impact_duration
            f_per_node  = decel_force / max(len(bottom), 1)
            for nid in bottom:
                f[3*nid+2] += f_per_node   # upward z

        # Predictor
        u_pred = u + dt*v + dt**2*(0.5-beta_nm)*a
        v_pred = v + dt*(1-gamma)*a

        # Effective force
        f_eff = f + M_diag*(c0*u_pred + c2*v + c3*a) \
                  + C_diag*(c1*u_pred + c4*v + c5*a)

        # Apply BCs to f_eff
        for nid in bottom:
            for d in [3*nid, 3*nid+1, 3*nid+2]:
                f_eff[d] = 0.0

        # Solve
        u_new = spsolve(K_eff_csr, f_eff)

        # Update acceleration and velocity
        a_new = c0*(u_new - u_pred) - c3*a
        v_new = v_pred + dt*gamma*a_new

        u = u_new
        v = v_new
        a = a_new

        # Track peak stress
        vm = compute_von_mises(nodes, elements, u, D)
        mask = vm > vm_peak
        vm_peak[mask] = vm[mask]

    print(f"  ✓ Peak σ_vm: {vm_peak.max():.1f} Pa ({vm_peak.max()/1000:.2f} kPa)")
    return vm_peak


# ══════════════════════════════════════════════════════
# SCENARIO 2: INTERNAL PRESSURE
# ══════════════════════════════════════════════════════
#
# Model: uniform pressure p applied outward on all
# surface nodes (like an overripe tomato filled with juice).
#
# For each surface triangle:
#   F_node = p · A_triangle / 3  (split equally to 3 corners)
#   Direction: outward normal of triangle
#
# Hoop stress in a thin sphere:
#   σ_hoop = p · R / (2 · t)
#
# We solve the static system K·u = f_pressure
# then compute Von Mises stress.

def simulate_pressure(nodes, elements, surface, D, pressure=2000.0):
    """
    Apply internal pressure to tomato surface and solve.

    Parameters
    ----------
    pressure : Pa — internal pressure
               Overripe tomato: ~1000–5000 Pa above ambient
    """
    print("\n" + "="*50)
    print("  SCENARIO 2: Internal Pressure")
    print("="*50)
    print(f"  Internal pressure: {pressure} Pa ({pressure/1000:.1f} kPa)")

    N    = len(nodes)
    ndof = 3 * N

    # Assemble K
    print("  Assembling K...")
    K = assemble_K(nodes, elements, D)

    # Build pressure load vector
    f = np.zeros(ndof)

    for tri in surface:
        p0, p1, p2 = nodes[tri[0]], nodes[tri[1]], nodes[tri[2]]

        # Triangle area and outward normal
        edge1  = p1 - p0
        edge2  = p2 - p0
        normal = np.cross(edge1, edge2)
        area   = np.linalg.norm(normal) / 2.0

        if area < 1e-12:
            continue

        # Ensure normal points outward (away from centroid)
        centroid = nodes.mean(axis=0)
        tri_center = (p0 + p1 + p2) / 3.0
        outward = tri_center - centroid
        if np.dot(normal, outward) < 0:
            normal = -normal

        n_hat = normal / (np.linalg.norm(normal) + 1e-12)

        # Force on each node of triangle = p * A / 3 * n_hat
        f_node = pressure * area / 3.0 * n_hat

        for nid in tri:
            f[3*nid  ] += f_node[0]
            f[3*nid+1] += f_node[1]
            f[3*nid+2] += f_node[2]

    # Fix a few nodes to prevent rigid body motion
    # (just the very bottom pole)
    z_min   = nodes[:,2].min()
    z_range = nodes[:,2].max() - z_min
    bottom  = np.where(nodes[:,2] < z_min + 0.05*z_range)[0]

    BIG = 1e20
    for nid in bottom:
        for d in [3*nid, 3*nid+1, 3*nid+2]:
            K[nid*3+d%3, nid*3+d%3] = BIG
            f[3*nid + d%3] = 0.0

    # Solve
    print("  Solving K·u = f_pressure ...")
    u = spsolve(csr_matrix(K), f)

    vm = compute_von_mises(nodes, elements, u, D)
    print(f"  ✓ Peak σ_vm: {vm.max():.1f} Pa ({vm.max()/1000:.2f} kPa)")

    return vm, u


# ══════════════════════════════════════════════════════
# SCENARIO 3: RUPTURE MAP
# ══════════════════════════════════════════════════════
#
# Combine both stress fields and check yield criterion:
#
#   If σ_vm(node) > σ_yield  → node has RUPTURED
#
# Real tomato skin properties (literature):
#   σ_yield ≈ 10,000 – 20,000 Pa
#   We use 15,000 Pa as baseline.
#
# Output: binary rupture map + safety factor per node
#   safety_factor = σ_yield / σ_vm
#   safety < 1.0  → ruptured
#   safety = 2.0  → twice as strong as needed

def compute_rupture(vm_impact, vm_pressure, sigma_yield=15000.0):
    """
    Combine stress fields and compute rupture zones.

    We take the worst case (max) of both scenarios per node.
    """
    print("\n" + "="*50)
    print("  SCENARIO 3: Rupture Analysis")
    print("="*50)
    print(f"  Yield stress σ_yield = {sigma_yield} Pa ({sigma_yield/1000:.0f} kPa)")

    # Worst-case stress at each node
    vm_combined = np.maximum(vm_impact, vm_pressure)

    # Safety factor
    safety = sigma_yield / (vm_combined + 1e-6)

    # Rupture flag
    ruptured = vm_combined > sigma_yield
    n_ruptured = ruptured.sum()

    print(f"  Max combined σ_vm:  {vm_combined.max():.1f} Pa")
    print(f"  Ruptured nodes:     {n_ruptured} / {len(vm_combined)}")
    print(f"  Min safety factor:  {safety.min():.2f}")
    print(f"  Avg safety factor:  {safety.mean():.2f}")

    if n_ruptured > 0:
        print(f"  🍅 TOMATO BURST! {n_ruptured} nodes exceeded yield stress!")
    else:
        pct = 100 * vm_combined.max() / sigma_yield
        print(f"  ✓ Tomato intact — at {pct:.1f}% of yield stress")

    return vm_combined, safety, ruptured


# ══════════════════════════════════════════════════════
# VISUALIZATION — 3 scenarios side by side
# ══════════════════════════════════════════════════════

def stress_color_array(vm, cmap, vmin=0, vmax=None):
    """Map stress values to colors using a colormap."""
    if vmax is None:
        vmax = vm.max()
    norm = plt.Normalize(vmin=vmin, vmax=max(vmax, 1e-6))
    return cmap(norm(vm))


def visualize_all(nodes, surface, vm_impact, vm_pressure, vm_combined, safety, ruptured, u_pressure):
    """
    3-panel visualization:
      Left:   Impact stress
      Middle: Pressure stress + deformed shape
      Right:  Rupture map (red = burst, green = safe)
    """
    N = len(nodes)
    disp_p = u_pressure[:3*N].reshape(N, 3) if len(u_pressure) >= 3*N else np.zeros((N,3))
    scale  = 30   # exaggerate deformation

    fig = plt.figure(figsize=(18, 6), facecolor="#0f0e0d")

    # ── Panel 1: Impact ──────────────────────────────────
    ax1 = fig.add_subplot(131, projection="3d", facecolor="#191714")
    ax1.set_title("1. Dynamic Impact\nPeak σ_vm", color="#f0ebe3", fontsize=11, pad=10)

    cmap1 = plt.cm.plasma
    surf_vm1 = np.array([vm_impact[tri].mean() for tri in surface])
    norm1 = plt.Normalize(0, max(surf_vm1.max() * 0.4, 1))
    polys1 = [nodes[tri] for tri in surface]
    pc1 = Poly3DCollection(polys1, alpha=0.85, linewidth=0.1)
    pc1.set_facecolor(cmap1(norm1(surf_vm1)))
    pc1.set_edgecolor("#00000015")
    ax1.add_collection3d(pc1)
    ax1.set_xlim(-1.2,1.2); ax1.set_ylim(-1.2,1.2); ax1.set_zlim(-1.2,1.2)
    ax1.tick_params(colors="#555")
    ax1.set_xlabel("X", color="#555"); ax1.set_ylabel("Y", color="#555"); ax1.set_zlabel("Z", color="#555")

    sm1 = plt.cm.ScalarMappable(cmap=cmap1, norm=norm1)
    sm1.set_array([])
    cb1 = plt.colorbar(sm1, ax=ax1, shrink=0.55, pad=0.1)
    cb1.set_label("σ_vm (Pa)", color="#7a7268")
    cb1.ax.yaxis.set_tick_params(color="#7a7268")
    plt.setp(cb1.ax.yaxis.get_ticklabels(), color="#7a7268")

    # ── Panel 2: Pressure + deformed ────────────────────
    ax2 = fig.add_subplot(132, projection="3d", facecolor="#191714")
    ax2.set_title("2. Internal Pressure\nDeformed shape (30×)", color="#f0ebe3", fontsize=11, pad=10)

    nodes_def = nodes + disp_p * scale
    cmap2 = plt.cm.inferno
    surf_vm2 = np.array([vm_pressure[tri].mean() for tri in surface])
    norm2 = plt.Normalize(0, max(surf_vm2.max() * 0.4, 1))
    polys2 = [nodes_def[tri] for tri in surface]
    pc2 = Poly3DCollection(polys2, alpha=0.85, linewidth=0.1)
    pc2.set_facecolor(cmap2(norm2(surf_vm2)))
    pc2.set_edgecolor("#00000015")
    ax2.add_collection3d(pc2)
    ax2.set_xlim(-1.5,1.5); ax2.set_ylim(-1.5,1.5); ax2.set_zlim(-1.5,1.5)
    ax2.tick_params(colors="#555")
    ax2.set_xlabel("X", color="#555"); ax2.set_ylabel("Y", color="#555"); ax2.set_zlabel("Z", color="#555")

    sm2 = plt.cm.ScalarMappable(cmap=cmap2, norm=norm2)
    sm2.set_array([])
    cb2 = plt.colorbar(sm2, ax=ax2, shrink=0.55, pad=0.1)
    cb2.set_label("σ_vm (Pa)", color="#7a7268")
    cb2.ax.yaxis.set_tick_params(color="#7a7268")
    plt.setp(cb2.ax.yaxis.get_ticklabels(), color="#7a7268")

    # ── Panel 3: Rupture map ─────────────────────────────
    ax3 = fig.add_subplot(133, projection="3d", facecolor="#191714")
    ax3.set_title("3. Rupture Map\n🔴 burst  🟢 safe", color="#f0ebe3", fontsize=11, pad=10)

    # Color: safety factor — green=safe, red=burst
    surf_safety = np.array([safety[tri].mean() for tri in surface])
    surf_rupt   = np.array([ruptured[tri].any() for tri in surface])

    # Custom colors: ruptured=red, safe=green gradient
    face_colors = []
    for i, tri in enumerate(surface):
        if surf_rupt[i]:
            face_colors.append([0.87, 0.15, 0.10, 0.9])   # red
        else:
            # Green gradient by safety factor (capped at 3)
            s = min(surf_safety[i] / 3.0, 1.0)
            face_colors.append([0.1, 0.3 + 0.5*s, 0.2 + 0.3*s, 0.85])

    polys3 = [nodes[tri] for tri in surface]
    pc3 = Poly3DCollection(polys3, alpha=0.9, linewidth=0.1)
    pc3.set_facecolor(face_colors)
    pc3.set_edgecolor("#00000015")
    ax3.add_collection3d(pc3)
    ax3.set_xlim(-1.2,1.2); ax3.set_ylim(-1.2,1.2); ax3.set_zlim(-1.2,1.2)
    ax3.tick_params(colors="#555")
    ax3.set_xlabel("X", color="#555"); ax3.set_ylabel("Y", color="#555"); ax3.set_zlabel("Z", color="#555")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#de2619', label=f'Ruptured ({ruptured.sum()} nodes)'),
        Patch(facecolor='#2ab870', label=f'Safe ({(~ruptured).sum()} nodes)'),
    ]
    ax3.legend(handles=legend_elements, loc='upper left',
               facecolor='#191714', labelcolor='#f0ebe3', fontsize=8)

    plt.tight_layout()
    plt.savefig("tomato_session3.png", dpi=150, bbox_inches="tight", facecolor="#0f0e0d")
    plt.show()
    print("\n📸 Saved: tomato_session3.png")


# ══════════════════════════════════════════════════════
# EXPORT FINAL JSON FOR JSX WEBSITE
# ══════════════════════════════════════════════════════

def export_final_json(nodes, elements, surface, vm_impact, vm_pressure,
                      vm_combined, safety, ruptured, u_pressure,
                      filename="tomato_final.json"):
    """
    Export everything the JSX website needs:
      - geometry (nodes, elements, surface)
      - 3 stress fields
      - rupture flags
      - safety factors
      - displacements
    """
    N    = len(nodes)
    disp = u_pressure[:3*N].reshape(N, 3).tolist() if len(u_pressure) >= 3*N else [[0,0,0]]*N

    def norm01(arr):
        mx = arr.max()
        return (arr / mx).tolist() if mx > 0 else arr.tolist()

    data = {
        "metadata": {
            "n_nodes":    N,
            "n_elements": len(elements),
            "n_surface":  len(surface),
            "session":    3,
            "sigma_yield_Pa": 15000,
            "n_ruptured": int(ruptured.sum()),
            "description": "Tomato FEM — Final (impact + pressure + rupture)"
        },
        "nodes":            nodes.tolist(),
        "elements":         elements.tolist(),
        "surface":          surface.tolist(),
        "displacement":     disp,
        "stress_impact":    norm01(vm_impact),
        "stress_pressure":  norm01(vm_pressure),
        "stress_combined":  norm01(vm_combined),
        "safety_factor":    safety.tolist(),
        "ruptured":         ruptured.tolist(),
    }

    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

    size_kb = len(json.dumps(data)) // 1024
    print(f"\n💾 Exported final JSON: '{filename}'  (~{size_kb} KB)")
    print(f"   This is what the JSX website will load!")


# ══════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 52)
    print("  FEM Tomato — Session 3: Impact + Pressure + Rupture")
    print("=" * 52)

    # Load mesh
    nodes, elements, surface = load_results("tomato_results.json")

    # Material
    D = make_D_matrix(E=80000, nu=0.47)

    # ── Scenario 1: Impact ───────────────────────────────
    vm_impact = simulate_impact(
        nodes, elements, D,
        drop_height=0.5    # ← drop from 50cm
    )

    # ── Scenario 2: Internal Pressure ───────────────────
    vm_pressure, u_pressure = simulate_pressure(
        nodes, elements, surface, D,
        pressure=3000.0    # ← 3 kPa internal pressure (overripe)
    )

    # ── Scenario 3: Rupture Map ──────────────────────────
    vm_combined, safety, ruptured = compute_rupture(
        vm_impact, vm_pressure,
        sigma_yield=15000.0   # ← tomato skin yield stress
    )

    # ── Export final JSON ────────────────────────────────
    export_final_json(
        nodes, elements, surface,
        vm_impact, vm_pressure, vm_combined,
        safety, ruptured, u_pressure,
        filename="tomato_final.json"
    )

    # ── Visualize ────────────────────────────────────────
    visualize_all(
        nodes, surface,
        vm_impact, vm_pressure,
        vm_combined, safety, ruptured,
        u_pressure
    )

    print("\n✅ Session 3 complete!")
    print("\n   Files ready for JSX website:")
    print("   → tomato_final.json     ← load this in the viewer")
    print("   → tomato_session3.png   ← static plot")
    print("\n   You now have a complete FEM pipeline:")
    print("   Mesh → K assembly → Solve → Stress → Rupture → Website")
