# 🍅 FEM Tomato — 3D Finite Element Analysis

A full 3D Finite Element Method (FEM) simulation of a tomato, built from scratch in Python and visualized in an interactive React website.

> *What happens when you squeeze, drop, or pressurize a tomato? This project answers that with real engineering math.*

![FEM Tomato](frontend-jsx/public/tomato_final.json)

---

## 🌐 Live Demo

👉 **[bulanarts.github.io/FEM-tomato](https://bulanarts.github.io/FEM-tomato)**

---

## 🔬 What is FEM?

Finite Element Method (FEM) is how engineers simulate stress and deformation in real objects — used in car crashes, airplane wings, bridges, and apparently tomatoes.

The idea:
1. Break the object into thousands of tiny **tetrahedra** (elements)
2. For each element, compute a **stiffness matrix K**
3. Assemble into one big system: **K · u = f**
4. Solve for **u** (displacement at every point)
5. Compute **Von Mises stress** to find where it might break

---

## 🍅 Three Scenarios

| Scenario | What happens | Key concept |
|---|---|---|
| **Compression** | Squeeze from top with a finger | Static stiffness matrix, boundary conditions |
| **Impact** | Drop the tomato from 50cm | Newmark-β time integration, stress waves |
| **Pressure** | Internal juice pressure (overripe) | Surface normal loads, hoop stress |

---

## 🏗️ Project Structure

```
FEM-tomato/
├── backend-python/
│   ├── Tomat.py          # Session 1: mesh generation
│   ├── Tomat2.py         # Session 2: stiffness matrix + solver
│   ├── Tomat3.py         # Session 3: impact + pressure + rupture
│   ├── server.py         # Flask API server
│   └── tomato_final.json # Precomputed FEM results
│
└── frontend-jsx/
    ├── src/
    │   └── App.jsx       # React 3D viewer
    └── public/
        └── tomato_final.json
```

---

## 🧮 The Math

**Stiffness matrix per element:**
```
K_e = Bᵀ · D · B · Volume
```

**Elasticity matrix D** (isotropic linear elastic):
```
E  = 80,000 Pa   (Young's modulus — tomato flesh stiffness)
ν  = 0.47        (Poisson's ratio — nearly incompressible)
```

**Von Mises stress:**
```
σ_vm = √( ½[(σxx-σyy)² + (σyy-σzz)² + (σzz-σxx)² + 6(σxy² + σyz² + σxz²)] )
```

**Rupture condition:**
```
σ_vm > σ_yield (15,000 Pa) → 🍅 TOMATO BURST
```

**Dynamic equation of motion (impact):**
```
M·ü + C·u̇ + K·u = f(t)
solved with Newmark-β method (β=0.25, γ=0.5)
```

---

## 🚀 Run Locally

### Backend (Python)

```bash
cd backend-python
pip3 install flask flask-cors numpy scipy matplotlib
python3 server.py
# → http://localhost:5000
```

### Frontend (React)

```bash
cd frontend-jsx
npm install
npm run dev
# → http://localhost:5173
```

---

## 📦 Dependencies

**Python:**
- `numpy` — matrix operations
- `scipy` — sparse solver + Delaunay tetrahedralization
- `matplotlib` — static plots
- `flask` — API server

**JavaScript:**
- `react` — UI framework
- `three` — 3D engine
- `@react-three/fiber` — React wrapper for Three.js
- `@react-three/drei` — OrbitControls, camera helpers
- `vite` — build tool

---

## 📚 What I Learned

- How FEM works from scratch — mesh → K assembly → solve → stress
- The B matrix (strain-displacement) and D matrix (elasticity tensor)
- Newmark-β time integration for dynamic problems
- How tomato skin ruptures under pressure (stem dimple = stress concentration)
- Full stack: Python scientific computing → React 3D visualization
- Git, GitHub, deployment pipeline

---

## 🛠️ Built With

- Python 3.14
- React + Vite
- Three.js / React Three Fiber
- Flask
- VS Code on macOS

---

*Made with curiosity and a lot of tomatoes 🍅*
