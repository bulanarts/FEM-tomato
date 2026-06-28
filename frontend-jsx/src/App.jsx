import { useState, useEffect, useRef, useMemo } from "react"
import * as THREE from "three"
import { Canvas } from "@react-three/fiber"
import { OrbitControls, PerspectiveCamera } from "@react-three/drei"

// ══════════════════════════════════════════════════════
// CONSTANTS
// ══════════════════════════════════════════════════════

const API = "http://127.0.0.1:5000/api"

const SCENARIOS = [
  { key: "stress_impact",   label: "1. Impact",   color: "#e8c84a" },
  { key: "stress_pressure", label: "2. Pressure", color: "#2ab8a0" },
  { key: "stress_combined", label: "3. Rupture",  color: "#e03c2e" },
]

// Stress color: blue → green → yellow → red
function stressToColor(t) {
  const clamped = Math.max(0, Math.min(1, t))
  if (clamped < 0.33) {
    const s = clamped / 0.33
    return new THREE.Color(0.1 + s * 0.1, 0.3 + s * 0.5, 0.8 - s * 0.3)
  } else if (clamped < 0.66) {
    const s = (clamped - 0.33) / 0.33
    return new THREE.Color(0.2 + s * 0.7, 0.8, 0.5 - s * 0.4)
  } else {
    const s = (clamped - 0.66) / 0.34
    return new THREE.Color(0.9, 0.8 - s * 0.6, 0.1)
  }
}

// Rupture color: red if ruptured, green if safe
function ruptureColor(ruptured, safety) {
  if (ruptured) return new THREE.Color(0.87, 0.15, 0.10)
  const s = Math.min((safety - 1) / 3, 1)
  return new THREE.Color(0.1, 0.3 + s * 0.5, 0.2 + s * 0.3)
}


// ══════════════════════════════════════════════════════
// TOMATO MESH COMPONENT
// Builds Three.js BufferGeometry from JSON data
// ══════════════════════════════════════════════════════

function TomatoMesh({ data, scenario, deformScale }) {
  const meshRef = useRef()

  // Build geometry from JSON every time data or scenario changes
  const geometry = useMemo(() => {
    if (!data) return null

    const { nodes, surface, displacement } = data
    const stressField = data[scenario]      // e.g. data["stress_impact"]
    const ruptured    = data.ruptured
    const safety      = data.safety_factor

    const geo = new THREE.BufferGeometry()

    // Build vertex positions + colors
    // Each surface triangle = 3 vertices
    const positions = []
    const colors    = []

    for (const tri of surface) {
      for (const nodeIdx of tri) {
        // Base position
        const [x, y, z] = nodes[nodeIdx]

        // Add deformation (scaled)
        const [dx, dy, dz] = displacement
          ? displacement[nodeIdx]
          : [0, 0, 0]

        positions.push(
          x + dx * deformScale,
          z + dz * deformScale,   // swap y/z: Python uses Z-up, Three.js uses Y-up
          y + dy * deformScale
        )

        // Color by stress or rupture
        let col
        if (scenario === "stress_combined" && ruptured && safety) {
          col = ruptureColor(ruptured[nodeIdx], safety[nodeIdx])
        } else if (stressField) {
          col = stressToColor(stressField[nodeIdx])
        } else {
          col = new THREE.Color(0.87, 0.24, 0.18)  // default tomato red
        }

        colors.push(col.r, col.g, col.b)
      }
    }

    geo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3))
    geo.setAttribute("color",    new THREE.Float32BufferAttribute(colors, 3))
    geo.computeVertexNormals()

    return geo
  }, [data, scenario, deformScale])

  if (!geometry) return null

  return (
    <mesh ref={meshRef} geometry={geometry}>
      <meshStandardMaterial
        vertexColors={true}
        side={THREE.DoubleSide}
        roughness={0.6}
        metalness={0.0}
      />
    </mesh>
  )
}


// ══════════════════════════════════════════════════════
// STATS BAR
// ══════════════════════════════════════════════════════

function StatsBar({ data, scenario }) {
  if (!data) return null

  const { metadata } = data
  const sigmaYield = metadata?.sigma_yield_Pa || 15000
  const nRuptured  = metadata?.n_ruptured || 0
  const nNodes     = metadata?.n_nodes || 0

  return (
    <div style={{
      display: "flex", gap: 24, flexWrap: "wrap",
      background: "#191714", border: "1px solid #2a2724",
      borderRadius: 8, padding: "10px 16px",
    }}>
      {[
        ["Nodes",         nNodes],
        ["Elements",      metadata?.n_elements],
        ["σ_yield",       `${sigmaYield / 1000} kPa`],
        ["Ruptured",      `${nRuptured} / ${nNodes}`],
        ["Safety min",    data.safety_factor
                            ? Math.min(...data.safety_factor).toFixed(2)
                            : "—"],
      ].map(([label, value]) => (
        <div key={label} style={{ display: "flex", flexDirection: "column", gap: 2 }}>
          <span style={{ fontSize: 10, color: "#7a7268", fontFamily: "monospace" }}>{label}</span>
          <span style={{ fontSize: 13, color: "#f0ebe3", fontWeight: 700, fontFamily: "monospace" }}>{value}</span>
        </div>
      ))}
    </div>
  )
}


// ══════════════════════════════════════════════════════
// PARAMETER PANEL
// Sliders to re-run FEM with new values
// ══════════════════════════════════════════════════════

function ParamPanel({ onRun, loading }) {
  const [params, setParams] = useState({
    E:           80000,
    nu:          0.47,
    load:        10,
    pressure:    3000,
    drop_height: 0.5,
    sigma_yield: 15000,
  })

  const set = (key, val) => setParams(p => ({ ...p, [key]: val }))

  const sliders = [
    { key: "E",           label: "Young's modulus E",  min: 10000,  max: 300000, step: 5000,  unit: "Pa",  tip: "Stiffness of flesh" },
    { key: "nu",          label: "Poisson's ratio ν",  min: 0.1,    max: 0.49,   step: 0.01,  unit: "",    tip: "Incompressibility" },
    { key: "load",        label: "Squeeze force",       min: 1,      max: 100,    step: 1,     unit: "N",   tip: "Finger press load" },
    { key: "pressure",    label: "Internal pressure",   min: 500,    max: 10000,  step: 500,   unit: "Pa",  tip: "Overripe juice pressure" },
    { key: "drop_height", label: "Drop height",         min: 0.1,    max: 2.0,    step: 0.1,   unit: "m",   tip: "Height before drop" },
    { key: "sigma_yield", label: "Yield stress",        min: 5000,   max: 50000,  step: 1000,  unit: "Pa",  tip: "Skin rupture threshold" },
  ]

  return (
    <div style={{
      background: "#191714", border: "1px solid #2a2724",
      borderRadius: 10, padding: 16,
      display: "flex", flexDirection: "column", gap: 12,
    }}>
      <div style={{ fontSize: 11, color: "#2ab8a0", fontWeight: 700,
                    letterSpacing: 1, textTransform: "uppercase", fontFamily: "monospace" }}>
        FEM Parameters
      </div>

      {sliders.map(({ key, label, min, max, step, unit, tip }) => (
        <div key={key} style={{ display: "flex", flexDirection: "column", gap: 3 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
            <span style={{ fontSize: 11, color: "#c4b8a8", fontFamily: "monospace" }}>{label}</span>
            <span style={{ fontSize: 11, color: "#f0ebe3", fontWeight: 700, fontFamily: "monospace" }}>
              {params[key]} {unit}
            </span>
          </div>
          <input
            type="range" min={min} max={max} step={step}
            value={params[key]}
            onChange={e => set(key, parseFloat(e.target.value))}
            style={{ accentColor: "#e03c2e", width: "100%" }}
          />
          <span style={{ fontSize: 9, color: "#7a7268", fontFamily: "monospace" }}>{tip}</span>
        </div>
      ))}

      <button
        onClick={() => onRun(params)}
        disabled={loading}
        style={{
          marginTop: 4,
          background: loading ? "#2a2724" : "#e03c2e",
          color: loading ? "#7a7268" : "#fff",
          border: "none", borderRadius: 6,
          padding: "10px 0", fontSize: 13,
          fontWeight: 700, fontFamily: "monospace",
          cursor: loading ? "not-allowed" : "pointer",
          transition: "background 0.2s",
        }}
      >
        {loading ? "⏳ Running FEM..." : "▶ Run FEM"}
      </button>
    </div>
  )
}


// ══════════════════════════════════════════════════════
// STRESS LEGEND
// ══════════════════════════════════════════════════════

function StressLegend({ scenario }) {
  const isRupture = scenario === "stress_combined"
  const stops = 12

  return (
    <div style={{
      position: "absolute", right: 16, top: "50%",
      transform: "translateY(-50%)",
      display: "flex", flexDirection: "column",
      alignItems: "center", gap: 4,
    }}>
      <span style={{ fontSize: 9, color: "#7a7268", fontFamily: "monospace" }}>
        {isRupture ? "ruptured" : "high σ"}
      </span>
      <div style={{ width: 14, height: 100, borderRadius: 4, overflow: "hidden",
                    display: "flex", flexDirection: "column" }}>
        {Array.from({ length: stops }, (_, i) => {
          const t = 1 - i / (stops - 1)
          const c = isRupture
            ? (t > 0.5 ? "rgb(222,38,25)" : "rgb(26,128,80)")
            : (() => {
                const col = stressToColor(t)
                return `rgb(${Math.round(col.r*255)},${Math.round(col.g*255)},${Math.round(col.b*255)})`
              })()
          return <div key={i} style={{ flex: 1, background: c }} />
        })}
      </div>
      <span style={{ fontSize: 9, color: "#7a7268", fontFamily: "monospace" }}>
        {isRupture ? "safe" : "low σ"}
      </span>
    </div>
  )
}


// ══════════════════════════════════════════════════════
// MAIN APP
// ══════════════════════════════════════════════════════

export default function App() {
  const [data,          setData]          = useState(null)
  const [scenario,      setScenario]      = useState("stress_impact")
  const [deformScale,   setDeformScale]   = useState(20)
  const [loading,       setLoading]       = useState(false)
  const [error,         setError]         = useState(null)
  const [backendStatus, setBackendStatus] = useState("checking")

  // ── Check backend is alive ──────────────────────────
  useEffect(() => {
    fetch(`${API}/status`)
      .then(r => r.json())
      .then(() => setBackendStatus("ok"))
      .catch(() => setBackendStatus("offline"))
  }, [])

  // ── Load precomputed mesh on startup ────────────────
  useEffect(() => {
    if (backendStatus !== "ok") return
    setLoading(true)
    fetch(`${API}/mesh`)
      .then(r => r.json())
      .then(d => { setData(d); setLoading(false) })
      .catch(e => { setError(e.message); setLoading(false) })
  }, [backendStatus])

  // ── Re-run FEM with new parameters ─────────────────
  const handleRun = (params) => {
    setLoading(true)
    setError(null)
    fetch(`${API}/run-fem`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(params),
    })
      .then(r => r.json())
      .then(d => {
        if (d.error) throw new Error(d.error)
        setData(d)
        setLoading(false)
      })
      .catch(e => { setError(e.message); setLoading(false) })
  }

  return (
    <div style={{
      background: "#0f0e0d", minHeight: "100vh",
      color: "#f0ebe3", fontFamily: "monospace",
      display: "flex", flexDirection: "column",
      padding: 16, gap: 12, boxSizing: "border-box",
    }}>

      {/* Header */}
      <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
        <span style={{ fontSize: 22, fontWeight: 800, color: "#e03c2e" }}>🍅 FEM Tomato</span>
        <span style={{ fontSize: 12, color: "#7a7268" }}>3D Finite Element Analysis</span>
        <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 6 }}>
          <div style={{
            width: 8, height: 8, borderRadius: "50%",
            background: backendStatus === "ok" ? "#2ab8a0" : "#e03c2e"
          }} />
          <span style={{ fontSize: 11, color: "#7a7268" }}>
            {backendStatus === "ok" ? "backend connected" : "backend offline — run server.py"}
          </span>
        </div>
      </div>

      {/* Scenario tabs */}
      <div style={{ display: "flex", gap: 8 }}>
        {SCENARIOS.map(s => (
          <button key={s.key} onClick={() => setScenario(s.key)}
            style={{
              background: scenario === s.key ? s.color : "#191714",
              color:      scenario === s.key ? "#000" : "#7a7268",
              border:     `1px solid ${scenario === s.key ? s.color : "#2a2724"}`,
              borderRadius: 6, padding: "6px 14px",
              fontSize: 12, fontWeight: 700,
              cursor: "pointer", fontFamily: "monospace",
            }}>
            {s.label}
          </button>
        ))}
      </div>

      {/* Main layout */}
      <div style={{ display: "flex", gap: 12, flex: 1, minHeight: 500 }}>

        {/* 3D Canvas */}
        <div style={{
          flex: 1, background: "#191714",
          border: "1px solid #2a2724", borderRadius: 10,
          position: "relative", overflow: "hidden",
          minHeight: 460,
        }}>
          {backendStatus === "offline" && (
            <div style={{
              position: "absolute", inset: 0,
              display: "flex", alignItems: "center", justifyContent: "center",
              flexDirection: "column", gap: 8,
            }}>
              <span style={{ fontSize: 32 }}>⚠️</span>
              <span style={{ color: "#e03c2e", fontSize: 13 }}>Backend offline</span>
              <span style={{ color: "#7a7268", fontSize: 11 }}>
                Run: cd backend && python3 server.py
              </span>
            </div>
          )}

          {loading && (
            <div style={{
              position: "absolute", inset: 0, zIndex: 10,
              display: "flex", alignItems: "center", justifyContent: "center",
              background: "#0f0e0dcc",
            }}>
              <span style={{ color: "#e8c84a", fontSize: 14 }}>⏳ Running FEM pipeline...</span>
            </div>
          )}

          {error && (
            <div style={{
              position: "absolute", bottom: 12, left: 12, right: 12, zIndex: 10,
              background: "#3a0f0d", border: "1px solid #e03c2e",
              borderRadius: 6, padding: 10, fontSize: 11, color: "#f0ebe3",
            }}>
              ❌ {error}
            </div>
          )}

          {data && (
            <Canvas style={{ width: "100%", height: "100%" }}>
              <PerspectiveCamera makeDefault position={[3, 2, 3]} fov={45} />
              <OrbitControls enablePan={true} enableZoom={true} enableRotate={true} />
              <ambientLight intensity={0.5} />
              <directionalLight position={[5, 5, 5]} intensity={1.0} />
              <directionalLight position={[-5, -3, -5]} intensity={0.3} />
              <TomatoMesh
                data={data}
                scenario={scenario}
                deformScale={deformScale / 1000}
              />
            </Canvas>
          )}

          <StressLegend scenario={scenario} />

          {/* Deform scale slider */}
          <div style={{
            position: "absolute", bottom: 12, left: 12,
            display: "flex", alignItems: "center", gap: 8,
          }}>
            <span style={{ fontSize: 10, color: "#7a7268" }}>Deform scale</span>
            <input
              type="range" min={0} max={100} value={deformScale}
              onChange={e => setDeformScale(+e.target.value)}
              style={{ accentColor: "#2ab8a0", width: 100 }}
            />
            <span style={{ fontSize: 10, color: "#f0ebe3" }}>{deformScale}×</span>
          </div>
        </div>

        {/* Right panel */}
        <div style={{ width: 240, display: "flex", flexDirection: "column", gap: 10 }}>
          <ParamPanel onRun={handleRun} loading={loading} />
        </div>
      </div>

      {/* Stats bar */}
      <StatsBar data={data} scenario={scenario} />

      <div style={{ fontSize: 10, color: "#2a2724", textAlign: "center" }}>
        FEM Tomato · Python backend + React frontend · drag to rotate · scroll to zoom
      </div>
    </div>
  )
}
