"""
src/visualization/visualizer.py

Step 6: Visualization — War Room Mission Control

Outputs a standalone HTML file with:
    - Animated 3D globe with live satellite positions (Three.js)
    - Conjunction event timeline
    - Stats bar
    - Satellite catalog table
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from src.models import Trajectory
from src.conjunction.conjunction_models import ConjunctionEvent

logger = logging.getLogger(__name__)

RISK_COLORS = {
    "red":    "#FF4B4B",
    "yellow": "#FFD700",
    "green":  "#00CC96",
    "none":   "#636EFA",
}

BG_COLOR    = "#0D1117"
PANEL_COLOR = "#161B22"
GRID_COLOR  = "#21262D"
TEXT_COLOR  = "#E6EDF3"
MUTED_COLOR = "#8B949E"


def _event_risk_level(event: ConjunctionEvent) -> str:
    if event.pc is None:
        return "none"
    if event.pc >= 1e-4:
        return "red"
    elif event.pc >= 1e-5:
        return "yellow"
    return "green"


class SatelliteVisualizer:

    def __init__(self, output_dir: str = "data/visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def render(
        self,
        trajectories: dict[int, Trajectory],
        events: list[ConjunctionEvent],
        output_file: str = "mission_control.html",
    ) -> str:
        logger.info("Rendering war room dashboard: %d sats, %d events",
                    len(trajectories), len(events))

        sat_data   = self._build_sat_data(trajectories)
        event_data = self._build_event_data(events)
        stats      = self._build_stats(trajectories, events)

        html = self._build_html(sat_data, event_data, stats)
        output_path = self.output_dir / output_file
        output_path.write_text(html, encoding="utf-8")
        logger.info("War room dashboard saved → %s", output_path)
        return str(output_path)

    def _build_sat_data(self, trajectories: dict[int, Trajectory]) -> list[dict]:
        data = []
        for traj in trajectories.values():
            positions = []
            for sv in traj.states[::6]:
                positions.append([
                    round(sv.position[0], 2),
                    round(sv.position[1], 2),
                    round(sv.position[2], 2),
                ])
            if not positions:
                continue
            alts = [np.linalg.norm(sv.position) - 6371.0 for sv in traj.states]
            speeds = [float(np.linalg.norm(sv.velocity)) for sv in traj.states]
            data.append({
                "id":       traj.norad_id,
                "name":     traj.name if traj.name != "UNKNOWN" else f"SAT-{traj.norad_id}",
                "positions": positions,
                "alt_min":  round(min(alts), 1),
                "alt_max":  round(max(alts), 1),
                "speed":    round(float(np.mean(speeds)), 3),
            })
        return data

    def _build_event_data(self, events: list[ConjunctionEvent]) -> list[dict]:
        data = []
        for e in events:
            risk = _event_risk_level(e)
            data.append({
                "primary":   e.primary_name if e.primary_name != "UNKNOWN" else f"SAT-{e.primary_id}",
                "secondary": e.secondary_name if e.secondary_name != "UNKNOWN" else f"SAT-{e.secondary_id}",
                "tca":       e.tca.strftime("%Y-%m-%d %H:%M:%S"),
                "miss":      round(e.miss_distance, 3),
                "vrel":      round(e.relative_velocity, 3),
                "pc":        f"{e.pc:.3e}" if e.pc else "N/A",
                "risk":      risk,
            })
        return data

    def _build_stats(self, trajectories, events) -> dict:
        n_red    = sum(1 for e in events if _event_risk_level(e) == "red")
        n_yellow = sum(1 for e in events if _event_risk_level(e) == "yellow")
        n_green  = sum(1 for e in events if _event_risk_level(e) == "green")
        n        = len(trajectories)
        pairs    = n * (n - 1) // 2
        return {
            "satellites": n,
            "pairs":      pairs,
            "events":     len(events),
            "red":        n_red,
            "yellow":     n_yellow,
            "green":      n_green,
            "timestamp":  datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        }

    def _build_html(self, sat_data, event_data, stats) -> str:
        sat_json   = json.dumps(sat_data)
        event_json = json.dumps(event_data)
        stats_json = json.dumps(stats)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SCRAS — War Room</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0D1117;color:#E6EDF3;font-family:'Courier New',monospace;overflow-x:hidden}}
header{{background:#0D1117;border-bottom:1px solid #21262D;padding:12px 28px;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:100}}
.logo{{font-size:1.1rem;letter-spacing:.15em;color:#58A6FF;font-weight:700}}
.logo span{{color:#8B949E;font-weight:400}}
.htime{{font-size:.75rem;color:#8B949E;letter-spacing:.08em}}
.stats-bar{{display:flex;gap:0;border-bottom:1px solid #21262D}}
.stat{{flex:1;padding:14px 20px;border-right:1px solid #21262D;text-align:center}}
.stat:last-child{{border-right:none}}
.stat .val{{font-size:1.5rem;font-weight:700;letter-spacing:.05em}}
.stat .lbl{{font-size:.65rem;color:#8B949E;letter-spacing:.12em;text-transform:uppercase;margin-top:3px}}
.red{{color:#FF4B4B}}.yellow{{color:#FFD700}}.green{{color:#00CC96}}.blue{{color:#58A6FF}}
.main{{display:grid;grid-template-columns:1fr 360px;height:calc(100vh - 110px)}}
#globe-wrap{{position:relative;overflow:hidden;background:#0D1117}}
canvas{{display:block}}
.side{{background:#161B22;border-left:1px solid #21262D;display:flex;flex-direction:column;overflow:hidden}}
.side-header{{padding:12px 16px;border-bottom:1px solid #21262D;font-size:.7rem;letter-spacing:.15em;text-transform:uppercase;color:#8B949E;display:flex;align-items:center;justify-content:space-between}}
.side-header .count{{background:#21262D;padding:2px 8px;border-radius:10px;color:#E6EDF3;font-size:.7rem}}
.events-list{{flex:1;overflow-y:auto;padding:8px}}
.events-list::-webkit-scrollbar{{width:4px}}
.events-list::-webkit-scrollbar-thumb{{background:#21262D;border-radius:2px}}
.event-card{{background:#0D1117;border:1px solid #21262D;border-radius:6px;padding:10px 12px;margin-bottom:6px;cursor:pointer;transition:border-color .2s}}
.event-card:hover{{border-color:#58A6FF}}
.event-card.red{{border-left:3px solid #FF4B4B}}
.event-card.yellow{{border-left:3px solid #FFD700}}
.event-card.green{{border-left:3px solid #00CC96}}
.event-card.none{{border-left:3px solid #636EFA}}
.ec-names{{font-size:.8rem;color:#E6EDF3;margin-bottom:4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.ec-meta{{display:flex;gap:12px;font-size:.7rem;color:#8B949E}}
.ec-miss{{color:#58A6FF}}.ec-pc{{color:#AB63FA}}
.no-events{{padding:40px 20px;text-align:center;color:#8B949E;font-size:.8rem;line-height:2}}
.no-events .ok{{font-size:1.5rem;color:#00CC96;margin-bottom:8px}}
.timeline-wrap{{border-top:1px solid #21262D;height:140px;padding:8px 16px;flex-shrink:0}}
.tl-label{{font-size:.65rem;letter-spacing:.12em;text-transform:uppercase;color:#8B949E;margin-bottom:6px}}
#tl-canvas{{width:100%;height:90px}}
.sat-table-wrap{{border-top:1px solid #21262D;max-height:220px;overflow-y:auto;flex-shrink:0}}
.sat-table-wrap::-webkit-scrollbar{{width:4px}}
.sat-table-wrap::-webkit-scrollbar-thumb{{background:#21262D;border-radius:2px}}
table{{width:100%;border-collapse:collapse;font-size:.7rem}}
th{{padding:6px 12px;text-align:left;color:#8B949E;letter-spacing:.1em;text-transform:uppercase;border-bottom:1px solid #21262D;position:sticky;top:0;background:#161B22}}
td{{padding:5px 12px;border-bottom:1px solid #21262D20;color:#E6EDF3}}
tr:hover td{{background:#21262D40}}
.hud{{position:absolute;top:12px;left:12px;font-size:.7rem;color:#8B949E;letter-spacing:.08em;line-height:2;pointer-events:none}}
.hud span{{color:#58A6FF}}
</style>
</head>
<body>

<header>
  <div class="logo">SCRAS <span>// SATELLITE COLLISION RISK ANALYSIS SYSTEM</span></div>
  <div class="htime" id="clock">{stats['timestamp']}</div>
</header>

<div class="stats-bar">
  <div class="stat"><div class="val blue">{stats['satellites']}</div><div class="lbl">Satellites</div></div>
  <div class="stat"><div class="val blue">{stats['pairs']:,}</div><div class="lbl">Pairs Screened</div></div>
  <div class="stat"><div class="val blue">5.0 km</div><div class="lbl">Threshold</div></div>
  <div class="stat"><div class="val blue">24h</div><div class="lbl">Window</div></div>
  <div class="stat"><div class="val {'red' if stats['red'] else 'green'}">{stats['red']}</div><div class="lbl">Red Events</div></div>
  <div class="stat"><div class="val {'yellow' if stats['yellow'] else 'green'}">{stats['yellow']}</div><div class="lbl">Yellow Events</div></div>
  <div class="stat"><div class="val green">{stats['green']}</div><div class="lbl">Green Events</div></div>
  <div class="stat"><div class="val blue">{stats['events']}</div><div class="lbl">Total Events</div></div>
</div>

<div class="main">
  <div id="globe-wrap">
    <canvas id="globe"></canvas>
    <div class="hud">
      FRAME: ECI &nbsp;|&nbsp; PROP: SGP4<br>
      SATS: <span id="hud-sats">—</span> &nbsp;|&nbsp; STEP: <span id="hud-step">—</span><br>
      DATA: SPACE-TRACK.ORG
    </div>
  </div>

  <div class="side">
    <div class="side-header">Conjunction Events <span class="count" id="ev-count">{stats['events']}</span></div>
    <div class="events-list" id="events-list"></div>

    <div class="timeline-wrap">
      <div class="tl-label">24h Conjunction Timeline</div>
      <canvas id="tl-canvas"></canvas>
    </div>

    <div class="sat-table-wrap">
      <table>
        <thead><tr><th>Satellite</th><th>Alt (km)</th><th>Speed (km/s)</th></tr></thead>
        <tbody id="sat-tbody"></tbody>
      </table>
    </div>
  </div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const SAT_DATA   = {sat_json};
const EVENT_DATA = {event_json};

// ── Clock ──────────────────────────────────────────────────────────────────
setInterval(()=>{{
  document.getElementById('clock').textContent =
    new Date().toISOString().replace('T',' ').slice(0,19)+' UTC';
}}, 1000);

// ── Events list ────────────────────────────────────────────────────────────
const evList = document.getElementById('events-list');
if(EVENT_DATA.length === 0){{
  evList.innerHTML = `<div class="no-events"><div class="ok">✓</div>No conjunctions detected<br>within 5.0 km threshold<br>across ${{SAT_DATA.length}} satellites</div>`;
}} else {{
  EVENT_DATA.forEach(e=>{{
    const d = document.createElement('div');
    d.className = 'event-card '+e.risk;
    d.innerHTML = `<div class="ec-names">${{e.primary}} / ${{e.secondary}}</div>
      <div class="ec-meta">
        <span>TCA: ${{e.tca.slice(11,19)}}</span>
        <span class="ec-miss">↔ ${{e.miss}} km</span>
        <span class="ec-pc">Pc: ${{e.pc}}</span>
      </div>`;
    evList.appendChild(d);
  }});
}}

// ── Satellite table ─────────────────────────────────────────────────────────
const tbody = document.getElementById('sat-tbody');
SAT_DATA.forEach(s=>{{
  const tr = document.createElement('tr');
  tr.innerHTML = `<td>${{s.name}}</td><td>${{s.alt_min}}–${{s.alt_max}}</td><td>${{s.speed}}</td>`;
  tbody.appendChild(tr);
}});

// ── Timeline canvas ────────────────────────────────────────────────────────
(function(){{
  const cv = document.getElementById('tl-canvas');
  const pr = window.devicePixelRatio||1;
  cv.width  = cv.offsetWidth  * pr;
  cv.height = cv.offsetHeight * pr;
  const ctx = cv.getContext('2d');
  ctx.scale(pr,pr);
  const w = cv.offsetWidth, h = cv.offsetHeight;

  ctx.strokeStyle = '#21262D';
  ctx.lineWidth = 0.5;
  for(let i=0;i<=4;i++){{
    const x = (w/4)*i;
    ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,h); ctx.stroke();
    ctx.fillStyle='#8B949E'; ctx.font='9px Courier New';
    ctx.fillText(i*6+'h', x+2, h-2);
  }}

  if(EVENT_DATA.length===0){{
    ctx.fillStyle='#00CC964D';
    ctx.fillRect(0, h/2-1, w, 2);
    ctx.fillStyle='#00CC96';
    ctx.font='10px Courier New';
    ctx.fillText('No events — clear window', w/2-60, h/2-6);
  }} else {{
    EVENT_DATA.forEach(e=>{{
      const tParts = e.tca.split(' ')[1].split(':');
      const tH = parseInt(tParts[0]) + parseInt(tParts[1])/60;
      const x = (tH/24)*w;
      const colors = {{red:'#FF4B4B',yellow:'#FFD700',green:'#00CC96',none:'#636EFA'}};
      ctx.fillStyle = colors[e.risk]||'#636EFA';
      ctx.beginPath();
      ctx.arc(x, h/2, 5, 0, Math.PI*2);
      ctx.fill();
    }});
  }}
}})();

// ── Three.js Globe ─────────────────────────────────────────────────────────
(function(){{
  const wrap  = document.getElementById('globe-wrap');
  const canvas= document.getElementById('globe');

  const renderer = new THREE.WebGLRenderer({{canvas, antialias:true, alpha:true}});
  renderer.setPixelRatio(window.devicePixelRatio);

  const scene  = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100000);
  camera.position.set(0, 0, 22000);

  function resize(){{
    const w = wrap.clientWidth, h = wrap.clientHeight;
    renderer.setSize(w,h);
    camera.aspect = w/h;
    camera.updateProjectionMatrix();
  }}
  resize();
  window.addEventListener('resize', resize);

  // Earth
  const earthGeo  = new THREE.SphereGeometry(6371, 48, 48);
  const texLoader = new THREE.TextureLoader();
  const earthTex  = texLoader.load('https://unpkg.com/three-globe/example/img/earth-blue-marble.jpg');
  const earthMat  = new THREE.MeshPhongMaterial({{
    map: earthTex,
    shininess: 15,
  }});const earth = new THREE.Mesh(earthGeo, earthMat);
  scene.add(earth);

  // Grid lines on earth
  const gridMat = new THREE.LineBasicMaterial({{color:0x1E3A5F, transparent:true, opacity:0.4}});
  for(let lat=-80;lat<=80;lat+=20){{
    const pts=[];
    const r=6371+2;
    for(let lon=0;lon<=360;lon+=5){{
      const phi=(90-lat)*Math.PI/180, theta=lon*Math.PI/180;
      pts.push(new THREE.Vector3(r*Math.sin(phi)*Math.cos(theta),r*Math.cos(phi),r*Math.sin(phi)*Math.sin(theta)));
    }}
    scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),gridMat));
  }}
  for(let lon=0;lon<360;lon+=30){{
    const pts=[];
    const r=6371+2;
    for(let lat=-90;lat<=90;lat+=5){{
      const phi=(90-lat)*Math.PI/180, theta=lon*Math.PI/180;
      pts.push(new THREE.Vector3(r*Math.sin(phi)*Math.cos(theta),r*Math.cos(phi),r*Math.sin(phi)*Math.sin(theta)));
    }}
    scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),gridMat));
  }}

  // Atmosphere glow ring
  const atmGeo = new THREE.SphereGeometry(6571, 32, 32);
  const atmMat = new THREE.MeshPhongMaterial({{
    color:0x1A6BFF, transparent:true, opacity:0.06, side:THREE.BackSide
  }});
  scene.add(new THREE.Mesh(atmGeo, atmMat));

  // Lights
  scene.add(new THREE.AmbientLight(0x334466, 1.2));
  const sun = new THREE.DirectionalLight(0x6699FF, 0.8);
  sun.position.set(30000,10000,20000);
  scene.add(sun);

  // Satellite orbits + dots
  const ORBIT_COLORS = [
    0x58A6FF,0x00CC96,0xAB63FA,0xFFA15A,
    0x19D3F3,0xFF6692,0xB6E880,0xFF97FF,
    0xFECB52,0x00B5F7,0xE4E4E4
  ];

  const satMeshes = [];
  const orbitStep = [];

  SAT_DATA.forEach((sat,idx)=>{{
    const col = ORBIT_COLORS[idx % ORBIT_COLORS.length];

    // Orbit trail
    if(sat.positions.length>1){{
      const pts = sat.positions.map(p=>new THREE.Vector3(p[0],p[2],p[1]));
      const lineMat = new THREE.LineBasicMaterial({{color:col, transparent:true, opacity:0.25}});
      const lineGeo = new THREE.BufferGeometry().setFromPoints(pts);
      scene.add(new THREE.Line(lineGeo, lineMat));
    }}

    // Satellite dot
    const dotGeo = new THREE.SphereGeometry(55, 8, 8);
    const dotMat = new THREE.MeshBasicMaterial({{color:col}});
    const dot    = new THREE.Mesh(dotGeo, dotMat);
    if(sat.positions.length>0){{
      dot.position.set(sat.positions[0][0], sat.positions[0][2], sat.positions[0][1]);
    }}
    scene.add(dot);
    satMeshes.push(dot);
    orbitStep.push(0);
  }});

  // Conjunction markers
  EVENT_DATA.forEach(e=>{{
    const col = {{red:0xFF4B4B,yellow:0xFFD700,green:0x00CC96,none:0x636EFA}}[e.risk];
    const geo = new THREE.OctahedronGeometry(150, 0);
    const mat = new THREE.MeshBasicMaterial({{color:col, wireframe:true}});
    const mesh= new THREE.Mesh(geo,mat);
    mesh.position.set(0,7000,0);
    scene.add(mesh);
  }});

  document.getElementById('hud-sats').textContent = SAT_DATA.length;
  document.getElementById('hud-step').textContent = '60s';

  // Mouse drag
  let isDragging=false, prevMouse={{x:0,y:0}};
  const globeGroup = new THREE.Group();
  globeGroup.add(earth);
  scene.add(globeGroup);

  wrap.addEventListener('mousedown',e=>{{isDragging=true;prevMouse={{x:e.clientX,y:e.clientY}}}});
  wrap.addEventListener('mouseup',  ()=>{{isDragging=false}});
  wrap.addEventListener('mousemove',e=>{{
    if(!isDragging) return;
    const dx=e.clientX-prevMouse.x, dy=e.clientY-prevMouse.y;
    earth.rotation.y += dx*0.005;
    earth.rotation.x += dy*0.005;
    prevMouse={{x:e.clientX,y:e.clientY}};
  }});
  wrap.addEventListener('wheel',e=>{{
    camera.position.z = Math.max(8000,Math.min(45000,camera.position.z+e.deltaY*8));
  }});

  // Animation
  let frame=0;
  function animate(){{
    requestAnimationFrame(animate);
    frame++;

    earth.rotation.y += 0.0003;

    satMeshes.forEach((dot,idx)=>{{
      const sat = SAT_DATA[idx];
      if(!sat.positions.length) return;
      const step = Math.floor(frame*0.3) % sat.positions.length;
      const p = sat.positions[step];
      dot.position.set(p[0], p[2], p[1]);
    }});

    renderer.render(scene,camera);
  }}
  animate();
}})();
</script>
</body>
</html>"""