// Persistent 2D Lidar SLAM WebSocket Client & Canvas Renderer

const canvas = document.getElementById("slam-canvas");
const ctx = canvas.getContext("2d");
const container = document.getElementById("canvas-container");

// UI Elements
const statusBadge = document.getElementById("status-badge");
const confBar = document.getElementById("confidence-bar");
const confVal = document.getElementById("confidence-val");
const poseVal = document.getElementById("pose-val");
const submapsVal = document.getElementById("submaps-val");
const keyframesVal = document.getElementById("keyframes-val");
const diagResidual = document.getElementById("diag-residual");
const diagDegen = document.getElementById("diag-degen");
const mapNameInput = document.getElementById("map-name-input");

// Viewport State (meters -> canvas pixels)
let scale = 40.0; // pixels per meter
let panX = 0;
let panY = 0;
let isDragging = false;
let startX, startY;

// SLAM telemetry data
let currentPose = [0, 0, 0];
let trajectory = [];
let livePts = [];
let submapsList = [];
let globalGridData = null;

function resizeCanvas() {
  canvas.width = container.clientWidth;
  canvas.height = container.clientHeight;
  if (panX === 0 && panY === 0) {
    panX = canvas.width / 2;
    panY = canvas.height / 2;
  }
}
window.addEventListener("resize", resizeCanvas);
resizeCanvas();

// Mouse Pan & Zoom
canvas.addEventListener("mousedown", (e) => {
  isDragging = true;
  startX = e.clientX - panX;
  startY = e.clientY - panY;
});
window.addEventListener("mouseup", () => { isDragging = false; });
canvas.addEventListener("mousemove", (e) => {
  if (isDragging) {
    panX = e.clientX - startX;
    panY = e.clientY - startY;
    render();
  }
});
canvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  const zoomFactor = e.deltaY < 0 ? 1.15 : 0.85;
  const mouseX = e.clientX - canvas.getBoundingClientRect().left;
  const mouseY = e.clientY - canvas.getBoundingClientRect().top;

  panX = mouseX - (mouseX - panX) * zoomFactor;
  panY = mouseY - (mouseY - panY) * zoomFactor;
  scale *= zoomFactor;
  render();
});

document.getElementById("btn-reset-view").addEventListener("click", () => {
  panX = canvas.width / 2;
  panY = canvas.height / 2;
  scale = 40.0;
  render();
});

// World to Screen transformation
function worldToScreen(wx, wy) {
  // Screen X = panX + wx * scale
  // Screen Y = panY - wy * scale (invert Y so world +Y is up)
  return [panX + wx * scale, panY - wy * scale];
}

// Render loop
function render() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // 1. Draw grid coordinate axes
  drawAxes();

  // 2. Draw submaps / occupancy cells
  if (submapsList && submapsList.length > 0) {
    drawSubmaps();
  }

  // 3. Draw trajectory line
  if (trajectory.length > 1) {
    ctx.beginPath();
    const [p0x, p0y] = worldToScreen(trajectory[0][0], trajectory[0][1]);
    ctx.moveTo(p0x, p0y);
    for (let i = 1; i < trajectory.length; i++) {
      const [px, py] = worldToScreen(trajectory[i][0], trajectory[i][1]);
      ctx.lineTo(px, py);
    }
    ctx.strokeStyle = "rgba(56, 189, 248, 0.7)";
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  // 4. Draw live lidar return points
  if (livePts && livePts.length > 0) {
    ctx.fillStyle = "#ef4444";
    for (let i = 0; i < livePts.length; i++) {
      const [sx, sy] = worldToScreen(livePts[i][0], livePts[i][1]);
      ctx.fillRect(sx - 1.5, sy - 1.5, 3, 3);
    }
  }

  // 5. Draw Robot Pose
  drawRobot(currentPose[0], currentPose[1], currentPose[2]);
}

function drawAxes() {
  ctx.strokeStyle = "#1e293b";
  ctx.lineWidth = 1;
  const step = scale * 1.0; // 1 meter grid
  
  const startGridX = panX % step;
  for (let x = startGridX; x < canvas.width; x += step) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, canvas.height);
    ctx.stroke();
  }
  const startGridY = panY % step;
  for (let y = startGridY; y < canvas.height; y += step) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(canvas.width, y);
    ctx.stroke();
  }
}

function drawSubmaps() {
  for (const sm of submapsList) {
    let color = "#38bdf8"; // saved
    if (sm.is_active) color = "#22c55e"; // active
    if (sm.is_unlinked) color = "#f97316"; // unlinked

    ctx.fillStyle = color;
    if (sm.cells) {
      for (const pt of sm.cells) {
        const [sx, sy] = worldToScreen(pt[0], pt[1]);
        ctx.fillRect(sx - 2, sy - 2, 4, 4);
      }
    }
  }
}

function drawRobot(rx, ry, rtheta) {
  const [sx, sy] = worldToScreen(rx, ry);
  const radius = 10;

  // Robot body
  ctx.beginPath();
  ctx.arc(sx, sy, radius, 0, Math.PI * 2);
  ctx.fillStyle = "#38bdf8";
  ctx.fill();
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 2;
  ctx.stroke();

  // Heading arrow
  const headingLength = 18;
  const hx = sx + Math.cos(-rtheta) * headingLength;
  const hy = sy + Math.sin(-rtheta) * headingLength;

  ctx.beginPath();
  ctx.moveTo(sx, sy);
  ctx.lineTo(hx, hy);
  ctx.strokeStyle = "#facc15";
  ctx.lineWidth = 3;
  ctx.stroke();
}

// WebSocket Connection
let ws = null;
function connectWebSocket() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

  ws.onopen = () => {
    document.getElementById("diag-conn").textContent = "Connected";
    document.getElementById("diag-conn").className = "diag-ok";
  };

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === "telemetry") {
      updateTelemetry(data);
    } else if (data.event === "saved") {
      alert(`Map '${data.name}' successfully saved!`);
    } else if (data.event === "loaded") {
      alert(`Map '${data.name}' successfully loaded into memory.`);
    } else if (data.event === "error") {
      alert(`Error: ${data.message}`);
    }
  };

  ws.onclose = () => {
    document.getElementById("diag-conn").textContent = "Disconnected (Retrying...)";
    document.getElementById("diag-conn").className = "diag-warn";
    setTimeout(connectWebSocket, 2000);
  };
}

function updateTelemetry(data) {
  currentPose = data.pose || [0, 0, 0];
  trajectory.push([currentPose[0], currentPose[1]]);
  if (trajectory.length > 500) trajectory.shift();

  if (data.pts_world) {
    livePts = data.pts_world;
  }
  if (data.submaps) {
    submapsList = data.submaps;
  }

  // Update status badge
  const status = data.status || "TRACKING";
  statusBadge.textContent = status;
  statusBadge.className = "status-badge " + status.toLowerCase().replace(/_/g, "-");

  // Update confidence
  const conf = Math.round((data.confidence || 0) * 100);
  confBar.style.width = `${conf}%`;
  confVal.textContent = `${conf}%`;

  // Update pose
  const deg = Math.round((currentPose[2] * 180 / Math.PI) * 10) / 10;
  poseVal.textContent = `X: ${currentPose[0].toFixed(2)}m | Y: ${currentPose[1].toFixed(2)}m | ${deg}°`;

  // Submaps & keyframes
  submapsVal.textContent = data.num_submaps || 1;
  keyframesVal.textContent = data.num_keyframes || 0;

  // Diagnostics
  diagResidual.textContent = `${(data.mean_residual || 0.02).toFixed(3)} m`;
  if (data.is_degenerate) {
    diagDegen.textContent = "DEGENERATE (Low Info)";
    diagDegen.className = "diag-warn";
  } else {
    diagDegen.textContent = "OK (Constrained)";
    diagDegen.className = "diag-ok";
  }

  render();
}

// Button actions
document.getElementById("btn-save").addEventListener("click", () => {
  const name = mapNameInput.value.trim() || "session_manual";
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ action: "save", name: name }));
  }
});

document.getElementById("btn-load").addEventListener("click", () => {
  const name = mapNameInput.value.trim() || "session_manual";
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ action: "load", name: name }));
  }
});

document.getElementById("btn-export").addEventListener("click", () => {
  const name = mapNameInput.value.trim() || "session_manual";
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ action: "export", name: name }));
  }
});

connectWebSocket();

