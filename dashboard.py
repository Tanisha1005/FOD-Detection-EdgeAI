"""
dashboard.py — Sentinel FOD Detection & Maintenance Priority System
────────────────────────────────────────────────────────────────────
Visualises live data from the Raspberry Pi 5 Sentinel node:
  • YOLOv10n camera detections (bounding box feed + log)
  • Radar sweep returns
  • PIR motion sensor zone triggers
  • Maintenance priority queue
  • Node system health

Run:
    streamlit run dashboard.py

Requires:
    pip install streamlit>=1.37 paho-mqtt>=2.0 pandas opencv-python-headless
"""

import base64
import json
import math
import queue
import time
from datetime import datetime

import pandas as pd
import paho.mqtt.client as mqtt
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
#  Page config  (MUST be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentinel — FOD Command Centre",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
#  Design constants
# ─────────────────────────────────────────────────────────────────────────────
RISK_HEX = {
    "CRITICAL": "#ef4444",
    "HIGH":     "#f97316",
    "MODERATE": "#eab308",
    "LOW":      "#22c55e",
}
RISK_BG = {
    "CRITICAL": "#3b0808",
    "HIGH":     "#3b1808",
    "MODERATE": "#38350a",
    "LOW":      "#0a3318",
}
RISK_EMOJI  = {"CRITICAL": "🔴", "HIGH": "🟠", "MODERATE": "🟡", "LOW": "🟢"}
RISK_WEIGHT = {"CRITICAL": 0, "HIGH": 1, "MODERATE": 2, "LOW": 3}

MAINT_ACTION = {
    "CRITICAL": "🚨 IMMEDIATE runway closure. Dispatch FOD team now.",
    "HIGH":     "⚠️  Alert ground crew. Clear within 15 minutes.",
    "MODERATE": "📋  Log for next scheduled maintenance sweep.",
    "LOW":      "✅  Monitor. No immediate action required.",
}

# ─────────────────────────────────────────────────────────────────────────────
#  Global CSS — aviation dark-ops aesthetic
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow+Condensed:wght@400;600;700&family=Barlow:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Barlow', sans-serif; }

[data-testid="stMetric"] {
    background: linear-gradient(135deg, #0f1117 0%, #161b27 100%);
    border-radius: 8px;
    padding: 14px 18px;
    border: 1px solid #1f2937;
    border-top: 2px solid #374151;
}
[data-testid="stMetricLabel"] {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.70rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #6b7280;
}
[data-testid="stMetricValue"] {
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.8rem;
    color: #f9fafb;
}
.det-card {
    border-radius: 6px;
    padding: 10px 14px;
    margin-bottom: 8px;
    border-left: 4px solid;
    background: #0d1117;
}
.det-class {
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 700;
    font-size: 1.0rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.det-meta {
    font-family: 'Share Tech Mono', monospace;
    color: #6b7280;
    font-size: 0.74rem;
    margin-top: 3px;
}
.risk-tag {
    float: right;
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.70rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    padding: 2px 10px;
    border-radius: 3px;
}
.sec-head {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.66rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #4b5563;
    margin-bottom: 10px;
    border-bottom: 1px solid #1f2937;
    padding-bottom: 5px;
}
.pill { display:inline-block; padding:3px 12px; border-radius:20px;
        font-family:'Share Tech Mono',monospace; font-size:0.72rem; }
.pill-online  { background:#052e16; color:#4ade80; border:1px solid #166534; }
.pill-offline { background:#3b0808; color:#f87171; border:1px solid #991b1b; }
.pill-unknown { background:#1f2937; color:#9ca3af; border:1px solid #374151; }
.maint-card {
    background: #0d1117;
    border-radius: 6px;
    padding: 10px 14px;
    margin-bottom: 8px;
    border: 1px solid #1f2937;
}
.scroll-pane { max-height: 400px; overflow-y: auto; }
.page-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 2.0rem; font-weight: 700;
    letter-spacing: 0.08em; color: #f9fafb;
}
.page-sub {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.70rem; color: #4b5563; letter-spacing: 0.1em;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Singleton MQTT interface
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_mqtt(broker: str, port: int, camera_id: str):
    msg_q: queue.Queue = queue.Queue(maxsize=300)

    def on_message(client, userdata, msg):
        try:
            payload = msg.payload.decode("utf-8")
            topic   = msg.topic
            if   "/telemetry" in topic: data = {**json.loads(payload), "_type": "telemetry"}
            elif "/image"     in topic: data = {"_type": "image",  "b64":   payload}
            elif "/status"    in topic: data = {"_type": "status", "value": payload}
            else: return
            if not msg_q.full():
                msg_q.put_nowait(data)
        except Exception:
            pass

    def on_connect(client, userdata, flags, rc, props=None):
        if rc == 0:
            client.subscribe(f"sentinel/v1/{camera_id}/#")

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_message = on_message
    client.on_connect = on_connect
    client.will_set(f"sentinel/v1/{camera_id}/status", "Offline", retain=True)
    try:
        client.connect(broker, port, keepalive=60)
        client.loop_start()
    except Exception as e:
        st.error(f"⛔ MQTT failed: {e}")
    return msg_q


# ─────────────────────────────────────────────────────────────────────────────
#  Radar SVG
# ─────────────────────────────────────────────────────────────────────────────
def radar_svg(sweep_angle: float, returns: list, size: int = 250) -> str:
    cx = cy = size // 2
    r  = size // 2 - 10
    lines = []

    for frac in [0.25, 0.5, 0.75, 1.0]:
        cr = int(r * frac)
        lines.append(
            f'<circle cx="{cx}" cy="{cy}" r="{cr}" fill="none" '
            f'stroke="#14532d" stroke-width="1" opacity="0.6"/>'
        )
        if frac < 1.0:
            lines.append(
                f'<text x="{cx+3}" y="{cy-cr+11}" fill="#166534" '
                f'font-size="8" font-family="monospace">{int(frac*100)}m</text>'
            )
    lines.append(f'<line x1="{cx}" y1="{cy-r}" x2="{cx}" y2="{cy+r}" stroke="#14532d" stroke-width="1" opacity="0.5"/>')
    lines.append(f'<line x1="{cx-r}" y1="{cy}" x2="{cx+r}" y2="{cy}" stroke="#14532d" stroke-width="1" opacity="0.5"/>')

    # Sweep trail
    for i in range(30):
        ang = (sweep_angle - i * 2) % 360
        rad = math.radians(ang - 90)
        ex  = cx + r * math.cos(rad)
        ey  = cy + r * math.sin(rad)
        op  = (30 - i) / 30 * 0.5
        lines.append(f'<line x1="{cx}" y1="{cy}" x2="{ex:.1f}" y2="{ey:.1f}" stroke="#4ade80" stroke-width="2" opacity="{op:.2f}"/>')

    # Active sweep line
    rad = math.radians(sweep_angle - 90)
    ex  = cx + r * math.cos(rad)
    ey  = cy + r * math.sin(rad)
    lines.append(f'<line x1="{cx}" y1="{cy}" x2="{ex:.1f}" y2="{ey:.1f}" stroke="#4ade80" stroke-width="2.5"/>')

    # Blips
    for ret in returns:
        frac   = min(ret["distance_m"] / 100.0, 0.97)
        rr     = frac * r
        brad   = math.radians(ret["angle_deg"] - 90)
        bx     = cx + rr * math.cos(brad)
        by     = cy + rr * math.sin(brad)
        br     = max(4, int(ret.get("intensity", 0.7) * 8))
        lines.append(f'<circle cx="{bx:.1f}" cy="{by:.1f}" r="{br}" fill="#ef4444" opacity="0.85"/>')
        lines.append(f'<circle cx="{bx:.1f}" cy="{by:.1f}" r="{br+4}" fill="none" stroke="#ef4444" stroke-width="1" opacity="0.3"/>')

    lines.append(f'<circle cx="{cx}" cy="{cy}" r="4" fill="#4ade80"/>')

    return (
        f'<svg width="{size}" height="{size}" xmlns="http://www.w3.org/2000/svg" '
        f'style="background:#030a03;border-radius:50%;border:2px solid #14532d;">'
        + "\n".join(lines) + "</svg>"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Session-state defaults
# ─────────────────────────────────────────────────────────────────────────────
_D = {
    "log":           [],
    "maint_queue":   {},
    "latest_b64":    None,
    "node_status":   "Unknown",
    "health":        {},
    "last_ts":       "—",
    "radar":         {"sweep_angle": 0, "returns": [], "range_m": 100},
    "motion":        {"triggered": False, "zone": None},
    "total_det":     0,
    "session_start": time.time(),
}
for k, v in _D.items():
    if k not in st.session_state:
        st.session_state[k] = v

MAX_LOG = 300


# ─────────────────────────────────────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="sec-head">⚙ Connection</p>', unsafe_allow_html=True)
    broker    = st.text_input("MQTT Broker", "localhost")
    port      = st.number_input("Port", value=1883, min_value=1, max_value=65535)
    camera_id = st.text_input("Camera / Node ID", "camera_01")
    conf_thr  = st.slider("Min Confidence", 0.30, 0.95, 0.45, 0.05, format="%.2f")

    st.divider()
    st.markdown('<p class="sec-head">📡 Node Health</p>', unsafe_allow_html=True)

    status   = st.session_state.node_status
    pill_cls = ("pill-online"  if status == "Online"  else
                "pill-offline" if status == "Offline" else "pill-unknown")
    st.markdown(
        f'Status &nbsp; <span class="pill {pill_cls}">{status}</span>',
        unsafe_allow_html=True
    )
    st.markdown("<br/>", unsafe_allow_html=True)

    h = st.session_state.health
    c1, c2 = st.columns(2)
    c1.metric("CPU Temp",  f"{h.get('cpu_temp_c','—')}°C")
    c2.metric("Inf. FPS",  h.get("inference_fps", "—"))
    c1.metric("CPU Load",  f"{h.get('cpu_usage_pct','—')}%")
    c2.metric("RAM",       f"{h.get('ram_used_mb','—')} MB")

    uptime_s = h.get("uptime_s")
    if uptime_s:
        m, s   = divmod(int(uptime_s), 60)
        hrs, m = divmod(m, 60)
        st.caption(f"⏱ Uptime: {hrs:02d}h {m:02d}m {s:02d}s")

    st.caption(f"Last packet: {st.session_state.last_ts}")

    st.divider()
    st.markdown('<p class="sec-head">⚡ Risk Levels</p>', unsafe_allow_html=True)
    for risk, color in RISK_HEX.items():
        st.markdown(
            f"<span style='color:{color};font-size:0.84rem'>■ {risk}</span><br>",
            unsafe_allow_html=True
        )

    st.divider()
    if st.button("🗑️ Clear All Data", use_container_width=True):
        for k, v in _D.items():
            st.session_state[k] = (
                [] if isinstance(v, list) else
                {} if isinstance(v, dict) else v
            )
        st.session_state.session_start = time.time()
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
#  Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    '<p class="page-title">🛡️ SENTINEL — FOD Command Centre</p>'
    f'<p class="page-sub">'
    f'NODE: {camera_id.upper()} &nbsp;|&nbsp; '
    f'TOPIC: sentinel/v1/{camera_id}/# &nbsp;|&nbsp; '
    f'SESSION: {datetime.fromtimestamp(st.session_state.session_start).strftime("%H:%M:%S")}'
    f'</p>',
    unsafe_allow_html=True
)
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
#  Static placeholder layout
# ─────────────────────────────────────────────────────────────────────────────
kpi_cols = st.columns(5)
ph_kpi   = [col.empty() for col in kpi_cols]
st.markdown("---")

col_feed, col_dets, col_radar = st.columns([3, 2, 2], gap="medium")
with col_feed:
    st.markdown('<p class="sec-head">📷 Live Camera Feed</p>', unsafe_allow_html=True)
    ph_feed = st.empty()
with col_dets:
    st.markdown('<p class="sec-head">⚠️ Active Detections</p>', unsafe_allow_html=True)
    ph_dets = st.empty()
with col_radar:
    st.markdown('<p class="sec-head">📡 Radar + Motion Sensor</p>', unsafe_allow_html=True)
    ph_radar  = st.empty()
    ph_motion = st.empty()

st.markdown("---")

tab_maint, tab_log, tab_chart = st.tabs(
    ["🔧 Maintenance Priority Queue", "📋 Detection Log", "📊 Analytics"]
)
with tab_maint: ph_maint = st.empty()
with tab_log:   ph_log   = st.empty()
with tab_chart: ph_chart = st.empty()


# ─────────────────────────────────────────────────────────────────────────────
#  Fragment — only this reruns every 500ms
# ─────────────────────────────────────────────────────────────────────────────
@st.fragment(run_every=0.5)
def live_consumer():
    q = get_mqtt(broker, int(port), camera_id)

    # 1. Drain MQTT queue ──────────────────────────────────────────────────
    while not q.empty():
        msg   = q.get_nowait()
        mtype = msg.get("_type")

        if mtype == "status":
            st.session_state.node_status = msg["value"]

        elif mtype == "image":
            st.session_state.latest_b64 = msg["b64"]

        elif mtype == "telemetry":
            ts = datetime.fromtimestamp(
                msg.get("timestamp", time.time())
            ).strftime("%H:%M:%S")
            st.session_state.last_ts = ts
            st.session_state.health  = msg.get("health", {})
            st.session_state.radar   = msg.get("radar",  st.session_state.radar)
            st.session_state.motion  = msg.get("motion", st.session_state.motion)

            for det in msg.get("detections", []):
                if det["confidence"] < conf_thr:
                    continue

                record = {
                    "time":     ts,
                    "class":    det["class"],
                    "conf_pct": round(det["confidence"] * 100, 1),
                    "risk":     det["risk"],
                    "x_m":      det["world_coords"][0],
                    "y_m":      det["world_coords"][1],
                }
                st.session_state.log.insert(0, record)
                st.session_state.total_det += 1

                # Maintenance queue (aggregate by class)
                cls = det["class"]
                mq  = st.session_state.maint_queue
                if cls not in mq:
                    mq[cls] = {"risk": det["risk"], "count": 0,
                               "last_seen": ts, "locations": []}
                mq[cls]["count"]     += 1
                mq[cls]["last_seen"]  = ts
                locs = mq[cls]["locations"]
                locs.append((det["world_coords"][0], det["world_coords"][1]))
                mq[cls]["locations"] = locs[-5:]

            st.session_state.log = st.session_state.log[:MAX_LOG]

    # 2. Derived stats ─────────────────────────────────────────────────────
    log     = st.session_state.log
    active  = sum(1 for r in log[:20] if r["risk"] in ("CRITICAL", "HIGH"))
    crit    = sum(1 for r in log      if r["risk"] == "CRITICAL")
    n_radar = len(st.session_state.radar.get("returns", []))

    # 3. KPIs ──────────────────────────────────────────────────────────────
    for ph, lbl, val in zip(
        ph_kpi,
        ["🔴 Active Threats", "📦 Total Detected", "💥 Critical",
         "📡 Radar Returns", "📝 Log Entries"],
        [active, st.session_state.total_det, crit, n_radar, len(log)]
    ):
        ph.metric(lbl, val)

    # 4. Camera feed ───────────────────────────────────────────────────────
    if st.session_state.latest_b64:
        ph_feed.image(
            base64.b64decode(st.session_state.latest_b64),
            caption="YOLOv10n annotated frame",
            use_container_width=True,
        )
    else:
        ph_feed.markdown(
            "<div style='height:260px;display:flex;align-items:center;"
            "justify-content:center;background:#0d1117;border-radius:8px;"
            "border:1px solid #1f2937;color:#374151;font-family:monospace;"
            "font-size:0.84rem'>⏳ Awaiting camera stream…</div>",
            unsafe_allow_html=True,
        )

    # 5. Detection cards ───────────────────────────────────────────────────
    recent = log[:8]
    if recent:
        html = "<div class='scroll-pane'>"
        for det in recent:
            risk  = det["risk"]
            color = RISK_HEX.get(risk, "#9ca3af")
            bg    = RISK_BG.get(risk, "#111")
            emoji = RISK_EMOJI.get(risk, "⚪")
            html += (
                f'<div class="det-card" style="border-color:{color};background:{bg}">'
                f'<span class="det-class" style="color:{color}">{emoji} {det["class"].replace("_"," ").upper()}</span>'
                f'<span class="risk-tag" style="color:{color};background:{RISK_BG.get(risk,"#222")}">{risk}</span>'
                f'<div class="det-meta">CONF {det["conf_pct"]:.1f}% &nbsp;·&nbsp; '
                f'POS ({det["x_m"]}m, {det["y_m"]}m) &nbsp;·&nbsp; {det["time"]}</div>'
                f'</div>'
            )
        html += "</div>"
        ph_dets.markdown(html, unsafe_allow_html=True)
    else:
        ph_dets.success("✅ No active threats detected")

    # 6. Radar ─────────────────────────────────────────────────────────────
    radar = st.session_state.radar
    svg   = radar_svg(radar.get("sweep_angle", 0), radar.get("returns", []), 240)
    ph_radar.markdown(
        f'<div style="text-align:center">{svg}'
        f'<br><span style="font-family:monospace;font-size:0.70rem;color:#4b5563;">'
        f'RANGE {radar.get("range_m",100)}m &nbsp;·&nbsp; '
        f'RETURNS {len(radar.get("returns",[]))}</span></div>',
        unsafe_allow_html=True,
    )

    # 7. Motion sensor ─────────────────────────────────────────────────────
    motion = st.session_state.motion
    if motion.get("triggered"):
        ph_motion.markdown(
            f'<div style="background:#3b1608;border:1px solid #92400e;'
            f'border-radius:6px;padding:8px 12px;margin-top:8px;'
            f'font-family:monospace;font-size:0.77rem;">'
            f'🚨 <b style="color:#f97316">MOTION DETECTED</b><br>'
            f'<span style="color:#9ca3af">{motion.get("zone","Unknown zone")}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        ph_motion.markdown(
            '<div style="background:#0a1f0a;border:1px solid #14532d;'
            'border-radius:6px;padding:8px 12px;margin-top:8px;'
            'font-family:monospace;font-size:0.77rem;">'
            '✅ <b style="color:#4ade80">MOTION CLEAR</b><br>'
            '<span style="color:#374151">All zones nominal</span></div>',
            unsafe_allow_html=True,
        )

    # 8. Maintenance priority queue ────────────────────────────────────────
    mq = st.session_state.maint_queue
    if mq:
        sorted_items = sorted(
            mq.items(),
            key=lambda x: (RISK_WEIGHT.get(x[1]["risk"], 9), -x[1]["count"])
        )
        html = "<div class='scroll-pane'>"
        for rank, (cls, data) in enumerate(sorted_items, 1):
            risk   = data["risk"]
            color  = RISK_HEX.get(risk, "#9ca3af")
            action = MAINT_ACTION.get(risk, "—")
            locs   = data["locations"]
            avg_x  = round(sum(l[0] for l in locs) / len(locs), 1) if locs else "—"
            avg_y  = round(sum(l[1] for l in locs) / len(locs), 1) if locs else "—"

            html += (
                f'<div class="maint-card" style="border-left:4px solid {color}">'
                f'<div style="display:flex;justify-content:space-between;align-items:center">'
                f'<span style="font-family:\'Barlow Condensed\',sans-serif;font-weight:700;'
                f'font-size:1.0rem;color:{color};text-transform:uppercase;">'
                f'#{rank} &nbsp; {cls.replace("_"," ")}</span>'
                f'<span style="font-family:monospace;font-size:0.70rem;color:#6b7280;">'
                f'{data["count"]}× &nbsp;·&nbsp; {data["last_seen"]}</span>'
                f'</div>'
                f'<div style="font-size:0.82rem;color:#d1d5db;margin-top:4px">{action}</div>'
                f'<div style="font-family:monospace;font-size:0.68rem;color:#4b5563;margin-top:4px">'
                f'Avg pos: ({avg_x}m, {avg_y}m) &nbsp;·&nbsp; '
                f'Risk: <span style="color:{color}">{risk}</span></div>'
                f'</div>'
            )
        html += "</div>"
        ph_maint.markdown(html, unsafe_allow_html=True)
    else:
        ph_maint.info("Maintenance queue empty — no detections yet.")

    # 9. Detection log table ───────────────────────────────────────────────
    if log:
        df = pd.DataFrame(log)
        df.columns = ["Time", "Object", "Conf (%)", "Risk", "X (m)", "Y (m)"]

        def style_risk(val):
            return {
                "CRITICAL": "background:#3b0808;color:#ef4444;font-weight:700",
                "HIGH":     "background:#3b1808;color:#f97316;font-weight:700",
                "MODERATE": "background:#38350a;color:#eab308;font-weight:700",
                "LOW":      "background:#0a3318;color:#22c55e;font-weight:700",
            }.get(val, "")

        ph_log.dataframe(
            df.style.applymap(style_risk, subset=["Risk"]),
            use_container_width=True, height=340
        )
    else:
        ph_log.info("No detections recorded yet.")

    # 10. Analytics ────────────────────────────────────────────────────────
    if log:
        df   = pd.DataFrame(log)
        ca, cb = st.columns(2)

        with ca:
            st.markdown('<p class="sec-head">Detections by Risk Level</p>',
                        unsafe_allow_html=True)
            rc = df["risk"].value_counts().reset_index()
            rc.columns = ["Risk Level", "Count"]
            st.bar_chart(rc.set_index("Risk Level"), height=220,
                         use_container_width=True)

        with cb:
            st.markdown('<p class="sec-head">Top 10 Detected Objects</p>',
                        unsafe_allow_html=True)
            cc = df["class"].value_counts().head(10).reset_index()
            cc.columns = ["Object", "Count"]
            st.bar_chart(cc.set_index("Object"), height=220,
                         use_container_width=True)

        st.markdown('<p class="sec-head">Spatial Scatter — World Coordinates</p>',
                    unsafe_allow_html=True)
        ph_chart.scatter_chart(
            df.rename(columns={"x_m": "Lateral (m)", "y_m": "Distance (m)"}),
            x="Lateral (m)", y="Distance (m)", color="risk",
            height=300, use_container_width=True,
        )
    else:
        ph_chart.info("Analytics populate as detections arrive.")


# ─────────────────────────────────────────────────────────────────────────────
#  Bootstrap
# ─────────────────────────────────────────────────────────────────────────────
get_mqtt(broker, int(port), camera_id)
live_consumer()
