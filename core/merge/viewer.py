"""HTML viewer for merged canonical JSON."""

from pathlib import Path
from typing import List
import json

from .schema import CanonicalPlan


def generate_viewer_html(plan: CanonicalPlan, output_path: Path) -> None:
    """Generate interactive HTML viewer for merged plan."""
    
    # Serialize plan to JSON for JavaScript
    plan_json = plan.model_dump_json(indent=2)
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Merged Plan Viewer</title>
    <style>
        body {{
            margin: 0;
            font-family: Arial, sans-serif;
            background: #1a1a1a;
            color: #fff;
        }}
        #controls {{
            position: fixed;
            top: 10px;
            left: 10px;
            background: rgba(0,0,0,0.8);
            padding: 15px;
            border-radius: 8px;
            z-index: 1000;
        }}
        #controls label {{
            display: block;
            margin: 8px 0;
            cursor: pointer;
        }}
        #controls input[type="checkbox"] {{
            margin-right: 8px;
        }}
        #canvas {{
            display: block;
            background: #2a2a2a;
        }}
        #info {{
            position: fixed;
            bottom: 10px;
            left: 10px;
            background: rgba(0,0,0,0.8);
            padding: 10px;
            border-radius: 8px;
            font-size: 12px;
            max-width: 300px;
        }}
    </style>
</head>
<body>
    <div id="controls">
        <h3 style="margin-top:0;">Layers</h3>
        <label><input type="checkbox" id="showWalls" checked> Walls</label>
        <label><input type="checkbox" id="showOpenings" checked> Doors/Windows</label>
        <label><input type="checkbox" id="showRooms" checked> Rooms</label>
        <label><input type="checkbox" id="showLabels" checked> Labels</label>
    </div>
    <canvas id="canvas"></canvas>
    <div id="info">Hover over openings to see sill/head metadata</div>

    <script>
        const planData = {plan_json};
        
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        
        let scale = 1.0;
        let offsetX = 0;
        let offsetY = 0;
        let isDragging = false;
        let lastX = 0;
        let lastY = 0;
        let hoverOpening = null;
        
        // Setup canvas
        function resizeCanvas() {{
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
            draw();
        }}
        
        window.addEventListener('resize', resizeCanvas);
        resizeCanvas();
        
        // Mouse controls
        canvas.addEventListener('mousedown', (e) => {{
            isDragging = true;
            lastX = e.clientX;
            lastY = e.clientY;
        }});
        
        canvas.addEventListener('mousemove', (e) => {{
            if (isDragging) {{
                offsetX += e.clientX - lastX;
                offsetY += e.clientY - lastY;
                lastX = e.clientX;
                lastY = e.clientY;
                draw();
            }} else {{
                hoverOpening = pickOpening(e.clientX, e.clientY);
                updateInfo();
            }}
        }});
        
        canvas.addEventListener('mouseup', () => {{
            isDragging = false;
        }});
        
        canvas.addEventListener('wheel', (e) => {{
            e.preventDefault();
            const delta = e.deltaY > 0 ? 0.9 : 1.1;
            scale *= delta;
            scale = Math.max(0.1, Math.min(10, scale));
            draw();
        }});
        
        // Convert meters to pixels (assuming 1m = 100px at scale 1)
        function mToPx(m) {{
            return m * 100 * scale;
        }}
        
        function pickOpening(screenX, screenY) {{
            const rect = canvas.getBoundingClientRect();
            const x = (screenX - rect.left - offsetX) / (100 * scale);
            const y = (screenY - rect.top - offsetY) / (100 * scale);

            for (const opening of planData.openings) {{
                const wall = planData.walls.find(w => w.id === opening.hostWallId);
                if (!wall || wall.polyline.length < 2) continue;

                const start = wall.polyline[0];
                const end = wall.polyline[wall.polyline.length - 1];
                const dx = end.x - start.x;
                const dy = end.y - start.y;
                const length = Math.sqrt(dx * dx + dy * dy);
                if (length === 0) continue;

                const px = start.x + opening.s * dx;
                const py = start.y + opening.s * dy;
                const sill = opening.sillHeight ?? 0;
                const overall = opening.overallHeight ?? opening.height ?? 0;
                const halfWidth = (opening.width ?? 0) / 2;
                const perpX = -dy / length;
                const perpY = dx / length;

                const p1x = px + perpX * halfWidth;
                const p1y = py + perpY * halfWidth;
                const p2x = px - perpX * halfWidth;
                const p2y = py - perpY * halfWidth;

                // Simple distance check to segment
                const t = ((x - p1x) * (p2x - p1x) + (y - p1y) * (p2y - p1y)) / ((p2x - p1x) ** 2 + (p2y - p1y) ** 2);
                if (t < 0 || t > 1) continue;
                const projX = p1x + t * (p2x - p1x);
                const projY = p1y + t * (p2y - p1y);
                const dist = Math.sqrt((x - projX) ** 2 + (y - projY) ** 2);
                if (dist < 0.1) {{
                    return {{ opening, sill, overall }};
                }}
            }}
            return null;
        }}

        function updateInfo() {{
            const info = document.getElementById('info');
            if (!hoverOpening) {{
                info.textContent = 'Hover over openings to see sill/head metadata';
                return;
            }}
            const {{ opening, sill, overall }} = hoverOpening;
            const head = opening.headHeight ?? (sill + overall);
            info.textContent = (
                opening.type.toUpperCase() +
                ' width=' + (opening.width ?? 0).toFixed(3) + 'm' +
                ' | sill=' + sill.toFixed(3) + 'm' +
                ' | head=' + head.toFixed(3) + 'm' +
                ' | height=' + (opening.overallHeight ?? opening.height ?? 0).toFixed(3) + 'm'
            );
        }}

        function draw() {{
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            const showWalls = document.getElementById('showWalls').checked;
            const showOpenings = document.getElementById('showOpenings').checked;
            const showRooms = document.getElementById('showRooms').checked;
            const showLabels = document.getElementById('showLabels').checked;
            
            ctx.save();
            ctx.translate(offsetX, offsetY);
            
            // Draw rooms
            if (showRooms) {{
                planData.rooms.forEach(room => {{
                    ctx.beginPath();
                    ctx.fillStyle = 'rgba(100, 150, 255, 0.2)';
                    ctx.strokeStyle = 'rgba(100, 150, 255, 0.5)';
                    ctx.lineWidth = 1;
                    
                    if (room.polygon.length > 0) {{
                        const first = room.polygon[0];
                        ctx.moveTo(mToPx(first.x), mToPx(first.y));
                        for (let i = 1; i < room.polygon.length; i++) {{
                            ctx.lineTo(mToPx(room.polygon[i].x), mToPx(room.polygon[i].y));
                        }}
                        ctx.closePath();
                        ctx.fill();
                        ctx.stroke();
                    }}
                    
                    if (showLabels && room.polygon.length > 0) {{
                        const center = room.polygon.reduce((acc, p) => ({{x: acc.x + p.x, y: acc.y + p.y}}), {{x: 0, y: 0}});
                        center.x /= room.polygon.length;
                        center.y /= room.polygon.length;
                        
                        ctx.fillStyle = '#fff';
                        ctx.font = '12px Arial';
                        ctx.fillText(room.name || room.id, mToPx(center.x), mToPx(center.y));
                    }}
                }});
            }}
            
            // Draw walls
            if (showWalls) {{
                planData.walls.forEach(wall => {{
                    if (wall.polyline.length < 2) return;
                    
                    ctx.beginPath();
                    ctx.strokeStyle = wall.isExternal ? '#ff6b6b' : '#4ecdc4';
                    ctx.lineWidth = Math.max(2, mToPx(wall.thickness));
                    
                    const first = wall.polyline[0];
                    ctx.moveTo(mToPx(first.x), mToPx(first.y));
                    for (let i = 1; i < wall.polyline.length; i++) {{
                        ctx.lineTo(mToPx(wall.polyline[i].x), mToPx(wall.polyline[i].y));
                    }}
                    ctx.stroke();
                    
                    if (showLabels && wall.polyline.length > 0) {{
                        const mid = wall.polyline[Math.floor(wall.polyline.length / 2)];
                        ctx.fillStyle = '#fff';
                        ctx.font = '10px Arial';
                        ctx.fillText(wall.id, mToPx(mid.x), mToPx(mid.y));
                    }}
                }});
            }}
            
            // Draw openings
            if (showOpenings) {{
                planData.openings.forEach(opening => {{
                    // Find host wall
                    const wall = planData.walls.find(w => w.id === opening.hostWallId);
                    if (!wall || wall.polyline.length < 2) return;
                    
                    const start = wall.polyline[0];
                    const end = wall.polyline[wall.polyline.length - 1];
                    const dx = end.x - start.x;
                    const dy = end.y - start.y;
                    const length = Math.sqrt(dx * dx + dy * dy);
                    
                    const s = opening.s;
                    const px = start.x + s * dx;
                    const py = start.y + s * dy;
                    
                    // Perpendicular offset
                    const perpX = -dy / length;
                    const perpY = dx / length;
                    
                    const halfWidth = opening.width / 2;
                    const p1x = px + perpX * halfWidth;
                    const p1y = py + perpY * halfWidth;
                    const p2x = px - perpX * halfWidth;
                    const p2y = py - perpY * halfWidth;
                    
                    ctx.beginPath();
                    ctx.strokeStyle = opening.type === 'door' ? '#ffd93d' : '#6bcf7f';
                    ctx.lineWidth = 3;
                    ctx.moveTo(mToPx(p1x), mToPx(p1y));
                    ctx.lineTo(mToPx(p2x), mToPx(p2y));
                    ctx.stroke();
                    
                    if (showLabels) {{
                        ctx.fillStyle = '#fff';
                        ctx.font = '10px Arial';
                        const sill = opening.sillHeight ?? 0;
                        const head = opening.headHeight ?? (sill + (opening.overallHeight ?? opening.height ?? 0));
                        const sillText = opening.sillHeight == null ? 'n/a' : sill.toFixed(2);
                        const headText = opening.headHeight == null ? 'n/a' : head.toFixed(2);
                        const label = opening.type + ' (' + sillText + 'm/' + headText + 'm)';
                        ctx.fillText(label, mToPx(px), mToPx(py));
                    }}
                }});
            }}
            
            ctx.restore();
        }}
        
        // Update on checkbox change
        ['showWalls', 'showOpenings', 'showRooms', 'showLabels'].forEach(id => {{
            document.getElementById(id).addEventListener('change', draw);
        }});
        
        // Initial draw
        draw();
    </script>
</body>
</html>"""
    
    output_path.write_text(html_content, encoding='utf-8')

