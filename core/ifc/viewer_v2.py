"""HTML viewer for cleaned predictions (V2 post-processing output)."""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import Any
import json


def generate_predictions_viewer_html(
    predictions: list[dict[str, Any]],
    output_path: Path | None = None,
    title: str = "Cleaned Predictions Viewer (V2)",
    *,
    return_string: bool = False,
) -> str | None:
    """
    Generate interactive HTML viewer for cleaned predictions.
    
    Args:
        predictions: List of prediction dicts with 'points', 'class', 'confidence'
        output_path: Path to save HTML file (if None and return_string=False, does nothing)
        title: Page title
        return_string: If True, return HTML as string instead of writing to file
        
    Returns:
        HTML string if return_string=True, None otherwise
    """
    # Serialize predictions to JSON for JavaScript
    predictions_json = json.dumps(predictions, indent=2)
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
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
            max-width: 250px;
        }}
        #controls h3 {{
            margin-top: 0;
            margin-bottom: 10px;
        }}
        #controls label {{
            display: block;
            margin: 8px 0;
            cursor: pointer;
        }}
        #controls input[type="checkbox"] {{
            margin-right: 8px;
        }}
        #stats {{
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid #444;
            font-size: 11px;
        }}
        #stats div {{
            margin: 4px 0;
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
        <h3>Layers</h3>
        <label><input type="checkbox" id="showWalls" checked> Walls</label>
        <label><input type="checkbox" id="showDoors" checked> Doors</label>
        <label><input type="checkbox" id="showWindows" checked> Windows</label>
        <label><input type="checkbox" id="showLabels" checked> Labels</label>
        <label><input type="checkbox" id="showConfidence" checked> Confidence</label>
        <div id="stats"></div>
    </div>
    <canvas id="canvas"></canvas>
    <div id="info">Hover over elements to see details</div>

    <script>
        const predictionsData = {predictions_json};
        
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        
        let scale = 1.0;
        let offsetX = 0;
        let offsetY = 0;
        let isDragging = false;
        let lastX = 0;
        let lastY = 0;
        let bounds = {{minX: 0, minY: 0, maxX: 1000, maxY: 1000}};
        
        // Calculate bounds from predictions
        function calculateBounds() {{
            let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
            predictionsData.forEach(pred => {{
                const points = pred.points || [];
                points.forEach(pt => {{
                    if (Array.isArray(pt) && pt.length >= 2) {{
                        minX = Math.min(minX, pt[0]);
                        minY = Math.min(minY, pt[1]);
                        maxX = Math.max(maxX, pt[0]);
                        maxY = Math.max(maxY, pt[1]);
                    }}
                }});
            }});
            if (isFinite(minX) && isFinite(minY) && isFinite(maxX) && isFinite(maxY)) {{
                bounds = {{minX, minY, maxX, maxY}};
                // Add padding
                const padding = Math.max((maxX - minX), (maxY - minY)) * 0.1;
                bounds.minX -= padding;
                bounds.minY -= padding;
                bounds.maxX += padding;
                bounds.maxY += padding;
            }}
        }}
        
        calculateBounds();
        
        // Color mapping
        function getColorForClass(className) {{
            const classLower = (className || '').toLowerCase();
            if (classLower.includes('wall')) {{
                return classLower.includes('external') ? '#ff6b6b' : '#4ecdc4';
            }}
            if (classLower.includes('door')) return '#ffd93d';
            if (classLower.includes('window')) return '#6bcf7f';
            return '#60a5fa';
        }}
        
        // Count by class
        function countByClass() {{
            const counts = {{}};
            predictionsData.forEach(pred => {{
                const cls = pred.class || pred.label || 'unknown';
                counts[cls] = (counts[cls] || 0) + 1;
            }});
            return counts;
        }}
        
        // Update stats
        function updateStats() {{
            const counts = countByClass();
            const statsDiv = document.getElementById('stats');
            statsDiv.innerHTML = '<div><strong>Statistics:</strong></div>' +
                Object.entries(counts).map(([cls, count]) => 
                    `<div>${{cls}}: ${{count}}</div>`
                ).join('') +
                `<div style="margin-top: 8px;"><strong>Total: ${{predictionsData.length}}</strong></div>`;
        }}
        
        updateStats();
        
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
        
        // Convert world coordinates to canvas
        function worldToCanvas(x, y) {{
            const width = bounds.maxX - bounds.minX;
            const height = bounds.maxY - bounds.minY;
            const scaleX = canvas.width / width;
            const scaleY = canvas.height / height;
            const useScale = Math.min(scaleX, scaleY) * scale;
            
            const centerX = canvas.width / 2;
            const centerY = canvas.height / 2;
            const worldCenterX = (bounds.minX + bounds.maxX) / 2;
            const worldCenterY = (bounds.minY + bounds.maxY) / 2;
            
            return {{
                x: centerX + (x - worldCenterX) * useScale + offsetX,
                y: centerY + (y - worldCenterY) * useScale + offsetY
            }};
        }}
        
        function draw() {{
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            const showWalls = document.getElementById('showWalls').checked;
            const showDoors = document.getElementById('showDoors').checked;
            const showWindows = document.getElementById('showWindows').checked;
            const showLabels = document.getElementById('showLabels').checked;
            const showConfidence = document.getElementById('showConfidence').checked;
            
            predictionsData.forEach((pred, idx) => {{
                const className = pred.class || pred.label || 'unknown';
                const classLower = className.toLowerCase();
                const points = pred.points || [];
                const confidence = pred.confidence || 0;
                
                // Filter by type
                if (classLower.includes('wall') && !showWalls) return;
                if (classLower.includes('door') && !showDoors) return;
                if (classLower.includes('window') && !showWindows) return;
                
                if (points.length < 2) return;
                
                const color = getColorForClass(className);
                const alpha = Math.max(0.3, confidence);
                
                // Draw polygon
                ctx.beginPath();
                const firstPt = worldToCanvas(points[0][0], points[0][1]);
                ctx.moveTo(firstPt.x, firstPt.y);
                
                for (let i = 1; i < points.length; i++) {{
                    const pt = worldToCanvas(points[i][0], points[i][1]);
                    ctx.lineTo(pt.x, pt.y);
                }}
                ctx.closePath();
                
                ctx.fillStyle = color + Math.floor(alpha * 51).toString(16).padStart(2, '0');
                ctx.fill();
                ctx.strokeStyle = color;
                ctx.lineWidth = 2;
                ctx.stroke();
                
                // Draw label and confidence
                if (showLabels || showConfidence) {{
                    // Calculate center
                    let sumX = 0, sumY = 0;
                    points.forEach(pt => {{
                        sumX += pt[0];
                        sumY += pt[1];
                    }});
                    const centerX = sumX / points.length;
                    const centerY = sumY / points.length;
                    const center = worldToCanvas(centerX, centerY);
                    
                    ctx.fillStyle = '#fff';
                    ctx.strokeStyle = '#000';
                    ctx.lineWidth = 3;
                    ctx.font = '12px Arial';
                    
                    let text = '';
                    if (showLabels) text += className;
                    if (showConfidence && confidence > 0) {{
                        if (text) text += ' ';
                        text += '(' + (confidence * 100).toFixed(0) + '%)';
                    }}
                    
                    if (text) {{
                        ctx.strokeText(text, center.x, center.y);
                        ctx.fillText(text, center.x, center.y);
                    }}
                }}
            }});
        }}
        
        // Update on checkbox change
        ['showWalls', 'showDoors', 'showWindows', 'showLabels', 'showConfidence'].forEach(id => {{
            document.getElementById(id).addEventListener('change', draw);
        }});
        
        // Initial draw
        draw();
    </script>
</body>
</html>"""
    
    if return_string:
        return html_content
    
    if output_path is not None:
        output_path.write_text(html_content, encoding='utf-8')
    
    return None

