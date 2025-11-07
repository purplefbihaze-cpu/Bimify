import json, httpx
payload = {
   predictions: [
    {
      id: test-1,
      class: WALL,
      confidence: 0.99,
      points: [[0,0],[3000,0],[3000,200],[0,200]],
      raw: {class: WALL}
    }
  ],
  image: None,
  storey_height_mm: 3000,
  door_height_mm: 2100,
  window_height_mm: 1000,
  window_head_elevation_mm: 2000,
  px_per_mm: 1.0,
  project_name: Test,
  storey_name: EG,
  calibration: None
}
resp = httpx.post(http://localhost:8000/v1/export-ifc, json=payload, timeout=180.0)
print(resp.status_code)
print(resp.text[:500])
