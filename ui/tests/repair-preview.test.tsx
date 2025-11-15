import { describe, it, expect, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import RepairPreview from "../components/RepairPreview";

vi.mock("../lib/api", async () => {
  const actual = await vi.importActual<Record<string, any>>("../lib/api");
  return {
    ...actual,
    repairIfcPreview: vi.fn(async () => ({
      preview_id: "test-preview",
      level: 1,
      overlay_url: "/__test__/overlay.geojson",
      metrics: { total_walls_src: 2, total_axes: 2, median_iou: 0.9 },
    })),
    repairIfcCommit: vi.fn(async () => ({
      file_name: "committed.ifc",
      ifc_url: "/files/committed.ifc",
      level: 1,
      topview_url: "/files/committed_topview.geojson",
    })),
  };
});

describe("RepairPreview", () => {
  it("renders and shows apply button", async () => {
    // Mock fetch for overlay
    global.fetch = vi.fn(async (url: RequestInfo) => {
      return new Response(
        JSON.stringify({
          type: "FeatureCollection",
          features: [],
        }),
        { status: 200 }
      ) as unknown as Response;
    }) as unknown as typeof fetch;

    const onApplied = vi.fn();
    render(<RepairPreview fileName="sample.ifc" onApplied={onApplied} />);

    expect(await screen.findByText(/Level 1 Vorschau/i)).toBeInTheDocument();

    const apply = await screen.findByRole("button", { name: /Übernehmen/i });
    expect(apply).toBeInTheDocument();

    // Do not click in unit test; integration covered in backend tests
  });
});


