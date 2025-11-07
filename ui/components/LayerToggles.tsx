type Layers = {
  walls: boolean;
  walls_axes: boolean;
  spaces: boolean;
};

export default function LayerToggles({ value, onChange }: { value: Layers; onChange: (v: Layers) => void }) {
  return (
    <div className="grid grid-cols-3 gap-2">
      {([
        ["walls", "Walls"],
        ["walls_axes", "Wall Axes"],
        ["spaces", "Spaces"],
      ] as const).map(([k, label]) => (
        <label key={k} className="flex items-center gap-2 text-sm">
          <input type="checkbox" checked={(value as any)[k]} onChange={(e) => onChange({ ...value, [k]: e.target.checked })} />
          <span>{label}</span>
        </label>
      ))}
    </div>
  );
}


