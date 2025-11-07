import GlassPanel from "@/components/GlassPanel";

export default function MetricCard({ label, value }: { label: string; value: string | number }) {
  return (
    <GlassPanel className="p-4">
      <div className="text-sm opacity-70">{label}</div>
      <div className="text-2xl font-semibold mt-1">{value}</div>
    </GlassPanel>
  );
}


