'use client';

import GlassPanel from "@/components/GlassPanel";
import { Area, AreaChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

type TimelinePoint = {
  step: string;
  durationMs: number;
  startedAt?: string;
  finishedAt?: string;
};

type Props = {
  title: string;
  subtitle?: string;
  data: TimelinePoint[];
};

export default function ChartCard({ title, subtitle, data }: Props) {
  const formatted = data.map((item) => ({
    name: labelize(item.step),
    value: item.durationMs,
    tooltip: item,
  }));

  return (
    <GlassPanel className="p-4">
      <div className="flex items-baseline justify-between">
        <div>
          <div className="text-xs uppercase tracking-[0.35em] opacity-70">Pipeline</div>
          <div className="mt-1 text-lg font-semibold text-accent-400">{title}</div>
          {subtitle && <div className="text-xs opacity-60">{subtitle}</div>}
        </div>
      </div>
      <div className="mt-4 h-48">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={formatted} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
            <defs>
              <linearGradient id="timeline" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="var(--dynamic-accent)" stopOpacity={0.8} />
                <stop offset="95%" stopColor="var(--dynamic-accent)" stopOpacity={0.05} />
              </linearGradient>
            </defs>
            <XAxis dataKey="name" stroke="rgba(255,255,255,0.35)" tickLine={false} axisLine={false} fontSize={11} />
            <YAxis
              tickFormatter={(v) => `${(v / 1000).toFixed(1)}s`}
              stroke="rgba(255,255,255,0.35)"
              tickLine={false}
              axisLine={false}
              fontSize={11}
            />
            <Tooltip content={<TimelineTooltip />} />
            <Area type="monotone" dataKey="value" stroke="var(--dynamic-accent)" strokeWidth={2} fill="url(#timeline)" />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </GlassPanel>
  );
}

function TimelineTooltip({ active, payload }: any) {
  if (!active || !payload?.length) return null;
  const item = payload[0].payload?.tooltip as TimelinePoint;
  if (!item) return null;
  return (
    <div className="glass rounded-lg px-3 py-2 text-xs shadow-soft">
      <div className="text-[10px] uppercase tracking-[0.35em] opacity-60">{labelize(item.step)}</div>
      <div className="mt-1 text-sm font-semibold text-accent-400">{(item.durationMs / 1000).toFixed(2)}s</div>
      {item.startedAt && <div className="opacity-60">Start: {formatTime(item.startedAt)}</div>}
      {item.finishedAt && <div className="opacity-60">Ende: {formatTime(item.finishedAt)}</div>}
    </div>
  );
}

function labelize(str: string) {
  return str
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase())
    .trim();
}

function formatTime(value: string) {
  try {
    return new Date(value).toLocaleTimeString();
  } catch {
    return value;
  }
}














