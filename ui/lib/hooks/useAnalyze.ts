import { useState } from "react";
import { analyzeImage, AnalyzeOptions, AnalyzeResponse } from "@/lib/api";
import { triggerHaptic } from "@/lib/haptics";

export function useAnalyze() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);

  async function run(file: File, options: AnalyzeOptions): Promise<AnalyzeResponse> {
    setLoading(true);
    setError(null);
    try {
      const res = await analyzeImage(file, options);
      setResult(res);
      triggerHaptic("success");
      return res;
    } catch (err: any) {
      const message = err?.message ? String(err.message) : "Analyse fehlgeschlagen";
      setError(message);
      triggerHaptic("error");
      throw err;
    } finally {
      setLoading(false);
    }
  }

  return { run, loading, error, result, reset: () => setResult(null) };
}
