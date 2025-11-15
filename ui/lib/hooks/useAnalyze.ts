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
      let message = err?.message ? String(err.message) : "Analyse fehlgeschlagen";
      
      // Parse and improve Roboflow 403 error messages
      if (message.includes("403") || message.includes("Forbidden")) {
        // Extract the helpful part if it's a detailed error
        if (message.includes("Roboflow API-Zugriff verweigert")) {
          // Keep the improved backend message
        } else if (message.includes("Roboflow inference failed")) {
          message = "Roboflow API-Zugriff verweigert (403). Bitte überprüfe deinen API-Key in den Einstellungen und stelle sicher, dass er Zugriff auf das Projekt hat.";
        }
      }
      
      setError(message);
      triggerHaptic("error");
      throw err;
    } finally {
      setLoading(false);
    }
  }

  return { run, loading, error, result, reset: () => setResult(null) };
}
