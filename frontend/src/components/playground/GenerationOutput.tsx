import React, { useState } from 'react';
import { Copy, Check, Sparkles, Clock, Hash, Scissors, CornerDownRight, Trash2 } from 'lucide-react';
import { GenerationResponse } from '../../types/generation';

interface GenerationOutputProps {
  result: GenerationResponse | null;
  isGenerating: boolean;
  onClear: () => void;
}

export const GenerationOutput: React.FC<GenerationOutputProps> = ({
  result,
  isGenerating,
  onClear,
}) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    if (!result) return;
    try {
      await navigator.clipboard.writeText(result.full_text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Fallback
    }
  };

  if (isGenerating) {
    return (
      <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-8 text-center space-y-4">
        <div className="w-12 h-12 rounded-2xl bg-brand-500/10 border border-brand-500/20 text-brand-400 flex items-center justify-center mx-auto animate-pulse">
          <Sparkles className="w-6 h-6 animate-spin" />
        </div>
        <div className="space-y-1">
          <h3 className="text-sm font-semibold text-slate-200">Autoregressive Sampling in Progress</h3>
          <p className="text-xs text-slate-400">Evaluating transformer logits token-by-token...</p>
        </div>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="rounded-2xl border border-dashed border-slate-800 bg-slate-900/30 p-12 text-center space-y-3">
        <div className="w-12 h-12 rounded-2xl bg-slate-800/80 border border-slate-700/60 text-slate-500 flex items-center justify-center mx-auto">
          <Sparkles className="w-6 h-6" />
        </div>
        <div className="space-y-1">
          <h3 className="text-sm font-semibold text-slate-300">Awaiting Your Prompt</h3>
          <p className="text-xs text-slate-500 max-w-sm mx-auto">
            Choose either the 25k or 100k model, enter an initial sentence with 5+ words, set your token limit, and click Generate.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-2xl border border-slate-800 bg-slate-900/80 shadow-xl overflow-hidden animate-fadeIn">
      {/* Output Header with Badges */}
      <div className="px-5 py-3.5 border-b border-slate-800 bg-slate-900/90 flex items-center justify-between gap-3 flex-wrap">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-xs font-semibold text-slate-200 flex items-center gap-1.5">
            <Sparkles className="w-3.5 h-3.5 text-brand-400" />
            Generated Story
          </span>

          <span className="px-2 py-0.5 rounded-full text-[10px] font-mono font-bold uppercase tracking-wider bg-indigo-500/10 text-indigo-300 border border-indigo-500/20">
            {result.model_id}
          </span>

          {result.truncated ? (
            <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-mono font-medium bg-amber-500/10 text-amber-300 border border-amber-500/30">
              <Scissors className="w-3 h-3 text-amber-400" /> Truncated at Limit
            </span>
          ) : (
            <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-mono font-medium bg-emerald-500/10 text-emerald-300 border border-emerald-500/30">
              <Check className="w-3 h-3 text-emerald-400" /> Finished at &lt;eos&gt;
            </span>
          )}
        </div>

        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={handleCopy}
            className="inline-flex items-center gap-1.5 px-3 py-1 text-xs font-medium rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 hover:text-white border border-slate-700 transition-colors"
          >
            {copied ? (
              <>
                <Check className="w-3.5 h-3.5 text-emerald-400" />
                <span>Copied!</span>
              </>
            ) : (
              <>
                <Copy className="w-3.5 h-3.5 text-slate-400" />
                <span>Copy Full Text</span>
              </>
            )}
          </button>

          <button
            type="button"
            onClick={onClear}
            className="p-1 text-slate-400 hover:text-slate-200 rounded-lg hover:bg-slate-800 transition-colors"
            title="Clear output"
          >
            <Trash2 className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Output Content */}
      <div className="p-6 text-sm sm:text-base leading-relaxed font-sans text-slate-200 space-y-4">
        <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800/80">
          <span className="font-semibold text-brand-300 bg-brand-500/10 px-1.5 py-0.5 rounded mr-1">
            {result.prompt}
          </span>
          <span className="text-slate-100">{result.completion}</span>
        </div>
      </div>

      {/* Output Metrics Footer */}
      <div className="px-5 py-3 border-t border-slate-800/80 bg-slate-950/40 grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs text-slate-400">
        <div className="flex items-center gap-2">
          <CornerDownRight className="w-3.5 h-3.5 text-brand-400 shrink-0" />
          <span>Prompt: <strong className="font-mono text-slate-200">{result.prompt_tokens}</strong> tokens</span>
        </div>
        <div className="flex items-center gap-2">
          <Hash className="w-3.5 h-3.5 text-indigo-400 shrink-0" />
          <span>Output: <strong className="font-mono text-slate-200">{result.completion_tokens}</strong> tokens</span>
        </div>
        <div className="flex items-center gap-2">
          <Scissors className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
          <span>Length: <strong className="font-mono text-slate-200">{result.character_count}</strong> chars</span>
        </div>
        <div className="flex items-center gap-2">
          <Clock className="w-3.5 h-3.5 text-amber-400 shrink-0" />
          <span>Latency: <strong className="font-mono text-slate-200">{result.inference_duration_ms}</strong> ms</span>
        </div>
      </div>
    </div>
  );
};
