import React from 'react';
import { Layers, CheckCircle2, TrendingDown, Cpu } from 'lucide-react';
import { ModelInfo } from '../../types/generation';

interface ModelSelectorProps {
  models: ModelInfo[];
  selectedModelId: string;
  onSelectModel: (modelId: string) => void;
  disabled?: boolean;
}

export const ModelSelector: React.FC<ModelSelectorProps> = ({
  models,
  selectedModelId,
  onSelectModel,
  disabled = false,
}) => {
  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <label className="text-sm font-semibold text-slate-200 flex items-center gap-2">
          <Layers className="w-4 h-4 text-brand-400" />
          Select Model Checkpoint
        </label>
        <span className="text-xs text-slate-400">2 checkpoints available in registry</span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {models.map((model) => {
          const isSelected = model.model_id === selectedModelId;
          const is100k = model.training_subset === '100k';

          return (
            <div
              key={model.model_id}
              onClick={() => !disabled && onSelectModel(model.model_id)}
              className={`relative cursor-pointer rounded-xl p-4 transition-all duration-200 border text-left ${
                disabled ? 'opacity-60 cursor-not-allowed' : ''
              } ${
                isSelected
                  ? 'border-brand-500 bg-brand-500/10 shadow-lg shadow-brand-500/10 ring-1 ring-brand-500/50'
                  : 'border-slate-800 bg-slate-900/60 hover:border-slate-700 hover:bg-slate-800/40'
              }`}
            >
              <div className="flex items-start justify-between gap-2">
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <span className="font-semibold text-white text-sm">{model.display_name}</span>
                    <span
                      className={`text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full ${
                        is100k
                          ? 'bg-indigo-500/20 text-indigo-300 border border-indigo-500/30'
                          : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                      }`}
                    >
                      {model.training_subset.toUpperCase()} SUBSET
                    </span>
                  </div>
                  <p className="text-xs text-slate-400 leading-relaxed line-clamp-2">
                    {model.description}
                  </p>
                </div>

                <div className="shrink-0 mt-0.5">
                  <div
                    className={`w-5 h-5 rounded-full flex items-center justify-center border transition-colors ${
                      isSelected
                        ? 'border-brand-500 bg-brand-500 text-white'
                        : 'border-slate-700 bg-slate-800'
                    }`}
                  >
                    {isSelected && <CheckCircle2 className="w-3.5 h-3.5" />}
                  </div>
                </div>
              </div>

              {/* Metrics bar */}
              <div className="mt-3 pt-3 border-t border-slate-800/80 grid grid-cols-3 gap-2 text-[11px]">
                <div>
                  <span className="text-slate-500 block">Val Loss</span>
                  <span className="font-mono font-medium text-slate-200">
                    {model.validation_loss ? model.validation_loss.toFixed(4) : 'N/A'}
                  </span>
                </div>
                <div>
                  <span className="text-slate-500 block">Perplexity</span>
                  <span className="font-mono font-medium text-slate-200">
                    {model.validation_perplexity ? model.validation_perplexity.toFixed(3) : 'N/A'}
                  </span>
                </div>
                <div>
                  <span className="text-slate-500 block">Batch Size</span>
                  <span className="font-mono font-medium text-slate-200">
                    {model.batch_size ? model.batch_size.toLocaleString() : 'N/A'}
                  </span>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
