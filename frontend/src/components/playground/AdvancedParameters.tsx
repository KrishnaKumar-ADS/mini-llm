import React, { useState } from 'react';
import { ChevronDown, ChevronUp, Settings2 } from 'lucide-react';

interface AdvancedParametersProps {
  temperature: number;
  onChangeTemperature: (val: number) => void;
  topK: number;
  onChangeTopK: (val: number) => void;
  topP: number;
  onChangeTopP: (val: number) => void;
  seed: number | null;
  onChangeSeed: (val: number | null) => void;
  stopAtEos: boolean;
  onChangeStopAtEos: (val: boolean) => void;
  disabled?: boolean;
}

export const AdvancedParameters: React.FC<AdvancedParametersProps> = ({
  temperature,
  onChangeTemperature,
  topK,
  onChangeTopK,
  topP,
  onChangeTopP,
  seed,
  onChangeSeed,
  stopAtEos,
  onChangeStopAtEos,
  disabled = false,
}) => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/30 overflow-hidden">
      <button
        type="button"
        disabled={disabled}
        onClick={() => setIsOpen(!isOpen)}
        className="w-full px-4 py-3 flex items-center justify-between text-xs font-semibold text-slate-300 hover:text-white hover:bg-slate-800/40 transition-colors"
      >
        <span className="flex items-center gap-2">
          <Settings2 className="w-4 h-4 text-slate-400" />
          Advanced Sampling Hyperparameters
        </span>
        {isOpen ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
      </button>

      {isOpen && (
        <div className="p-4 border-t border-slate-800/80 space-y-4 text-xs">
          {/* Temperature */}
          <div className="space-y-1">
            <div className="flex justify-between items-center">
              <span className="text-slate-300">Temperature: {temperature.toFixed(2)}</span>
              <span className="text-slate-500 text-[11px]">(0.0 = greedy, 1.0 = creative)</span>
            </div>
            <input
              type="range"
              min={0.0}
              max={1.5}
              step={0.05}
              value={temperature}
              disabled={disabled}
              onChange={(e) => onChangeTemperature(parseFloat(e.target.value))}
              className="w-full h-1.5 bg-slate-800 rounded appearance-none cursor-pointer accent-brand-500"
            />
          </div>

          {/* Top-K and Top-P in 2 cols */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div className="space-y-1">
              <div className="flex justify-between items-center">
                <span className="text-slate-300">Top-k: {topK}</span>
                <span className="text-slate-500 text-[11px]">(vocab candidate pool)</span>
              </div>
              <input
                type="range"
                min={1}
                max={100}
                step={1}
                value={topK}
                disabled={disabled}
                onChange={(e) => onChangeTopK(parseInt(e.target.value, 10))}
                className="w-full h-1.5 bg-slate-800 rounded appearance-none cursor-pointer accent-brand-500"
              />
            </div>

            <div className="space-y-1">
              <div className="flex justify-between items-center">
                <span className="text-slate-300">Top-p (Nucleus): {topP.toFixed(2)}</span>
                <span className="text-slate-500 text-[11px]">(cumulative threshold)</span>
              </div>
              <input
                type="range"
                min={0.1}
                max={1.0}
                step={0.05}
                value={topP}
                disabled={disabled}
                onChange={(e) => onChangeTopP(parseFloat(e.target.value))}
                className="w-full h-1.5 bg-slate-800 rounded appearance-none cursor-pointer accent-brand-500"
              />
            </div>
          </div>

          {/* Seed and Stop-at-EOS */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-1">
            <div className="flex items-center justify-between">
              <label className="text-slate-300">Random Seed:</label>
              <input
                type="number"
                placeholder="e.g. 42 (optional)"
                value={seed ?? ''}
                disabled={disabled}
                onChange={(e) => {
                  const val = parseInt(e.target.value, 10);
                  onChangeSeed(isNaN(val) ? null : val);
                }}
                className="w-32 px-2 py-1 text-xs rounded bg-slate-800 border border-slate-700 text-slate-200"
              />
            </div>

            <div className="flex items-center justify-between">
              <label className="text-slate-300 cursor-pointer flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={stopAtEos}
                  disabled={disabled}
                  onChange={(e) => onChangeStopAtEos(e.target.checked)}
                  className="rounded border-slate-700 bg-slate-800 text-brand-500"
                />
                Stop at &lt;|eos|&gt; token
              </label>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
