import React from 'react';
import { Sliders, Scissors, ShieldAlert } from 'lucide-react';

interface TokenLimitSliderProps {
  maxTokens: number;
  onChangeMaxTokens: (tokens: number) => void;
  maxCharacters?: number | null;
  onChangeMaxCharacters: (chars: number | null) => void;
  disabled?: boolean;
}

const PRESET_TOKEN_VALUES = [50, 150, 300, 600, 1200, 3000, 5000];

export const TokenLimitSlider: React.FC<TokenLimitSliderProps> = ({
  maxTokens,
  onChangeMaxTokens,
  maxCharacters,
  onChangeMaxCharacters,
  disabled = false,
}) => {
  const [enableCharLimit, setEnableCharLimit] = React.useState<boolean>(maxCharacters !== null && maxCharacters !== undefined);

  const handleCharLimitToggle = (checked: boolean) => {
    setEnableCharLimit(checked);
    if (!checked) {
      onChangeMaxCharacters(null);
    } else {
      onChangeMaxCharacters(maxCharacters || 500);
    }
  };

  return (
    <div className="space-y-4 rounded-xl border border-slate-800 bg-slate-900/40 p-4">
      {/* Token Generation Limit */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <label htmlFor="token-slider" className="text-sm font-semibold text-slate-200 flex items-center gap-2">
            <Sliders className="w-4 h-4 text-brand-400" />
            Generation Limit (Max Tokens)
          </label>
          <div className="flex items-center gap-2">
            <input
              type="number"
              min={1}
              max={5000}
              value={maxTokens}
              disabled={disabled}
              onChange={(e) => {
                const val = parseInt(e.target.value, 10);
                if (!isNaN(val)) {
                  onChangeMaxTokens(Math.min(5000, Math.max(1, val)));
                }
              }}
              className="w-20 px-2.5 py-1 text-right text-xs font-mono font-semibold rounded-lg bg-slate-800 border border-slate-700 text-brand-300 focus:outline-none focus:border-brand-500"
            />
            <span className="text-xs text-slate-400">tokens</span>
          </div>
        </div>

        {/* Range slider */}
        <input
          id="token-slider"
          type="range"
          min={10}
          max={5000}
          step={10}
          value={maxTokens}
          disabled={disabled}
          onChange={(e) => onChangeMaxTokens(Number(e.target.value))}
          className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-brand-500 disabled:opacity-50"
        />

        {/* Presets and warning */}
        <div className="flex items-center justify-between gap-2 flex-wrap pt-1">
          <div className="flex items-center gap-1.5 flex-wrap">
            <span className="text-[11px] text-slate-500">Presets:</span>
            {PRESET_TOKEN_VALUES.map((preset) => (
              <button
                key={preset}
                type="button"
                disabled={disabled}
                onClick={() => onChangeMaxTokens(preset)}
                className={`text-[11px] px-2 py-0.5 rounded-md font-mono transition-colors ${
                  maxTokens === preset
                    ? 'bg-brand-500/20 text-brand-300 border border-brand-500/40 font-semibold'
                    : 'bg-slate-800/80 hover:bg-slate-700/80 text-slate-400 border border-slate-700/50'
                }`}
              >
                {preset}
              </button>
            ))}
          </div>
          <span className="text-[11px] text-slate-500 font-mono">Max limit: 5,000</span>
        </div>
      </div>

      {/* Optional Character Truncation */}
      <div className="pt-3 border-t border-slate-800/80 space-y-2">
        <div className="flex items-center justify-between">
          <label className="text-xs font-medium text-slate-300 flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={enableCharLimit}
              disabled={disabled}
              onChange={(e) => handleCharLimitToggle(e.target.checked)}
              className="rounded border-slate-700 bg-slate-800 text-brand-500 focus:ring-brand-500/20"
            />
            <Scissors className="w-3.5 h-3.5 text-indigo-400" />
            Enable Character Truncation Limit
          </label>

          {enableCharLimit && (
            <div className="flex items-center gap-1.5">
              <input
                type="number"
                min={10}
                max={25000}
                value={maxCharacters || 500}
                disabled={disabled}
                onChange={(e) => {
                  const val = parseInt(e.target.value, 10);
                  onChangeMaxCharacters(isNaN(val) ? null : Math.max(1, val));
                }}
                className="w-20 px-2 py-0.5 text-right text-xs font-mono rounded bg-slate-800 border border-slate-700 text-indigo-300 focus:outline-none focus:border-indigo-500"
              />
              <span className="text-[11px] text-slate-400">chars</span>
            </div>
          )}
        </div>
        {enableCharLimit && (
          <p className="text-[11px] text-slate-400 pl-5">
            Output will be truncated if the generated continuation exceeds {maxCharacters || 500} characters.
          </p>
        )}
      </div>

      {/* Context window advisory note */}
      <div className="flex items-start gap-2 p-2.5 rounded-lg bg-slate-800/40 border border-slate-800 text-[11px] text-slate-400">
        <ShieldAlert className="w-4 h-4 text-brand-400 shrink-0 mt-0.5" />
        <span>
          <strong>Architecture Note:</strong> Model context window is 128 tokens. Generation autoregressively slides its context across larger limits seamlessly up to your configured token truncation point.
        </span>
      </div>
    </div>
  );
};
