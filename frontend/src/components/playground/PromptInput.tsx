import React from 'react';
import { Edit3, CheckCircle2, AlertCircle, Sparkles } from 'lucide-react';
import { PromptValidationResult } from '../../utils/validation';

interface PromptInputProps {
  prompt: string;
  onChangePrompt: (newPrompt: string) => void;
  validation: PromptValidationResult;
  disabled?: boolean;
}

const SAMPLE_PROMPTS = [
  'Once upon a time in a quiet little forest',
  'The curious puppy wandered into the colorful vegetable garden',
  'A brave little girl found an old mysterious treasure box',
  'The sunny morning was perfect for a big adventure outside',
];

export const PromptInput: React.FC<PromptInputProps> = ({
  prompt,
  onChangePrompt,
  validation,
  disabled = false,
}) => {
  const { isValid, wordCount, errorMessage } = validation;

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <label htmlFor="prompt-input" className="text-sm font-semibold text-slate-200 flex items-center gap-2">
          <Edit3 className="w-4 h-4 text-brand-400" />
          Initial Seed Prompt
          <span className="text-xs font-normal text-slate-400">(min 5 words)</span>
        </label>

        {/* Word count pill */}
        <div className="flex items-center gap-2">
          <span
            className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-mono font-medium border ${
              wordCount === 0
                ? 'bg-slate-800 text-slate-400 border-slate-700'
                : isValid
                ? 'bg-emerald-500/10 text-emerald-300 border-emerald-500/30'
                : 'bg-amber-500/10 text-amber-300 border-amber-500/30'
            }`}
          >
            {isValid ? (
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
            ) : (
              <AlertCircle className="w-3.5 h-3.5 text-amber-400" />
            )}
            {wordCount} {wordCount === 1 ? 'word' : 'words'}
          </span>
        </div>
      </div>

      <div className="relative">
        <textarea
          id="prompt-input"
          rows={3}
          value={prompt}
          onChange={(e) => onChangePrompt(e.target.value)}
          disabled={disabled}
          placeholder="Enter at least 5 to 7 words to start the story (e.g. 'Once upon a time in a sunny meadow...')"
          className={`w-full rounded-xl bg-slate-900/80 border px-4 py-3 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 transition-all duration-200 resize-none font-sans leading-relaxed ${
            disabled ? 'opacity-60 cursor-not-allowed' : ''
          } ${
            prompt.length > 0 && !isValid
              ? 'border-amber-500/50 focus:border-amber-500 focus:ring-amber-500/20'
              : 'border-slate-800 focus:border-brand-500 focus:ring-brand-500/20'
          }`}
        />
      </div>

      {/* Validation warning */}
      {prompt.length > 0 && errorMessage && (
        <p className="text-xs text-amber-400/90 flex items-center gap-1.5 animate-fadeIn">
          <AlertCircle className="w-3.5 h-3.5 shrink-0" />
          {errorMessage}
        </p>
      )}

      {/* Quick starter suggestions */}
      <div className="flex items-center gap-1.5 flex-wrap pt-1">
        <span className="text-[11px] text-slate-500 flex items-center gap-1">
          <Sparkles className="w-3 h-3 text-slate-400" /> Suggestions:
        </span>
        {SAMPLE_PROMPTS.map((sample, idx) => (
          <button
            key={idx}
            type="button"
            disabled={disabled}
            onClick={() => onChangePrompt(sample)}
            className="text-[11px] px-2.5 py-1 rounded-lg bg-slate-800/80 hover:bg-slate-700/80 text-slate-300 hover:text-white border border-slate-700/50 transition-colors"
          >
            "{sample.slice(0, 32)}..."
          </button>
        ))}
      </div>
    </div>
  );
};
