import React, { useState } from 'react';
import { Play, Sparkles, RefreshCw } from 'lucide-react';
import { ModelSelector } from './ModelSelector';
import { PromptInput } from './PromptInput';
import { TokenLimitSlider } from './TokenLimitSlider';
import { AdvancedParameters } from './AdvancedParameters';
import { GenerationOutput } from './GenerationOutput';
import { ErrorAlert } from '../common/ErrorAlert';
import { LoadingSpinner } from '../common/LoadingSpinner';
import { useTextGeneration } from '../../hooks/useTextGeneration';
import { validatePrompt, validateTokenLimit } from '../../utils/validation';

export const PlaygroundPage: React.FC = () => {
  const {
    models,
    selectedModelId,
    setSelectedModelId,
    isLoadingModels,
    isGenerating,
    generationResult,
    error,
    generate,
    clearResult,
    clearError,
    refreshModels,
  } = useTextGeneration();

  // Form states
  const [prompt, setPrompt] = useState<string>('Once upon a time in a sunny meadow');
  const [maxTokens, setMaxTokens] = useState<number>(200);
  const [maxCharacters, setMaxCharacters] = useState<number | null>(null);

  // Advanced sampling states
  const [temperature, setTemperature] = useState<number>(0.8);
  const [topK, setTopK] = useState<number>(20);
  const [topP, setTopP] = useState<number>(0.9);
  const [seed, setSeed] = useState<number | null>(null);
  const [stopAtEos, setStopAtEos] = useState<boolean>(true);

  // Validations
  const promptValidation = validatePrompt(prompt, 5);
  const tokenValidation = validateTokenLimit(maxTokens, 5000);
  const isFormValid = promptValidation.isValid && tokenValidation.isValid && !isGenerating;

  const handleGenerate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!isFormValid) return;

    try {
      await generate({
        model_id: selectedModelId,
        prompt: prompt.trim(),
        max_tokens: maxTokens,
        max_characters: maxCharacters,
        temperature,
        top_k: topK,
        top_p: topP,
        seed: seed ?? undefined,
        stop_at_eos: stopAtEos,
      });
    } catch {
      // Error handled in hook state
    }
  };

  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      {/* Intro hero banner */}
      <div className="text-center space-y-2">
        <h2 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
          Interactive Transformer Playground
        </h2>
        <p className="text-sm text-slate-400 max-w-2xl mx-auto">
          Test and compare next-token narrative completions between the{' '}
          <span className="text-emerald-400 font-semibold">25k Baseline</span> and{' '}
          <span className="text-indigo-400 font-semibold">100k Advanced</span> Mini LLM models.
        </p>
      </div>

      {/* Global Error Banner */}
      {error && (
        <ErrorAlert
          message={error}
          onDismiss={clearError}
          className="animate-fadeIn"
        />
      )}

      {/* Loading models state */}
      {isLoadingModels ? (
        <div className="p-12 text-center rounded-2xl border border-slate-800 bg-slate-900/40">
          <LoadingSpinner size="lg" label="Loading models and checkpoints from backend..." />
        </div>
      ) : (
        <form onSubmit={handleGenerate} className="space-y-6">
          {/* Model selection card group */}
          <ModelSelector
            models={models}
            selectedModelId={selectedModelId}
            onSelectModel={setSelectedModelId}
            disabled={isGenerating}
          />

          {/* Seed Prompt Input with validation */}
          <PromptInput
            prompt={prompt}
            onChangePrompt={setPrompt}
            validation={promptValidation}
            disabled={isGenerating}
          />

          {/* Token generation limit slider */}
          <TokenLimitSlider
            maxTokens={maxTokens}
            onChangeMaxTokens={setMaxTokens}
            maxCharacters={maxCharacters}
            onChangeMaxCharacters={setMaxCharacters}
            disabled={isGenerating}
          />

          {/* Advanced Sampling Parameters */}
          <AdvancedParameters
            temperature={temperature}
            onChangeTemperature={setTemperature}
            topK={topK}
            onChangeTopK={setTopK}
            topP={topP}
            onChangeTopP={setTopP}
            seed={seed}
            onChangeSeed={setSeed}
            stopAtEos={stopAtEos}
            onChangeStopAtEos={setStopAtEos}
            disabled={isGenerating}
          />

          {/* Submit action button */}
          <div className="flex items-center justify-between gap-4 pt-2">
            <button
              type="button"
              onClick={refreshModels}
              disabled={isGenerating}
              className="px-4 py-2.5 rounded-xl border border-slate-800 bg-slate-900/60 hover:bg-slate-800 text-xs font-medium text-slate-400 hover:text-slate-200 transition-colors flex items-center gap-1.5"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              Reload Models
            </button>

            <button
              type="submit"
              disabled={!isFormValid}
              className={`flex-1 sm:flex-initial inline-flex items-center justify-center gap-2 px-8 py-3.5 rounded-xl font-semibold text-sm transition-all duration-200 shadow-lg ${
                isFormValid
                  ? 'bg-gradient-to-r from-brand-500 to-indigo-600 hover:from-brand-400 hover:to-indigo-500 text-white shadow-brand-500/25 cursor-pointer active:scale-[0.98]'
                  : 'bg-slate-800/80 text-slate-500 border border-slate-700/50 cursor-not-allowed shadow-none'
              }`}
            >
              {isGenerating ? (
                <>
                  <LoadingSpinner size="sm" />
                  <span>Generating Completion...</span>
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 fill-current" />
                  <span>Generate Completion</span>
                </>
              )}
            </button>
          </div>
        </form>
      )}

      {/* Generation Results Section */}
      <div className="pt-4">
        <GenerationOutput
          result={generationResult}
          isGenerating={isGenerating}
          onClear={clearResult}
        />
      </div>
    </div>
  );
};
