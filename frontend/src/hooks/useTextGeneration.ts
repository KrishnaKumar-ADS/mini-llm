/**
 * Custom React hook encapsulating model discovery and text generation state.
 */

import { useCallback, useEffect, useState } from 'react';
import { fetchHealth, fetchModels, postGenerate } from '../api/generationApi';
import { GenerationRequest, GenerationResponse, HealthStatus, ModelInfo } from '../types/generation';

export function useTextGeneration() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<string>('mini-llm-100k');
  const [health, setHealth] = useState<HealthStatus | null>(null);

  const [isLoadingModels, setIsLoadingModels] = useState<boolean>(true);
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [generationResult, setGenerationResult] = useState<GenerationResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const loadInitialData = useCallback(async () => {
    setIsLoadingModels(true);
    setError(null);
    try {
      const [fetchedModels, fetchedHealth] = await Promise.all([
        fetchModels(),
        fetchHealth().catch(() => null),
      ]);

      setModels(fetchedModels);
      setHealth(fetchedHealth);

      // Default to 100k if available, else first available model
      if (fetchedModels.length > 0) {
        const has100k = fetchedModels.some((m) => m.model_id === 'mini-llm-100k');
        setSelectedModelId(has100k ? 'mini-llm-100k' : fetchedModels[0].model_id);
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to connect to Mini LLM Backend';
      setError(
        `${message}. Please verify the backend server is running and accessible at the configured VITE_API_BASE_URL.`
      );
    } finally {
      setIsLoadingModels(false);
    }
  }, []);

  useEffect(() => {
    loadInitialData();
  }, [loadInitialData]);

  const generate = useCallback(async (request: GenerationRequest) => {
    setIsGenerating(true);
    setError(null);
    try {
      const result = await postGenerate(request);
      setGenerationResult(result);
      return result;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Generation request failed';
      setError(message);
      throw err;
    } finally {
      setIsGenerating(false);
    }
  }, []);

  const clearResult = useCallback(() => {
    setGenerationResult(null);
  }, []);

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  return {
    models,
    selectedModelId,
    setSelectedModelId,
    health,
    isLoadingModels,
    isGenerating,
    generationResult,
    error,
    generate,
    clearResult,
    clearError,
    refreshModels: loadInitialData,
  };
}
