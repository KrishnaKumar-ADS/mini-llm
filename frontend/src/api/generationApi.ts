/**
 * API client for interacting with the Mini LLM FastAPI backend.
 */

import {
  GenerationRequest,
  GenerationResponse,
  HealthStatus,
  ModelInfo,
  ModelListResponse,
} from '../types/generation';

// Reads from .env (VITE_API_BASE_URL), defaults to standard local FastAPI port
const BASE_URL: string =
  (import.meta.env.VITE_API_BASE_URL as string) || 'http://localhost:8000/api/v1';

/**
 * Normalizes error messages from HTTP responses into human-readable strings.
 */
function extractErrorMessage(errorData: unknown, status: number): string {
  if (typeof errorData === 'object' && errorData !== null) {
    const data = errorData as Record<string, unknown>;
    if (typeof data.detail === 'string') {
      return data.detail;
    }
    if (Array.isArray(data.detail)) {
      // Pydantic validation error array
      return data.detail.map((err) => (err as { msg?: string }).msg || JSON.stringify(err)).join(', ');
    }
    if (typeof data.message === 'string') {
      return data.message;
    }
  }
  return `Server returned HTTP status ${status}`;
}

/**
 * Fetch health and device diagnostics from backend.
 */
export async function fetchHealth(): Promise<HealthStatus> {
  const response = await fetch(`${BASE_URL}/health`, {
    method: 'GET',
    headers: { 'Content-Type': 'application/json' },
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => null);
    throw new Error(extractErrorMessage(errorData, response.status));
  }

  return response.json();
}

/**
 * Fetch list of registered models with training metadata.
 */
export async function fetchModels(): Promise<ModelInfo[]> {
  const response = await fetch(`${BASE_URL}/models`, {
    method: 'GET',
    headers: { 'Content-Type': 'application/json' },
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => null);
    throw new Error(extractErrorMessage(errorData, response.status));
  }

  const data: ModelListResponse = await response.json();
  return data.models;
}

/**
 * Send text generation request to the chosen model.
 */
export async function postGenerate(request: GenerationRequest): Promise<GenerationResponse> {
  const response = await fetch(`${BASE_URL}/generate`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => null);
    throw new Error(extractErrorMessage(errorData, response.status));
  }

  return response.json();
}
