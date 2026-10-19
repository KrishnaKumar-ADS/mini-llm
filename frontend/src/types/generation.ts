/**
 * TypeScript type definitions for Mini LLM API communication and state.
 */

export interface ModelInfo {
  model_id: string;
  display_name: string;
  description: string;
  training_subset: '25k' | '100k' | string;
  parameter_count: number;
  context_length: number;
  validation_loss?: number;
  validation_perplexity?: number;
  batch_size?: number;
  training_steps: number;
}

export interface ModelListResponse {
  models: ModelInfo[];
}

export interface GenerationRequest {
  model_id: string;
  prompt: string;
  max_tokens: number;
  max_characters?: number | null;
  temperature?: number;
  top_k?: number | null;
  top_p?: number;
  seed?: number | null;
  stop_at_eos?: boolean;
}

export interface GenerationResponse {
  model_id: string;
  prompt: string;
  completion: string;
  full_text: string;
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  character_count: number;
  truncated: boolean;
  inference_duration_ms: number;
}

export interface HealthStatus {
  status: string;
  device: string;
  cuda_available: boolean;
  cuda_device_name?: string | null;
  loaded_models: string[];
}

export interface ApiErrorDetail {
  detail: string | { msg: string; loc: string[] }[];
  error_code?: string;
  path?: string;
}
