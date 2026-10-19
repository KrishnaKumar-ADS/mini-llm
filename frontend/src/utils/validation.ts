/**
 * Client-side validation utilities for generation prompts and parameters.
 */

export interface PromptValidationResult {
  isValid: boolean;
  wordCount: number;
  characterCount: number;
  errorMessage: string | null;
}

export interface TokenValidationResult {
  isValid: boolean;
  errorMessage: string | null;
}

/**
 * Counts whitespace-delimited words in a string.
 */
export function countWords(text: string): number {
  const trimmed = text.trim();
  if (!trimmed) return 0;
  return trimmed.split(/\s+/).length;
}

/**
 * Validates the user's prompt text.
 * Requires at least 5 words as specified in project requirements.
 */
export function validatePrompt(prompt: string, minWords = 5): PromptValidationResult {
  const trimmed = prompt.trim();
  const wordCount = countWords(trimmed);
  const characterCount = trimmed.length;

  if (characterCount === 0) {
    return {
      isValid: false,
      wordCount: 0,
      characterCount: 0,
      errorMessage: 'Please enter an initial starter prompt.',
    };
  }

  if (wordCount < minWords) {
    const diff = minWords - wordCount;
    return {
      isValid: false,
      wordCount,
      characterCount,
      errorMessage: `Prompt requires at least ${minWords} words (current: ${wordCount}). Please add ${diff} more word${diff > 1 ? 's' : ''}.`,
    };
  }

  return {
    isValid: true,
    wordCount,
    characterCount,
    errorMessage: null,
  };
}

/**
 * Validates the token generation limit (must be between 1 and 5000).
 */
export function validateTokenLimit(tokens: number, maxLimit = 5000): TokenValidationResult {
  if (isNaN(tokens) || !Number.isInteger(tokens)) {
    return { isValid: false, errorMessage: 'Token limit must be a valid integer.' };
  }
  if (tokens < 1) {
    return { isValid: false, errorMessage: 'Token limit must be at least 1 token.' };
  }
  if (tokens > maxLimit) {
    return { isValid: false, errorMessage: `Token limit cannot exceed ${maxLimit} tokens.` };
  }
  return { isValid: true, errorMessage: null };
}
