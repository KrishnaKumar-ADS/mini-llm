# Mini LLM Studio — Frontend

Modern, responsive web playground for the Mini LLM inference platform built with **React 18**, **TypeScript**, **Vite**, and **Tailwind CSS**.

## Features

- **Model Selection**: Switch seamlessly between the `mini-llm-25k` Baseline and `mini-llm-100k` Advanced checkpoints.
- **Dynamic Diagnostics**: Live inspection of model parameters, training subsets, batch sizes, validation loss, and perplexity.
- **Real-Time Word Validation**: Live validation badge ensuring initial prompts contain at least 5 words with starter suggestions.
- **Dual Truncation Controls**: Interactive slider and numeric inputs for token bounds (up to 5,000 tokens) with optional exact character-level truncation.
- **Advanced Sampling**: Collapsible controls for temperature, top-k, nucleus top-p, reproducibility seeds, and EOS handling.
- **Rich Results View**: Distinct visual highlighting for prompts vs generated continuations, token counters, latency metrics, and one-click copy.
- **Environment-Driven**: Configured via `.env` (`VITE_API_BASE_URL`) for decoupled, independent deployment.

---

## Directory Structure

```
frontend/
├── src/
│   ├── api/
│   │   └── generationApi.ts       # Backend API client with error handling
│   ├── components/
│   │   ├── common/
│   │   │   ├── ErrorAlert.tsx     # Dismissible error banner
│   │   │   └── LoadingSpinner.tsx # Reusable loading spinner
│   │   ├── layout/
│   │   │   ├── Header.tsx         # App bar with health/device badge
│   │   │   └── Footer.tsx         # Footer with architecture specs
│   │   └── playground/
│   │       ├── AdvancedParameters.tsx # Sampling hyperparameters panel
│   │       ├── GenerationOutput.tsx   # Results card with metrics & copy
│   │       ├── ModelSelector.tsx      # Checkpoint card picker
│   │       ├── PlaygroundPage.tsx     # Main playground layout
│   │       ├── PromptInput.tsx        # Textarea with live word counter
│   │       └── TokenLimitSlider.tsx   # Max tokens & character limits
│   ├── hooks/
│   │   └── useTextGeneration.ts   # Custom hook for model state & API calls
│   ├── types/
│   │   └── generation.ts          # TypeScript interfaces & types
│   ├── utils/
│   │   └── validation.ts          # Prompt and token bound validators
│   ├── App.tsx                    # Root application component
│   ├── index.css                  # Tailwind styles and animations
│   ├── main.tsx                   # React root mount
│   └── vite-env.d.ts              # Vite environment typings
├── .env                           # Active environment configuration
├── .env.example                   # Reference environment template
├── index.html                     # HTML document template
├── package.json                   # Dependencies and scripts
├── tsconfig.json                  # TypeScript compiler settings
└── vite.config.ts                 # Vite bundler configuration
```

---

## Setup & Running

### 1. Install Dependencies
```bash
npm install
```

### 2. Configure Environment
Create `.env` from template if not already present:
```bash
cp .env.example .env
```

Ensure `VITE_API_BASE_URL` matches your running backend URL:
```dotenv
VITE_API_BASE_URL=http://localhost:8000/api/v1
```

### 3. Development Server
Start the Vite local development server:
```bash
npm run dev
```

The app will be available at `http://localhost:5173`.

### 4. Production Build
```bash
npm run build
```

The optimized static assets will be output to `frontend/dist/`, ready to deploy to Vercel, Netlify, Cloudflare Pages, S3, or Nginx.
