import React from 'react';
import { Sparkles, Cpu, Layers } from 'lucide-react';
import { HealthStatus } from '../../types/generation';

interface HeaderProps {
  health: HealthStatus | null;
  isLoading: boolean;
}

export const Header: React.FC<HeaderProps> = ({ health, isLoading }) => {
  return (
    <header className="border-b border-slate-800 bg-slate-900/60 backdrop-blur-md sticky top-0 z-40">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-brand-600 to-indigo-500 flex items-center justify-center shadow-lg shadow-brand-500/20">
            <Sparkles className="w-5 h-5 text-white" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-lg font-bold text-white tracking-tight">Mini LLM Studio</h1>
              <span className="px-2 py-0.5 text-xs font-semibold uppercase tracking-wider rounded-full bg-brand-500/10 text-brand-400 border border-brand-500/20">
                v1.0
              </span>
            </div>
            <p className="text-xs text-slate-400">Autoregressive Transformer Playground</p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          {/* Health indicator */}
          <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-800/80 border border-slate-700/60 text-xs">
            <span
              className={`w-2 h-2 rounded-full ${
                health?.status === 'healthy' ? 'bg-emerald-400 animate-pulse' : 'bg-amber-400'
              }`}
            />
            <span className="text-slate-300">
              {isLoading ? 'Connecting...' : health?.status === 'healthy' ? 'API Online' : 'Offline'}
            </span>
          </div>

          {/* Device diagnostic badge */}
          {health && (
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-800/80 border border-slate-700/60 text-xs text-slate-300">
              <Cpu className="w-3.5 h-3.5 text-brand-400" />
              <span className="font-mono uppercase font-semibold text-brand-300">
                {health.cuda_available ? (health.cuda_device_name ? 'GPU' : 'CUDA') : 'CPU'}
              </span>
              {health.cuda_device_name && (
                <span className="hidden md:inline text-slate-400 text-[11px] truncate max-w-[150px]">
                  ({health.cuda_device_name})
                </span>
              )}
            </div>
          )}

          {/* Model count badge */}
          {health?.loaded_models && health.loaded_models.length > 0 && (
            <div className="hidden sm:flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800/80 border border-slate-700/60 text-xs text-slate-300">
              <Layers className="w-3.5 h-3.5 text-indigo-400" />
              <span>{health.loaded_models.length} Models Loaded</span>
            </div>
          )}
        </div>
      </div>
    </header>
  );
};
