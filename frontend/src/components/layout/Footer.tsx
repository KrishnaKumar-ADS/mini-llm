import React from 'react';

export const Footer: React.FC = () => {
  return (
    <footer className="border-t border-slate-800/80 bg-slate-900/40 py-6 mt-16 text-center text-xs text-slate-500">
      <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-3">
        <p>Mini LLM Laboratory • Decoder-Only Transformer Architecture</p>
        <div className="flex items-center gap-4 text-slate-400">
          <span>Byte-Level Tokenizer (259 Vocab)</span>
          <span>•</span>
          <span>Context Window: 128 Tokens</span>
        </div>
      </div>
    </footer>
  );
};
