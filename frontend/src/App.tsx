import React from 'react';
import { Header } from './components/layout/Header';
import { Footer } from './components/layout/Footer';
import { PlaygroundPage } from './components/playground/PlaygroundPage';
import { useTextGeneration } from './hooks/useTextGeneration';

export const App: React.FC = () => {
  const { health, isLoadingModels } = useTextGeneration();

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col antialiased selection:bg-brand-500 selection:text-white">
      {/* Background ambient gradient glow */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden">
        <div className="absolute -top-40 -left-40 w-96 h-96 bg-brand-500/10 rounded-full blur-3xl" />
        <div className="absolute top-1/3 -right-40 w-96 h-96 bg-indigo-500/10 rounded-full blur-3xl" />
      </div>

      <Header health={health} isLoading={isLoadingModels} />

      <main className="flex-1 relative z-10">
        <PlaygroundPage />
      </main>

      <Footer />
    </div>
  );
};

export default App;
