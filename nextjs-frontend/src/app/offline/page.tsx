'use client';

import { WifiOff, RefreshCw } from 'lucide-react';

export default function OfflinePage() {
  const handleRefresh = () => {
    window.location.reload();
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-stone-50 via-slate-50 to-stone-100 p-6">
      <div className="text-center max-w-md">
        <div className="w-24 h-24 bg-gradient-to-br from-orange-200 to-amber-200 rounded-3xl flex items-center justify-center mx-auto mb-8 shadow-lg border border-orange-300">
          <WifiOff className="w-12 h-12 text-orange-600" strokeWidth={2} />
        </div>

        <h1 className="text-3xl font-bold text-slate-800 mb-4">You're Offline</h1>

        <p className="text-slate-500 mb-8 text-lg">
          It seems you've lost your internet connection. Please check your connection and try again.
        </p>

        <button
          onClick={handleRefresh}
          className="inline-flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-orange-500 to-amber-500 text-white rounded-xl font-semibold hover:from-orange-600 hover:to-amber-600 transition-all shadow-lg"
        >
          <RefreshCw className="w-5 h-5" />
          Try Again
        </button>

        <div className="mt-12 p-4 bg-white rounded-xl border border-slate-200 shadow-sm">
          <p className="text-sm text-slate-600">
            <strong className="text-slate-700">Raksham</strong> - Mine Safety System
          </p>
          <p className="text-xs text-slate-400 mt-1">
            Some features may be available offline
          </p>
        </div>
      </div>
    </div>
  );
}
