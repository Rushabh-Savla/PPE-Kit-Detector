'use client';

import { ReactNode } from 'react';
import { useSidebarStore } from '@/lib/store';

interface GateVerificationLayoutProps {
  videoPanel: ReactNode;
  workerFigure: ReactNode;
  statusPanel: ReactNode;
}

export function GateVerificationLayout({
  videoPanel,
  workerFigure,
  statusPanel,
}: GateVerificationLayoutProps) {
  const { isCollapsed } = useSidebarStore();

  return (
    <div className="h-full min-h-[600px]">
      {/* Desktop Layout - 3 columns */}
      <div className="hidden lg:grid lg:grid-cols-3 gap-6 h-full">
        {/* Left - Video Panel */}
        <div className="flex flex-col">
          <h2 className="text-lg font-semibold text-stone-700 mb-3">Live Camera Feed</h2>
          <div className="flex-1">
            {videoPanel}
          </div>
        </div>

        {/* Center - Worker Figure */}
        <div className="flex flex-col items-center justify-center py-8">
          <h2 className="text-lg font-semibold text-stone-700 mb-6">Verification Status</h2>
          <div className="flex-1 flex items-center justify-center w-full max-w-[300px]">
            {workerFigure}
          </div>
          <div className="mt-4 text-center text-sm text-stone-500">
            <p>RFID + ML Detection</p>
            <p className="text-xs mt-1">Hardware and software verification</p>
          </div>
        </div>

        {/* Right - Status Panel */}
        <div className="flex flex-col">
          <h2 className="text-lg font-semibold text-stone-700 mb-3">Status & Controls</h2>
          <div className="flex-1">
            {statusPanel}
          </div>
        </div>
      </div>

      {/* Tablet Layout - 2 columns */}
      <div className="hidden md:grid md:grid-cols-2 lg:hidden gap-6 h-full">
        {/* Left - Video + Worker Figure */}
        <div className="flex flex-col gap-6">
          <div>
            <h2 className="text-lg font-semibold text-stone-700 mb-3">Live Camera</h2>
            {videoPanel}
          </div>
          <div className="flex-1 flex flex-col items-center justify-center">
            <h2 className="text-lg font-semibold text-stone-700 mb-4">Verification</h2>
            <div className="max-w-[250px]">
              {workerFigure}
            </div>
          </div>
        </div>

        {/* Right - Status Panel */}
        <div className="flex flex-col">
          <h2 className="text-lg font-semibold text-stone-700 mb-3">Status</h2>
          <div className="flex-1">
            {statusPanel}
          </div>
        </div>
      </div>

      {/* Mobile Layout - Stacked */}
      <div className="md:hidden flex flex-col gap-6">
        {/* Status Panel First on Mobile */}
        <div>
          <h2 className="text-lg font-semibold text-stone-700 mb-3">Status</h2>
          {statusPanel}
        </div>

        {/* Video Panel */}
        <div>
          <h2 className="text-lg font-semibold text-stone-700 mb-3">Camera</h2>
          {videoPanel}
        </div>

        {/* Worker Figure - Compact on Mobile */}
        <div className="flex flex-col items-center">
          <h2 className="text-lg font-semibold text-stone-700 mb-3">Checks</h2>
          <div className="w-full max-w-[200px]">
            {workerFigure}
          </div>
        </div>
      </div>
    </div>
  );
}
