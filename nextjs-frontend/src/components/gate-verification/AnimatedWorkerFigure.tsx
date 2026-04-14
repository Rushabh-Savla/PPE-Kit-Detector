'use client';

import { VerificationIndicator } from './VerificationIndicator';
import type { VerificationItem, VerificationItemType } from '@/types';

interface AnimatedWorkerFigureProps {
  items: Record<VerificationItemType, VerificationItem>;
  className?: string;
}

export function AnimatedWorkerFigure({ items, className = '' }: AnimatedWorkerFigureProps) {
  return (
    <div className={`relative flex items-center justify-center ${className}`}>
      {/* SVG Worker Figure */}
      <svg
        viewBox="0 0 200 320"
        className="w-full max-w-[200px] h-auto"
        style={{ filter: 'drop-shadow(0 4px 6px rgba(0,0,0,0.1))' }}
      >
        {/* Hard Hat / Helmet */}
        <ellipse cx="100" cy="45" rx="40" ry="20" className="fill-amber-400 stroke-amber-600 stroke-2" />
        <path d="M60 45 Q60 25 100 20 Q140 25 140 45" className="fill-amber-500 stroke-amber-600 stroke-2" />
        <rect x="55" y="40" width="90" height="8" rx="2" className="fill-amber-600" />

        {/* Face */}
        <circle cx="100" cy="75" r="28" className="fill-amber-100 stroke-amber-300 stroke-2" />
        {/* Eyes */}
        <circle cx="90" cy="70" r="4" className="fill-stone-700" />
        <circle cx="110" cy="70" r="4" className="fill-stone-700" />
        {/* Smile */}
        <path d="M88 82 Q100 92 112 82" className="fill-none stroke-stone-600 stroke-2" strokeLinecap="round" />

        {/* Neck */}
        <rect x="90" y="100" width="20" height="15" className="fill-amber-100" />

        {/* Safety Vest - Body */}
        <path
          d="M60 115 L80 110 L80 200 L60 200 Z"
          className="fill-orange-500 stroke-orange-600 stroke-2"
        />
        <path
          d="M140 115 L120 110 L120 200 L140 200 Z"
          className="fill-orange-500 stroke-orange-600 stroke-2"
        />
        <rect x="80" y="110" width="40" height="90" className="fill-orange-400 stroke-orange-500 stroke-2" />

        {/* Reflective Stripes on Vest */}
        <rect x="60" y="140" width="80" height="6" className="fill-yellow-300" />
        <rect x="60" y="170" width="80" height="6" className="fill-yellow-300" />

        {/* Arms */}
        <rect x="45" y="115" width="18" height="60" rx="8" className="fill-stone-500 stroke-stone-600 stroke-2" />
        <rect x="137" y="115" width="18" height="60" rx="8" className="fill-stone-500 stroke-stone-600 stroke-2" />

        {/* Hands */}
        <circle cx="54" cy="180" r="10" className="fill-amber-200 stroke-amber-300 stroke-2" />
        <circle cx="146" cy="180" r="10" className="fill-amber-200 stroke-amber-300 stroke-2" />

        {/* Pants */}
        <rect x="70" y="200" width="25" height="70" rx="4" className="fill-stone-600 stroke-stone-700 stroke-2" />
        <rect x="105" y="200" width="25" height="70" rx="4" className="fill-stone-600 stroke-stone-700 stroke-2" />

        {/* Safety Boots */}
        <path
          d="M65 270 L65 300 Q65 310 75 310 L95 310 Q100 310 100 305 L100 270"
          className="fill-stone-800 stroke-stone-900 stroke-2"
        />
        <path
          d="M100 270 L100 305 Q100 310 105 310 L125 310 Q135 310 135 300 L135 270"
          className="fill-stone-800 stroke-stone-900 stroke-2"
        />
        {/* Boot soles */}
        <rect x="65" y="305" width="35" height="8" rx="2" className="fill-amber-700" />
        <rect x="100" y="305" width="35" height="8" rx="2" className="fill-amber-700" />
      </svg>

      {/* Verification Indicators positioned around the figure */}

      {/* Helmet Indicator - Top */}
      <div className="absolute top-0 left-1/2 transform -translate-x-1/2 -translate-y-2">
        <div className="flex flex-col items-center">
          <VerificationIndicator
            rfidStatus={items.helmet.rfidStatus}
            mlStatus={items.helmet.mlStatus}
            size="md"
          />
          <span className="text-xs font-semibold text-stone-600 mt-1 bg-white/80 px-2 py-0.5 rounded">
            HELMET
          </span>
        </div>
      </div>

      {/* Face Indicator - Right of head */}
      <div className="absolute top-[18%] right-0 transform translate-x-2">
        <div className="flex items-center gap-2">
          <div className="w-8 h-0.5 bg-stone-300" />
          <div className="flex flex-col items-center">
            <VerificationIndicator
              rfidStatus={items.face.rfidStatus}
              mlStatus={items.face.mlStatus}
              showRfid={false}
              size="md"
            />
            <span className="text-xs font-semibold text-stone-600 mt-1 bg-white/80 px-2 py-0.5 rounded">
              FACE
            </span>
          </div>
        </div>
      </div>

      {/* Vest Indicator - Left of body */}
      <div className="absolute top-[45%] left-0 transform -translate-x-2">
        <div className="flex items-center gap-2">
          <div className="flex flex-col items-center">
            <VerificationIndicator
              rfidStatus={items.vest.rfidStatus}
              mlStatus={items.vest.mlStatus}
              size="md"
            />
            <span className="text-xs font-semibold text-stone-600 mt-1 bg-white/80 px-2 py-0.5 rounded">
              VEST
            </span>
          </div>
          <div className="w-8 h-0.5 bg-stone-300" />
        </div>
      </div>

      {/* Shoes Indicator - Bottom */}
      <div className="absolute bottom-0 left-1/2 transform -translate-x-1/2 translate-y-2">
        <div className="flex flex-col items-center">
          <span className="text-xs font-semibold text-stone-600 mb-1 bg-white/80 px-2 py-0.5 rounded">
            SHOES
          </span>
          <VerificationIndicator
            rfidStatus={items.shoes.rfidStatus}
            mlStatus={items.shoes.mlStatus}
            size="md"
          />
        </div>
      </div>
    </div>
  );
}

// Compact version for smaller displays
export function WorkerFigureCompact({
  items,
  className = '',
}: AnimatedWorkerFigureProps) {
  return (
    <div className={`grid grid-cols-2 gap-3 p-4 bg-stone-50 rounded-xl ${className}`}>
      {/* Helmet */}
      <div className="flex items-center gap-2 p-2 bg-white rounded-lg shadow-sm">
        <div className="w-8 h-8 rounded-full bg-amber-100 flex items-center justify-center">
          <span className="text-lg">🪖</span>
        </div>
        <div className="flex-1">
          <div className="text-xs font-medium text-stone-600">Helmet</div>
          <div className="flex gap-1 mt-1">
            <StatusDot status={items.helmet.rfidStatus} label="RFID" />
            <StatusDot status={items.helmet.mlStatus} label="ML" />
          </div>
        </div>
      </div>

      {/* Face */}
      <div className="flex items-center gap-2 p-2 bg-white rounded-lg shadow-sm">
        <div className="w-8 h-8 rounded-full bg-amber-100 flex items-center justify-center">
          <span className="text-lg">👤</span>
        </div>
        <div className="flex-1">
          <div className="text-xs font-medium text-stone-600">Face</div>
          <div className="flex gap-1 mt-1">
            <StatusDot status={items.face.mlStatus} label="ML" />
          </div>
        </div>
      </div>

      {/* Vest */}
      <div className="flex items-center gap-2 p-2 bg-white rounded-lg shadow-sm">
        <div className="w-8 h-8 rounded-full bg-orange-100 flex items-center justify-center">
          <span className="text-lg">🦺</span>
        </div>
        <div className="flex-1">
          <div className="text-xs font-medium text-stone-600">Vest</div>
          <div className="flex gap-1 mt-1">
            <StatusDot status={items.vest.rfidStatus} label="RFID" />
            <StatusDot status={items.vest.mlStatus} label="ML" />
          </div>
        </div>
      </div>

      {/* Shoes */}
      <div className="flex items-center gap-2 p-2 bg-white rounded-lg shadow-sm">
        <div className="w-8 h-8 rounded-full bg-stone-200 flex items-center justify-center">
          <span className="text-lg">👢</span>
        </div>
        <div className="flex-1">
          <div className="text-xs font-medium text-stone-600">Shoes</div>
          <div className="flex gap-1 mt-1">
            <StatusDot status={items.shoes.rfidStatus} label="RFID" />
            <StatusDot status={items.shoes.mlStatus} label="ML" />
          </div>
        </div>
      </div>
    </div>
  );
}

function StatusDot({ status, label }: { status: string; label: string }) {
  const colors: Record<string, string> = {
    pending: 'bg-stone-300',
    checking: 'bg-amber-400 animate-pulse',
    passed: 'bg-green-500',
    failed: 'bg-red-500',
    warning: 'bg-orange-500',
  };

  return (
    <div className="flex items-center gap-1" title={`${label}: ${status}`}>
      <div className={`w-2 h-2 rounded-full ${colors[status] || colors.pending}`} />
      <span className="text-[10px] text-stone-500">{label}</span>
    </div>
  );
}
