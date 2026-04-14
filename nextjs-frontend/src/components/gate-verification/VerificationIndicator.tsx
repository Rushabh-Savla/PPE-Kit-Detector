'use client';

import { Cpu, Brain, Check, X, Loader2 } from 'lucide-react';
import type { VerificationStatus } from '@/types';

interface VerificationIndicatorProps {
  rfidStatus: VerificationStatus;
  mlStatus: VerificationStatus;
  showRfid?: boolean; // false for face (no RFID)
  size?: 'sm' | 'md' | 'lg';
  label?: string;
}

const STATUS_COLORS: Record<VerificationStatus, string> = {
  pending: 'bg-stone-200 text-stone-400 border-stone-300',
  checking: 'bg-amber-100 text-amber-600 border-amber-400 animate-pulse',
  passed: 'bg-green-100 text-green-600 border-green-500',
  failed: 'bg-red-100 text-red-600 border-red-500',
  warning: 'bg-orange-100 text-orange-600 border-orange-500',
};

const SIZE_CLASSES = {
  sm: 'w-6 h-6',
  md: 'w-8 h-8',
  lg: 'w-10 h-10',
};

const ICON_SIZES = {
  sm: 12,
  md: 16,
  lg: 20,
};

function StatusIcon({ status, size }: { status: VerificationStatus; size: 'sm' | 'md' | 'lg' }) {
  const iconSize = ICON_SIZES[size];

  switch (status) {
    case 'passed':
      return <Check size={iconSize} className="text-green-600" />;
    case 'failed':
      return <X size={iconSize} className="text-red-600" />;
    case 'checking':
      return <Loader2 size={iconSize} className="animate-spin" />;
    default:
      return null;
  }
}

export function VerificationIndicator({
  rfidStatus,
  mlStatus,
  showRfid = true,
  size = 'md',
  label,
}: VerificationIndicatorProps) {
  return (
    <div className="flex flex-col items-center gap-1">
      <div className="flex items-center gap-2">
        {/* RFID Indicator */}
        {showRfid && (
          <div
            className={`${SIZE_CLASSES[size]} rounded-full border-2 flex items-center justify-center transition-all duration-300 ${STATUS_COLORS[rfidStatus]}`}
            title={`RFID: ${rfidStatus}`}
          >
            {rfidStatus === 'passed' || rfidStatus === 'failed' ? (
              <StatusIcon status={rfidStatus} size={size} />
            ) : (
              <Cpu size={ICON_SIZES[size]} />
            )}
          </div>
        )}

        {/* ML Indicator */}
        <div
          className={`${SIZE_CLASSES[size]} rounded-full border-2 flex items-center justify-center transition-all duration-300 ${STATUS_COLORS[mlStatus]}`}
          title={`ML Detection: ${mlStatus}`}
        >
          {mlStatus === 'passed' || mlStatus === 'failed' ? (
            <StatusIcon status={mlStatus} size={size} />
          ) : (
            <Brain size={ICON_SIZES[size]} />
          )}
        </div>
      </div>

      {/* Label */}
      {label && (
        <span className="text-xs font-medium text-stone-600 uppercase tracking-wide">
          {label}
        </span>
      )}
    </div>
  );
}

// Compact version for inline display
export function VerificationBadge({
  rfidStatus,
  mlStatus,
  showRfid = true,
}: {
  rfidStatus: VerificationStatus;
  mlStatus: VerificationStatus;
  showRfid?: boolean;
}) {
  const rfidPassed = rfidStatus === 'passed';
  const mlPassed = mlStatus === 'passed';
  const allPassed = showRfid ? rfidPassed && mlPassed : mlPassed;
  const anyFailed = rfidStatus === 'failed' || mlStatus === 'failed';

  let bgColor = 'bg-stone-100';
  let textColor = 'text-stone-600';

  if (allPassed) {
    bgColor = 'bg-green-100';
    textColor = 'text-green-700';
  } else if (anyFailed) {
    bgColor = 'bg-red-100';
    textColor = 'text-red-700';
  } else if (rfidStatus === 'checking' || mlStatus === 'checking') {
    bgColor = 'bg-amber-100';
    textColor = 'text-amber-700';
  }

  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium ${bgColor} ${textColor}`}>
      {showRfid && (
        <span className={rfidPassed ? 'text-green-600' : rfidStatus === 'failed' ? 'text-red-600' : ''}>
          {rfidPassed ? '✓' : rfidStatus === 'failed' ? '✗' : '○'}
        </span>
      )}
      <span className={mlPassed ? 'text-green-600' : mlStatus === 'failed' ? 'text-red-600' : ''}>
        {mlPassed ? '✓' : mlStatus === 'failed' ? '✗' : '○'}
      </span>
    </span>
  );
}
