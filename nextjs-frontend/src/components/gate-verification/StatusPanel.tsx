'use client';

import { User, CheckCircle, XCircle, AlertTriangle, Play, RotateCcw, Shield, Camera } from 'lucide-react';
import { CountdownTimer } from './CountdownTimer';
import type { Worker } from '@/types';

interface StatusPanelProps {
  overallStatus: 'idle' | 'verifying' | 'passed' | 'failed' | 'warning';
  timeRemaining: number;
  isTimerRunning: boolean;
  identifiedWorker: Worker | null;
  passedCount: number;
  totalChecks: number;
  canOverride: boolean;
  attendanceMarked: boolean;
  onStart: () => void;
  onReset: () => void;
  onOverride: () => void;
  onCaptureSnapshot?: () => void;
  isCapturingSnapshot?: boolean;
  disabled?: boolean;
}

export function StatusPanel({
  overallStatus,
  timeRemaining,
  isTimerRunning,
  identifiedWorker,
  passedCount,
  totalChecks,
  canOverride,
  attendanceMarked,
  onStart,
  onReset,
  onOverride,
  onCaptureSnapshot,
  isCapturingSnapshot = false,
  disabled = false,
}: StatusPanelProps) {
  const getStatusDisplay = () => {
    switch (overallStatus) {
      case 'idle':
        return {
          icon: <Play className="text-stone-400" size={32} />,
          text: 'Ready to Verify',
          bgColor: 'bg-stone-100',
          textColor: 'text-stone-600',
        };
      case 'verifying':
        return {
          icon: <Shield className="text-amber-500 animate-pulse" size={32} />,
          text: 'Verifying...',
          bgColor: 'bg-amber-50',
          textColor: 'text-amber-700',
        };
      case 'passed':
        return {
          icon: <CheckCircle className="text-green-500" size={32} />,
          text: 'VERIFIED - PASS',
          bgColor: 'bg-green-50',
          textColor: 'text-green-700',
        };
      case 'failed':
        return {
          icon: <XCircle className="text-red-500" size={32} />,
          text: 'VERIFICATION FAILED',
          bgColor: 'bg-red-50',
          textColor: 'text-red-700',
        };
      case 'warning':
        return {
          icon: <AlertTriangle className="text-orange-500" size={32} />,
          text: 'PARTIAL VERIFICATION',
          bgColor: 'bg-orange-50',
          textColor: 'text-orange-700',
        };
      default:
        return {
          icon: <Shield className="text-stone-400" size={32} />,
          text: 'Unknown',
          bgColor: 'bg-stone-100',
          textColor: 'text-stone-600',
        };
    }
  };

  const status = getStatusDisplay();

  return (
    <div className="flex flex-col gap-4 h-full">
      {/* Worker Info Card */}
      <div className="bg-white rounded-xl border border-stone-200 p-4 shadow-sm">
        <h3 className="text-sm font-semibold text-stone-500 uppercase tracking-wide mb-3">
          Worker
        </h3>
        {identifiedWorker ? (
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 rounded-full bg-gradient-to-br from-orange-400 to-amber-500 flex items-center justify-center text-white">
              <User size={24} />
            </div>
            <div className="flex-1 min-w-0">
              <p className="font-semibold text-stone-800 truncate">
                {identifiedWorker.name}
              </p>
              <p className="text-sm text-stone-500">
                ID: {identifiedWorker.employee_id}
              </p>
              {attendanceMarked && (
                <span className="inline-flex items-center gap-1 text-xs text-green-600 mt-1">
                  <CheckCircle size={12} />
                  Attendance Marked
                </span>
              )}
            </div>
          </div>
        ) : (
          <div className="flex items-center gap-3 text-stone-400">
            <div className="w-12 h-12 rounded-full bg-stone-100 flex items-center justify-center">
              <User size={24} />
            </div>
            <p className="text-sm">Waiting for identification...</p>
          </div>
        )}
      </div>

      {/* Timer */}
      {isTimerRunning && (
        <div className="bg-white rounded-xl border border-stone-200 p-4 shadow-sm flex flex-col items-center">
          <h3 className="text-sm font-semibold text-stone-500 uppercase tracking-wide mb-3">
            Time Remaining
          </h3>
          <CountdownTimer timeRemaining={timeRemaining} size="lg" />
        </div>
      )}

      {/* Progress */}
      <div className="bg-white rounded-xl border border-stone-200 p-4 shadow-sm">
        <h3 className="text-sm font-semibold text-stone-500 uppercase tracking-wide mb-3">
          Verification Progress
        </h3>
        <div className="flex items-center gap-3">
          <div className="flex-1 h-3 bg-stone-200 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full transition-all duration-300 ${
                overallStatus === 'passed'
                  ? 'bg-green-500'
                  : overallStatus === 'failed'
                  ? 'bg-red-500'
                  : 'bg-gradient-to-r from-orange-400 to-amber-500'
              }`}
              style={{ width: `${(passedCount / totalChecks) * 100}%` }}
            />
          </div>
          <span className="text-sm font-semibold text-stone-700 tabular-nums">
            {passedCount}/{totalChecks}
          </span>
        </div>
      </div>

      {/* Status Display */}
      <div className={`rounded-xl p-4 ${status.bgColor} flex items-center gap-3`}>
        {status.icon}
        <span className={`font-bold text-lg ${status.textColor}`}>
          {status.text}
        </span>
      </div>

      {/* Action Buttons */}
      <div className="mt-auto flex flex-col gap-2">
        {overallStatus === 'idle' && (
          <button
            onClick={onStart}
            disabled={disabled}
            className="w-full py-4 px-6 bg-gradient-to-r from-orange-500 to-amber-500 text-white rounded-xl font-bold text-lg hover:from-orange-600 hover:to-amber-600 transition-all shadow-lg hover:shadow-xl disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          >
            <Camera size={24} />
            START ML VERIFICATION
          </button>
        )}

        {overallStatus === 'verifying' && (
          <div className="flex flex-col gap-2">
            {/* Snapshot Capture Button - prominent placement */}
            {onCaptureSnapshot && (
              <button
                onClick={onCaptureSnapshot}
                disabled={isCapturingSnapshot}
                className={`w-full py-4 px-6 rounded-xl font-bold text-lg transition-all flex items-center justify-center gap-2 ${
                  isCapturingSnapshot
                    ? 'bg-stone-400 text-white cursor-not-allowed'
                    : 'bg-gradient-to-r from-green-500 to-emerald-500 text-white hover:from-green-600 hover:to-emerald-600 shadow-lg hover:shadow-xl'
                }`}
              >
                {isCapturingSnapshot ? (
                  <>
                    <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    Detecting...
                  </>
                ) : (
                  <>
                    <Camera size={22} />
                    Capture & Detect
                  </>
                )}
              </button>
            )}

            <button
              onClick={onReset}
              className="w-full py-3 px-4 bg-stone-600 text-white rounded-xl font-semibold hover:bg-stone-700 transition-all flex items-center justify-center gap-2"
            >
              <RotateCcw size={18} />
              Reset
            </button>
          </div>
        )}

        {(overallStatus === 'passed' || overallStatus === 'failed' || overallStatus === 'warning') && (
          <button
            onClick={onReset}
            className="w-full py-3 px-6 bg-stone-600 text-white rounded-xl font-semibold hover:bg-stone-700 transition-all flex items-center justify-center gap-2"
          >
            <RotateCcw size={20} />
            Reset & Start New
          </button>
        )}

        {overallStatus === 'warning' && canOverride && (
          <button
            onClick={onOverride}
            className="w-full py-3 px-6 bg-orange-500 text-white rounded-xl font-semibold hover:bg-orange-600 transition-all flex items-center justify-center gap-2"
          >
            <Shield size={20} />
            Manager Override
          </button>
        )}
      </div>
    </div>
  );
}
