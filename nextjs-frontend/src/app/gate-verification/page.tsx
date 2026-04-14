'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import { useRouter } from 'next/navigation';
import { ScanLine, ChevronDown, Volume2, VolumeX, Maximize2, CheckCircle, Terminal, User, RotateCcw } from 'lucide-react';
import AppLayout from '@/components/AppLayout';
import {
  GateVerificationLayout,
  LiveVideoPanel,
  AnimatedWorkerFigure,
  WorkerFigureCompact,
  StatusPanel,
  OverrideModal,
} from '@/components/gate-verification';
import { useGateVerification } from '@/hooks/useGateVerification';
import { useAuthStore } from '@/lib/store';
import { mineApi, gateEntryApi } from '@/lib/api';
import type { Mine, Gate, Worker } from '@/types';

export default function GateVerificationPage() {
  const router = useRouter();
  const { isAuthenticated, hasMinRole, getMineIds } = useAuthStore();

  // Selection state
  const [mines, setMines] = useState<Mine[]>([]);
  const [selectedMineId, setSelectedMineId] = useState<string | null>(null);
  const [selectedGateId, setSelectedGateId] = useState<string | null>(null);
  const [gates, setGates] = useState<Gate[]>([]);

  // UI state
  const [showOverrideModal, setShowOverrideModal] = useState(false);
  const [soundEnabled, setSoundEnabled] = useState(true);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [showSuccessAnimation, setShowSuccessAnimation] = useState(false);
  const [showDebugLog, setShowDebugLog] = useState(true); // Show debug log by default for troubleshooting
  const [attendanceNotification, setAttendanceNotification] = useState<string | null>(null);

  // Ref to store current frame for snapshot capture from StatusPanel
  const currentFrameRef = useRef<string | null>(null);

  // Verification hook
  const verification = useGateVerification({
    gateId: selectedGateId,
    mineId: selectedMineId,
    onVerificationComplete: (success, worker) => {
      if (success) {
        playSound('success');
        // Play gate-open sound when verification is successful
        setTimeout(() => playSound('gate-open'), 300);
        setShowSuccessAnimation(true);
        setTimeout(() => setShowSuccessAnimation(false), 3000);
      } else {
        playSound('error');
      }
    },
    onGateOpen: () => {
      // Gate open signal - play gate-open sound
      playSound('gate-open');
      console.log('Gate opening...');
    },
    onWorkerIdentified: (worker) => {
      playSound('scan');
      console.log('Worker identified:', worker.name);
    },
    onAttendanceMarked: () => {
      setAttendanceNotification('Attendance marked successfully!');
      setTimeout(() => setAttendanceNotification(null), 3000);
    },
  });

  // Handler for frame capture from LiveVideoPanel - stores in ref for StatusPanel use
  const handleFrameCapture = useCallback((frame: string) => {
    currentFrameRef.current = frame;
    verification.setCurrentFrame(frame);
  }, [verification]);

  // Handler for snapshot capture from StatusPanel
  const handleSnapshotCapture = useCallback(() => {
    // Call captureSnapshot - it will handle error logging if no frame is available
    verification.captureSnapshot(currentFrameRef.current || undefined);
  }, [verification]);

  // Sound effects
  const playSound = useCallback((type: 'success' | 'error' | 'scan' | 'gate-open') => {
    if (!soundEnabled) return;

    // Play gate-open.mp3 for gate opening sound
    if (type === 'gate-open') {
      try {
        const audio = new Audio('/gate-open.mp3');
        audio.volume = 0.7;
        audio.play().catch(e => console.log('Audio play failed:', e));
      } catch (e) {
        console.log('Gate open sound not available');
      }
      return;
    }

    // Using Web Audio API for simple beeps
    try {
      const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
      const oscillator = audioContext.createOscillator();
      const gainNode = audioContext.createGain();

      oscillator.connect(gainNode);
      gainNode.connect(audioContext.destination);

      if (type === 'success') {
        oscillator.frequency.value = 800;
        gainNode.gain.value = 0.3;
      } else if (type === 'error') {
        oscillator.frequency.value = 300;
        gainNode.gain.value = 0.3;
      } else {
        oscillator.frequency.value = 600;
        gainNode.gain.value = 0.2;
      }

      oscillator.start();
      setTimeout(() => {
        oscillator.stop();
        audioContext.close();
      }, type === 'success' ? 200 : 150);
    } catch (e) {
      // Audio not supported
    }
  }, [soundEnabled]);

  // Load mines on mount
  useEffect(() => {
    const loadMines = async () => {
      try {
        const userMineIds = getMineIds();
        const { mines: allMines } = await mineApi.list();

        // Filter mines based on user access and active status
        const accessibleMines = allMines.filter(
          (mine) => mine.is_active && (userMineIds.length === 0 || userMineIds.includes(mine.id))
        );

        setMines(accessibleMines);

        // Don't auto-select - let user choose or use "Any" option
        // if (accessibleMines.length > 0 && !selectedMineId) {
        //   setSelectedMineId(accessibleMines[0].id);
        // }
      } catch (error) {
        console.error('Failed to load mines:', error);
      }
    };

    if (isAuthenticated) {
      loadMines();
    }
  }, [isAuthenticated, getMineIds, selectedMineId]);

  // Load gates when mine changes
  useEffect(() => {
    if (selectedMineId) {
      const mine = mines.find((m) => m.id === selectedMineId);
      if (mine) {
        const activeGates = mine.gates.filter((g) => g.is_active && g.has_camera);
        setGates(activeGates);

        // Don't auto-select - let user choose or use "Any" option
        // if (activeGates.length > 0) {
        //   setSelectedGateId(activeGates[0].id);
        // } else {
        //   setSelectedGateId(null);
        // }
      }
    }
  }, [selectedMineId, mines]);

  // Handle override
  const handleOverride = async (reason: string) => {
    try {
      // In a real implementation, you would call the override API
      // await gateEntryApi.override(entryId, reason);

      setShowOverrideModal(false);
      playSound('success');
      setShowSuccessAnimation(true);
      setTimeout(() => setShowSuccessAnimation(false), 3000);

      // Reset verification
      setTimeout(() => verification.resetVerification(), 3000);
    } catch (error) {
      console.error('Override failed:', error);
    }
  };

  // Toggle fullscreen
  const toggleFullscreen = () => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen();
      setIsFullscreen(true);
    } else {
      document.exitFullscreen();
      setIsFullscreen(false);
    }
  };

  // Check authorization
  const canAccess = hasMinRole('shift_incharge');
  const canOverride = hasMinRole('manager');

  if (!canAccess) {
    return (
      <AppLayout>
        <div className="flex items-center justify-center h-full">
          <div className="text-center">
            <ScanLine size={48} className="mx-auto mb-4 text-stone-400" />
            <h2 className="text-xl font-semibold text-stone-700">Access Denied</h2>
            <p className="text-stone-500 mt-2">
              You need Shift Incharge or higher role to access gate verification.
            </p>
          </div>
        </div>
      </AppLayout>
    );
  }

  return (
    <AppLayout>
      <div className="h-full flex flex-col">
        {/* Header */}
        <div className="flex flex-wrap items-center justify-between gap-4 mb-6">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-orange-500 to-amber-500 flex items-center justify-center text-white">
              <ScanLine size={24} />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-stone-800">Gate Verification</h1>
              <p className="text-sm text-stone-500">Two-factor PPE & Attendance Check</p>
            </div>
          </div>

          {/* Controls */}
          <div className="flex items-center gap-3">
            {/* Mine Selector */}
            <div className="relative">
              <select
                value={selectedMineId || ''}
                onChange={(e) => setSelectedMineId(e.target.value || null)}
                className="appearance-none pl-4 pr-10 py-2 bg-white border border-stone-200 rounded-lg text-sm font-medium text-stone-700 focus:outline-none focus:ring-2 focus:ring-orange-500"
              >
                <option value="">Any Mine (Auto)</option>
                {mines.map((mine) => (
                  <option key={mine.id} value={mine.id}>
                    {mine.name}
                  </option>
                ))}
              </select>
              <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 text-stone-400 pointer-events-none" size={16} />
            </div>

            {/* Gate Selector */}
            <div className="relative">
              <select
                value={selectedGateId || ''}
                onChange={(e) => setSelectedGateId(e.target.value || null)}
                className="appearance-none pl-4 pr-10 py-2 bg-white border border-stone-200 rounded-lg text-sm font-medium text-stone-700 focus:outline-none focus:ring-2 focus:ring-orange-500"
              >
                <option value="">Any Gate (Auto)</option>
                {gates.map((gate) => (
                  <option key={gate.id} value={gate.id}>
                    {gate.name}
                  </option>
                ))}
              </select>
              <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 text-stone-400 pointer-events-none" size={16} />
            </div>

            {/* Sound Toggle */}
            <button
              onClick={() => setSoundEnabled(!soundEnabled)}
              className={`p-2 rounded-lg transition-colors ${
                soundEnabled
                  ? 'bg-orange-100 text-orange-600'
                  : 'bg-stone-100 text-stone-400'
              }`}
              title={soundEnabled ? 'Mute sounds' : 'Enable sounds'}
            >
              {soundEnabled ? <Volume2 size={20} /> : <VolumeX size={20} />}
            </button>

            {/* Fullscreen Toggle */}
            <button
              onClick={toggleFullscreen}
              className="p-2 rounded-lg bg-stone-100 text-stone-600 hover:bg-stone-200 transition-colors"
              title="Toggle fullscreen"
            >
              <Maximize2 size={20} />
            </button>

            {/* Debug Log Toggle */}
            <button
              onClick={() => setShowDebugLog(!showDebugLog)}
              className={`p-2 rounded-lg transition-colors ${
                showDebugLog
                  ? 'bg-purple-100 text-purple-600'
                  : 'bg-stone-100 text-stone-400'
              }`}
              title="Toggle debug log"
            >
              <Terminal size={20} />
            </button>

            {/* Reset Button */}
            <button
              onClick={verification.resetVerification}
              className="flex items-center gap-2 px-4 py-2 bg-red-100 text-red-600 hover:bg-red-200 rounded-lg transition-colors font-medium"
              title="Reset verification"
            >
              <RotateCcw size={18} />
              Reset
            </button>
          </div>
        </div>

        {/* Main Content */}
        <div className="flex-1 min-h-0">
          <GateVerificationLayout
            videoPanel={
              <LiveVideoPanel
                isActive={verification.isTimerRunning}
                onFrameCapture={handleFrameCapture}
                onSnapshotCapture={verification.captureSnapshot}
                onResult={(result) => {
                  if (result?.detections) {
                    verification.processDetectionResponse({ success: true, detections: result.detections });
                  }
                }}
                identifiedWorker={verification.identifiedWorker}
                attendanceMarked={verification.attendanceMarked}
                showSnapshotButton={true}
                isCapturing={verification.isSnapshotCapturing}
              />
            }
            workerFigure={
              <AnimatedWorkerFigure items={verification.items} className="py-4" />
            }
            statusPanel={
              <StatusPanel
                overallStatus={verification.overallStatus}
                timeRemaining={verification.timeRemaining}
                isTimerRunning={verification.isTimerRunning}
                identifiedWorker={verification.identifiedWorker}
                passedCount={verification.passedCount}
                totalChecks={verification.totalChecks}
                canOverride={canOverride && verification.canOverride}
                attendanceMarked={verification.attendanceMarked}
                onStart={verification.startVerification}
                onReset={verification.resetVerification}
                onOverride={() => setShowOverrideModal(true)}
                onCaptureSnapshot={handleSnapshotCapture}
                isCapturingSnapshot={verification.isSnapshotCapturing}
                disabled={false}  // Gate selection is now optional
              />
            }
          />
        </div>

        {/* Success Animation Overlay */}
        {showSuccessAnimation && (
          <div className="fixed inset-0 z-50 flex items-center justify-center pointer-events-none">
            <div className="bg-green-500 text-white p-8 rounded-3xl shadow-2xl animate-bounce">
              <CheckCircle size={80} className="mx-auto mb-4" />
              <p className="text-3xl font-bold text-center">VERIFIED</p>
              <p className="text-center text-green-100 mt-2">Gate Opening...</p>
            </div>
          </div>
        )}

        {/* Override Modal */}
        <OverrideModal
          isOpen={showOverrideModal}
          onClose={() => setShowOverrideModal(false)}
          onConfirm={handleOverride}
          items={verification.items}
          passedCount={verification.passedCount}
          totalChecks={verification.totalChecks}
        />


        {/* Debug Log Panel */}
        {/* {showDebugLog && (
          <div className="fixed bottom-4 right-4 w-96 max-h-64 bg-stone-900 text-green-400 rounded-lg shadow-2xl overflow-hidden z-40">
            <div className="flex items-center justify-between px-3 py-2 bg-stone-800 border-b border-stone-700">
              <span className="text-xs font-semibold text-stone-300">Detection Log</span>
              <button
                onClick={() => setShowDebugLog(false)}
                className="text-stone-400 hover:text-white"
              >
                ×
              </button>
            </div>
            <div className="p-3 overflow-y-auto max-h-52 font-mono text-xs">
              {verification.detectionLog.length === 0 ? (
                <p className="text-stone-500">No logs yet. Start verification to see activity.</p>
              ) : (
                verification.detectionLog.map((log, index) => (
                  <p key={index} className="mb-1">{log}</p>
                ))
              )}
            </div>
          </div>
        )} */}

        {/* Attendance Notification */}
        {attendanceNotification && (
          <div className="fixed top-4 right-4 bg-green-500 text-white px-6 py-3 rounded-lg shadow-lg flex items-center gap-3 animate-bounce z-50">
            <CheckCircle size={24} />
            <span className="font-semibold">{attendanceNotification}</span>
          </div>
        )}
      </div>
    </AppLayout>
  );
}
