'use client';

import { useRef, useCallback, useState, useEffect } from 'react';
import Webcam from 'react-webcam';
import { Camera as CameraIcon, Play, Square, Settings, AlertTriangle, CheckCircle } from 'lucide-react';
import { Spinner } from './Loading';
import { dataURLtoFile } from '@/lib/utils';
import type { DetectionResult } from '@/types';

interface LiveCameraProps {
  onDetection: (result: DetectionResult) => void;
  onError?: (error: string) => void;
  detectEndpoint: (file: File) => Promise<DetectionResult>;
  intervalMs?: number;
  disabled?: boolean;
  showSnapshotButton?: boolean;
  onSnapshotCapture?: (result: DetectionResult) => void;
}

export default function LiveCamera({
  onDetection,
  onError,
  detectEndpoint,
  intervalMs = 2000,
  disabled = false,
  showSnapshotButton = true,
  onSnapshotCapture,
}: LiveCameraProps) {
  const webcamRef = useRef<Webcam>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  const [isLive, setIsLive] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isSnapshotCapturing, setIsSnapshotCapturing] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [lastResult, setLastResult] = useState<DetectionResult | null>(null);
  const [detectionInterval, setDetectionInterval] = useState(intervalMs);
  const [showSettings, setShowSettings] = useState(false);
  const [frameCount, setFrameCount] = useState(0);
  const [fps, setFps] = useState(0);

  // FPS calculation
  const fpsRef = useRef({ lastTime: Date.now(), frames: 0 });

  const captureAndDetect = useCallback(async () => {
    if (!webcamRef.current || isProcessing) return;

    const imageSrc = webcamRef.current.getScreenshot({
      width: 1280,
      height: 720,
    });

    if (!imageSrc) return;

    setIsProcessing(true);

    try {
      const file = dataURLtoFile(imageSrc, 'live-capture.png');
      const result = await detectEndpoint(file);

      setLastResult(result);
      setFrameCount(prev => prev + 1);
      onDetection(result);

      // Calculate FPS
      fpsRef.current.frames++;
      const now = Date.now();
      const elapsed = now - fpsRef.current.lastTime;
      if (elapsed >= 1000) {
        setFps(Math.round((fpsRef.current.frames * 1000) / elapsed));
        fpsRef.current = { lastTime: now, frames: 0 };
      }
    } catch (err) {
      console.error('Live detection error:', err);
      onError?.('Detection failed. Retrying...');
    } finally {
      setIsProcessing(false);
    }
  }, [isProcessing, detectEndpoint, onDetection, onError]);

  const startLiveDetection = useCallback(() => {
    if (intervalRef.current) return;

    setIsLive(true);
    setFrameCount(0);
    fpsRef.current = { lastTime: Date.now(), frames: 0 };

    // Immediate first detection
    captureAndDetect();

    // Set up interval for continuous detection
    intervalRef.current = setInterval(() => {
      captureAndDetect();
    }, detectionInterval);
  }, [captureAndDetect, detectionInterval]);

  const stopLiveDetection = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    setIsLive(false);
    setFps(0);
  }, []);

  // Clean up on unmount
  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, []);

  // Restart interval when detection interval changes
  useEffect(() => {
    if (isLive) {
      stopLiveDetection();
      startLiveDetection();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [detectionInterval]);

  const handleCameraError = () => {
    setCameraError('Unable to access camera. Please check permissions.');
    stopLiveDetection();
  };

  const toggleLive = () => {
    if (isLive) {
      stopLiveDetection();
    } else {
      startLiveDetection();
    }
  };

  // Manual snapshot capture - captures current frame and runs full detection
  const captureSnapshot = useCallback(async () => {
    if (!webcamRef.current || isSnapshotCapturing) return;

    const imageSrc = webcamRef.current.getScreenshot({
      width: 1280,
      height: 720,
    });

    if (!imageSrc) {
      onError?.('Failed to capture frame');
      return;
    }

    setIsSnapshotCapturing(true);

    try {
      const file = dataURLtoFile(imageSrc, 'snapshot-capture.png');
      const result = await detectEndpoint(file);

      setLastResult(result);
      onDetection(result);
      onSnapshotCapture?.(result);

      return result;
    } catch (err) {
      console.error('Snapshot detection error:', err);
      onError?.('Snapshot detection failed');
      return null;
    } finally {
      setIsSnapshotCapturing(false);
    }
  }, [isSnapshotCapturing, detectEndpoint, onDetection, onSnapshotCapture, onError]);

  return (
    <div className="space-y-4">
      {/* Camera View */}
      <div className="relative aspect-video bg-stone-900 rounded-lg overflow-hidden">
        {cameraError ? (
          <div className="flex items-center justify-center h-full text-white text-center p-4">
            <p>{cameraError}</p>
          </div>
        ) : (
          <>
            <Webcam
              ref={webcamRef}
              audio={false}
              screenshotFormat="image/png"
              mirrored={false}
              videoConstraints={{
                width: 1280,
                height: 720,
                facingMode: 'user',
              }}
              onUserMediaError={handleCameraError}
              className="w-full h-full object-cover"
            />

            {/* Live Overlay */}
            {isLive && (
              <>
                {/* Live indicator */}
                <div className="absolute top-3 left-3 flex items-center gap-2 bg-red-600 text-white px-3 py-1 rounded-full text-sm font-medium">
                  <span className="w-2 h-2 bg-white rounded-full animate-pulse" />
                  LIVE
                </div>

                {/* Stats overlay */}
                <div className="absolute top-3 right-3 bg-black/60 text-white px-3 py-2 rounded-lg text-sm">
                  <div>Frames: {frameCount}</div>
                  <div>FPS: {fps}</div>
                </div>

                {/* Processing indicator */}
                {isProcessing && (
                  <div className="absolute bottom-3 left-3 flex items-center gap-2 bg-orange-600 text-white px-3 py-1 rounded-full text-sm">
                    <Spinner size="sm" className="border-white/30 border-t-white" />
                    Detecting...
                  </div>
                )}

                {/* Compliance status overlay */}
                {lastResult && (
                  <div className={`absolute bottom-3 right-3 flex items-center gap-2 px-4 py-2 rounded-lg text-white font-medium ${lastResult.detections.summary.safety_compliant
                      ? 'bg-green-600'
                      : 'bg-red-600'
                    }`}>
                    {lastResult.detections.summary.safety_compliant ? (
                      <>
                        <CheckCircle size={20} />
                        Compliant
                      </>
                    ) : (
                      <>
                        <AlertTriangle size={20} />
                        {lastResult.detections.summary.total_violations} Violation(s)
                      </>
                    )}
                  </div>
                )}
              </>
            )}

            {/* Snapshot capturing indicator */}
            {isSnapshotCapturing && (
              <div className="absolute inset-0 bg-white/20 flex items-center justify-center">
                <div className="bg-blue-600 text-white px-6 py-3 rounded-xl flex items-center gap-3 shadow-lg">
                  <Spinner size="md" className="border-white/30 border-t-white" />
                  <span className="font-medium">Capturing & Detecting...</span>
                </div>
              </div>
            )}
          </>
        )}
      </div>

      {/* Controls */}
      <div className="flex gap-2">
        <button
          onClick={toggleLive}
          disabled={disabled || !!cameraError}
          className={`flex-1 py-3 px-4 rounded-lg font-medium transition-colors flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed ${isLive
              ? 'bg-red-600 text-white hover:bg-red-700'
              : 'bg-green-600 text-white hover:bg-green-700'
            }`}
        >
          {isLive ? (
            <>
              <Square size={18} />
              Stop Live Detection
            </>
          ) : (
            <>
              <Play size={18} />
              Start Live Detection
            </>
          )}
        </button>

        {/* Snapshot Capture Button */}
        {showSnapshotButton && (
          <button
            onClick={captureSnapshot}
            disabled={disabled || !!cameraError || isSnapshotCapturing}
            className={`py-3 px-4 rounded-lg font-medium transition-colors flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed ${isSnapshotCapturing
                ? 'bg-stone-400 text-white'
                : 'bg-gradient-to-r from-blue-500 to-indigo-500 text-white hover:from-blue-600 hover:to-indigo-600'
              }`}
            title="Capture snapshot for detection"
          >
            {isSnapshotCapturing ? (
              <>
                <Spinner size="sm" className="border-white/30 border-t-white" />
                Detecting...
              </>
            ) : (
              <>
                <CameraIcon size={18} />
                Capture & Detect
              </>
            )}
          </button>
        )}

        <button
          onClick={() => setShowSettings(!showSettings)}
          className="py-3 px-4 bg-stone-200 text-stone-700 rounded-lg font-medium hover:bg-stone-300 transition-colors"
        >
          <Settings size={18} />
        </button>
      </div>

      {/* Settings Panel */}
      {showSettings && (
        <div className="p-4 bg-stone-50 rounded-lg space-y-4">
          <h4 className="font-medium text-stone-800">Detection Settings</h4>

          <div>
            <label className="block text-sm font-medium text-stone-700 mb-2">
              Detection Interval: {detectionInterval / 1000}s
            </label>
            <input
              type="range"
              min={500}
              max={5000}
              step={250}
              value={detectionInterval}
              onChange={(e) => setDetectionInterval(Number(e.target.value))}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-stone-500 mt-1">
              <span>0.5s (Fast)</span>
              <span>5s (Slow)</span>
            </div>
          </div>

          <p className="text-xs text-stone-500">
            Faster intervals provide more real-time feedback but use more resources.
            Recommended: 1-2 seconds for balanced performance.
          </p>
        </div>
      )}

      {/* Last Detection Result Summary */}
      {lastResult && !isLive && (
        <div className={`p-4 rounded-lg ${lastResult.detections.summary.safety_compliant ? 'bg-green-50' : 'bg-red-50'
          }`}>
          <div className="flex items-center gap-3">
            {lastResult.detections.summary.safety_compliant ? (
              <CheckCircle className="text-green-500" size={24} />
            ) : (
              <AlertTriangle className="text-red-500" size={24} />
            )}
            <div>
              <p className={`font-medium ${lastResult.detections.summary.safety_compliant ? 'text-green-800' : 'text-red-800'
                }`}>
                Last Detection: {lastResult.detections.summary.safety_compliant ? 'Compliant' : 'Violations Found'}
              </p>
              <p className="text-sm text-stone-600">
                {lastResult.detections.summary.total_ppe_items} PPE items detected, {lastResult.detections.summary.total_violations} violations
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
