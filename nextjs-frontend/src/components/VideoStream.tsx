'use client';

import { useRef, useState, useEffect, useCallback } from 'react';
import { Play, Square, Settings, AlertTriangle, CheckCircle, Wifi, WifiOff, Camera } from 'lucide-react';
import { Spinner } from './Loading';
import { videoApi, detectionApi } from '@/lib/api';
import { dataURLtoFile } from '@/lib/utils';
import type { DetectionResult } from '@/types';

interface Detection {
  label: string;
  confidence: number;
  bbox: number[];
  is_violation: boolean;
}

interface StreamResult {
  detections: Detection[];
  violations: Detection[];
  timestamp: string;
  compliant: boolean;
}

interface VideoStreamProps {
  onDetection?: (result: StreamResult) => void;
  onError?: (error: string) => void;
  showSnapshotButton?: boolean;
  onSnapshotCapture?: (result: DetectionResult) => void;
  snapshotDetectEndpoint?: (imageData: string) => Promise<DetectionResult>;
}

export default function VideoStream({
  onDetection,
  onError,
  showSnapshotButton = true,
  onSnapshotCapture,
  snapshotDetectEndpoint,
}: VideoStreamProps) {
  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [isSnapshotCapturing, setIsSnapshotCapturing] = useState(false);
  const [streamAvailable, setStreamAvailable] = useState<boolean | null>(null);
  const [currentFrame, setCurrentFrame] = useState<string | null>(null);
  const [latestResult, setLatestResult] = useState<StreamResult | null>(null);
  const [showSettings, setShowSettings] = useState(false);
  const [maxFps, setMaxFps] = useState(15);
  const [frameCount, setFrameCount] = useState(0);
  const [connectionError, setConnectionError] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Check if streaming is available on mount
  useEffect(() => {
    checkStreamStatus();
  }, []);

  const checkStreamStatus = async () => {
    try {
      const status = await videoApi.getStatus();
      setStreamAvailable(status.available ?? false);
      if (!status.available) {
        setConnectionError(status.message ?? 'Video streaming not available');
      }
    } catch (err) {
      console.error('Failed to check stream status:', err);
      setStreamAvailable(false);
      setConnectionError('Failed to connect to server');
    }
  };

  const [logs, setLogs] = useState<string[]>([]);

  const addLog = (message: string) => {
    const timestamp = new Date().toLocaleTimeString();
    const logMessage = `[${timestamp}] ${message}`;
    console.log(logMessage);
    setLogs((prev) => [...prev.slice(-19), logMessage]); // Keep last 20 logs
  };

  const connect = useCallback(async () => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      addLog('WebSocket already connected, skipping...');
      return;
    }

    setIsConnecting(true);
    setConnectionError(null);
    addLog('Starting connection process...');

    try {
      // Start the video stream on the server
      addLog(`Calling /video/start with source=0, maxFps=${maxFps}...`);
      const startResponse = await videoApi.start(0, maxFps);
      addLog(`Server response: ${JSON.stringify(startResponse)}`);

      // Connect to WebSocket
      const wsUrl = videoApi.getWebSocketUrl();
      addLog(`Connecting to WebSocket: ${wsUrl}`);
      const ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        addLog('WebSocket connection opened successfully!');
        setIsConnected(true);
        setIsConnecting(false);
        setConnectionError(null);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          if (data.type === 'frame') {
            // 1. DIRECT DOM UPDATE for the image (Zero re-renders for the video)
            if (imgRef.current) {
              imgRef.current.src = `data:image/jpeg;base64,${data.frame}`;
            }

            setFrameCount((prev) => prev + 1);

            // 2. CONDITIONAL STATE UPDATE (Only re-render if results changed)
            // We compare simple features to decide if we need a full React update
            const newViolationsCount = data.result?.violations?.length || 0;
            const hasStatusChanged = (latestResult?.compliant !== data.result?.compliant) ||
              (latestResult?.violations?.length !== newViolationsCount);

            if (hasStatusChanged) {
              setLatestResult(data.result);
            }

            // 3. MINIMAL LOGGING
            if (frameCount % 60 === 0) { // Every ~2 seconds at 30fps
              console.log(`Stream heartbeat: Received frame #${frameCount + 1}`);
            }

            if (onDetection && data.result && hasStatusChanged) {
              onDetection(data.result);
            }
          } else if (data.type === 'heartbeat') {
            addLog('Received heartbeat from server');
          } else if (data.error) {
            addLog(`Server error: ${data.error}`);
            setConnectionError(data.error);
            onError?.(data.error);
          } else {
            addLog(`Unknown message type: ${JSON.stringify(data).slice(0, 100)}`);
          }
        } catch (err) {
          addLog(`Failed to parse message: ${err}`);
        }
      };

      ws.onerror = (error) => {
        addLog(`WebSocket error occurred: ${JSON.stringify(error)}`);
        setConnectionError('Connection error - check browser console');
        setIsConnecting(false);
      };

      ws.onclose = (event) => {
        addLog(`WebSocket closed. Code: ${event.code}, Reason: ${event.reason || 'none'}, Clean: ${event.wasClean}`);
        setIsConnected(false);
        setIsConnecting(false);
      };

      wsRef.current = ws;
      addLog('WebSocket object created, waiting for connection...');
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : String(err);
      addLog(`Failed to start stream: ${errorMessage}`);
      setConnectionError(`Failed to start: ${errorMessage}`);
      setIsConnecting(false);
      onError?.('Failed to start video stream');
    }
  }, [maxFps, onDetection, onError, frameCount]);

  const disconnect = useCallback(async () => {
    // Clear reconnect timeout
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    // Close WebSocket
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    // Stop the video stream on the server
    try {
      await videoApi.stop();
    } catch (err) {
      console.error('Failed to stop stream:', err);
    }

    setIsConnected(false);
    if (imgRef.current) imgRef.current.src = "";
    setLatestResult(null);
    setFrameCount(0);
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  const toggleStream = () => {
    if (isConnected) {
      disconnect();
    } else {
      connect();
    }
  };

  // Manual snapshot capture - captures current frame and runs full detection
  const captureSnapshot = useCallback(async () => {
    if (!currentFrame || isSnapshotCapturing) return;

    setIsSnapshotCapturing(true);
    addLog('Capturing snapshot for detection...');

    try {
      // Use the ref to get the current frame content
      const frameData = imgRef.current?.src;
      if (!frameData) {
        onError?.('No frame available to capture');
        return;
      }
      const file = dataURLtoFile(frameData, 'snapshot-capture.jpg');

      let result: DetectionResult;

      if (snapshotDetectEndpoint) {
        // Use custom endpoint if provided (still expects base64 for backwards compatibility)
        const base64Data = currentFrame.replace(/^data:image\/\w+;base64,/, '');
        result = await snapshotDetectEndpoint(base64Data);
      } else {
        // Use default detection API with File object
        result = await detectionApi.detect(file);
      }

      addLog(`Snapshot detection complete: ${result.detections?.summary?.safety_compliant ? 'Compliant' : 'Violations found'}`);
      onSnapshotCapture?.(result);

      return result;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : String(err);
      addLog(`Snapshot detection error: ${errorMessage}`);
      onError?.('Snapshot detection failed');
      return null;
    } finally {
      setIsSnapshotCapturing(false);
    }
  }, [currentFrame, isSnapshotCapturing, snapshotDetectEndpoint, onSnapshotCapture, onError]);

  // If streaming not available, show message
  if (streamAvailable === false) {
    return (
      <div className="space-y-4">
        <div className="aspect-video bg-stone-900 rounded-lg flex items-center justify-center">
          <div className="text-center text-white p-6">
            <WifiOff size={48} className="mx-auto mb-4 text-stone-500" />
            <p className="text-lg font-medium mb-2">Video Streaming Not Available</p>
            <p className="text-sm text-stone-400">
              {connectionError || 'Install the inference package on the server'}
            </p>
            <p className="text-xs text-stone-500 mt-2">
              Run: pip install inference
            </p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Video Display */}
      <div className="relative aspect-video bg-stone-900 rounded-lg overflow-hidden">
        {isConnected || isConnecting ? (
          <div className="w-full h-full relative">
            <img
              ref={imgRef}
              alt="Live video stream"
              className="w-full h-full object-contain"
              style={{ display: isConnected ? 'block' : 'none' }}
            />
            {!isConnected && isConnecting && (
              <div className="absolute inset-0 flex items-center justify-center bg-stone-900/50">
                <div className="text-center text-white">
                  <Spinner size="lg" className="mx-auto mb-4" />
                  <p>Connecting to video stream...</p>
                </div>
              </div>
            )}
          </div>
        ) : (
          <div className="flex items-center justify-center h-full text-white">
            <div className="text-center">
              <Play size={48} className="mx-auto mb-4 text-stone-500" />
              <p className="text-stone-400">Click Start to begin live detection</p>
            </div>
          </div>
        )}

        {/* Connection Status Overlay */}
        {isConnected && (
          <>
            {/* Live indicator */}
            <div className="absolute top-3 left-3 flex items-center gap-2 bg-red-600 text-white px-3 py-1 rounded-full text-sm font-medium">
              <span className="w-2 h-2 bg-white rounded-full animate-pulse" />
              LIVE
            </div>

            {/* Connection status */}
            <div className="absolute top-3 right-3 flex items-center gap-2 bg-green-600 text-white px-3 py-1 rounded-full text-sm">
              <Wifi size={14} />
              Connected
            </div>

            {/* Stats overlay */}
            <div className="absolute bottom-3 left-3 bg-black/60 text-white px-3 py-2 rounded-lg text-sm">
              <div>Frames: {frameCount}</div>
              {latestResult && <div>Time: {latestResult.timestamp}</div>}
            </div>

            {/* Compliance status overlay */}
            {latestResult && (
              <div
                className={`absolute bottom-3 right-3 flex items-center gap-2 px-4 py-2 rounded-lg text-white font-medium ${latestResult.compliant ? 'bg-green-600' : 'bg-red-600'
                  }`}
              >
                {latestResult.compliant ? (
                  <>
                    <CheckCircle size={20} />
                    Compliant
                  </>
                ) : (
                  <>
                    <AlertTriangle size={20} />
                    {latestResult.violations.length} Violation(s)
                  </>
                )}
              </div>
            )}
          </>
        )}

        {/* Error overlay */}
        {connectionError && !isConnected && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/80">
            <div className="text-center text-white p-4">
              <AlertTriangle size={48} className="mx-auto mb-4 text-red-500" />
              <p className="text-lg font-medium mb-2">Connection Error</p>
              <p className="text-sm text-stone-400">{connectionError}</p>
            </div>
          </div>
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
      </div>

      {/* Controls */}
      <div className="flex gap-2">
        <button
          onClick={toggleStream}
          disabled={isConnecting || streamAvailable === null}
          className={`flex-1 py-3 px-4 rounded-lg font-medium transition-colors flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed ${isConnected
              ? 'bg-red-600 text-white hover:bg-red-700'
              : 'bg-green-600 text-white hover:bg-green-700'
            }`}
        >
          {isConnecting ? (
            <>
              <Spinner size="sm" className="border-white/30 border-t-white" />
              Connecting...
            </>
          ) : isConnected ? (
            <>
              <Square size={18} />
              Stop Stream
            </>
          ) : (
            <>
              <Play size={18} />
              Start Live Stream
            </>
          )}
        </button>

        {/* Snapshot Capture Button */}
        {showSnapshotButton && (
          <button
            onClick={captureSnapshot}
            disabled={!isConnected || isSnapshotCapturing}
            className={`py-3 px-4 rounded-lg font-medium transition-colors flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed ${isSnapshotCapturing
                ? 'bg-stone-400 text-white'
                : 'bg-gradient-to-r from-blue-500 to-indigo-500 text-white hover:from-blue-600 hover:to-indigo-600'
              }`}
            title="Capture snapshot for full detection"
          >
            {isSnapshotCapturing ? (
              <>
                <Spinner size="sm" className="border-white/30 border-t-white" />
                Detecting...
              </>
            ) : (
              <>
                <Camera size={18} />
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
          <h4 className="font-medium text-stone-800">Stream Settings</h4>

          <div>
            <label className="block text-sm font-medium text-stone-700 mb-2">
              Max FPS: {maxFps}
            </label>
            <input
              type="range"
              min={5}
              max={30}
              step={5}
              value={maxFps}
              onChange={(e) => setMaxFps(Number(e.target.value))}
              disabled={isConnected}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-stone-500 mt-1">
              <span>5 (Low)</span>
              <span>30 (High)</span>
            </div>
          </div>

          <p className="text-xs text-stone-500">
            Higher FPS provides smoother video but uses more bandwidth.
            Settings can only be changed when stream is stopped.
          </p>
        </div>
      )}

      {/* Detection Summary */}
      {latestResult && !isConnected && (
        <div
          className={`p-4 rounded-lg ${latestResult.compliant ? 'bg-green-50' : 'bg-red-50'
            }`}
        >
          <div className="flex items-center gap-3">
            {latestResult.compliant ? (
              <CheckCircle className="text-green-500" size={24} />
            ) : (
              <AlertTriangle className="text-red-500" size={24} />
            )}
            <div>
              <p
                className={`font-medium ${latestResult.compliant ? 'text-green-800' : 'text-red-800'
                  }`}
              >
                Last Detection:{' '}
                {latestResult.compliant ? 'Compliant' : 'Violations Found'}
              </p>
              <p className="text-sm text-stone-600">
                {latestResult.detections.length} detections,{' '}
                {latestResult.violations.length} violations
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
