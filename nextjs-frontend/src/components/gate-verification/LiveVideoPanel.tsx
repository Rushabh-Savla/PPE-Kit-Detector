'use client';

import { useRef, useState, useEffect, useCallback } from 'react';
import { Play, Square, Wifi, WifiOff, Camera, User, CheckCircle, AlertCircle, Video } from 'lucide-react';
import { Spinner } from '../Loading';

interface LiveVideoPanelProps {
  isActive: boolean;
  onFrameCapture?: (frame: string) => void;
  onSnapshotCapture?: (frame: string) => void;
  identifiedWorker?: { name: string; employee_id: string } | null;
  attendanceMarked?: boolean;
  showSnapshotButton?: boolean;
  isCapturing?: boolean;
  onResult?: (result: any) => void;
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export function LiveVideoPanel({
  isActive,
  onFrameCapture,
  onSnapshotCapture,
  identifiedWorker,
  attendanceMarked,
  showSnapshotButton = false,
  isCapturing = false,
  onResult,
}: LiveVideoPanelProps) {
  const [isStreaming, setIsStreaming] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentFrame, setCurrentFrame] = useState<string | null>(null);
  const [useBrowserCamera, setUseBrowserCamera] = useState(false);
  const [browserCameraActive, setBrowserCameraActive] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);

  const onFrameCaptureRef = useRef(onFrameCapture);
  onFrameCaptureRef.current = onFrameCapture;

  const onResultRef = useRef(onResult);
  onResultRef.current = onResult;

  // Get WebSocket URL
  const getWebSocketUrl = () => {
    try {
      // Use the URL constructor to handle parsing robustly
      const baseUrl = new URL(API_URL);
      const wsProtocol = baseUrl.protocol === 'https:' ? 'wss:' : 'ws:';
      const host = baseUrl.host; // includes port
      return `${wsProtocol}//${host}/ws/video`;
    } catch (e) {
      // Fallback
      const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const apiHost = API_URL.replace(/^https?:\/\//, '').replace(/\/$/, '');
      return `${wsProtocol}//${apiHost}/ws/video`;
    }
  };

  // Connect to WebSocket
  const connectWebSocket = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    const wsUrl = getWebSocketUrl();
    console.log('Connecting to WebSocket:', wsUrl);

    try {
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log('WebSocket connected');
        setIsConnected(true);
        setError(null);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          if (data.type === 'frame' && data.frame) {
            const frameData = `data:image/jpeg;base64,${data.frame}`;
            setCurrentFrame(frameData);

            // Send frame to parent for detection if active
            if (isActive) {
              onFrameCaptureRef.current?.(frameData);

              // Forward detection results if available in the WebSocket message
              if (data.result) {
                onResultRef.current?.(data.result);
              }
            }
          } else if (data.type === 'heartbeat') {
            // Heartbeat received, connection is alive
          } else if (data.error) {
            console.error('WebSocket error:', data.error);
            setError(data.error);
          }
        } catch (e) {
          console.error('Failed to parse WebSocket message:', e);
        }
      };

      ws.onerror = (event) => {
        console.error('WebSocket error:', event);
        setError(`Connection error to ${wsUrl}. Check if server is running on port 8000.`);
        setIsConnected(false);
      };

      ws.onclose = () => {
        console.log('WebSocket closed');
        setIsConnected(false);
        wsRef.current = null;

        // Attempt to reconnect if streaming is still enabled
        if (isStreaming) {
          reconnectTimeoutRef.current = setTimeout(() => {
            console.log('Attempting to reconnect...');
            connectWebSocket();
          }, 2000);
        }
      };
    } catch (e) {
      console.error('Failed to create WebSocket:', e);
      setError('Failed to connect');
    }
  }, [isStreaming, isActive]);

  // Disconnect WebSocket
  const disconnectWebSocket = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    setIsConnected(false);
    setCurrentFrame(null);
  }, []);

  // Start streaming
  const startStreaming = useCallback(() => {
    setIsStreaming(true);
    setError(null);
    connectWebSocket();
  }, [connectWebSocket]);

  // Stop streaming
  const stopStreaming = useCallback(() => {
    setIsStreaming(false);
    disconnectWebSocket();
  }, [disconnectWebSocket]);

  // Auto-start when verification becomes active
  useEffect(() => {
    if (isActive && !isStreaming) {
      startStreaming();
    }
  }, [isActive, isStreaming, startStreaming]);

  // Cleanup function that stops all camera/stream resources
  const cleanupAllResources = useCallback(() => {
    console.log('[LiveVideoPanel] Cleaning up all resources...');

    // Clear any pending reconnect
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    // Close WebSocket
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    // Stop browser camera - this is the critical part!
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(track => {
        console.log('[LiveVideoPanel] Stopping track:', track.kind, track.label);
        track.stop();
      });
      mediaStreamRef.current = null;
    }

    // Clear video element source
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    // Handle page unload/refresh - ensures camera stops even if component doesn't unmount cleanly
    const handleBeforeUnload = () => {
      cleanupAllResources();
    };

    window.addEventListener('beforeunload', handleBeforeUnload);

    return () => {
      window.removeEventListener('beforeunload', handleBeforeUnload);
      cleanupAllResources();
    };
  }, [cleanupAllResources]);

  // Also cleanup when navigating away (Next.js router)
  useEffect(() => {
    const handleRouteChange = () => {
      cleanupAllResources();
    };

    // Listen for visibility change (tab switch, minimize)
    const handleVisibilityChange = () => {
      if (document.visibilityState === 'hidden') {
        // Optional: stop camera when tab is hidden to save resources
        // Uncomment if you want this behavior:
        // cleanupAllResources();
      }
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);

    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, [cleanupAllResources]);

  // Browser camera functions
  const startBrowserCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: 640, height: 480 }
      });
      mediaStreamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      setBrowserCameraActive(true);
      setError(null);
      console.log('[LiveVideoPanel] Browser camera started');
    } catch (err) {
      console.error('[LiveVideoPanel] Failed to start browser camera:', err);
      setError('Failed to access camera. Please allow camera permissions.');
    }
  }, []);

  const stopBrowserCamera = useCallback(() => {
    console.log('[LiveVideoPanel] Stopping browser camera...');
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(track => {
        console.log('[LiveVideoPanel] Stopping track:', track.kind, track.label);
        track.stop();
      });
      mediaStreamRef.current = null;
    }
    // Clear the video element source
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setBrowserCameraActive(false);
  }, []);

  const captureBrowserFrame = useCallback((): string | null => {
    if (!videoRef.current || !canvasRef.current) return null;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    if (!ctx) return null;

    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    ctx.drawImage(video, 0, 0);

    return canvas.toDataURL('image/jpeg', 0.8);
  }, []);

  const toggleStream = () => {
    if (useBrowserCamera) {
      if (browserCameraActive) {
        stopBrowserCamera();
        setIsStreaming(false);
      } else {
        startBrowserCamera();
        setIsStreaming(true);
      }
    } else {
      if (isStreaming) {
        stopStreaming();
      } else {
        startStreaming();
      }
    }
  };

  // Handle snapshot capture
  const handleSnapshotCapture = useCallback(async () => {
    let frameToCapture = currentFrame;

    // If using browser camera, capture frame from video element
    if (useBrowserCamera && browserCameraActive) {
      frameToCapture = captureBrowserFrame();
      if (frameToCapture) {
        setCurrentFrame(frameToCapture);
      }
    }

    if (frameToCapture && onSnapshotCapture) {
      console.log('[LiveVideoPanel] Capturing snapshot with frame data, size:', frameToCapture.length);
      try {
        // Call the snapshot handler (may be async)
        await onSnapshotCapture(frameToCapture);
      } catch (err) {
        console.error('[LiveVideoPanel] Snapshot capture error:', err);
        setError(`Detection failed: ${err instanceof Error ? err.message : String(err)}`);
      }
    } else {
      console.log('[LiveVideoPanel] Cannot capture snapshot:', {
        hasFrame: !!frameToCapture,
        hasCallback: !!onSnapshotCapture,
        useBrowserCamera,
        browserCameraActive
      });
      // Show error to user
      if (!frameToCapture) {
        setError('No video frame available. Please start the camera first.');
      }
    }
  }, [currentFrame, onSnapshotCapture, useBrowserCamera, browserCameraActive, captureBrowserFrame]);

  return (
    <div className="flex flex-col gap-3">
      {/* Hidden canvas for browser camera capture */}
      <canvas ref={canvasRef} className="hidden" />

      {/* Video Display */}
      <div className="relative aspect-video bg-stone-900 rounded-xl overflow-hidden">
        {/* Browser Camera Video Element */}
        {useBrowserCamera && (
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className={`w-full h-full object-cover ${browserCameraActive ? 'block' : 'hidden'}`}
          />
        )}

        {/* WebSocket Frame Display */}
        {!useBrowserCamera && currentFrame ? (
          <img
            src={currentFrame}
            alt="Live video feed"
            className="w-full h-full object-cover"
          />
        ) : null}

        {/* Placeholder when no video */}
        {!browserCameraActive && !currentFrame && (
          <div className="absolute inset-0 flex items-center justify-center">
            {isStreaming ? (
              <div className="text-center text-white">
                <Spinner size="lg" className="mx-auto mb-4" />
                <p>Connecting to camera...</p>
              </div>
            ) : (
              <div className="text-center text-white">
                <Camera size={48} className="mx-auto mb-4 text-stone-400" />
                <p className="text-stone-300">Click Start to begin verification</p>
              </div>
            )}
          </div>
        )}

        {/* Overlays */}
        {(isStreaming || browserCameraActive) && (
          <>
            {/* Live badge */}
            <div className="absolute top-3 left-3 flex items-center gap-2 bg-red-600 text-white px-3 py-1 rounded-full text-sm font-medium">
              <span className="w-2 h-2 bg-white rounded-full animate-pulse" />
              LIVE {useBrowserCamera ? '(Browser)' : '(Server)'}
            </div>

            {/* Connection Status */}
            <div className={`absolute top-3 right-3 flex items-center gap-2 px-3 py-1 rounded-full text-sm ${(useBrowserCamera ? browserCameraActive : isConnected) ? 'bg-green-600/90 text-white' : 'bg-yellow-600/90 text-white'
              }`}>
              {(useBrowserCamera ? browserCameraActive : isConnected) ? <Wifi size={14} /> : <WifiOff size={14} />}
              {(useBrowserCamera ? browserCameraActive : isConnected) ? 'Connected' : 'Reconnecting...'}
            </div>
          </>
        )}

        {/* Scanning indicator when active */}
        {isActive && isStreaming && (
          <div className="absolute inset-0 border-4 border-orange-500 animate-pulse rounded-xl pointer-events-none" />
        )}

        {/* Worker Identified Notification */}
        {identifiedWorker && (
          <div className="absolute bottom-3 left-3 right-3 bg-green-600/95 text-white p-3 rounded-lg">
            <div className="flex items-center gap-3">
              <div className="w-12 h-12 rounded-full bg-white/20 flex items-center justify-center">
                <User size={28} />
              </div>
              <div className="flex-1">
                <p className="font-bold text-lg">{identifiedWorker.name}</p>
                <p className="text-sm text-green-100">ID: {identifiedWorker.employee_id}</p>
              </div>
              {attendanceMarked && (
                <div className="flex flex-col items-center gap-1 bg-white/20 px-3 py-2 rounded-lg">
                  <CheckCircle size={24} />
                  <span className="text-xs font-semibold">ATTENDANCE</span>
                  <span className="text-xs">MARKED</span>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Error display */}
        {error && (
          <div className="absolute bottom-3 left-3 right-3 bg-red-600/95 text-white p-3 rounded-lg">
            <div className="flex items-center gap-2">
              <AlertCircle size={20} />
              <span>{error}</span>
            </div>
          </div>
        )}
      </div>

      {/* Camera Mode Toggle */}
      <div className="flex gap-2 p-1 bg-stone-100 rounded-lg">
        <button
          onClick={() => {
            if (!useBrowserCamera) return; // Already on server mode
            stopBrowserCamera();
            setCurrentFrame(null);
            setUseBrowserCamera(false);
          }}
          className={`flex-1 py-2 px-3 rounded-md text-sm font-medium transition-colors flex items-center justify-center gap-2 ${!useBrowserCamera
            ? 'bg-white text-orange-600 shadow-sm'
            : 'text-stone-600 hover:text-stone-800'
            }`}
        >
          <Wifi size={16} />
          Server Camera
        </button>
        <button
          onClick={() => {
            if (useBrowserCamera) return; // Already on browser mode
            disconnectWebSocket();
            setIsStreaming(false);
            setCurrentFrame(null);
            setUseBrowserCamera(true);
          }}
          className={`flex-1 py-2 px-3 rounded-md text-sm font-medium transition-colors flex items-center justify-center gap-2 ${useBrowserCamera
            ? 'bg-white text-orange-600 shadow-sm'
            : 'text-stone-600 hover:text-stone-800'
            }`}
        >
          <Video size={16} />
          Browser Camera
        </button>
      </div>

      {/* Controls */}
      <div className="flex gap-2">
        <button
          onClick={toggleStream}
          className={`flex-1 py-3 px-4 rounded-lg font-medium transition-all flex items-center justify-center gap-2 ${(isStreaming || browserCameraActive)
            ? 'bg-red-600 text-white hover:bg-red-700'
            : 'bg-gradient-to-r from-orange-500 to-amber-500 text-white hover:from-orange-600 hover:to-amber-600'
            }`}
        >
          {(isStreaming || browserCameraActive) ? (
            <>
              <Square size={18} />
              Stop Camera
            </>
          ) : (
            <>
              <Play size={18} />
              Start Camera
            </>
          )}
        </button>

        {/* Snapshot Button - show when streaming/browser camera active */}
        {showSnapshotButton && (isStreaming || browserCameraActive) && (
          <button
            onClick={handleSnapshotCapture}
            disabled={isCapturing}
            className={`py-3 px-4 rounded-lg font-medium transition-all flex items-center justify-center gap-2 ${isCapturing
              ? 'bg-stone-400 text-white cursor-not-allowed'
              : 'bg-gradient-to-r from-blue-500 to-indigo-500 text-white hover:from-blue-600 hover:to-indigo-600'
              }`}
            title="Capture snapshot for detection"
          >
            {isCapturing ? (
              <>
                <Spinner size="sm" className="text-white" />
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
      </div>
    </div>
  );
}
