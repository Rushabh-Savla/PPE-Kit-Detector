'use client';

import { useState, useEffect, useCallback } from 'react';
import { Shield, AlertTriangle, CheckCircle, Clock, RefreshCw, HardHat, Eye, Hand, Radio, Camera as CameraIcon, Video } from 'lucide-react';
import AppLayout from '@/components/AppLayout';
import Camera from '@/components/Camera';
import LiveCamera from '@/components/LiveCamera';
import VideoStream from '@/components/VideoStream';
import { Card } from '@/components/Card';
import { Spinner } from '@/components/Loading';
import { detectionApi, violationsApi } from '@/lib/api';
import { formatTime } from '@/lib/utils';
import type { DetectionResult, PPEViolationRecord } from '@/types';

export default function PPEDetectionPage() {
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState<DetectionResult | null>(null);
  const [location, setLocation] = useState('');
  const [recentViolations, setRecentViolations] = useState<PPEViolationRecord[]>([]);
  const [capturedFile, setCapturedFile] = useState<File | null>(null);
  const [logViolations, setLogViolations] = useState(true);
  const [detectionMode, setDetectionMode] = useState<'manual' | 'live' | 'stream'>('manual');
  const [liveViolationCount, setLiveViolationCount] = useState(0);

  useEffect(() => {
    loadRecentViolations();
  }, []);

  const loadRecentViolations = async () => {
    try {
      const data = await violationsApi.getToday();
      setRecentViolations(data.violations.slice(0, 10));
    } catch (err) {
      console.error('Failed to load violations:', err);
    }
  };

  const handleCapture = (file: File) => {
    setCapturedFile(file);
    setResult(null);
  };

  const handleSubmit = async () => {
    if (!capturedFile) return;

    setIsProcessing(true);
    setResult(null);

    try {
      const response = await detectionApi.detectAndLog(capturedFile, logViolations, location || undefined);
      setResult(response);

      if (response.violations_logged) {
        loadRecentViolations();
      }
    } catch (err) {
      console.error('Detection failed:', err);
    } finally {
      setIsProcessing(false);
    }
  };

  const resetForm = () => {
    setCapturedFile(null);
    setResult(null);
  };

  // Live detection handler
  const handleLiveDetection = useCallback(async (file: File): Promise<DetectionResult> => {
    const response = await detectionApi.detectAndLog(file, logViolations, location || undefined);

    // Track violations in live mode
    if (response.detections.summary.total_violations > 0) {
      setLiveViolationCount(prev => prev + response.detections.summary.total_violations);
    }

    // Refresh violations list periodically during live mode
    if (response.violations_logged) {
      loadRecentViolations();
    }

    return response;
  }, [logViolations, location]);

  const handleLiveResult = useCallback((result: DetectionResult) => {
    setResult(result);
  }, []);

  // Handler for manual snapshot capture during live mode
  const handleSnapshotCapture = useCallback((result: DetectionResult) => {
    setResult(result);
    // Refresh violations if logged
    if (result.violations_logged) {
      loadRecentViolations();
    }
  }, []);

  const getPPEIcon = (label: string) => {
    const lowerLabel = label.toLowerCase();
    if (lowerLabel.includes('helmet')) return <HardHat size={16} />;
    if (lowerLabel.includes('goggles') || lowerLabel.includes('glasses')) return <Eye size={16} />;
    if (lowerLabel.includes('glove')) return <Hand size={16} />;
    return <Shield size={16} />;
  };

  return (
    <AppLayout>
      <div className="space-y-6">
        {/* Page Header */}
        <div>
          <h1 className="text-2xl font-bold text-stone-800">PPE Detection</h1>
          <p className="text-stone-500 mt-1">Detect personal protective equipment compliance</p>
        </div>

        {/* Mode Toggle */}
        <div className="flex flex-wrap gap-2 p-1 bg-stone-100 rounded-lg w-fit">
          <button
            onClick={() => {
              setDetectionMode('manual');
              setLiveViolationCount(0);
            }}
            className={`flex items-center gap-2 px-4 py-2 rounded-md font-medium transition-colors ${
              detectionMode === 'manual'
                ? 'bg-white text-orange-600 shadow-sm'
                : 'text-stone-600 hover:text-stone-800'
            }`}
          >
            <CameraIcon size={18} />
            Manual
          </button>
          <button
            onClick={() => {
              setDetectionMode('live');
              setResult(null);
              setCapturedFile(null);
            }}
            className={`flex items-center gap-2 px-4 py-2 rounded-md font-medium transition-colors ${
              detectionMode === 'live'
                ? 'bg-white text-orange-600 shadow-sm'
                : 'text-stone-600 hover:text-stone-800'
            }`}
          >
            <Radio size={18} />
            Live (Browser)
          </button>
          <button
            onClick={() => {
              setDetectionMode('stream');
              setResult(null);
              setCapturedFile(null);
              setLiveViolationCount(0);
            }}
            className={`flex items-center gap-2 px-4 py-2 rounded-md font-medium transition-colors ${
              detectionMode === 'stream'
                ? 'bg-white text-orange-600 shadow-sm'
                : 'text-stone-600 hover:text-stone-800'
            }`}
          >
            <Video size={18} />
            Live (Server)
          </button>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left Column - Camera and Detection */}
          <div className="space-y-4">
            {/* Manual Mode */}
            {detectionMode === 'manual' && (
              <Card title="Capture Image">
                <Camera onCapture={handleCapture} disabled={isProcessing} />

                {/* Options */}
                <div className="mt-4 space-y-3">
                  <div>
                    <label className="block text-sm font-medium text-stone-700 mb-1">
                      Location (optional)
                    </label>
                    <input
                      type="text"
                      value={location}
                      onChange={(e) => setLocation(e.target.value)}
                      placeholder="e.g., Entrance Gate A"
                      className="w-full"
                    />
                  </div>

                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={logViolations}
                      onChange={(e) => setLogViolations(e.target.checked)}
                      className="w-4 h-4 text-orange-600 rounded"
                    />
                    <span className="text-sm text-stone-700">Log violations to database</span>
                  </label>
                </div>

                {capturedFile && !result && (
                  <button
                    onClick={handleSubmit}
                    disabled={isProcessing}
                    className="w-full mt-4 py-3 px-4 bg-orange-600 text-white rounded-lg font-medium hover:bg-orange-700 transition-colors flex items-center justify-center gap-2 disabled:opacity-50"
                  >
                    {isProcessing ? (
                      <>
                        <Spinner size="sm" className="border-white/30 border-t-white" />
                        Analyzing...
                      </>
                    ) : (
                      <>
                        <Shield size={20} />
                        Detect PPE
                      </>
                    )}
                  </button>
                )}
              </Card>
            )}

            {/* Live Mode (Browser-side) */}
            {detectionMode === 'live' && (
              <Card
                title="Live PPE Detection (Browser)"
                description="Captures frames in browser, sends to API for detection. Use 'Capture & Detect' for manual snapshots."
              >
                <LiveCamera
                  onDetection={handleLiveResult}
                  detectEndpoint={handleLiveDetection}
                  intervalMs={2000}
                  showSnapshotButton={true}
                  onSnapshotCapture={handleSnapshotCapture}
                />

                {/* Live Mode Options */}
                <div className="mt-4 space-y-3 pt-4 border-t border-stone-100">
                  <div>
                    <label className="block text-sm font-medium text-stone-700 mb-1">
                      Location (optional)
                    </label>
                    <input
                      type="text"
                      value={location}
                      onChange={(e) => setLocation(e.target.value)}
                      placeholder="e.g., Entrance Gate A"
                      className="w-full"
                    />
                  </div>

                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={logViolations}
                      onChange={(e) => setLogViolations(e.target.checked)}
                      className="w-4 h-4 text-orange-600 rounded"
                    />
                    <span className="text-sm text-stone-700">Log violations to database</span>
                  </label>

                  {liveViolationCount > 0 && (
                    <div className="flex items-center gap-2 p-3 bg-red-50 rounded-lg">
                      <AlertTriangle className="text-red-500" size={20} />
                      <span className="text-red-700 font-medium">
                        {liveViolationCount} violations detected this session
                      </span>
                    </div>
                  )}
                </div>
              </Card>
            )}

            {/* Stream Mode (Server-side with bounding boxes) */}
            {detectionMode === 'stream' && (
              <Card
                title="Live PPE Detection (Server)"
                description="Real-time video stream with bounding boxes drawn on server. Use 'Capture & Detect' for full analysis."
              >
                <VideoStream
                  onDetection={(result) => {
                    if (result.violations.length > 0) {
                      setLiveViolationCount((prev) => prev + result.violations.length);
                      loadRecentViolations();
                    }
                  }}
                  onError={(error) => console.error('Stream error:', error)}
                  showSnapshotButton={true}
                  onSnapshotCapture={handleSnapshotCapture}
                />

                {liveViolationCount > 0 && (
                  <div className="mt-4 flex items-center gap-2 p-3 bg-red-50 rounded-lg">
                    <AlertTriangle className="text-red-500" size={20} />
                    <span className="text-red-700 font-medium">
                      {liveViolationCount} violations detected this session
                    </span>
                  </div>
                )}

                <div className="mt-4 p-3 bg-blue-50 rounded-lg">
                  <p className="text-sm text-blue-700">
                    <strong>Server-side streaming:</strong> Video is processed on the server using
                    Roboflow InferencePipeline. Bounding boxes are drawn directly on the video frames
                    for smooth, real-time visualization.
                  </p>
                </div>
              </Card>
            )}

            {/* Detection Result */}
            {result && (
              <Card title="Detection Result">
                {/* Result Image */}
                <img
                  src={result.image}
                  alt="Detection result"
                  className="w-full rounded-lg mb-4"
                />

                {/* Compliance Status */}
                <div className={`p-4 rounded-lg mb-4 ${
                  result.detections.summary.safety_compliant ? 'bg-green-50' : 'bg-red-50'
                }`}>
                  <div className="flex items-center gap-3">
                    {result.detections.summary.safety_compliant ? (
                      <CheckCircle className="text-green-500" size={24} />
                    ) : (
                      <AlertTriangle className="text-red-500" size={24} />
                    )}
                    <div>
                      <p className={`font-medium ${
                        result.detections.summary.safety_compliant ? 'text-green-800' : 'text-red-800'
                      }`}>
                        {result.detections.summary.safety_compliant ? 'Safety Compliant' : 'Violations Detected'}
                      </p>
                      <p className={`text-sm ${
                        result.detections.summary.safety_compliant ? 'text-green-600' : 'text-red-600'
                      }`}>
                        {result.detections.summary.total_violations} violation(s) found
                      </p>
                    </div>
                  </div>
                </div>

                {/* PPE Detected */}
                {result.detections.summary.total_ppe_items > 0 && (
                  <div className="mb-4">
                    <h4 className="text-sm font-medium text-stone-700 mb-2">PPE Detected:</h4>
                    <div className="flex flex-wrap gap-2">
                      {Object.entries(result.detections.summary.ppe_detected).map(([item, count]) => (
                        <span key={item} className="badge badge-success flex items-center gap-1">
                          {getPPEIcon(item)}
                          {item} ({count})
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {/* Violations */}
                {result.detections.summary.total_violations > 0 && (
                  <div className="mb-4">
                    <h4 className="text-sm font-medium text-stone-700 mb-2">Violations:</h4>
                    <div className="flex flex-wrap gap-2">
                      {Object.entries(result.detections.summary.violations).map(([item, count]) => (
                        <span key={item} className="badge badge-danger flex items-center gap-1">
                          <AlertTriangle size={14} />
                          {item} ({count})
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {/* Identified Persons */}
                {result.detections.summary.identified_persons.length > 0 && (
                  <div className="mb-4">
                    <h4 className="text-sm font-medium text-stone-700 mb-2">Identified Persons:</h4>
                    <div className="flex flex-wrap gap-2">
                      {result.detections.summary.identified_persons.map((person) => (
                        <span key={person} className="badge badge-info">
                          {person}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {result.violations_logged && (
                  <p className="text-sm text-green-600 mb-4">
                    ✓ Violation logged to database
                  </p>
                )}

                {detectionMode === 'manual' && (
                  <button
                    onClick={resetForm}
                    className="w-full py-2 px-4 bg-stone-100 text-stone-700 rounded-lg font-medium hover:bg-stone-200 transition-colors flex items-center justify-center gap-2"
                  >
                    <RefreshCw size={18} />
                    New Detection
                  </button>
                )}

                {detectionMode !== 'manual' && (
                  <p className="text-xs text-stone-500 text-center">
                    Results update automatically during live detection
                  </p>
                )}
              </Card>
            )}
          </div>

          {/* Right Column - Recent Violations */}
          <Card title="Today's Violations" description="Recent PPE violations detected">
            <div className="space-y-3 max-h-[600px] overflow-y-auto">
              {recentViolations.length === 0 ? (
                <p className="text-center text-stone-500 py-8">No violations today</p>
              ) : (
                recentViolations.map((violation) => (
                  <div
                    key={violation.id}
                    className="p-4 bg-stone-50 rounded-lg"
                  >
                    <div className="flex items-start justify-between mb-2">
                      <div>
                        <p className="font-medium text-stone-800">
                          {violation.employee_name || 'Unknown Person'}
                        </p>
                        {violation.location && (
                          <p className="text-xs text-stone-500">{violation.location}</p>
                        )}
                      </div>
                      <div className="flex items-center gap-1 text-stone-500 text-sm">
                        <Clock size={14} />
                        {formatTime(violation.timestamp)}
                      </div>
                    </div>
                    <div className="flex flex-wrap gap-1">
                      {violation.violations.map((v, i) => (
                        <span key={i} className="badge badge-danger text-xs">
                          {v.label}
                        </span>
                      ))}
                    </div>
                  </div>
                ))
              )}
            </div>

            <button
              onClick={loadRecentViolations}
              className="w-full mt-4 py-2 px-4 bg-stone-100 text-stone-700 rounded-lg font-medium hover:bg-stone-200 transition-colors flex items-center justify-center gap-2"
            >
              <RefreshCw size={18} />
              Refresh
            </button>
          </Card>
        </div>

        {/* PPE Legend */}
        <Card title="PPE Requirements" description="Required safety equipment">
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
            {[
              { name: 'Helmet', icon: <HardHat size={24} />, color: 'bg-blue-50 text-blue-600' },
              { name: 'Vest', icon: <Shield size={24} />, color: 'bg-orange-50 text-orange-600' },
              { name: 'Gloves', icon: <Hand size={24} />, color: 'bg-green-50 text-green-600' },
              { name: 'Goggles', icon: <Eye size={24} />, color: 'bg-purple-50 text-purple-600' },
              { name: 'Mask', icon: <Shield size={24} />, color: 'bg-pink-50 text-pink-600' },
              { name: 'Safety Shoes', icon: <Shield size={24} />, color: 'bg-yellow-50 text-yellow-600' },
            ].map((item) => (
              <div
                key={item.name}
                className={`p-4 rounded-lg ${item.color} flex flex-col items-center gap-2`}
              >
                {item.icon}
                <span className="text-sm font-medium">{item.name}</span>
              </div>
            ))}
          </div>
        </Card>
      </div>
    </AppLayout>
  );
}
