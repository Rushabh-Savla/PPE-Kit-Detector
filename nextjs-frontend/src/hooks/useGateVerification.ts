'use client';

import { useEffect, useCallback, useRef, useState } from 'react';
import { useGateVerificationStore } from '@/lib/store';
import { gateEntryApi, attendanceApi, workerApi } from '@/lib/api';
import type { VerificationItemType, Worker } from '@/types';

interface UseGateVerificationOptions {
  gateId: string | null;
  mineId: string | null;
  onVerificationComplete?: (success: boolean, worker: Worker | null) => void;
  onGateOpen?: () => void;
  onWorkerIdentified?: (worker: Worker) => void;
  onAttendanceMarked?: () => void;
}

// PPE Detection Service URL (ppe_detection.py - runs on port 8002)
// DISABLED - service not running
// const PPE_DETECTION_URL = process.env.NEXT_PUBLIC_PPE_DETECTION_URL || 'http://localhost:8002';

// PPE Detection response from the new service - DISABLED
// interface PPEDetectionResponse { ... }

interface DetectionResponse {
  success: boolean;
  message?: string;
  image?: string;
  detections?: {
    ppe?: Array<{
      label: string;
      confidence: number;
      is_violation: boolean;
      bbox?: number[];
    }>;
    faces?: Array<{
      employee_id: string | null;
      name: string | null;
      confidence: number;
      bbox?: number[];
    }>;
    violations?: Array<{
      label: string;
      confidence: number;
    }>;
    summary?: {
      ppe_detected: Record<string, number>;
      violations: Record<string, number>;
      total_ppe_items: number;
      total_violations: number;
      faces_detected: number;
      identified_persons: string[];
      identified_names?: string[];
      safety_compliant: boolean;
    };
  };
  entry?: {
    id?: string;
    worker_id?: string;
    worker_name?: string;
    employee_id?: string;
    status?: string;
  };
}

// Map PPE labels to verification items (supports both lowercase and normalized formats)
const PPE_LABEL_MAP: Record<string, VerificationItemType> = {
  // Lowercase versions
  helmet: 'helmet',
  hardhat: 'helmet',
  'hard hat': 'helmet',
  'hard-hat': 'helmet',
  vest: 'vest',
  'safety vest': 'vest',
  'safety-vest': 'vest',
  'high-vis vest': 'vest',
  'hi-vis': 'vest',
  shoes: 'shoes',
  'safety shoes': 'shoes',
  'safety-shoes': 'shoes',
  boots: 'shoes',
  'safety boots': 'shoes',
  'safety-boots': 'shoes',
  // Normalized versions from backend (capitalized)
  'Helmet': 'helmet',
  'Hardhat': 'helmet',
  'Vest': 'vest',
  'Safety Vest': 'vest',
  'Safety Shoes': 'shoes',
  'Shoes': 'shoes',
  'Boots': 'shoes',
  'Safety Boots': 'shoes',
};

export function useGateVerification({
  gateId,
  mineId,
  onVerificationComplete,
  onGateOpen,
  onWorkerIdentified,
  onAttendanceMarked,
}: UseGateVerificationOptions) {
  const store = useGateVerificationStore();
  const isDetectingRef = useRef(false);
  const timerIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const detectionIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const frameRef = useRef<string | null>(null);
  const [detectionLog, setDetectionLog] = useState<string[]>([]);
  const [isSnapshotCapturing, setIsSnapshotCapturing] = useState(false);

  // Add to detection log
  const addLog = useCallback((message: string) => {
    const timestamp = new Date().toLocaleTimeString();
    setDetectionLog(prev => [...prev.slice(-9), `[${timestamp}] ${message}`]);
    console.log(`[GateVerification] ${message}`);
  }, []);

  // Start verification session (ML-based only)
  const startVerification = useCallback(async () => {
    // Gate and mine IDs are now optional - will use defaults if not provided
    if (!gateId || !mineId) {
      console.log('[useGateVerification] Starting verification without gate/mine selection - will use defaults');
      addLog('Starting verification (using default gate)...');
    }

    setDetectionLog([]);
    addLog('Verification started (ML-based)');
    store.startVerification(gateId || 'default', mineId || 'default');
  }, [gateId, mineId, store, addLog]);

  // Reset verification
  const resetVerification = useCallback(() => {
    store.resetVerification();
    frameRef.current = null;
    setDetectionLog([]);
    addLog('Verification reset');
  }, [store, addLog]);

  // Set current video frame (called from LiveVideoPanel)
  const setCurrentFrame = useCallback((frame: string) => {
    frameRef.current = frame;
  }, []);

  // Process detection response
  const processDetectionResponse = useCallback(async (response: DetectionResponse) => {
    if (!response.success) {
      addLog(`Detection failed: ${response.message || 'Unknown error'}`);
      return;
    }

    const { ppe, faces, summary } = response.detections || {};

    // Process PPE detections
    if (ppe && ppe.length > 0) {
      const detectedItems = new Set<VerificationItemType>();

      ppe.forEach((item) => {
        const label = item.label.toLowerCase().trim();

        // Try to match the label
        let itemType: VerificationItemType | undefined;

        // Direct match
        if (PPE_LABEL_MAP[label]) {
          itemType = PPE_LABEL_MAP[label];
        } else {
          // Partial match
          for (const [key, value] of Object.entries(PPE_LABEL_MAP)) {
            if (label.includes(key) || key.includes(label)) {
              itemType = value;
              break;
            }
          }
        }

        if (itemType && !item.is_violation && item.confidence > 0.4) {
          detectedItems.add(itemType);
        }
      });

      // Update ML status for detected items
      detectedItems.forEach((itemType) => {
        const currentStatus = store.items[itemType].mlStatus;
        if (currentStatus !== 'passed') {
          store.updateMLStatus(itemType, 'passed', 0.9);
          addLog(`PPE detected: ${itemType}`);
        }
      });
    }

    // Also check summary for PPE
    if (summary?.ppe_detected) {
      Object.keys(summary.ppe_detected).forEach((key) => {
        const label = key.toLowerCase();
        const itemType = PPE_LABEL_MAP[label];
        if (itemType && summary.ppe_detected[key] > 0) {
          const currentStatus = store.items[itemType].mlStatus;
          if (currentStatus !== 'passed') {
            store.updateMLStatus(itemType, 'passed', 0.9);
            addLog(`PPE detected (summary): ${itemType}`);
          }
        }
      });
    }

    // Process face recognition
    if (faces && faces.length > 0) {
      addLog(`Faces detected: ${faces.length}`);

      // Find identified face - backend returns employee_id and name separately
      const identifiedFace = faces.find((f) =>
        (f.employee_id || f.name) &&
        f.name !== 'Unknown' &&
        f.confidence > 0.3
      );

      if (identifiedFace && !store.identifiedWorker) {
        const employeeId = identifiedFace.employee_id || identifiedFace.name || '';

        // Face recognized - update face ML status
        store.updateMLStatus('face', 'passed', identifiedFace.confidence);

        // Try to get worker details from entry or create from face data
        let worker: Worker | null = null;

        if (response.entry?.worker_id && response.entry?.worker_name) {
          // We have entry data with worker info from backend (includes actual name from DB)
          worker = {
            id: response.entry.worker_id,
            employee_id: response.entry.employee_id || employeeId,
            name: response.entry.worker_name,
            mine_id: mineId || '',
            assigned_shift: 'day',
            face_registered: true,
            is_active: true,
            created_at: new Date().toISOString(),
            compliance_score: 100,
            total_violations: 0,
            badges: [],
          };
          addLog(`Face identified: ${worker.name} (ID: ${employeeId}, conf: ${(identifiedFace.confidence * 100).toFixed(1)}%)`);
        } else if (employeeId) {
          // No worker_name from entry - fetch worker details from API using employee_id
          try {
            addLog(`Fetching worker details for ${employeeId}...`);
            const workersResponse = await workerApi.list({ search: employeeId, limit: 1 });
            if (workersResponse.workers && workersResponse.workers.length > 0) {
              const fetchedWorker = workersResponse.workers[0];
              worker = {
                id: fetchedWorker.id,
                employee_id: fetchedWorker.employee_id,
                name: fetchedWorker.name,
                mine_id: fetchedWorker.mine_id || mineId || '',
                assigned_shift: fetchedWorker.assigned_shift || 'day',
                face_registered: fetchedWorker.face_registered ?? true,
                is_active: fetchedWorker.is_active ?? true,
                created_at: fetchedWorker.created_at || new Date().toISOString(),
                compliance_score: fetchedWorker.compliance_score ?? 100,
                total_violations: fetchedWorker.total_violations ?? 0,
                badges: fetchedWorker.badges || [],
              };
              addLog(`Face identified: ${worker.name} (ID: ${employeeId}, conf: ${(identifiedFace.confidence * 100).toFixed(1)}%)`);
            }
          } catch (err) {
            addLog(`Failed to fetch worker details: ${err}`);
          }

          // Fallback if API call failed or returned empty
          if (!worker) {
            const displayName = identifiedFace.name && identifiedFace.name !== employeeId
              ? identifiedFace.name
              : employeeId;
            worker = {
              id: '',
              employee_id: employeeId,
              name: displayName,
              mine_id: mineId || '',
              assigned_shift: 'day',
              face_registered: true,
              is_active: true,
              created_at: new Date().toISOString(),
              compliance_score: 100,
              total_violations: 0,
              badges: [],
            };
            addLog(`Face identified (fallback): ${displayName} (ID: ${employeeId}, conf: ${(identifiedFace.confidence * 100).toFixed(1)}%)`);
          }
        }

        if (worker) {
          store.setIdentifiedWorker(worker, identifiedFace.confidence);
          onWorkerIdentified?.(worker);

          // Mark attendance in background
          await markAttendance(worker);
        }
      }
    }

    // Also check summary for identified persons (fallback if faces array didn't match)
    if (summary?.identified_persons && summary.identified_persons.length > 0 && !store.identifiedWorker) {
      const employeeId = summary.identified_persons[0];

      if (employeeId && employeeId !== 'Unknown') {
        store.updateMLStatus('face', 'passed', 0.85);

        let worker: Worker | null = null;

        // Try to fetch worker details from API using employee_id
        try {
          addLog(`Fetching worker details for ${employeeId} (summary)...`);
          const workersResponse = await workerApi.list({ search: employeeId, limit: 1 });
          if (workersResponse.workers && workersResponse.workers.length > 0) {
            const fetchedWorker = workersResponse.workers[0];
            worker = {
              id: fetchedWorker.id,
              employee_id: fetchedWorker.employee_id,
              name: fetchedWorker.name,
              mine_id: fetchedWorker.mine_id || mineId || '',
              assigned_shift: fetchedWorker.assigned_shift || 'day',
              face_registered: fetchedWorker.face_registered ?? true,
              is_active: fetchedWorker.is_active ?? true,
              created_at: fetchedWorker.created_at || new Date().toISOString(),
              compliance_score: fetchedWorker.compliance_score ?? 100,
              total_violations: fetchedWorker.total_violations ?? 0,
              badges: fetchedWorker.badges || [],
            };
            addLog(`Person identified (summary): ${worker.name} (ID: ${employeeId})`);
          }
        } catch (err) {
          addLog(`Failed to fetch worker details (summary): ${err}`);
        }

        // Fallback if API call failed or returned empty
        if (!worker) {
          const displayName = summary.identified_names?.[0] || employeeId;
          worker = {
            id: '',
            employee_id: employeeId,
            name: displayName,
            mine_id: mineId || '',
            assigned_shift: 'day',
            face_registered: true,
            is_active: true,
            created_at: new Date().toISOString(),
            compliance_score: 100,
            total_violations: 0,
            badges: [],
          };
          addLog(`Person identified (summary fallback): ${displayName} (ID: ${employeeId})`);
        }

        store.setIdentifiedWorker(worker, 0.85);
        onWorkerIdentified?.(worker);

        // Mark attendance in background
        await markAttendance(worker);
      }
    }

    // Check if all checks are complete
    checkCompletion();
  }, [store, mineId, addLog, onWorkerIdentified]);

  // Mark attendance
  const markAttendance = useCallback(async (worker?: Worker) => {
    if (store.attendanceMarked || !frameRef.current) {
      addLog('Attendance already marked or no frame available');
      return;
    }

    try {
      addLog('Marking attendance...');

      // Convert base64 to File
      const response = await fetch(frameRef.current);
      const blob = await response.blob();
      const file = new File([blob], 'face-capture.jpg', { type: 'image/jpeg' });

      const result = await attendanceApi.checkIn(file);

      store.markAttendance();
      addLog('Attendance marked successfully!');
      onAttendanceMarked?.();

      return result;
    } catch (error) {
      console.error('Failed to mark attendance:', error);
      addLog(`Attendance error: ${error}`);
      // Still mark as done to avoid repeated attempts
      store.markAttendance();
    }
  }, [store, addLog, onAttendanceMarked]);

  // Check if verification is complete
  const checkCompletion = useCallback(() => {
    const passedCount = store.getPassedCount();
    const totalChecks = store.getTotalChecks();

    addLog(`Progress: ${passedCount}/${totalChecks} checks passed`);

    if (passedCount === totalChecks) {
      // All checks passed - determine outcome immediately
      addLog('All checks passed!');
      store.determineOutcome();
    }
  }, [store, addLog]);

  // Perform PPE detection using the new PPE detection service
  // DISABLED - PPE detection service on port 8002 is not running
  // The gate entry detect API already includes PPE detection
  const performPPEDetection = useCallback(async () => {
    // PPE detection service is disabled - skip API call
    return;
  }, []);

  // Perform detection on current frame (face recognition + legacy PPE)
  const performDetection = useCallback(async () => {
    if (!gateId || !frameRef.current || isDetectingRef.current) {
      if (!frameRef.current) {
        addLog('No frame available for detection');
      }
      return;
    }

    isDetectingRef.current = true;

    try {
      addLog('Sending frame for detection...');

      // Run PPE detection in parallel with face recognition
      performPPEDetection();

      // Convert base64 to File for face recognition
      const fetchResponse = await fetch(frameRef.current);
      const blob = await fetchResponse.blob();
      const file = new File([blob], 'frame.jpg', { type: 'image/jpeg' });

      addLog(`Frame size: ${(blob.size / 1024).toFixed(1)}KB`);

      // Pass empty string if no gate selected - backend will use default gate
      const result = await gateEntryApi.detect(gateId || '', file, 'entry');

      addLog(`Detection response received: ${result.success ? 'SUCCESS' : 'FAILED'}`);

      // Log what was detected
      if (result.detections) {
        const ppeCount = result.detections.ppe?.length || 0;
        const faceCount = result.detections.faces?.length || 0;
        addLog(`Detected: ${ppeCount} PPE items, ${faceCount} faces`);

        if (result.detections.ppe && result.detections.ppe.length > 0) {
          const labels = result.detections.ppe.map((p: any) => p.label).join(', ');
          addLog(`PPE labels: ${labels}`);
        }

        if (result.detections.faces && result.detections.faces.length > 0) {
          result.detections.faces.forEach((f: any) => {
            addLog(`Face: ${f.name || 'Unknown'} (ID: ${f.employee_id || 'N/A'}, conf: ${(f.confidence * 100).toFixed(0)}%)`);
          });
        }
      }

      await processDetectionResponse(result);
    } catch (error: any) {
      console.error('Detection failed:', error);
      const errorMessage = error?.response?.data?.detail || error?.message || String(error);
      addLog(`Detection error: ${errorMessage}`);
    } finally {
      isDetectingRef.current = false;
    }
  }, [gateId, processDetectionResponse, addLog, performPPEDetection]);

  // Timer tick effect
  useEffect(() => {
    if (store.isTimerRunning) {
      timerIntervalRef.current = setInterval(() => {
        store.tick();
      }, 100); // Update every 100ms for smooth countdown
    } else {
      if (timerIntervalRef.current) {
        clearInterval(timerIntervalRef.current);
        timerIntervalRef.current = null;
      }
    }

    return () => {
      if (timerIntervalRef.current) {
        clearInterval(timerIntervalRef.current);
      }
    };
  }, [store.isTimerRunning, store]);

  // Detection interval effect
  useEffect(() => {
    if (store.isTimerRunning && store.overallStatus === 'verifying') {
      // RFID is DISABLED - auto-pass all RFID checks
      // Only ML detection is active
      ['helmet', 'vest', 'shoes'].forEach((item) => {
        const itemType = item as VerificationItemType;
        // Auto-pass RFID since it's disabled
        if (store.items[itemType].rfidStatus === 'pending') {
          store.updateRFIDStatus(itemType, 'passed', 'RFID-DISABLED');
        }
        // Set ML to checking
        if (store.items[itemType].mlStatus === 'pending') {
          store.updateMLStatus(itemType, 'checking');
        }
      });

      // Auto-pass shoes ML since the new model focuses on helmet and vest
      if (store.items.shoes.mlStatus === 'pending' || store.items.shoes.mlStatus === 'checking') {
        store.updateMLStatus('shoes', 'passed', 1.0);
        addLog('Shoes ML: Auto-passed (model focuses on helmet/vest)');
      }

      if (store.items.face.mlStatus === 'pending') {
        store.updateMLStatus('face', 'checking');
      }

      // Start detection at 1 second intervals (reduced from 500ms to avoid overwhelming)
      detectionIntervalRef.current = setInterval(() => {
        performDetection();
      }, 1000);

      // Also run immediately
      performDetection();
    } else {
      if (detectionIntervalRef.current) {
        clearInterval(detectionIntervalRef.current);
        detectionIntervalRef.current = null;
      }
    }

    return () => {
      if (detectionIntervalRef.current) {
        clearInterval(detectionIntervalRef.current);
      }
    };
  }, [store.isTimerRunning, store.overallStatus, performDetection, store, addLog]);

  // Handle verification outcome
  useEffect(() => {
    if (store.overallStatus === 'passed' || store.overallStatus === 'failed' || store.overallStatus === 'warning') {
      const success = store.overallStatus === 'passed';
      addLog(`Verification complete: ${store.overallStatus.toUpperCase()}`);

      onVerificationComplete?.(success, store.identifiedWorker);

      if (success) {
        onGateOpen?.();
      }
    }
  }, [store.overallStatus, store.identifiedWorker, onVerificationComplete, onGateOpen, addLog]);

  // Capture snapshot and perform full detection (PPE + face) with attendance marking
  const captureSnapshot = useCallback(async (frameData?: string) => {
    // Use provided frameData or fall back to stored frame
    const frame = frameData || frameRef.current;

    console.log('[useGateVerification] captureSnapshot called', { gateId, hasFrame: !!frame, frameLength: frame?.length });

    // Gate ID is now optional - backend will use default if not provided
    if (!gateId) {
      addLog('No gate selected - using default gate');
      console.log('[useGateVerification] No gate selected, backend will use default');
    }

    if (!frame) {
      const msg = 'Cannot capture snapshot: no video frame available. Please start the camera first.';
      addLog(msg);
      console.error('[useGateVerification]', msg);
      return null;
    }

    setIsSnapshotCapturing(true);
    addLog('Capturing snapshot for detection...');
    console.log('[useGateVerification] Starting detection with gateId:', gateId || 'default');

    try {
      // Store the frame for later use
      frameRef.current = frame;

      // Run PPE detection in parallel
      performPPEDetection();

      // Convert base64 to File for gate entry detection
      const fetchResponse = await fetch(frame);
      const blob = await fetchResponse.blob();
      const file = new File([blob], 'snapshot.jpg', { type: 'image/jpeg' });

      addLog(`Snapshot size: ${(blob.size / 1024).toFixed(1)}KB - Sending for detection...`);

      // Call the gate entry detect API (includes face recognition + PPE + attendance)
      // Pass empty string if no gate selected - backend will use default gate
      const result = await gateEntryApi.detect(gateId || '', file, 'entry');

      addLog(`Snapshot detection: ${result.success ? 'SUCCESS' : 'FAILED'}`);

      if (result.success) {
        // Process detection response (updates store with PPE and face results)
        await processDetectionResponse(result);

        // Log detailed results
        if (result.detections) {
          const ppeCount = result.detections.ppe?.length || 0;
          const faceCount = result.detections.faces?.length || 0;
          const violations = result.detections.violations?.length || 0;

          addLog(`Detected: ${ppeCount} PPE items, ${faceCount} faces, ${violations} violations`);

          if (result.detections.summary) {
            const { identified_persons, safety_compliant } = result.detections.summary;
            if (identified_persons?.length > 0) {
              addLog(`Identified: ${identified_persons.join(', ')}`);
            }
            addLog(`Safety compliant: ${safety_compliant ? 'YES' : 'NO'}`);
          }
        }

        // If worker was identified and attendance not yet marked, mark it
        if (result.entry?.worker_id && !store.attendanceMarked) {
          addLog('Attendance marked via snapshot detection');
          store.markAttendance();
          onAttendanceMarked?.();
        }

        // Check if verification is now complete
        checkCompletion();
      }

      return result;
    } catch (error: any) {
      const errorMessage = error?.response?.data?.detail || error?.message || String(error);
      addLog(`Snapshot detection error: ${errorMessage}`);
      console.error('[useGateVerification] Snapshot detection failed:', error);
      console.error('[useGateVerification] Error details:', {
        status: error?.response?.status,
        statusText: error?.response?.statusText,
        data: error?.response?.data,
        message: error?.message
      });
      return null;
    } finally {
      setIsSnapshotCapturing(false);
    }
  }, [gateId, store, addLog, processDetectionResponse, performPPEDetection, checkCompletion, onAttendanceMarked]);

  return {
    // State
    items: store.items,
    overallStatus: store.overallStatus,
    identifiedWorker: store.identifiedWorker,
    timeRemaining: store.timeRemaining,
    isTimerRunning: store.isTimerRunning,
    attendanceMarked: store.attendanceMarked,
    isSnapshotCapturing,

    // Computed
    passedCount: store.getPassedCount(),
    totalChecks: store.getTotalChecks(),
    canOverride: store.canOverride(),

    // Debug
    detectionLog,

    // Actions
    startVerification,
    resetVerification,
    setCurrentFrame,
    processDetectionResponse,
    determineOutcome: store.determineOutcome,
    markAttendance,
    captureSnapshot,
  };
}
