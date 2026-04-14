/**
 * useHelmetTracking Hook
 *
 * Connects to helmet.py REST API and implements Pedestrian Dead Reckoning (PDR)
 * using step detection for smooth, jitter-free position tracking.
 */

import { useEffect, useRef, useCallback, useState } from 'react';

// ==================== TYPES ====================
export interface IMUData {
  accelX: number;      // mg (milli-g)
  accelY: number;      // mg
  accelZ: number;      // mg
  gyroX: number;       // mdps
  gyroY: number;       // mdps
  gyroZ: number;       // mdps
  roll: number;        // degrees
  pitch: number;       // degrees
  yaw: number;         // degrees
  timestamp: number;   // ms
}

export interface Position {
  x: number;
  y: number;
  z: number;
}

export interface TrackingState {
  position: Position;
  heading: number;           // radians
  isConnected: boolean;
  isSimulating: boolean;
  stepCount: number;
  lastIMU: IMUData | null;
}

// ==================== CONSTANTS ====================
// Uses main.py backend endpoint (not helmet.py)
const HELMET_API_URL = 'http://localhost:8000/api/helmet/imu-data';

// Polling interval for REST API
const POLL_INTERVAL = 200; // ms

// Step detection parameters (tuned for walking)
const STEP_THRESHOLD = 1050;      // mg - lowered for real data sensitivity
const STEP_COOLDOWN = 350;        // ms - minimum time between steps
const STEP_LENGTH = 2.0;          // meters per step - increased for visibility
const VARIANCE_THRESHOLD = 150;   // mg - detect steps via acceleration variance

// Smoothing filter parameters (EMA - Exponential Moving Average)
const HEADING_FILTER_ALPHA = 0.08;   // Lower = smoother heading
const POSITION_FILTER_ALPHA = 0.12;  // Lower = smoother position

// Mock simulation settings
const MOCK_UPDATE_INTERVAL = 100; // ms
const MOCK_STEP_INTERVAL = 600;   // ms between simulated steps

// ==================== UTILITY FUNCTIONS ====================

/**
 * Exponential Moving Average filter
 */
function emaFilter(current: number, previous: number, alpha: number): number {
  return alpha * current + (1 - alpha) * previous;
}

/**
 * Convert degrees to radians
 */
function degToRad(degrees: number): number {
  return degrees * (Math.PI / 180);
}

/**
 * Calculate acceleration magnitude from 3-axis data
 */
function calculateAccelMagnitude(ax: number, ay: number, az: number): number {
  return Math.sqrt(ax * ax + ay * ay + az * az);
}

// ==================== MAIN HOOK ====================

interface UseHelmetTrackingOptions {
  enabled: boolean;
  initialPosition?: Position;
  onPositionUpdate?: (position: Position, heading: number) => void;
  onConnectionChange?: (connected: boolean, simulating: boolean) => void;
  onStepDetected?: () => void;
}

export function useHelmetTracking({
  enabled,
  initialPosition = { x: 0, y: 1.7, z: 0 },
  onPositionUpdate,
  onConnectionChange,
  onStepDetected,
}: UseHelmetTrackingOptions): TrackingState {
  // State
  const [state, setState] = useState<TrackingState>({
    position: initialPosition,
    heading: 0,
    isConnected: false,
    isSimulating: false,
    stepCount: 0,
    lastIMU: null,
  });

  // Refs for mutable values (don't trigger re-renders)
  const pollIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const mockIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const mockStepIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const failCountRef = useRef<number>(0);
  const isConnectedRef = useRef<boolean>(false);

  // PDR state refs
  const positionRef = useRef<Position>(initialPosition);
  const filteredPositionRef = useRef<Position>(initialPosition);
  const headingRef = useRef<number>(0);
  const filteredHeadingRef = useRef<number>(0);
  const lastStepTimeRef = useRef<number>(0);
  const stepCountRef = useRef<number>(0);
  const lastAccelMagnitudeRef = useRef<number>(1000);
  const wasAboveThresholdRef = useRef<boolean>(false);
  const accelHistoryRef = useRef<number[]>([]);  // For variance calculation
  const logCounterRef = useRef<number>(0);

  // Callbacks refs (to avoid stale closures)
  const onPositionUpdateRef = useRef(onPositionUpdate);
  const onConnectionChangeRef = useRef(onConnectionChange);
  const onStepDetectedRef = useRef(onStepDetected);

  useEffect(() => {
    onPositionUpdateRef.current = onPositionUpdate;
    onConnectionChangeRef.current = onConnectionChange;
    onStepDetectedRef.current = onStepDetected;
  }, [onPositionUpdate, onConnectionChange, onStepDetected]);

  /**
   * Process incoming IMU data and update position
   */
  const processIMUData = useCallback((imu: IMUData) => {
    const now = Date.now();

    // Update heading from yaw (convert to radians, apply filter)
    const rawHeading = degToRad(imu.yaw);
    filteredHeadingRef.current = emaFilter(
      rawHeading,
      filteredHeadingRef.current,
      HEADING_FILTER_ALPHA
    );
    headingRef.current = filteredHeadingRef.current;

    // Calculate acceleration magnitude for step detection
    const accelMag = calculateAccelMagnitude(imu.accelX, imu.accelY, imu.accelZ);

    // Log every 10th reading to see values
    logCounterRef.current += 1;
    if (logCounterRef.current % 10 === 0) {
      console.log(`[IMU] mag=${accelMag.toFixed(0)} yaw=${imu.yaw.toFixed(1)} ax=${imu.accelX} ay=${imu.accelY} az=${imu.accelZ}`);
    }

    // Update acceleration history for variance detection
    accelHistoryRef.current.push(accelMag);
    if (accelHistoryRef.current.length > 10) {
      accelHistoryRef.current.shift();
    }

    // Calculate variance in recent readings
    const avgMag = accelHistoryRef.current.reduce((a, b) => a + b, 0) / accelHistoryRef.current.length;
    const variance = accelHistoryRef.current.reduce((sum, val) => sum + Math.abs(val - avgMag), 0) / accelHistoryRef.current.length;

    // Step detection: either magnitude spike OR high variance
    const isAboveThreshold = accelMag > STEP_THRESHOLD;
    const hasHighVariance = variance > VARIANCE_THRESHOLD;
    const timeSinceLastStep = now - lastStepTimeRef.current;

    const stepDetected = (
      (isAboveThreshold && !wasAboveThresholdRef.current) ||
      (hasHighVariance && timeSinceLastStep > STEP_COOLDOWN * 1.5)
    ) && timeSinceLastStep > STEP_COOLDOWN;

    if (stepDetected) {
      // Step detected!
      lastStepTimeRef.current = now;
      stepCountRef.current += 1;

      console.log(`[STEP] #${stepCountRef.current} mag=${accelMag.toFixed(0)} var=${variance.toFixed(0)} yaw=${imu.yaw.toFixed(1)}`);

      // Update position based on heading
      const heading = filteredHeadingRef.current;
      positionRef.current = {
        x: positionRef.current.x + STEP_LENGTH * Math.sin(heading),
        y: positionRef.current.y,
        z: positionRef.current.z - STEP_LENGTH * Math.cos(heading),
      };

      // Notify step detected
      onStepDetectedRef.current?.();

      // Reset variance tracking after step
      accelHistoryRef.current = [];
    }

    wasAboveThresholdRef.current = isAboveThreshold;
    lastAccelMagnitudeRef.current = accelMag;

    // Apply position smoothing filter
    filteredPositionRef.current = {
      x: emaFilter(positionRef.current.x, filteredPositionRef.current.x, POSITION_FILTER_ALPHA),
      y: positionRef.current.y,
      z: emaFilter(positionRef.current.z, filteredPositionRef.current.z, POSITION_FILTER_ALPHA),
    };

    // Update state and notify
    const newPosition = filteredPositionRef.current;
    const newHeading = filteredHeadingRef.current;

    setState(prev => ({
      ...prev,
      position: newPosition,
      heading: newHeading,
      stepCount: stepCountRef.current,
      lastIMU: imu,
    }));

    onPositionUpdateRef.current?.(newPosition, newHeading);
  }, []);

  /**
   * Parse REST API response from helmet.py
   */
  const parseHelmetData = useCallback((json: Record<string, unknown>): IMUData | null => {
    try {
      if (!json || Object.keys(json).length === 0) {
        return null;
      }

      return {
        accelX: (json['Accel X (mg)'] as number) ?? 0,
        accelY: (json['Accel Y (mg)'] as number) ?? 0,
        accelZ: (json['Accel Z (mg)'] as number) ?? 1000,
        gyroX: (json['Gyro X (mdps)'] as number) ?? 0,
        gyroY: (json['Gyro Y (mdps)'] as number) ?? 0,
        gyroZ: (json['Gyro Z (mdps)'] as number) ?? 0,
        roll: (json['Roll (Deg)'] as number) ?? 0,
        pitch: (json['Pitch (Deg)'] as number) ?? 0,
        yaw: (json['Yaw (Deg)'] as number) ?? 0,
        timestamp: (json['Timestamp (ms)'] as number) ?? Date.now(),
      };
    } catch (e) {
      console.error('[HelmetTracking] Failed to parse data:', e);
      return null;
    }
  }, []);

  /**
   * Start mock simulation when helmet is disconnected
   */
  const startMockSimulation = useCallback(() => {
    if (mockIntervalRef.current) return; // Already running

    console.log('[HelmetTracking] Starting mock simulation');

    let mockYaw = 0;
    let mockTime = 0;

    // Update mock IMU data at regular intervals
    mockIntervalRef.current = setInterval(() => {
      mockTime += MOCK_UPDATE_INTERVAL;

      // Gentle heading drift for realistic movement
      mockYaw += (Math.random() - 0.5) * 2;
      mockYaw = mockYaw % 360;

      const mockIMU: IMUData = {
        accelX: (Math.random() - 0.5) * 100,
        accelY: (Math.random() - 0.5) * 100,
        accelZ: 980 + (Math.random() - 0.5) * 50,
        gyroX: (Math.random() - 0.5) * 50,
        gyroY: (Math.random() - 0.5) * 50,
        gyroZ: (Math.random() - 0.5) * 50,
        roll: (Math.random() - 0.5) * 5,
        pitch: (Math.random() - 0.5) * 5,
        yaw: mockYaw,
        timestamp: mockTime,
      };

      // Update heading only (not triggering steps)
      const rawHeading = degToRad(mockIMU.yaw);
      filteredHeadingRef.current = emaFilter(
        rawHeading,
        filteredHeadingRef.current,
        HEADING_FILTER_ALPHA
      );

      setState(prev => ({
        ...prev,
        heading: filteredHeadingRef.current,
        lastIMU: mockIMU,
      }));
    }, MOCK_UPDATE_INTERVAL);

    // Simulate steps at regular intervals
    mockStepIntervalRef.current = setInterval(() => {
      const heading = filteredHeadingRef.current;
      stepCountRef.current += 1;

      positionRef.current = {
        x: positionRef.current.x + STEP_LENGTH * Math.sin(heading),
        y: positionRef.current.y,
        z: positionRef.current.z - STEP_LENGTH * Math.cos(heading),
      };

      filteredPositionRef.current = {
        x: emaFilter(positionRef.current.x, filteredPositionRef.current.x, POSITION_FILTER_ALPHA),
        y: positionRef.current.y,
        z: emaFilter(positionRef.current.z, filteredPositionRef.current.z, POSITION_FILTER_ALPHA),
      };

      const newPosition = filteredPositionRef.current;

      setState(prev => ({
        ...prev,
        position: newPosition,
        stepCount: stepCountRef.current,
      }));

      onPositionUpdateRef.current?.(newPosition, filteredHeadingRef.current);
      onStepDetectedRef.current?.();
    }, MOCK_STEP_INTERVAL);

    setState(prev => ({
      ...prev,
      isSimulating: true,
      isConnected: false,
    }));

    onConnectionChangeRef.current?.(false, true);
  }, []);

  /**
   * Stop mock simulation
   */
  const stopMockSimulation = useCallback(() => {
    if (mockIntervalRef.current) {
      clearInterval(mockIntervalRef.current);
      mockIntervalRef.current = null;
    }
    if (mockStepIntervalRef.current) {
      clearInterval(mockStepIntervalRef.current);
      mockStepIntervalRef.current = null;
    }
  }, []);

  /**
   * Poll the REST API for sensor data
   */
  const pollSensorData = useCallback(async () => {
    try {
      const response = await fetch(HELMET_API_URL);

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const json = await response.json();
      const imu = parseHelmetData(json);

      if (imu) {
        // Successfully got data
        failCountRef.current = 0;

        if (!isConnectedRef.current) {
          console.log('[HelmetTracking] ✓ Connected to helmet.py - REAL DATA MODE');
          isConnectedRef.current = true;
          stopMockSimulation();
          setState(prev => ({
            ...prev,
            isConnected: true,
            isSimulating: false,
          }));
          onConnectionChangeRef.current?.(true, false);
        }

        processIMUData(imu);
      }
    } catch (error) {
      failCountRef.current += 1;

      // After 3 consecutive failures, switch to simulation
      if (failCountRef.current >= 3 && isConnectedRef.current) {
        console.log('[HelmetTracking] Connection lost, switching to simulation');
        isConnectedRef.current = false;
        setState(prev => ({
          ...prev,
          isConnected: false,
        }));
        onConnectionChangeRef.current?.(false, false);
        startMockSimulation();
      } else if (failCountRef.current >= 3 && !mockIntervalRef.current) {
        // Start simulation if not already running
        startMockSimulation();
      }
    }
  }, [parseHelmetData, processIMUData, startMockSimulation, stopMockSimulation]);

  /**
   * Start polling
   */
  const startPolling = useCallback(() => {
    if (pollIntervalRef.current) return;

    console.log('[HelmetTracking] Starting REST API polling');
    failCountRef.current = 0;

    // Initial poll
    pollSensorData();

    // Set up interval
    pollIntervalRef.current = setInterval(pollSensorData, POLL_INTERVAL);
  }, [pollSensorData]);

  /**
   * Stop polling
   */
  const stopPolling = useCallback(() => {
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current);
      pollIntervalRef.current = null;
    }
    stopMockSimulation();
    isConnectedRef.current = false;
    failCountRef.current = 0;

    setState(prev => ({
      ...prev,
      isConnected: false,
      isSimulating: false,
    }));
  }, [stopMockSimulation]);

  /**
   * Reset position to initial/entrance
   */
  const resetPosition = useCallback((newPosition?: Position) => {
    const pos = newPosition || initialPosition;
    positionRef.current = pos;
    filteredPositionRef.current = pos;
    stepCountRef.current = 0;

    setState(prev => ({
      ...prev,
      position: pos,
      stepCount: 0,
    }));
  }, [initialPosition]);

  // Effect: Start/stop polling based on enabled state
  useEffect(() => {
    if (enabled) {
      startPolling();
    } else {
      stopPolling();
    }

    return () => {
      stopPolling();
    };
  }, [enabled, startPolling, stopPolling]);

  // Effect: Update initial position when it changes
  useEffect(() => {
    if (!enabled) {
      positionRef.current = initialPosition;
      filteredPositionRef.current = initialPosition;
      setState(prev => ({
        ...prev,
        position: initialPosition,
      }));
    }
  }, [initialPosition, enabled]);

  return state;
}

// Export reset function separately for imperative usage
export function createHelmetTrackingController() {
  let resetFn: ((pos?: Position) => void) | null = null;

  return {
    setResetFunction: (fn: (pos?: Position) => void) => {
      resetFn = fn;
    },
    reset: (pos?: Position) => {
      resetFn?.(pos);
    },
  };
}

export default useHelmetTracking;
