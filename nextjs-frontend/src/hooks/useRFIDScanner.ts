'use client';

import { useEffect, useCallback, useRef, useState } from 'react';
import type { VerificationItemType } from '@/types';

// RFID WebSocket server URL (rfid.py backend - runs on port 8001)
const RFID_WEBSOCKET_URL = process.env.NEXT_PUBLIC_RFID_WS_URL || 'ws://localhost:8001/ws';

interface RFIDScanEvent {
  tagId: string;
  tagType: VerificationItemType;
  timestamp: Date;
}

interface UseRFIDScannerOptions {
  enabled: boolean;
  onScan: (event: RFIDScanEvent) => void;
  mockMode?: boolean; // Enable keyboard shortcuts for testing (fallback if WebSocket fails)
}

interface RFIDScannerStateInternal {
  isConnected: boolean;
  lastScan: RFIDScanEvent | null;
  error: string | null;
  scanning: boolean;
  gateStatus: 'OPEN' | 'CLOSED';
  result: 'WAITING' | 'SCANNING' | 'PASS' | 'FAIL';
}

interface RFIDScannerState extends RFIDScannerStateInternal {
  startRFIDScan: () => Promise<boolean>;
}

// PPE state from rfid.py backend
interface RFIDBackendState {
  scanning: boolean;
  gate: 'OPEN' | 'CLOSED';
  result: 'WAITING' | 'SCANNING' | 'PASS' | 'FAIL';
  ppe: {
    helmet: boolean;
    vest: boolean;
    boots: boolean; // Note: backend uses 'boots', frontend uses 'shoes'
  };
  log: string;
}

/**
 * Hook for RFID scanner integration with rfid.py backend
 *
 * Connects to the rfid.py FastAPI WebSocket server to receive real-time
 * PPE scanning updates from the ESP32 RFID reader.
 *
 * Mock mode keyboard shortcuts (fallback for development):
 * - Press 'H' to simulate helmet RFID scan
 * - Press 'V' to simulate vest RFID scan
 * - Press 'S' to simulate shoes RFID scan
 */
export function useRFIDScanner({ enabled, onScan, mockMode = false }: UseRFIDScannerOptions): RFIDScannerState {
  const [state, setState] = useState<RFIDScannerStateInternal>({
    isConnected: false,
    lastScan: null,
    error: null,
    scanning: false,
    gateStatus: 'CLOSED',
    result: 'WAITING',
  });

  const onScanRef = useRef(onScan);
  onScanRef.current = onScan;

  // Track previous PPE state to detect changes
  const prevPPERef = useRef<{ helmet: boolean; vest: boolean; boots: boolean }>({
    helmet: false,
    vest: false,
    boots: false,
  });

  // Generate mock tag ID
  const generateMockTagId = useCallback((type: VerificationItemType): string => {
    const prefix = {
      helmet: 'HLM',
      vest: 'VST',
      shoes: 'SHS',
      face: 'FCE', // Not used for RFID, but included for completeness
    };
    return `${prefix[type]}-${Math.random().toString(36).substring(2, 8).toUpperCase()}`;
  }, []);

  // Handle mock keyboard input
  useEffect(() => {
    if (!enabled || !mockMode) return;

    const handleKeyDown = (event: KeyboardEvent) => {
      // Ignore if user is typing in an input field
      if (
        event.target instanceof HTMLInputElement ||
        event.target instanceof HTMLTextAreaElement
      ) {
        return;
      }

      let tagType: VerificationItemType | null = null;

      switch (event.key.toLowerCase()) {
        case 'h':
          tagType = 'helmet';
          break;
        case 'v':
          tagType = 'vest';
          break;
        case 's':
          tagType = 'shoes';
          break;
      }

      if (tagType) {
        const scanEvent: RFIDScanEvent = {
          tagId: generateMockTagId(tagType),
          tagType,
          timestamp: new Date(),
        };

        setState((prev) => ({
          ...prev,
          lastScan: scanEvent,
        }));

        onScanRef.current(scanEvent);
      }
    };

    window.addEventListener('keydown', handleKeyDown);

    // Simulate connection established
    setState((prev) => ({
      ...prev,
      isConnected: true,
      error: null,
    }));

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      setState((prev) => ({
        ...prev,
        isConnected: false,
      }));
    };
  }, [enabled, mockMode, generateMockTagId]);

  // Real RFID WebSocket integration with rfid.py backend
  useEffect(() => {
    if (!enabled) return;

    let ws: WebSocket | null = null;
    let reconnectTimeout: NodeJS.Timeout | null = null;
    let isCleaningUp = false;

    const connect = () => {
      if (isCleaningUp) return;

      try {
        console.log(`[RFID] Connecting to ${RFID_WEBSOCKET_URL}...`);
        ws = new WebSocket(RFID_WEBSOCKET_URL);

        ws.onopen = () => {
          console.log('[RFID] WebSocket connected');
          setState((prev) => ({
            ...prev,
            isConnected: true,
            error: null,
          }));
          // Reset previous PPE state on new connection
          prevPPERef.current = { helmet: false, vest: false, boots: false };
        };

        ws.onmessage = (event) => {
          try {
            const data: RFIDBackendState = JSON.parse(event.data);
            console.log('[RFID] Received:', data);

            // Update state
            setState((prev) => ({
              ...prev,
              scanning: data.scanning,
              gateStatus: data.gate,
              result: data.result,
            }));

            // Check for PPE changes and trigger scan events
            const ppeItems: Array<{ key: 'helmet' | 'vest' | 'boots'; type: VerificationItemType }> = [
              { key: 'helmet', type: 'helmet' },
              { key: 'vest', type: 'vest' },
              { key: 'boots', type: 'shoes' }, // Map boots -> shoes
            ];

            ppeItems.forEach(({ key, type }) => {
              // Trigger scan event when item changes from false to true
              if (data.ppe[key] && !prevPPERef.current[key]) {
                const scanEvent: RFIDScanEvent = {
                  tagId: `RFID-${type.toUpperCase()}-${Date.now()}`,
                  tagType: type,
                  timestamp: new Date(),
                };

                console.log(`[RFID] ${type} scanned!`, scanEvent);

                setState((prev) => ({
                  ...prev,
                  lastScan: scanEvent,
                }));

                onScanRef.current(scanEvent);
              }
            });

            // Update previous state
            prevPPERef.current = { ...data.ppe };
          } catch (parseError) {
            console.error('[RFID] Failed to parse message:', parseError);
          }
        };

        ws.onerror = (error) => {
          console.error('[RFID] WebSocket error:', error);
          setState((prev) => ({
            ...prev,
            error: 'RFID connection error',
          }));
        };

        ws.onclose = () => {
          console.log('[RFID] WebSocket closed');
          setState((prev) => ({
            ...prev,
            isConnected: false,
          }));

          // Attempt to reconnect after 3 seconds if not cleaning up
          if (!isCleaningUp) {
            reconnectTimeout = setTimeout(() => {
              console.log('[RFID] Attempting to reconnect...');
              connect();
            }, 3000);
          }
        };
      } catch (error) {
        console.error('[RFID] Failed to create WebSocket:', error);
        setState((prev) => ({
          ...prev,
          isConnected: false,
          error: 'Failed to connect to RFID server',
        }));
      }
    };

    connect();

    return () => {
      isCleaningUp = true;
      if (reconnectTimeout) {
        clearTimeout(reconnectTimeout);
      }
      if (ws) {
        ws.close();
      }
    };
  }, [enabled]);

  // Function to start RFID scanning via the backend
  const startRFIDScan = useCallback(async () => {
    const baseUrl = RFID_WEBSOCKET_URL.replace('ws://', 'http://').replace('/ws', '');
    console.log(`[RFID] Starting scan via ${baseUrl}/start-scan`);

    try {
      const response = await fetch(`${baseUrl}/start-scan`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      console.log(`[RFID] Response status: ${response.status}`);
      const data = await response.json();
      console.log('[RFID] Response data:', data);

      if (response.ok) {
        console.log('[RFID] Scan started successfully');
        return true;
      } else {
        console.error('[RFID] Failed to start scan:', response.statusText);
        return false;
      }
    } catch (error) {
      console.error('[RFID] Error starting scan:', error);
      return false;
    }
  }, []);

  return {
    ...state,
    startRFIDScan,
  };
}

// Helper function to manually trigger a scan (for UI buttons)
export function createManualScanTrigger(
  onScan: (event: RFIDScanEvent) => void
): (tagType: VerificationItemType) => void {
  return (tagType: VerificationItemType) => {
    const prefix = {
      helmet: 'HLM',
      vest: 'VST',
      shoes: 'SHS',
      face: 'FCE',
    };

    const scanEvent: RFIDScanEvent = {
      tagId: `${prefix[tagType]}-${Math.random().toString(36).substring(2, 8).toUpperCase()}`,
      tagType,
      timestamp: new Date(),
    };

    onScan(scanEvent);
  };
}
