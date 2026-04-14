import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { User, Worker, UserRole, VerificationStatus, VerificationItemType, VerificationItem, GateVerificationState } from '@/types';

interface AuthState {
  token: string | null;
  user: User | null;
  worker: Worker | null;
  userType: 'staff' | 'worker' | null;
  isAuthenticated: boolean;
  _hasHydrated: boolean;

  // Legacy fields for backward compatibility
  username: string | null;

  // Actions
  setStaffAuth: (token: string, user: User) => void;
  setWorkerAuth: (token: string, worker: Worker) => void;
  logout: () => void;
  setHasHydrated: (state: boolean) => void;

  // Legacy action
  setAuth: (token: string, username: string) => void;

  // Helpers
  getRole: () => UserRole | null;
  hasRole: (roles: UserRole[]) => boolean;
  hasMinRole: (minRole: UserRole) => boolean;
  getMineId: () => string | null;
  getMineIds: () => string[];
}

// Role hierarchy for comparison
const ROLE_HIERARCHY: Record<UserRole, number> = {
  worker: 1,
  shift_incharge: 2,
  safety_officer: 3,
  manager: 4,
  area_safety_officer: 5,
  general_manager: 6,
  super_admin: 7,
};

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      token: null,
      user: null,
      worker: null,
      userType: null,
      isAuthenticated: false,
      _hasHydrated: false,
      username: null,

      setHasHydrated: (state: boolean) => {
        set({ _hasHydrated: state });
      },

      setStaffAuth: (token: string, user: User) => {
        if (typeof window !== 'undefined') {
          localStorage.setItem('token', token);
        }
        set({
          token,
          user,
          worker: null,
          userType: 'staff',
          isAuthenticated: true,
          username: user.username,
        });
      },

      setWorkerAuth: (token: string, worker: Worker) => {
        if (typeof window !== 'undefined') {
          localStorage.setItem('token', token);
        }
        set({
          token,
          user: null,
          worker,
          userType: 'worker',
          isAuthenticated: true,
          username: worker.employee_id,
        });
      },

      logout: () => {
        if (typeof window !== 'undefined') {
          localStorage.removeItem('token');
          localStorage.removeItem('auth-storage');
        }
        set({
          token: null,
          user: null,
          worker: null,
          userType: null,
          isAuthenticated: false,
          username: null,
        });
      },

      // Legacy action for backward compatibility
      setAuth: (token: string, username: string) => {
        if (typeof window !== 'undefined') {
          localStorage.setItem('token', token);
        }
        set({ token, username, isAuthenticated: true });
      },

      getRole: () => {
        const state = get();
        if (state.userType === 'worker') {
          return 'worker';
        }
        return state.user?.role || null;
      },

      hasRole: (roles: UserRole[]) => {
        const currentRole = get().getRole();
        if (!currentRole) return false;
        return roles.includes(currentRole);
      },

      hasMinRole: (minRole: UserRole) => {
        const currentRole = get().getRole();
        if (!currentRole) return false;
        return ROLE_HIERARCHY[currentRole] >= ROLE_HIERARCHY[minRole];
      },

      getMineId: () => {
        const state = get();
        if (state.userType === 'worker') {
          return state.worker?.mine_id || null;
        }
        return state.user?.mine_id || null;
      },

      getMineIds: () => {
        const state = get();
        if (state.user?.mine_ids && state.user.mine_ids.length > 0) {
          return state.user.mine_ids;
        }
        const mineId = get().getMineId();
        return mineId ? [mineId] : [];
      },
    }),
    {
      name: 'auth-storage',
      onRehydrateStorage: () => (state) => {
        state?.setHasHydrated(true);
      },
    }
  )
);

// Helper hook for role-based access
export function useRole() {
  const { getRole, hasRole, hasMinRole, userType, user, worker } = useAuthStore();

  return {
    role: getRole(),
    userType,
    user,
    worker,
    isWorker: userType === 'worker',
    isStaff: userType === 'staff',
    hasRole,
    hasMinRole,
    isShiftIncharge: hasRole(['shift_incharge']),
    isSafetyOfficer: hasRole(['safety_officer']),
    isManager: hasRole(['manager']),
    isAreaSafetyOfficer: hasRole(['area_safety_officer']),
    isGeneralManager: hasRole(['general_manager']),
    isSuperAdmin: hasRole(['super_admin']),
    canManageWorkers: hasMinRole('shift_incharge'),
    canManageUsers: hasMinRole('manager'),
    canViewAllMines: hasRole(['super_admin', 'general_manager']),
  };
}

// Navigation items based on role
export function getNavigationForRole(role: UserRole | null): {
  name: string;
  href: string;
  icon: string;
}[] {
  if (!role) return [];

  const baseItems = [
    { name: 'Dashboard', href: '/', icon: 'LayoutDashboard' },
  ];

  switch (role) {
    case 'super_admin':
      return [
        ...baseItems,
        { name: 'Users', href: '/users', icon: 'Users' },
        { name: 'Workers', href: '/workers', icon: 'HardHat' },
        { name: 'Mines', href: '/mines', icon: 'Mountain' },
        { name: 'Gate Entry', href: '/gate-verification', icon: 'ScanLine' },
        // { name: 'Gate Monitoring', href: '/gate-monitoring', icon: 'DoorOpen' },
        // { name: 'PPE Detection', href: '/ppe-detection', icon: 'Shield' },
        { name: 'Gas Monitoring', href: '/gas-monitoring', icon: 'Wind' },
        { name: 'Helmet Monitoring', href: '/helmet-monitoring', icon: 'HardHat' },
        { name: 'SOS Alerts', href: '/sos-alerts', icon: 'AlertCircle' },
        { name: 'Performance Tracker', href: '/performance', icon: 'Brain' },
        { name: 'Alerts', href: '/alerts', icon: 'Bell' },
        { name: 'Reports', href: '/reports', icon: 'FileText' },
        { name: '3D Map', href: '/3d-map-generator', icon: 'Map' },
      ];

    case 'general_manager':
      return [
        ...baseItems,
        { name: 'Mines Overview', href: '/mines', icon: 'Mountain' },
        // { name: 'PPE Detection', href: '/ppe-detection', icon: 'Shield' },
        // { name: 'Gas Monitoring', href: '/gas-monitoring', icon: 'Wind' },
        { name: 'Helmet Monitoring', href: '/helmet-monitoring', icon: 'HardHat' },
        { name: 'SOS Alerts', href: '/sos-alerts', icon: 'AlertCircle' },
        { name: 'Performance Tracker', href: '/performance', icon: 'Brain' },
        { name: 'Reports', href: '/reports', icon: 'FileText' },
        { name: 'Alerts', href: '/alerts', icon: 'Bell' },
      ];

    case 'area_safety_officer':
      return [
        ...baseItems,
        { name: 'Mines Comparison', href: '/mines', icon: 'Mountain' },
        // { name: 'PPE Detection', href: '/ppe-detection', icon: 'Shield' },
        { name: 'Risk Analysis', href: '/risk-analysis', icon: 'AlertTriangle' },
        // { name: 'Gas Monitoring', href: '/gas-monitoring', icon: 'Wind' },
        { name: 'Helmet Monitoring', href: '/helmet-monitoring', icon: 'HardHat' },
        { name: 'SOS Alerts', href: '/sos-alerts', icon: 'AlertCircle' },
        { name: 'Performance Tracker', href: '/performance', icon: 'Brain' },
        { name: 'Alerts', href: '/alerts', icon: 'Bell' },
        { name: 'Reports', href: '/reports', icon: 'FileText' },
      ];

    case 'manager':
      return [
        ...baseItems,
        { name: 'Workers', href: '/workers', icon: 'HardHat' },
        { name: 'Gate Entry', href: '/gate-verification', icon: 'ScanLine' },
        // { name: 'PPE Detection', href: '/ppe-detection', icon: 'Shield' },
        { name: 'Shift Performance', href: '/shifts', icon: 'Clock' },
        // { name: 'Gas Monitoring', href: '/gas-monitoring', icon: 'Wind' },
        { name: 'Helmet Monitoring', href: '/helmet-monitoring', icon: 'HardHat' },
        { name: 'SOS Alerts', href: '/sos-alerts', icon: 'AlertCircle' },
        { name: 'Performance Tracker', href: '/performance', icon: 'Brain' },
        { name: 'Alerts', href: '/alerts', icon: 'Bell' },
        { name: 'Reports', href: '/reports', icon: 'FileText' },
      ];

    case 'safety_officer':
      return [
        ...baseItems,
        { name: 'Compliance', href: '/compliance', icon: 'Shield' },
        { name: 'Workers', href: '/workers', icon: 'HardHat' },
        { name: 'Gate Entry', href: '/gate-verification', icon: 'ScanLine' },
        // { name: 'PPE Detection', href: '/ppe-detection', icon: 'Shield' },
        // { name: 'PPE Config', href: '/ppe-config', icon: 'Settings' },
        // { name: 'Gas Monitoring', href: '/gas-monitoring', icon: 'Wind' },
        { name: 'Helmet Monitoring', href: '/helmet-monitoring', icon: 'HardHat' },
        { name: 'SOS Alerts', href: '/sos-alerts', icon: 'AlertCircle' },
        { name: 'Performance Tracker', href: '/performance', icon: 'Brain' },
        { name: 'Alerts', href: '/alerts', icon: 'Bell' },
        { name: 'Reports', href: '/reports', icon: 'FileText' },
      ];

    case 'shift_incharge':
      return [
        ...baseItems,
        { name: 'Gate Entry', href: '/gate-verification', icon: 'ScanLine' },
        // { name: 'Gate Monitoring', href: '/gate-monitoring', icon: 'DoorOpen' },
        // { name: 'PPE Detection', href: '/ppe-detection', icon: 'Shield' },
        // { name: 'Gas Monitoring', href: '/gas-monitoring', icon: 'Wind' },
        { name: 'Helmet Monitoring', href: '/helmet-monitoring', icon: 'HardHat' },
        { name: 'SOS Alerts', href: '/sos-alerts', icon: 'AlertCircle' },
        { name: 'Workers', href: '/workers', icon: 'HardHat' },
        { name: 'Alerts', href: '/alerts', icon: 'Bell' },
        { name: 'Shift Report', href: '/shift-report', icon: 'FileText' },
      ];

    case 'worker':
      return [
        { name: 'My Dashboard', href: '/', icon: 'LayoutDashboard' },
        { name: 'My Profile', href: '/worker-profile', icon: 'User' },
        { name: 'Violations', href: '/my-violations', icon: 'AlertCircle' },
        { name: 'Training', href: '/training', icon: 'GraduationCap' },
      ];

    default:
      return baseItems;
  }
}

// Sidebar state store
interface SidebarState {
  isCollapsed: boolean;
  toggle: () => void;
  setCollapsed: (collapsed: boolean) => void;
}

export const useSidebarStore = create<SidebarState>()(
  persist(
    (set) => ({
      isCollapsed: false,
      toggle: () => set((state) => ({ isCollapsed: !state.isCollapsed })),
      setCollapsed: (collapsed: boolean) => set({ isCollapsed: collapsed }),
    }),
    {
      name: 'sidebar-storage',
    }
  )
);

// Gate Verification Store
const VERIFICATION_TIMEOUT = 60; // seconds (1 minute for ML detection)

const createInitialItems = (): Record<VerificationItemType, VerificationItem> => ({
  helmet: { type: 'helmet', rfidStatus: 'pending', mlStatus: 'pending' },
  face: { type: 'face', rfidStatus: 'pending', mlStatus: 'pending' }, // rfidStatus stays pending (no RFID for face)
  vest: { type: 'vest', rfidStatus: 'pending', mlStatus: 'pending' },
  shoes: { type: 'shoes', rfidStatus: 'pending', mlStatus: 'pending' },
});

interface GateVerificationStore extends GateVerificationState {
  // Actions
  startVerification: (gateId: string, mineId: string) => void;
  resetVerification: () => void;

  // RFID updates
  updateRFIDStatus: (item: VerificationItemType, status: VerificationStatus, tagId?: string) => void;

  // ML updates
  updateMLStatus: (item: VerificationItemType, status: VerificationStatus, confidence?: number) => void;

  // Worker identification
  setIdentifiedWorker: (worker: Worker, confidence: number) => void;
  markAttendance: () => void;

  // Timer
  tick: () => void;
  stopTimer: () => void;

  // Outcome
  determineOutcome: () => void;

  // Computed helpers
  getPassedCount: () => number;
  getTotalChecks: () => number;
  canOverride: () => boolean;
}

export const useGateVerificationStore = create<GateVerificationStore>()((set, get) => ({
  // Initial state
  items: createInitialItems(),
  overallStatus: 'idle',
  identifiedWorker: null,
  identificationConfidence: 0,
  attendanceMarked: false,
  timeRemaining: VERIFICATION_TIMEOUT,
  isTimerRunning: false,
  startTime: null,
  sessionId: null,
  selectedGateId: null,
  selectedMineId: null,

  // Start verification session
  startVerification: (gateId: string, mineId: string) => {
    set({
      items: createInitialItems(),
      overallStatus: 'verifying',
      identifiedWorker: null,
      identificationConfidence: 0,
      attendanceMarked: false,
      timeRemaining: VERIFICATION_TIMEOUT,
      isTimerRunning: true,
      startTime: Date.now(),
      sessionId: `session-${Date.now()}`,
      selectedGateId: gateId,
      selectedMineId: mineId,
    });
  },

  // Reset to idle state
  resetVerification: () => {
    set({
      items: createInitialItems(),
      overallStatus: 'idle',
      identifiedWorker: null,
      identificationConfidence: 0,
      attendanceMarked: false,
      timeRemaining: VERIFICATION_TIMEOUT,
      isTimerRunning: false,
      startTime: null,
      sessionId: null,
    });
  },

  // Update RFID status for an item
  updateRFIDStatus: (itemType: VerificationItemType, status: VerificationStatus, tagId?: string) => {
    set((state) => ({
      items: {
        ...state.items,
        [itemType]: {
          ...state.items[itemType],
          rfidStatus: status,
          rfidTagId: tagId,
        },
      },
    }));
  },

  // Update ML status for an item
  updateMLStatus: (itemType: VerificationItemType, status: VerificationStatus, confidence?: number) => {
    set((state) => ({
      items: {
        ...state.items,
        [itemType]: {
          ...state.items[itemType],
          mlStatus: status,
          mlConfidence: confidence,
        },
      },
    }));
  },

  // Set identified worker from face recognition
  setIdentifiedWorker: (worker: Worker, confidence: number) => {
    set({
      identifiedWorker: worker,
      identificationConfidence: confidence,
    });
  },

  // Mark attendance
  markAttendance: () => {
    set({ attendanceMarked: true });
  },

  // Timer tick - calculates remaining time from startTime
  tick: () => {
    const state = get();
    if (!state.isTimerRunning || !state.startTime) return;

    const elapsed = Math.floor((Date.now() - state.startTime) / 1000);
    const remaining = Math.max(0, VERIFICATION_TIMEOUT - elapsed);
    set({ timeRemaining: remaining });

    // Auto-determine outcome when time runs out
    if (remaining === 0) {
      get().determineOutcome();
    }
  },

  // Stop timer
  stopTimer: () => {
    set({ isTimerRunning: false });
  },

  // Determine final outcome
  determineOutcome: () => {
    const state = get();
    const passedCount = get().getPassedCount();
    const totalChecks = get().getTotalChecks();

    let outcome: 'passed' | 'failed' | 'warning';

    if (passedCount === totalChecks) {
      outcome = 'passed';
    } else if (passedCount >= 5) {
      outcome = 'warning';
    } else {
      outcome = 'failed';
    }

    set({
      overallStatus: outcome,
      isTimerRunning: false,
    });
  },

  // Count passed checks (both RFID and ML)
  getPassedCount: () => {
    const state = get();
    let count = 0;

    Object.values(state.items).forEach((item) => {
      // For face, only count ML (no RFID)
      if (item.type === 'face') {
        if (item.mlStatus === 'passed') count++;
      } else {
        // For other items, count both RFID and ML
        if (item.rfidStatus === 'passed') count++;
        if (item.mlStatus === 'passed') count++;
      }
    });

    return count;
  },

  // Total checks: helmet(2) + face(1) + vest(2) + shoes(2) = 7
  getTotalChecks: () => 7,

  // Can override if 5+ checks passed
  canOverride: () => {
    return get().getPassedCount() >= 5;
  },
}));

// ==================== MINE TRACKING STORE ====================

// Types for mine tracking
export interface WallData {
  start: { x: number; z: number };
  end: { x: number; z: number };
  height: number;
  length: number;
}

export interface RoomData {
  center: { x: number; z: number };
  width: number;
  depth: number;
  area: number;
}

export interface MineData {
  walls: WallData[];
  rooms: RoomData[];
  bounds: { minX: number; maxX: number; minZ: number; maxZ: number };
  wallHeight: number;
  imageWidth?: number;
  imageHeight?: number;
}

export interface Position3D {
  x: number;
  y: number;
  z: number;
}

interface MineTrackingState {
  // Mine data (persisted to localStorage)
  mineData: MineData | null;
  entrancePosition: { x: number; z: number };
  mineName: string;

  // Worker tracking (session only - not persisted)
  workerPosition: Position3D;
  workerHeading: number; // radians
  isTracking: boolean;
  stepCount: number;

  // View mode
  viewMode: 'track' | 'explore';

  // Connection status
  connectionStatus: 'disconnected' | 'connected' | 'simulating';

  // Actions
  setMineData: (data: MineData, name?: string) => void;
  setEntrancePosition: (pos: { x: number; z: number }) => void;
  updateWorkerPosition: (pos: Position3D, heading: number) => void;
  incrementStepCount: () => void;
  setViewMode: (mode: 'track' | 'explore') => void;
  setConnectionStatus: (status: 'disconnected' | 'connected' | 'simulating') => void;
  setIsTracking: (tracking: boolean) => void;
  resetWorkerPosition: () => void;
  clearMineData: () => void;
}

// Calculate default entrance position from mine bounds
function calculateDefaultEntrance(bounds: MineData['bounds']): { x: number; z: number } {
  // Place entrance at center of max-Z edge (typically the mine entrance)
  return {
    x: (bounds.minX + bounds.maxX) / 2,
    z: bounds.maxZ + 2, // Slightly outside the max boundary
  };
}

export const useMineTrackingStore = create<MineTrackingState>()(
  persist(
    (set, get) => ({
      // Initial state
      mineData: null,
      entrancePosition: { x: 0, z: 0 },
      mineName: 'Untitled Mine',
      workerPosition: { x: 0, y: 1.7, z: 0 },
      workerHeading: 0,
      isTracking: false,
      stepCount: 0,
      viewMode: 'explore',
      connectionStatus: 'disconnected',

      // Set mine data and calculate default entrance
      setMineData: (data: MineData, name?: string) => {
        const entrance = calculateDefaultEntrance(data.bounds);
        set({
          mineData: data,
          mineName: name || 'Untitled Mine',
          entrancePosition: entrance,
          workerPosition: { x: entrance.x, y: 1.7, z: entrance.z },
        });
      },

      // Set custom entrance position
      setEntrancePosition: (pos: { x: number; z: number }) => {
        set({
          entrancePosition: pos,
          workerPosition: { x: pos.x, y: 1.7, z: pos.z },
        });
      },

      // Update worker position and heading (called from tracking hook)
      updateWorkerPosition: (pos: Position3D, heading: number) => {
        set({
          workerPosition: pos,
          workerHeading: heading,
        });
      },

      // Increment step count
      incrementStepCount: () => {
        set((state) => ({ stepCount: state.stepCount + 1 }));
      },

      // Set view mode (track worker or explore freely)
      setViewMode: (mode: 'track' | 'explore') => {
        set({ viewMode: mode });
      },

      // Set connection status
      setConnectionStatus: (status) => {
        set({ connectionStatus: status });
      },

      // Set tracking enabled/disabled
      setIsTracking: (tracking: boolean) => {
        set({ isTracking: tracking });
      },

      // Reset worker position to entrance
      resetWorkerPosition: () => {
        const state = get();
        set({
          workerPosition: {
            x: state.entrancePosition.x,
            y: 1.7,
            z: state.entrancePosition.z,
          },
          workerHeading: 0,
          stepCount: 0,
        });
      },

      // Clear all mine data
      clearMineData: () => {
        set({
          mineData: null,
          entrancePosition: { x: 0, z: 0 },
          mineName: 'Untitled Mine',
          workerPosition: { x: 0, y: 1.7, z: 0 },
          workerHeading: 0,
          isTracking: false,
          stepCount: 0,
          viewMode: 'explore',
          connectionStatus: 'disconnected',
        });
      },
    }),
    {
      name: 'mine-tracking-storage',
      // Only persist mine data and entrance, not worker position
      partialize: (state) => ({
        mineData: state.mineData,
        entrancePosition: state.entrancePosition,
        mineName: state.mineName,
      }),
    }
  )
);
