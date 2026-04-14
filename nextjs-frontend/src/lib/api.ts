import axios from 'axios';
import type {
  Employee,
  AttendanceRecord,
  TodayAttendance,
  PPEViolationRecord,
  DetectionResult,
  DashboardStats,
  AttendanceReport,
  ViolationReport,
  User,
  UserCreate,
  Worker,
  WorkerCreate,
  Mine,
  GateEntry,
  LiveEntriesResponse,
  Alert,
  Warning,
  ShiftInchargeDashboard,
  SafetyOfficerDashboard,
  ManagerDashboard,
  AreaSafetyOfficerDashboard,
  GeneralManagerDashboard,
  WorkerDashboard,
  MineVisualization,
  LoginResponse,
  WorkerLoginResponse,
  ShiftType,
  UserRole,
  AlertSeverity,
  EntryStatus,
  WorkerPrediction,
  AtRiskWorkersResponse,
  PredictionTrendsResponse,
  WorkerPredictionDetail,
  RiskCategory,
  HealthReading,
  WorkerHealthSummary,
  HealthDashboardResponse,
  ReportTypeInfo,
  GenerateReportRequest,
  GenerateReportResponse,
  ReportSchedule,
  CreateScheduleRequest,
  UpdateScheduleRequest,
} from '@/types';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// Create axios instance
const api = axios.create({
  baseURL: API_URL,
});

// Add auth token to requests
api.interceptors.request.use((config) => {
  if (typeof window !== 'undefined') {
    let token = localStorage.getItem('token');

    if (!token) {
      try {
        const authStorage = localStorage.getItem('auth-storage');
        if (authStorage) {
          const parsed = JSON.parse(authStorage);
          token = parsed?.state?.token;
        }
      } catch (e) {
        console.error('Failed to parse auth storage:', e);
      }
    }

    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
  }
  return config;
});

// Handle auth errors
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      if (typeof window !== 'undefined') {
        localStorage.removeItem('token');
        localStorage.removeItem('auth-storage');
        window.location.href = '/login';
      }
    }
    return Promise.reject(error);
  }
);

// ==================== Auth APIs ====================

export const authApi = {
  // Staff login
  login: async (username: string, password: string): Promise<LoginResponse> => {
    const response = await api.post('/auth/login', { username, password });
    return response.data;
  },

  // Worker login
  workerLogin: async (employeeId: string, password: string): Promise<WorkerLoginResponse> => {
    const response = await api.post('/auth/worker/login', {
      employee_id: employeeId,
      password,
    });
    return response.data;
  },

  verify: async () => {
    const response = await api.get('/auth/verify');
    return response.data;
  },

  getMe: async () => {
    const response = await api.get('/auth/me');
    return response.data;
  },

  changePassword: async (currentPassword: string, newPassword: string) => {
    const response = await api.post('/auth/change-password', {
      current_password: currentPassword,
      new_password: newPassword,
    });
    return response.data;
  },
};

// ==================== User APIs ====================

export const userApi = {
  list: async (params?: {
    skip?: number;
    limit?: number;
    role?: UserRole;
    mine_id?: string;
    search?: string;
    is_active?: boolean;
  }) => {
    const searchParams = new URLSearchParams();
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined) searchParams.append(key, String(value));
      });
    }
    const response = await api.get(`/users?${searchParams}`);
    return response.data as { users: User[]; total: number };
  },

  get: async (userId: string): Promise<User> => {
    const response = await api.get(`/users/${userId}`);
    return response.data;
  },

  create: async (data: UserCreate): Promise<User> => {
    const response = await api.post('/users', data);
    return response.data;
  },

  update: async (userId: string, data: Partial<UserCreate>): Promise<User> => {
    const response = await api.put(`/users/${userId}`, data);
    return response.data;
  },

  delete: async (userId: string) => {
    const response = await api.delete(`/users/${userId}`);
    return response.data;
  },

  resetPassword: async (userId: string, newPassword: string) => {
    const response = await api.post(`/users/${userId}/reset-password`, null, {
      params: { new_password: newPassword },
    });
    return response.data;
  },

  getAvailableRoles: async () => {
    const response = await api.get('/users/roles/available');
    return response.data as { roles: { value: string; label: string }[] };
  },
};

// ==================== Worker APIs ====================

export const workerApi = {
  list: async (params?: {
    skip?: number;
    limit?: number;
    mine_id?: string;
    zone_id?: string;
    shift?: ShiftType;
    department?: string;
    search?: string;
    is_active?: boolean;
    face_registered?: boolean;
  }) => {
    const searchParams = new URLSearchParams();
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined) searchParams.append(key, String(value));
      });
    }
    const response = await api.get(`/workers?${searchParams}`);
    return response.data as { workers: Worker[]; total: number };
  },

  get: async (workerId: string): Promise<Worker> => {
    const response = await api.get(`/workers/${workerId}`);
    return response.data;
  },

  create: async (data: WorkerCreate): Promise<Worker> => {
    const response = await api.post('/workers', data);
    return response.data;
  },

  update: async (workerId: string, data: Partial<WorkerCreate>): Promise<Worker> => {
    const response = await api.put(`/workers/${workerId}`, data);
    return response.data;
  },

  delete: async (workerId: string) => {
    const response = await api.delete(`/workers/${workerId}`);
    return response.data;
  },

  resetPassword: async (workerId: string, newPassword: string) => {
    const formData = new FormData();
    formData.append('new_password', newPassword);
    const response = await api.post(`/workers/${workerId}/reset-password`, formData);
    return response.data;
  },

  registerFace: async (workerId: string, image: File) => {
    const formData = new FormData();
    formData.append('file', image);
    const response = await api.post(`/workers/${workerId}/register-face`, formData);
    return response.data;
  },

  getViolations: async (workerId: string, skip = 0, limit = 20) => {
    const response = await api.get(`/workers/${workerId}/violations`, {
      params: { skip, limit },
    });
    return response.data;
  },

  getAttendance: async (workerId: string, params?: {
    start_date?: string;
    end_date?: string;
    skip?: number;
    limit?: number;
  }) => {
    const searchParams = new URLSearchParams();
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined) searchParams.append(key, String(value));
      });
    }
    const response = await api.get(`/workers/${workerId}/attendance?${searchParams}`);
    return response.data;
  },
};

// ==================== Mine APIs ====================

export const mineApi = {
  list: async (params?: {
    skip?: number;
    limit?: number;
    search?: string;
    is_active?: boolean;
  }) => {
    const searchParams = new URLSearchParams();
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined) searchParams.append(key, String(value));
      });
    }
    const response = await api.get(`/mines?${searchParams}`);
    return response.data as { mines: Mine[]; total: number };
  },

  get: async (mineId: string): Promise<Mine> => {
    const response = await api.get(`/mines/${mineId}`);
    return response.data;
  },

  create: async (data: { name: string; location: string; description?: string }): Promise<Mine> => {
    const response = await api.post('/mines', data);
    return response.data;
  },

  update: async (mineId: string, data: { name: string; location: string; description?: string }): Promise<Mine> => {
    const response = await api.put(`/mines/${mineId}`, data);
    return response.data;
  },

  delete: async (mineId: string) => {
    const response = await api.delete(`/mines/${mineId}`);
    return response.data;
  },

  getVisualization: async (mineId: string): Promise<MineVisualization> => {
    const response = await api.get(`/mines/${mineId}/visualization`);
    return response.data;
  },

  // Zone endpoints
  listZones: async (mineId: string) => {
    const response = await api.get(`/mines/${mineId}/zones`);
    return response.data;
  },

  createZone: async (mineId: string, data: {
    name: string;
    description?: string;
    risk_level?: string;
    coordinates?: object;
  }) => {
    const response = await api.post(`/mines/${mineId}/zones`, data);
    return response.data;
  },

  updateZone: async (mineId: string, zoneId: string, data: {
    name: string;
    description?: string;
    risk_level?: string;
    coordinates?: object;
  }) => {
    const response = await api.put(`/mines/${mineId}/zones/${zoneId}`, data);
    return response.data;
  },

  deleteZone: async (mineId: string, zoneId: string) => {
    const response = await api.delete(`/mines/${mineId}/zones/${zoneId}`);
    return response.data;
  },

  // Gate endpoints
  listGates: async (mineId: string) => {
    const response = await api.get(`/mines/${mineId}/gates`);
    return response.data;
  },

  createGate: async (mineId: string, data: {
    name: string;
    gate_type?: string;
    zone_id?: string;
    location?: string;
    has_camera?: boolean;
  }) => {
    const response = await api.post(`/mines/${mineId}/gates`, data);
    return response.data;
  },

  updateGate: async (mineId: string, gateId: string, data: {
    name: string;
    gate_type?: string;
    zone_id?: string;
    location?: string;
    has_camera?: boolean;
  }) => {
    const response = await api.put(`/mines/${mineId}/gates/${gateId}`, data);
    return response.data;
  },

  deleteGate: async (mineId: string, gateId: string) => {
    const response = await api.delete(`/mines/${mineId}/gates/${gateId}`);
    return response.data;
  },
};

// ==================== Gate Entry APIs ====================

export const gateEntryApi = {
  detect: async (gateId: string, image: File, entryType: 'entry' | 'exit' = 'entry') => {
    const formData = new FormData();
    // Only append gate_id if it's a valid ID (not empty or 'default')
    if (gateId && gateId !== 'default') {
      formData.append('gate_id', gateId);
    }
    formData.append('file', image);
    formData.append('entry_type', entryType);
    const response = await api.post('/gate-entries/detect', formData);
    return response.data;
  },

  override: async (entryId: string, reason: string) => {
    const formData = new FormData();
    formData.append('reason', reason);
    const response = await api.post(`/gate-entries/${entryId}/override`, formData);
    return response.data;
  },

  list: async (params?: {
    skip?: number;
    limit?: number;
    gate_id?: string;
    mine_id?: string;
    worker_id?: string;
    shift?: ShiftType;
    status?: EntryStatus;
    entry_type?: string;
    has_violations?: boolean;
    start_date?: string;
    end_date?: string;
  }) => {
    const searchParams = new URLSearchParams();
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined) searchParams.append(key, String(value));
      });
    }
    const response = await api.get(`/gate-entries?${searchParams}`);
    return response.data as { entries: GateEntry[]; total: number };
  },

  getLive: async (params?: {
    gate_id?: string;
    mine_id?: string;
    limit?: number;
  }): Promise<LiveEntriesResponse> => {
    const searchParams = new URLSearchParams();
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined) searchParams.append(key, String(value));
      });
    }
    const response = await api.get(`/gate-entries/live?${searchParams}`);
    return response.data;
  },

  get: async (entryId: string) => {
    const response = await api.get(`/gate-entries/${entryId}`);
    return response.data;
  },
};

// ==================== Alert APIs ====================

export const alertApi = {
  create: async (data: {
    alert_type: string;
    severity: AlertSeverity;
    message: string;
    mine_id: string;
    zone_id?: string;
    gate_id?: string;
    worker_id?: string;
    metadata?: object;
  }): Promise<Alert> => {
    const response = await api.post('/alerts', data);
    return response.data;
  },

  list: async (params?: {
    skip?: number;
    limit?: number;
    mine_id?: string;
    status?: string;
    severity?: AlertSeverity;
    alert_type?: string;
    start_date?: string;
    end_date?: string;
  }) => {
    const searchParams = new URLSearchParams();
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined) searchParams.append(key, String(value));
      });
    }
    const response = await api.get(`/alerts?${searchParams}`);
    return response.data as { alerts: Alert[]; total: number };
  },

  getActive: async (params?: {
    mine_id?: string;
    severity?: AlertSeverity;
    limit?: number;
  }) => {
    const searchParams = new URLSearchParams();
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined) searchParams.append(key, String(value));
      });
    }
    const response = await api.get(`/alerts/active?${searchParams}`);
    return response.data;
  },

  acknowledge: async (alertId: string) => {
    const response = await api.post(`/alerts/${alertId}/acknowledge`);
    return response.data;
  },

  resolve: async (alertId: string, resolutionNotes: string) => {
    const formData = new FormData();
    formData.append('resolution_notes', resolutionNotes);
    const response = await api.post(`/alerts/${alertId}/resolve`, formData);
    return response.data;
  },

  triggerEmergency: async (data: {
    mine_id: string;
    emergency_type: string;
    description: string;
    zone_id?: string;
  }) => {
    const formData = new FormData();
    formData.append('mine_id', data.mine_id);
    formData.append('emergency_type', data.emergency_type);
    formData.append('description', data.description);
    if (data.zone_id) formData.append('zone_id', data.zone_id);
    const response = await api.post('/alerts/emergency', formData);
    return response.data;
  },

  // Warning endpoints
  issueWarning: async (data: {
    worker_id: string;
    warning_type: string;
    description: string;
    severity?: string;
  }): Promise<Warning> => {
    const response = await api.post('/alerts/warnings', data);
    return response.data;
  },

  listWarnings: async (params?: {
    skip?: number;
    limit?: number;
    worker_id?: string;
    warning_type?: string;
    severity?: string;
    acknowledged?: boolean;
  }) => {
    const searchParams = new URLSearchParams();
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined) searchParams.append(key, String(value));
      });
    }
    const response = await api.get(`/alerts/warnings?${searchParams}`);
    return response.data as { warnings: Warning[]; total: number };
  },

  acknowledgeWarning: async (warningId: string) => {
    const response = await api.post(`/alerts/warnings/${warningId}/acknowledge`);
    return response.data;
  },
};

// ==================== Dashboard APIs ====================

export const dashboardApi = {
  // Legacy dashboard
  getStats: async (): Promise<DashboardStats> => {
    const response = await api.get('/dashboard/stats');
    return response.data;
  },

  // Super Admin dashboard - optimized endpoint
  getSuperAdmin: async () => {
    const response = await api.get('/dashboard/super-admin');
    return response.data;
  },

  // Role-specific dashboards
  getShiftIncharge: async (): Promise<ShiftInchargeDashboard> => {
    const response = await api.get('/dashboard/shift-incharge');
    return response.data;
  },

  getSafetyOfficer: async (): Promise<SafetyOfficerDashboard> => {
    const response = await api.get('/dashboard/safety-officer');
    return response.data;
  },

  getManager: async (): Promise<ManagerDashboard> => {
    const response = await api.get('/dashboard/manager');
    return response.data;
  },

  getAreaSafetyOfficer: async (): Promise<AreaSafetyOfficerDashboard> => {
    const response = await api.get('/dashboard/area-safety-officer');
    return response.data;
  },

  getGeneralManager: async (): Promise<GeneralManagerDashboard> => {
    const response = await api.get('/dashboard/general-manager');
    return response.data;
  },

  getWorker: async (): Promise<WorkerDashboard> => {
    const response = await api.get('/dashboard/worker');
    return response.data;
  },
};

// ==================== Legacy APIs (backward compatibility) ====================

export const employeeApi = {
  list: async (skip = 0, limit = 50, search?: string) => {
    const params = new URLSearchParams({ skip: String(skip), limit: String(limit) });
    if (search) params.append('search', search);
    const response = await api.get(`/employees?${params}`);
    return response.data as { employees: Employee[]; total: number };
  },
  get: async (employeeId: string) => {
    const response = await api.get(`/employees/${employeeId}`);
    return response.data as Employee;
  },
  create: async (name: string, employeeId: string, department?: string, faceImage?: File) => {
    const formData = new FormData();
    formData.append('name', name);
    formData.append('employee_id', employeeId);
    if (department) formData.append('department', department);
    if (faceImage) formData.append('file', faceImage);
    const response = await api.post('/employees', formData);
    return response.data;
  },
  update: async (employeeId: string, name?: string, department?: string) => {
    const formData = new FormData();
    if (name) formData.append('name', name);
    if (department !== undefined) formData.append('department', department);
    const response = await api.put(`/employees/${employeeId}`, formData);
    return response.data;
  },
  delete: async (employeeId: string) => {
    const response = await api.delete(`/employees/${employeeId}`);
    return response.data;
  },
  registerFace: async (employeeId: string, image: File) => {
    const formData = new FormData();
    formData.append('file', image);
    const response = await api.post(`/employees/${employeeId}/register-face`, formData);
    return response.data;
  },
};

export const detectionApi = {
  detect: async (image: File) => {
    const formData = new FormData();
    formData.append('file', image);
    const response = await api.post('/detect', formData);
    return response.data as DetectionResult;
  },
  detectAndLog: async (image: File, logViolations = true, location?: string) => {
    const formData = new FormData();
    formData.append('file', image);
    formData.append('log_violations', String(logViolations));
    if (location) formData.append('location', location);
    const response = await api.post('/detect-and-log', formData);
    return response.data as DetectionResult;
  },
};

export const attendanceApi = {
  checkIn: async (image: File) => {
    const formData = new FormData();
    formData.append('file', image);
    const response = await api.post('/attendance/check-in', formData);
    return response.data;
  },
  checkOut: async (image: File) => {
    const formData = new FormData();
    formData.append('file', image);
    const response = await api.post('/attendance/check-out', formData);
    return response.data;
  },
  getRecords: async (params?: {
    date?: string;
    employee_id?: string;
    start_date?: string;
    end_date?: string;
    skip?: number;
    limit?: number;
  }) => {
    const searchParams = new URLSearchParams();
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined) searchParams.append(key, String(value));
      });
    }
    const response = await api.get(`/attendance?${searchParams}`);
    return response.data as { records: AttendanceRecord[]; total: number };
  },
  getToday: async () => {
    const response = await api.get('/attendance/today');
    return response.data as TodayAttendance;
  },
};

export const violationsApi = {
  getRecords: async (params?: {
    date?: string;
    employee_id?: string;
    start_date?: string;
    end_date?: string;
    skip?: number;
    limit?: number;
  }) => {
    const searchParams = new URLSearchParams();
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined) searchParams.append(key, String(value));
      });
    }
    const response = await api.get(`/violations?${searchParams}`);
    return response.data as { violations: PPEViolationRecord[]; total: number };
  },
  getToday: async () => {
    const response = await api.get('/violations/today');
    return response.data;
  },
};

export const reportsApi = {
  getAttendanceReport: async (startDate: string, endDate: string, employeeId?: string) => {
    const params = new URLSearchParams({ start_date: startDate, end_date: endDate });
    if (employeeId) params.append('employee_id', employeeId);
    const response = await api.get(`/reports/attendance?${params}`);
    return response.data as AttendanceReport;
  },
  getViolationsReport: async (startDate: string, endDate: string, employeeId?: string) => {
    const params = new URLSearchParams({ start_date: startDate, end_date: endDate });
    if (employeeId) params.append('employee_id', employeeId);
    const response = await api.get(`/reports/violations?${params}`);
    return response.data as ViolationReport;
  },

  // New report generation endpoints
  getReportTypes: async () => {
    const response = await api.get('/reports/types');
    return response.data as { report_types: ReportTypeInfo[] };
  },

  generateReport: async (request: GenerateReportRequest) => {
    const response = await api.post('/reports/generate', request);
    return response.data as GenerateReportResponse;
  },

  downloadReport: async (reportId: string): Promise<Blob> => {
    const response = await api.get(`/reports/download/${reportId}`, {
      responseType: 'blob'
    });
    return response.data;
  },

  // Schedule management
  getSchedules: async () => {
    const response = await api.get('/reports/schedules');
    return response.data as { schedules: ReportSchedule[] };
  },

  createSchedule: async (schedule: CreateScheduleRequest) => {
    const response = await api.post('/reports/schedules', schedule);
    return response.data;
  },

  updateSchedule: async (id: string, updates: UpdateScheduleRequest) => {
    const response = await api.put(`/reports/schedules/${id}`, updates);
    return response.data;
  },

  deleteSchedule: async (id: string) => {
    const response = await api.delete(`/reports/schedules/${id}`);
    return response.data;
  },

  testSchedule: async (id: string) => {
    const response = await api.post(`/reports/schedules/${id}/test`);
    return response.data;
  },

  // Role-specific data endpoints
  getShiftInchargeSummary: async (shift: string, date?: string, gateId?: string) => {
    const params = new URLSearchParams({ shift });
    if (date) params.append('date', date);
    if (gateId) params.append('gate_id', gateId);
    const response = await api.get(`/reports/shift-incharge/summary?${params}`);
    return response.data;
  },

  getSafetyOfficerCompliance: async (startDate: string, endDate: string, groupBy = 'day') => {
    const params = new URLSearchParams({
      start_date: startDate,
      end_date: endDate,
      group_by: groupBy
    });
    const response = await api.get(`/reports/safety-officer/compliance?${params}`);
    return response.data;
  },

  getManagerOperations: async (startDate: string, endDate: string) => {
    const params = new URLSearchParams({ start_date: startDate, end_date: endDate });
    const response = await api.get(`/reports/manager/operations?${params}`);
    return response.data;
  },

  getWorkerComplianceCard: async (workerId: string) => {
    const response = await api.get(`/reports/worker/${workerId}/compliance-card`);
    return response.data;
  },
};

// ==================== Gas Sensor APIs ====================

export const gasSensorApi = {
  // Create a new gas reading
  createReading: async (data: {
    mine_id: string;
    zone_id?: string;
    gate_id?: string;
    methane_ppm: number;
    co_ppm: number;
    pressure_hpa?: number;
    altitude_m?: number;
    temperature_c?: number;
    humidity?: number;
    sensor_id?: string;
  }) => {
    const response = await api.post('/api/gas-sensors/readings', null, {
      params: data,
    });
    return response.data;
  },

  // Get gas readings with filters
  getReadings: async (params?: {
    mine_id?: string;
    zone_id?: string;
    gate_id?: string;
    start_date?: string;
    end_date?: string;
    severity?: string;
    limit?: number;
    skip?: number;
  }) => {
    const searchParams = new URLSearchParams();
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined) searchParams.append(key, String(value));
      });
    }
    const response = await api.get(`/api/gas-sensors/readings?${searchParams}`);
    return response.data;
  },

  // Get latest readings from all sensors
  getLatest: async (mine_id?: string) => {
    const params = mine_id ? `?mine_id=${mine_id}` : '';
    const response = await api.get(`/api/gas-sensors/latest${params}`);
    return response.data;
  },

  // Get gas monitoring statistics
  getStats: async (mine_id?: string, hours: number = 24) => {
    const params = new URLSearchParams({ hours: String(hours) });
    if (mine_id) params.append('mine_id', mine_id);
    const response = await api.get(`/api/gas-sensors/stats?${params}`);
    return response.data;
  },

  // Get gas-related alerts
  getAlerts: async (params?: {
    mine_id?: string;
    status?: string;
    limit?: number;
  }) => {
    const searchParams = new URLSearchParams();
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined) searchParams.append(key, String(value));
      });
    }
    const response = await api.get(`/api/gas-sensors/alerts?${searchParams}`);
    return response.data;
  },
};

// ==================== Utility APIs ====================

export const utilityApi = {
  getRoles: async () => {
    const response = await api.get('/api/roles');
    return response.data;
  },
  getShifts: async () => {
    const response = await api.get('/api/shifts');
    return response.data;
  },
};

// ==================== Prediction APIs ====================

export const predictionApi = {
  // Get prediction for a specific worker
  getWorkerPrediction: async (workerId: string): Promise<WorkerPrediction> => {
    const response = await api.get(`/predictions/worker/${workerId}`);
    return response.data;
  },

  // Get list of at-risk workers
  getAtRiskWorkers: async (params?: {
    mine_id?: string;
    risk_category?: RiskCategory;
    limit?: number;
  }): Promise<AtRiskWorkersResponse> => {
    const searchParams = new URLSearchParams();
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined) searchParams.append(key, String(value));
      });
    }
    const response = await api.get(`/predictions/at-risk-workers?${searchParams}`);
    return response.data;
  },

  // Generate predictions for all workers
  generateAll: async (params?: {
    mine_id?: string;
    force_refresh?: boolean;
  }) => {
    const searchParams = new URLSearchParams();
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined) searchParams.append(key, String(value));
      });
    }
    const response = await api.post(`/predictions/generate-all?${searchParams}`);
    return response.data;
  },

  // Get prediction trends over time
  getTrends: async (params?: {
    mine_id?: string;
    days_back?: number;
  }): Promise<PredictionTrendsResponse> => {
    const searchParams = new URLSearchParams();
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined) searchParams.append(key, String(value));
      });
    }
    const response = await api.get(`/predictions/trends?${searchParams}`);
    return response.data;
  },

  // Get worker prediction detail with recommendations
  getWorkerPredictionDetail: async (workerId: string): Promise<WorkerPredictionDetail> => {
    const response = await api.get(`/workers/${workerId}/prediction`);
    return response.data;
  },
};

// ==================== Health Monitoring APIs ====================

export const healthApi = {
  // Get health dashboard data (stats, trends, at-risk workers)
  getDashboard: async (params?: {
    mine_id?: string;
    days_back?: number;
  }): Promise<HealthDashboardResponse> => {
    const searchParams = new URLSearchParams();
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined) searchParams.append(key, String(value));
      });
    }
    const response = await api.get(`/api/health/dashboard?${searchParams}`);
    return response.data;
  },

  // Get health readings with filters
  getReadings: async (params?: {
    worker_id?: string;
    mine_id?: string;
    status?: string;
    start_date?: string;
    end_date?: string;
    limit?: number;
    skip?: number;
  }): Promise<{ readings: HealthReading[]; total: number }> => {
    const searchParams = new URLSearchParams();
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined) searchParams.append(key, String(value));
      });
    }
    const response = await api.get(`/api/health/readings?${searchParams}`);
    return response.data;
  },

  // Get worker health summary
  getWorkerHealth: async (workerId: string): Promise<WorkerHealthSummary> => {
    const response = await api.get(`/api/health/worker/${workerId}`);
    return response.data;
  },

  // Get all worker health summaries
  getWorkersSummary: async (params?: {
    mine_id?: string;
    status?: string;
    limit?: number;
  }): Promise<{ workers: WorkerHealthSummary[]; total: number }> => {
    const searchParams = new URLSearchParams();
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined) searchParams.append(key, String(value));
      });
    }
    const response = await api.get(`/api/health/workers?${searchParams}`);
    return response.data;
  },

  // Create a health reading (from sensor)
  createReading: async (data: {
    worker_id: string;
    spo2: number;
    systolic_bp: number;
    diastolic_bp: number;
    heart_rate: number;
    body_temperature?: number;
    sensor_id?: string;
  }): Promise<HealthReading> => {
    const response = await api.post('/api/health/readings', data);
    return response.data;
  },

  // Get latest readings for all workers
  getLatestReadings: async (mine_id?: string): Promise<HealthReading[]> => {
    const params = mine_id ? `?mine_id=${mine_id}` : '';
    const response = await api.get(`/api/health/latest${params}`);
    return response.data;
  },
};

// ==================== Video Stream APIs ====================

export const videoApi = {
  // Get video stream status
  getStatus: async () => {
    const response = await api.get('/video/status');
    return response.data;
  },

  // Start video stream
  start: async (cameraIndex: number = 0, maxFps: number = 30) => {
    const formData = new FormData();
    formData.append('video_source', cameraIndex.toString());
    formData.append('max_fps', maxFps.toString());
    const response = await api.post('/video/start', formData);
    return response.data;
  },

  // Stop video stream
  stop: async () => {
    const response = await api.post('/video/stop');
    return response.data;
  },

  // Get WebSocket URL for video stream
  getWebSocketUrl: () => {
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const apiHost = API_URL.replace(/^https?:\/\//, '');
    return `${wsProtocol}//${apiHost}/ws/video`;
  },
};

// ==================== Helmet Sensor APIs ====================

export interface HelmetReading {
  worker_id: string;
  worker_name: string;
  mine_id?: string;
  zone_id?: string;
  accel_x: number;
  accel_y: number;
  accel_z: number;
  gyro_x: number;
  gyro_y: number;
  gyro_z: number;
  roll: number;
  pitch: number;
  yaw: number;
  methane_ppm: number;
  co_raw: number;
  battery_voltage: number;
  battery_low: boolean;
  fsr_force: number;
  fsr_state: string;
  system_state: number;
  sos_active: boolean;
  manual_override: boolean;
  heart_rate: number;
  spo2: number;
  severity: 'normal' | 'medium' | 'high' | 'critical';
  timestamp: string;
}

export interface HelmetStats {
  active_helmets: number;
  total_workers: number;
  coverage_percent: number;
  avg_battery_voltage: number;
  avg_heart_rate: number;
  avg_spo2: number;
  low_battery_count: number;
  sos_active_count: number;
  critical_alerts: number;
  high_alerts: number;
  alerts_in_period: number;
  time_range_hours: number;
  severity_distribution: {
    normal: number;
    medium: number;
    high: number;
    critical: number;
  };
  is_safe: boolean;
}

export interface HelmetAlert {
  id: string;
  severity: 'medium' | 'high' | 'critical';
  status: 'active' | 'acknowledged' | 'resolved';
  message: string;
  worker_id: string;
  worker_name: string;
  details: {
    methane_ppm?: number;
    battery_voltage?: number;
    heart_rate?: number;
    spo2?: number;
    sos_active?: boolean;
  };
  created_at: string;
  acknowledged_at?: string;
  acknowledged_by?: string;
}

export const helmetApi = {
  // Get latest readings from all active helmets
  getLatest: async (mineId?: string): Promise<{ readings: HelmetReading[] }> => {
    const params = mineId ? `?mine_id=${mineId}` : '';
    const response = await api.get(`/api/helmet/latest${params}`);
    return response.data;
  },

  // Get latest reading for a specific worker
  getLatestByWorker: async (workerId: string): Promise<HelmetReading> => {
    const response = await api.get(`/api/helmet/latest/${workerId}`);
    return response.data;
  },

  // Get historical readings (only threshold breaches are stored)
  getReadings: async (params?: {
    worker_id?: string;
    mine_id?: string;
    start_date?: string;
    end_date?: string;
    severity?: string;
    limit?: number;
    skip?: number;
  }): Promise<{ readings: HelmetReading[]; total: number }> => {
    const searchParams = new URLSearchParams();
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined) searchParams.append(key, String(value));
      });
    }
    const response = await api.get(`/api/helmet/readings?${searchParams}`);
    return response.data;
  },

  // Get helmet statistics
  getStats: async (params?: {
    mine_id?: string;
    hours?: number;
  }): Promise<HelmetStats> => {
    const searchParams = new URLSearchParams();
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined) searchParams.append(key, String(value));
      });
    }
    const response = await api.get(`/api/helmet/stats?${searchParams}`);
    return response.data;
  },

  // Get helmet alerts
  getAlerts: async (params?: {
    worker_id?: string;
    status?: string;
    severity?: string;
    limit?: number;
  }): Promise<{ alerts: HelmetAlert[] }> => {
    const searchParams = new URLSearchParams();
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined) searchParams.append(key, String(value));
      });
    }
    const response = await api.get(`/api/helmet/alerts?${searchParams}`);
    return response.data;
  },

  // Acknowledge an alert
  acknowledgeAlert: async (alertId: string): Promise<{ success: boolean; message: string }> => {
    const response = await api.post(`/api/helmet/alerts/${alertId}/acknowledge`);
    return response.data;
  },

  // Resolve an alert
  resolveAlert: async (alertId: string, resolutionNote?: string): Promise<{ success: boolean; message: string }> => {
    const response = await api.post(`/api/helmet/alerts/${alertId}/resolve`, { resolution_note: resolutionNote });
    return response.data;
  },
};

// Generic API client for direct REST calls
export const apiClient = {
  get: async (url: string, token?: string | null) => {
    const response = await api.get(url);
    return response.data;
  },
  post: async (url: string, data?: any, token?: string | null) => {
    const response = await api.post(url, data);
    return response.data;
  },
  put: async (url: string, data?: any, token?: string | null) => {
    const response = await api.put(url, data);
    return response.data;
  },
  delete: async (url: string, token?: string | null) => {
    const response = await api.delete(url);
    return response.data;
  },
};

export default api;
