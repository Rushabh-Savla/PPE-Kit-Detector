// ==================== Enums ====================

export type UserRole =
  | 'super_admin'
  | 'general_manager'
  | 'area_safety_officer'
  | 'manager'
  | 'safety_officer'
  | 'shift_incharge'
  | 'worker';

export type ShiftType = 'day' | 'afternoon' | 'night';

export type GateType = 'entry' | 'exit' | 'both';

export type AlertSeverity = 'low' | 'medium' | 'high' | 'critical';

export type AlertStatus = 'active' | 'acknowledged' | 'resolved';

export type EntryStatus = 'pending' | 'approved' | 'denied' | 'override';

// ==================== User Types ====================

export interface User {
  id: string;
  username: string;
  full_name: string;
  email?: string;
  phone?: string;
  role: UserRole;
  mine_id?: string;
  mine_ids?: string[];
  assigned_shift?: ShiftType;
  assigned_gate_id?: string;
  is_active: boolean;
  created_at: string;
  last_login?: string;
}

export interface UserCreate {
  username: string;
  password: string;
  full_name: string;
  email?: string;
  phone?: string;
  role: UserRole;
  mine_id?: string;
  mine_ids?: string[];
  assigned_shift?: ShiftType;
  assigned_gate_id?: string;
}

// ==================== Worker Types ====================

export interface Worker {
  id: string;
  employee_id: string;
  name: string;
  department?: string;
  mine_id: string;
  mine_name?: string;
  zone_id?: string;
  zone_name?: string;
  assigned_shift: ShiftType;
  phone?: string;
  emergency_contact?: string;
  face_registered: boolean;
  is_active: boolean;
  created_at: string;
  compliance_score: number;
  total_violations: number;
  badges: string[];
}

export interface WorkerCreate {
  employee_id: string;
  name: string;
  password: string;
  department?: string;
  mine_id: string;
  zone_id?: string;
  assigned_shift: ShiftType;
  phone?: string;
  emergency_contact?: string;
}

// ==================== Mine Types ====================

export interface Zone {
  id: string;
  name: string;
  description?: string;
  risk_level: string;
  coordinates?: { x: number; y: number; width: number; height: number };
  worker_count: number;
}

export interface Gate {
  id: string;
  name: string;
  gate_type: GateType;
  zone_id?: string;
  location?: string;
  has_camera: boolean;
  is_active: boolean;
}

export interface Mine {
  id: string;
  name: string;
  location: string;
  description?: string;
  zones: Zone[];
  gates: Gate[];
  created_at: string;
  is_active: boolean;
}

// ==================== Gate Entry Types ====================

export interface GateEntry {
  id: string;
  gate_id: string;
  gate_name?: string;
  worker_id?: string;
  worker_name?: string;
  employee_id?: string;
  entry_type: 'entry' | 'exit';
  status: EntryStatus;
  ppe_status: Record<string, boolean>;
  violations: string[];
  timestamp: string;
  shift: ShiftType;
  override_by?: string;
  override_reason?: string;
}

export interface LiveEntriesResponse {
  current_shift: ShiftType;
  shift_name: string;
  shift_start: string;
  shift_end: string;
  entries: GateEntry[];
  summary: {
    total_entries: number;
    total_exits: number;
    currently_inside: number;
    total_violations: number;
    compliance_rate: number;
  };
}

// ==================== Alert Types ====================

export interface Alert {
  id: string;
  alert_type: string;
  severity: AlertSeverity;
  status: AlertStatus;
  message: string;
  mine_id: string;
  mine_name?: string;
  zone_id?: string;
  gate_id?: string;
  worker_id?: string;
  worker_name?: string;
  created_at: string;
  acknowledged_by?: string;
  acknowledged_at?: string;
  resolved_by?: string;
  resolved_at?: string;
  resolution_notes?: string;
}

export interface Warning {
  id: string;
  worker_id: string;
  worker_name: string;
  employee_id: string;
  warning_type: string;
  description: string;
  severity: string;
  issued_by: string;
  issued_by_name: string;
  issued_at: string;
  acknowledged: boolean;
  acknowledged_at?: string;
}

// ==================== Dashboard Types ====================

export interface ShiftInchargeDashboard {
  mine_id: string;
  mine_name: string;
  current_shift: ShiftType;
  shift_name: string;
  shift_start: string;
  shift_end: string;
  statistics: {
    expected_workers: number;
    workers_entered: number;
    workers_exited: number;
    currently_inside: number;
    ppe_compliant: number;
    ppe_non_compliant: number;
    violations_this_shift: number;
    compliance_rate: number;
    pending_alerts: number;
  };
  recent_entries: GateEntry[];
  active_alerts: Alert[];
}

export interface SafetyOfficerDashboard {
  mine_id: string;
  mine_name: string;
  compliance_rates: {
    today: number;
    this_week: number;
    this_month: number;
  };
  violations: {
    today: number;
    this_week: number;
  };
  violation_trends: Record<string, number>;
  high_risk_workers: Worker[];
  zone_risk_analysis: {
    zone_id: string;
    zone_name: string;
    risk_level: string;
    violations_this_week: number;
    worker_count: number;
  }[];
  recent_alerts: Alert[];
}

export interface ManagerDashboard {
  mine_id: string;
  mine_name: string;
  overview: {
    total_workers: number;
    active_workers_today: number;
    compliance_rate: number;
    pending_escalations: number;
    avg_entry_delay_minutes: number;
  };
  shift_performance: Record<ShiftType, number>;
  top_compliant_workers: Worker[];
}

export interface AreaSafetyOfficerDashboard {
  overall_compliance_rate: number;
  total_mines: number;
  mines_overview: {
    mine_id: string;
    mine_name: string;
    location?: string;
    compliance_rate: number;
    worker_count: number;
    violations_this_week: number;
    active_alerts: number;
  }[];
  critical_alerts: Alert[];
  risk_heatmap: {
    mine_id: string;
    mine_name: string;
    zone_id: string;
    zone_name: string;
    violations: number;
    risk_level: string;
  }[];
}

export interface GeneralManagerDashboard {
  organization_overview: {
    total_mines: number;
    total_workers: number;
    compliance_rate: number;
  };
  kpi_summary: {
    monthly_entries: number;
    monthly_violations: number;
    monthly_compliance_rate: number;
  };
  mine_performance: {
    mine_id: string;
    mine_name: string;
    compliance_rate: number;
    total_entries: number;
    violations: number;
  }[];
  strategic_alerts: Alert[];
  financial_insights: {
    estimated_cost_savings: number;
    violations_prevented_this_week: number;
  };
  regulatory_status: {
    compliance_threshold: number;
    current_compliance: number;
    status: string;
  };
}

export interface WorkerDashboard {
  worker: {
    id: string;
    employee_id: string;
    name: string;
    department?: string;
    mine_name?: string;
    zone_name?: string;
  };
  compliance: {
    score: number;
    total_violations: number;
    current_streak_days: number;
  };
  badges: string[];
  statistics: {
    total_entries: number;
  };
  recent_violations: {
    date: string;
    time: string;
    violations: string[];
  }[];
  shift_info: {
    assigned_shift: ShiftType;
    shift_name: string;
    start_time: string;
    end_time: string;
    is_current_shift: boolean;
  };
  notifications: {
    type: string;
    id: string;
    message: string;
    date: string;
    severity: string;
  }[];
}

// ==================== Mine Visualization Types ====================

export interface MineVisualization {
  mine_id: string;
  mine_name: string;
  zones: {
    id: string;
    name: string;
    description?: string;
    risk_level: string;
    coordinates: { x: number; y: number; width: number; height: number };
    worker_count: number;
    violations_today: number;
  }[];
  gates: {
    id: string;
    name: string;
    gate_type: string;
    zone_id?: string;
    location?: string;
    has_camera: boolean;
    position: { x: number; y: number };
  }[];
  workers_positions: {
    id: string;
    employee_id: string;
    name: string;
    zone_id?: string;
    zone_name?: string;
    last_entry_time: string;
    ppe_compliant: boolean;
  }[];
  risk_zones: {
    zone_id: string;
    zone_name: string;
    risk_level: string;
    violations_today: number;
  }[];
  statistics: {
    total_workers: number;
    workers_inside: number;
    total_zones: number;
    total_gates: number;
    violations_today: number;
    compliance_rate: number;
  };
}

// ==================== Auth Types ====================

export interface AuthState {
  isAuthenticated: boolean;
  token: string | null;
  user: User | null;
  worker: Worker | null;
  userType: 'staff' | 'worker' | null;
}

export interface LoginResponse {
  access_token: string;
  token_type: string;
  user: User;
}

export interface WorkerLoginResponse {
  access_token: string;
  token_type: string;
  worker: Worker;
}

// ==================== Legacy Types (for backward compatibility) ====================

export interface Employee {
  id: string;
  name: string;
  employee_id: string;
  department?: string;
  created_at: string;
  face_registered: boolean;
}

export interface AttendanceRecord {
  id: string;
  employee_id: string;
  employee_name: string;
  type: 'check_in' | 'check_out';
  timestamp: string;
  date: string;
  image?: string;
}

export interface TodayAttendance {
  date: string;
  total_employees: number;
  present: number;
  absent: number;
  attendance: {
    employee_id: string;
    employee_name: string;
    check_ins: string[];
    check_outs: string[];
    is_present: boolean;
  }[];
}

export interface Violation {
  label: string;
  confidence: number;
  bbox: number[];
  is_violation: boolean;
}

export interface PPEViolationRecord {
  id: string;
  employee_id?: string;
  employee_name?: string;
  violations: Violation[];
  timestamp: string;
  location?: string;
  image?: string;
}

export interface DetectionSummary {
  ppe_detected: Record<string, number>;
  violations: Record<string, number>;
  total_ppe_items: number;
  total_violations: number;
  faces_detected: number;
  identified_persons: string[];
  safety_compliant: boolean;
}

export interface DetectionResult {
  success: boolean;
  image: string;
  detections: {
    ppe: Violation[];
    faces: {
      name: string | null;
      confidence: number;
      bbox: number[];
    }[];
    violations: Violation[];
    summary: DetectionSummary;
  };
  violations_logged?: boolean;
}

export interface DashboardStats {
  total_employees: number;
  present_today: number;
  absent_today: number;
  violations_today: number;
  violations_this_week: number;
  compliance_rate: number;
  recent_attendance?: AttendanceRecord[];
  recent_violations?: PPEViolationRecord[];
}

export interface AttendanceReportRecord {
  employee_id: string;
  employee_name: string;
  date: string;
  check_ins: string[];
  check_outs: string[];
  total_hours?: number;
}

export interface AttendanceReport {
  start_date: string;
  end_date: string;
  records: AttendanceReportRecord[];
  summary: {
    unique_employees: number;
    total_days: number;
    total_records: number;
    total_hours_worked: number;
  };
}

export interface ViolationReport {
  start_date: string;
  end_date: string;
  violations: PPEViolationRecord[];
  summary: {
    total_violations: number;
    violation_breakdown: Record<string, number>;
    employees_with_violations: number;
  };
}

// ==================== Utility Types ====================

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
}

export interface ShiftInfo {
  value: ShiftType;
  label: string;
  start_time: string;
  end_time: string;
}

export interface RoleInfo {
  value: UserRole;
  label: string;
  description: string;
}

// ==================== Prediction Types ====================

export type RiskCategory = 'low' | 'medium' | 'high' | 'critical';
export type AttendancePattern = 'regular' | 'irregular' | 'declining';

export interface RiskFactor {
  factor: string;
  impact: number;
  description: string;
}

export interface WorkerPrediction {
  worker_id: string;
  employee_id: string;
  worker_name?: string;
  prediction_date: string;

  // Risk Scores (0-100)
  overall_risk_score: number;
  risk_category: RiskCategory;
  violation_risk_score: number;
  attendance_risk_score: number;
  compliance_trend_score: number;

  // Predictions
  predicted_violations_count: number;
  predicted_absent_days: number;
  high_risk_ppe_items: string[];

  // Classification
  requires_intervention: boolean;
  attendance_pattern: AttendancePattern;
  consecutive_absence_risk: number;
  attendance_rate_30d: number;

  // Explainability
  risk_factors: RiskFactor[];
  confidence: number;

  // Metadata
  model_version: string;
  created_at: string;
  expires_at: string;
}

export interface AtRiskWorkerSummary {
  worker_id: string;
  employee_id: string;
  worker_name: string;
  risk_score: number;
  risk_category: RiskCategory;
  main_issue: string;
  requires_intervention: boolean;
}

export interface AtRiskWorkersResponse {
  total_at_risk: number;
  by_category: Record<string, number>;
  workers: AtRiskWorkerSummary[];
}

export interface PredictionTrend {
  date: string;
  low?: { count: number; avg_risk_score: number };
  medium?: { count: number; avg_risk_score: number };
  high?: { count: number; avg_risk_score: number };
  critical?: { count: number; avg_risk_score: number };
}

export interface PredictionTrendsResponse {
  trends: Record<string, Record<string, { count: number; avg_risk_score: number }>>;
}

export interface WorkerPredictionDetail {
  worker_id: string;
  employee_id: string;
  name: string;
  prediction: WorkerPrediction;
  recommendations: string[];
}

// ==================== Health Monitoring Types ====================

export type HealthStatus = 'normal' | 'warning' | 'critical';

export interface HealthReading {
  id: string;
  worker_id: string;
  worker_name?: string;
  employee_id?: string;
  mine_id?: string;
  mine_name?: string;
  // Vital signs from sensors
  spo2: number;           // Blood oxygen saturation (%)
  systolic_bp: number;    // Systolic blood pressure (mmHg)
  diastolic_bp: number;   // Diastolic blood pressure (mmHg)
  heart_rate: number;     // Heart rate (bpm)
  body_temperature?: number; // Optional: Body temperature (°C)
  // Metadata
  timestamp: string;
  sensor_id?: string;
  status: HealthStatus;
  alerts: string[];
}

export interface WorkerHealthSummary {
  worker_id: string;
  worker_name: string;
  employee_id: string;
  mine_id?: string;
  mine_name?: string;
  department?: string;
  // Average readings
  avg_spo2: number;
  avg_systolic_bp: number;
  avg_diastolic_bp: number;
  avg_heart_rate: number;
  // Latest reading
  latest_reading?: HealthReading;
  // Health status
  overall_status: HealthStatus;
  readings_count: number;
  alerts_count: number;
  last_updated: string;
}

export interface HealthDashboardStats {
  total_workers_monitored: number;
  workers_normal: number;
  workers_warning: number;
  workers_critical: number;
  avg_spo2: number;
  avg_systolic_bp: number;
  avg_diastolic_bp: number;
  avg_heart_rate: number;
  readings_today: number;
  alerts_today: number;
}

export interface HealthTrendPoint {
  date: string;
  avg_spo2: number;
  avg_systolic_bp: number;
  avg_diastolic_bp: number;
  avg_heart_rate: number;
  normal_count: number;
  warning_count: number;
  critical_count: number;
}

export interface MineHealthSummary {
  mine_id: string;
  mine_name: string;
  workers_monitored: number;
  avg_spo2: number;
  avg_systolic_bp: number;
  avg_diastolic_bp: number;
  avg_heart_rate: number;
  workers_normal: number;
  workers_warning: number;
  workers_critical: number;
  health_score: number;  // 0-100 overall health score
}

export interface HealthDashboardResponse {
  stats: HealthDashboardStats;
  mine_summaries: MineHealthSummary[];
  trends: HealthTrendPoint[];
  at_risk_workers: WorkerHealthSummary[];
  recent_alerts: {
    worker_id: string;
    worker_name: string;
    alert_type: string;
    message: string;
    timestamp: string;
    severity: HealthStatus;
  }[];
}

// ==================== Gate Verification Types ====================

export type VerificationStatus = 'pending' | 'checking' | 'passed' | 'failed' | 'warning';

export type VerificationItemType = 'helmet' | 'face' | 'vest' | 'shoes';

export interface VerificationItem {
  type: VerificationItemType;
  rfidStatus: VerificationStatus;  // 'pending' for face (no RFID)
  mlStatus: VerificationStatus;
  rfidTagId?: string;
  mlConfidence?: number;
}

export interface GateVerificationState {
  // Verification items
  items: Record<VerificationItemType, VerificationItem>;

  // Overall state
  overallStatus: 'idle' | 'verifying' | 'passed' | 'failed' | 'warning';

  // Worker identification
  identifiedWorker: Worker | null;
  identificationConfidence: number;
  attendanceMarked: boolean;

  // Timer
  timeRemaining: number;
  isTimerRunning: boolean;
  startTime: number | null;

  // Session info
  sessionId: string | null;
  selectedGateId: string | null;
  selectedMineId: string | null;
}

export interface RFIDScanEvent {
  tagId: string;
  tagType: 'helmet' | 'vest' | 'shoes';
  timestamp: Date;
}

export interface VerificationResult {
  success: boolean;
  passedChecks: number;
  totalChecks: number;
  worker: Worker | null;
  violations: string[];
  canOverride: boolean;
}

// ==================== Report Types ====================

export type ReportType =
  // Shift Incharge Reports
  | 'shift_summary'
  | 'shift_handover'
  | 'entry_exit_log'
  | 'alert_resolution'
  // Safety Officer Reports
  | 'weekly_compliance'
  | 'high_risk_workers'
  | 'zone_risk_analysis'
  | 'violation_trends'
  // Manager Reports
  | 'daily_operations'
  | 'shift_performance'
  | 'worker_rankings'
  | 'monthly_summary'
  | 'escalation_report'
  // Area Safety Officer Reports
  | 'mine_comparison'
  | 'risk_heatmap'
  | 'critical_incidents'
  | 'compliance_leaderboard'
  // General Manager Reports
  | 'executive_summary'
  | 'kpi_dashboard'
  | 'regulatory_compliance'
  | 'financial_impact'
  // Worker Reports
  | 'compliance_card'
  | 'worker_monthly';

export type ReportFormat = 'pdf' | 'excel' | 'csv';

export type ScheduleFrequency = 'daily' | 'weekly' | 'monthly';

export interface ReportTypeInfo {
  id: ReportType;
  name: string;
  description: string;
  available_formats: ReportFormat[];
  min_role: string;
  parameters: string[];
}

export interface EmailRecipient {
  email: string;
  name: string;
  type: 'to' | 'cc' | 'bcc';
}

export interface ScheduleConfig {
  frequency: ScheduleFrequency;
  time: string;           // "HH:MM" format (UTC)
  day_of_week?: number;   // 1-7 for weekly schedules
  day_of_month?: number;  // 1-31 for monthly schedules
}

export interface ReportScheduleConfig {
  format: ReportFormat[];
  date_range: 'previous_shift' | 'previous_day' | 'previous_week' | 'previous_month';
  filters?: Record<string, string | number | boolean>;
}

export interface GenerateReportRequest {
  report_type: ReportType;
  format: ReportFormat;
  start_date: string;
  end_date: string;
  mine_id?: string;
  filters?: Record<string, string | number | boolean>;
}

export interface GenerateReportResponse {
  success: boolean;
  report_id: string;
  download_url?: string;
  file_name: string;
  format: ReportFormat;
  generated_at: string;
  message: string;
}

export interface ReportSchedule {
  id: string;
  name: string;
  report_type: ReportType;
  role_target: UserRole;
  created_by: string;
  mine_id?: string;
  schedule: ScheduleConfig;
  recipients: EmailRecipient[];
  config: ReportScheduleConfig;
  is_active: boolean;
  next_run: string;
  last_run?: string;
  created_at: string;
}

export interface CreateScheduleRequest {
  name: string;
  report_type: ReportType;
  mine_id?: string;
  schedule: ScheduleConfig;
  recipients: EmailRecipient[];
  config: ReportScheduleConfig;
}

export interface UpdateScheduleRequest {
  name?: string;
  schedule?: ScheduleConfig;
  recipients?: EmailRecipient[];
  config?: ReportScheduleConfig;
  is_active?: boolean;
}

export interface ReportHistory {
  id: string;
  schedule_id?: string;
  report_type: ReportType;
  format: ReportFormat;
  generated_by: string;
  mine_id?: string;
  file_path: string;
  file_size: number;
  status: 'success' | 'failed';
  error_message?: string;
  generated_at: string;
  email_sent: boolean;
  recipients_count: number;
}
