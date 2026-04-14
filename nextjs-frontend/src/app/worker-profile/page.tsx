'use client';

import { useState, useEffect } from 'react';
import {
  User,
  Shield,
  Award,
  AlertTriangle,
  Clock,
  Calendar,
  TrendingUp,
  TrendingDown,
  Bell,
  CheckCircle,
  Star,
  Flame,
  Heart,
  Activity,
  Target,
  ChevronRight,
  ChevronLeft,
  Phone,
  MapPin,
  Briefcase,
  Sun,
  Moon,
  Sunset,
  Zap,
  Trophy,
  Medal,
  Crown,
  Sparkles,
  FileText,
  BarChart3,
  PieChart,
  CalendarDays,
  ClipboardCheck,
  ShieldCheck,
  AlertCircle,
  Info,
  ArrowUpRight,
  ArrowDownRight,
  X,
  Home,
  Settings,
  LogOut,
  ChevronDown,
  ExternalLink,
  Share2,
  Download,
  Eye,
  Lock,
  Unlock,
} from 'lucide-react';
import AppLayout from '@/components/AppLayout';
import { Card } from '@/components/Card';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  PieChart as RechartsPie,
  Pie,
  Cell,
  LineChart,
  Line,
} from 'recharts';

// ==================== MOCK DATA ====================

const mockWorkerProfile = {
  id: 'w-001',
  employee_id: 'EMP-2024-001',
  name: 'Rushabh Savla',
  department: 'Mining Operations',
  designation: 'Senior Mine Worker',
  mine_name: 'Jharia Coal Mine',
  zone_name: 'Zone A - Extraction',
  phone: '+91 98765 43210',
  emergency_contact: '+91 98765 43211',
  joined_date: '2022-03-15',
  shift: 'day' as const,
  avatar: null,
  blood_group: 'B+',
  age: 28,
};

const mockComplianceData = {
  score: 92,
  trend: 'up' as const,
  trendValue: 3,
  total_violations: 4,
  current_streak_days: 23,
  best_streak_days: 45,
  rank: 12,
  total_workers: 156,
  percentile: 92,
};

const mockAttendanceData = {
  present_days: 24,
  total_working_days: 26,
  attendance_percentage: 92.3,
  late_arrivals: 2,
  early_departures: 1,
  overtime_hours: 18,
  leaves_taken: 2,
  leaves_remaining: 10,
};

const mockWeeklyAttendance = [
  { day: 'Mon', hours: 8, status: 'present', date: '2024-12-09' },
  { day: 'Tue', hours: 9, status: 'present', date: '2024-12-10' },
  { day: 'Wed', hours: 8, status: 'present', date: '2024-12-11' },
  { day: 'Thu', hours: 0, status: 'absent', date: '2024-12-12' },
  { day: 'Fri', hours: 8.5, status: 'present', date: '2024-12-13' },
  { day: 'Sat', hours: 8, status: 'present', date: '2024-12-14' },
  { day: 'Sun', hours: 0, status: 'off', date: '2024-12-15' },
];

const mockMonthlyProgress = [
  { week: 'Week 1', compliance: 88, attendance: 100, safety: 90 },
  { week: 'Week 2', compliance: 90, attendance: 95, safety: 92 },
  { week: 'Week 3', compliance: 94, attendance: 100, safety: 95 },
  { week: 'Week 4', compliance: 92, attendance: 90, safety: 94 },
];

const mockComplianceTrend = [
  { date: 'Dec 5', score: 88 },
  { date: 'Dec 6', score: 89 },
  { date: 'Dec 7', score: 91 },
  { date: 'Dec 8', score: 90 },
  { date: 'Dec 9', score: 92 },
  { date: 'Dec 10', score: 91 },
  { date: 'Dec 11', score: 92 },
];

const mockHealthAnalysis = {
  risk_level: 'low' as const,
  risk_score: 18,
  stress_indicator: 'normal' as const,
  last_health_check: '2024-12-01',
  next_health_check: '2025-01-01',
  vitals: {
    spo2: 98,
    heart_rate: 72,
    bp_systolic: 120,
    bp_diastolic: 80,
  },
  recommendations: [
    'Maintain current PPE compliance habits',
    'Stay hydrated during shifts',
    'Take regular breaks as scheduled',
  ],
  health_alerts: [],
};

const mockViolationBreakdown = [
  { name: 'Helmet', value: 1, color: '#ef4444' },
  { name: 'Vest', value: 2, color: '#f97316' },
  { name: 'Gloves', value: 1, color: '#eab308' },
  { name: 'Boots', value: 0, color: '#3b82f6' },
];

const mockEmergencyHistory = [
  {
    id: 'sos-001',
    date: '2024-12-12',
    time: '10:30 AM',
    type: 'Gas Emergency',
    zone: 'Zone A - Extraction',
    role: 'Evacuated Worker',
    response_time: '2 min 34 sec',
    status: 'safely_evacuated' as const,
    gas_type: 'Methane',
    gas_level: '15,200 PPM',
    details: 'Methane spike detected at 15,200 PPM. Successfully evacuated following emergency protocol. All safety procedures followed correctly.',
  }
];

const mockShiftSchedule = [
  { date: '2024-12-11', shift: 'day', status: 'current', start: '06:00', end: '14:00' },
  { date: '2024-12-12', shift: 'day', status: 'upcoming', start: '06:00', end: '14:00' },
  { date: '2024-12-13', shift: 'day', status: 'upcoming', start: '06:00', end: '14:00' },
  { date: '2024-12-14', shift: 'off', status: 'off', start: '-', end: '-' },
  { date: '2024-12-15', shift: 'day', status: 'upcoming', start: '06:00', end: '14:00' },
  { date: '2024-12-16', shift: 'afternoon', status: 'upcoming', start: '14:00', end: '22:00' },
  { date: '2024-12-17', shift: 'afternoon', status: 'upcoming', start: '14:00', end: '22:00' },
];

const mockBadges = [
  { id: 'safety_star', name: 'Safety Star', description: 'Maintained 95%+ compliance for 30 days', icon: Star, color: 'amber', earned: true, date: '2024-11-15', xp: 500 },
  { id: 'perfect_week', name: 'Perfect Week', description: 'Zero violations for 7 consecutive days', icon: CheckCircle, color: 'emerald', earned: true, date: '2024-12-01', xp: 200 },
  { id: 'streak_7', name: '7 Day Streak', description: 'Attended work for 7 days in a row', icon: Flame, color: 'orange', earned: true, date: '2024-12-05', xp: 100 },
  { id: 'streak_30', name: '30 Day Streak', description: 'Attended work for 30 days in a row', icon: Flame, color: 'red', earned: false, progress: 77, xp: 500 },
  { id: 'top_performer', name: 'Top 10%', description: 'Ranked in top 10% of all workers', icon: Trophy, color: 'purple', earned: true, date: '2024-11-30', xp: 300 },
  { id: 'zero_violations', name: 'Clean Record', description: 'No violations this month', icon: ShieldCheck, color: 'cyan', earned: false, progress: 75, xp: 250 },
  { id: 'early_bird', name: 'Early Bird', description: 'Arrived on time for 20 consecutive days', icon: Sun, color: 'yellow', earned: true, date: '2024-11-20', xp: 150 },
  { id: 'team_player', name: 'Team Player', description: 'Helped train 5 new workers', icon: Award, color: 'blue', earned: false, progress: 40, xp: 400 },
];

const mockRecentViolations = [
  { id: 'v1', date: '2024-12-08', time: '09:15 AM', type: 'NO-VEST', gate: 'Gate A', shift: 'day', severity: 'medium', image: null, acknowledged: true },
  { id: 'v2', date: '2024-11-28', time: '02:30 PM', type: 'NO-HELMET', gate: 'Gate B', shift: 'afternoon', severity: 'high', image: null, acknowledged: true },
  { id: 'v3', date: '2024-11-15', time: '07:45 AM', type: 'NO-GLOVES', gate: 'Gate A', shift: 'day', severity: 'low', image: null, acknowledged: true },
  { id: 'v4', date: '2024-11-02', time: '03:00 PM', type: 'NO-VEST', gate: 'Gate C', shift: 'afternoon', severity: 'medium', image: null, acknowledged: true },
];

const mockNotifications = [
  { id: '1', type: 'info', title: 'Shift Update', message: 'Your shift schedule has been updated for next week', date: '2024-12-11', read: false },
  { id: '2', type: 'warning', title: 'Health Check Due', message: 'Your monthly health check-up is due in 20 days', date: '2024-12-10', read: false },
  { id: '3', type: 'success', title: 'Badge Earned!', message: 'Congratulations! You earned the Perfect Week badge', date: '2024-12-01', read: true },
  { id: '4', type: 'info', title: 'Training Available', message: 'New safety training module is now available', date: '2024-11-28', read: true },
];

const mockAnnouncements = [
  { id: '1', title: 'Safety Drill Tomorrow', message: 'Mandatory safety drill scheduled for all workers at 10:00 AM. Please assemble at Gate A.', priority: 'high', date: '2024-12-11', author: 'Safety Officer' },
  { id: '2', title: 'New PPE Guidelines', message: 'Updated helmet standards will be effective from Dec 15. New helmets will be distributed.', priority: 'medium', date: '2024-12-09', author: 'Management' },
  { id: '3', title: 'Holiday Schedule', message: 'Mine operations will be closed on Dec 25-26 for Christmas holidays.', priority: 'low', date: '2024-12-08', author: 'HR Department' },
];

const mockTargets = {
  monthly_compliance_target: 95,
  current_compliance: 92,
  attendance_target: 95,
  current_attendance: 92.3,
  violations_target: 0,
  current_violations: 2,
};

const mockLeaderboard = [
  { rank: 1, name: 'Tanush Maloo', score: 98, department: 'Mining Operations' },
  { rank: 2, name: 'Rahul Verma', score: 97, department: 'Safety' },
  { rank: 3, name: 'Amit Singh', score: 96, department: 'Mining Operations' },
  { rank: 11, name: 'Priya Sharma', score: 93, department: 'Operations' },
  { rank: 12, name: 'Rushabh Savla', score: 92, department: 'Mining Operations', isCurrentUser: true },
  { rank: 13, name: 'Vikram Patel', score: 91, department: 'Maintenance' },
];

// ==================== HELPER FUNCTIONS ====================

const getShiftIcon = (shift: string) => {
  switch (shift) {
    case 'day': return <Sun className="w-4 h-4 text-amber-500" />;
    case 'afternoon': return <Sunset className="w-4 h-4 text-orange-500" />;
    case 'night': return <Moon className="w-4 h-4 text-indigo-500" />;
    default: return <Clock className="w-4 h-4 text-slate-500" />;
  }
};

const getShiftLabel = (shift: string) => {
  switch (shift) {
    case 'day': return 'Day Shift';
    case 'afternoon': return 'Afternoon Shift';
    case 'night': return 'Night Shift';
    case 'off': return 'Day Off';
    default: return shift;
  }
};

const getShiftTime = (shift: string) => {
  switch (shift) {
    case 'day': return '6:00 AM - 2:00 PM';
    case 'afternoon': return '2:00 PM - 10:00 PM';
    case 'night': return '10:00 PM - 6:00 AM';
    default: return '-';
  }
};

const getRiskColor = (level: string) => {
  switch (level) {
    case 'low': return 'text-emerald-600 bg-emerald-50 border-emerald-200';
    case 'medium': return 'text-amber-600 bg-amber-50 border-amber-200';
    case 'high': return 'text-orange-600 bg-orange-50 border-orange-200';
    case 'critical': return 'text-red-600 bg-red-50 border-red-200';
    default: return 'text-slate-600 bg-slate-50 border-slate-200';
  }
};

const getSeverityColor = (severity: string) => {
  switch (severity) {
    case 'low': return 'bg-yellow-100 text-yellow-700 border-yellow-300';
    case 'medium': return 'bg-orange-100 text-orange-700 border-orange-300';
    case 'high': return 'bg-red-100 text-red-700 border-red-300';
    default: return 'bg-slate-100 text-slate-700 border-slate-300';
  }
};

const formatDate = (dateStr: string) => {
  return new Date(dateStr).toLocaleDateString('en-IN', {
    day: 'numeric',
    month: 'short',
  });
};

const formatFullDate = (dateStr: string) => {
  return new Date(dateStr).toLocaleDateString('en-IN', {
    weekday: 'long',
    day: 'numeric',
    month: 'long',
    year: 'numeric',
  });
};

const getDayName = (dateStr: string) => {
  return new Date(dateStr).toLocaleDateString('en-IN', { weekday: 'short' });
};

// ==================== MODAL COMPONENT ====================

function Modal({ isOpen, onClose, title, children }: {
  isOpen: boolean;
  onClose: () => void;
  title: string;
  children: React.ReactNode;
}) {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-end sm:items-center justify-center p-0 sm:p-4">
      <div className="absolute inset-0 bg-black/50 backdrop-blur-sm" onClick={onClose} />
      <div className="relative bg-white w-full sm:w-auto sm:min-w-[400px] sm:max-w-lg max-h-[85vh] rounded-t-3xl sm:rounded-2xl shadow-2xl overflow-hidden animate-slide-up sm:animate-scale-in">
        {/* Handle bar for mobile */}
        <div className="sm:hidden flex justify-center pt-3 pb-1">
          <div className="w-10 h-1 bg-slate-300 rounded-full" />
        </div>
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-slate-100">
          <h3 className="text-lg font-semibold text-slate-800">{title}</h3>
          <button
            onClick={onClose}
            className="p-2 hover:bg-slate-100 rounded-xl transition-colors"
          >
            <X className="w-5 h-5 text-slate-500" />
          </button>
        </div>
        {/* Content */}
        <div className="p-5 overflow-y-auto max-h-[60vh]">
          {children}
        </div>
      </div>
    </div>
  );
}

// ==================== BOTTOM SHEET COMPONENT ====================

function BottomSheet({ isOpen, onClose, title, children }: {
  isOpen: boolean;
  onClose: () => void;
  title?: string;
  children: React.ReactNode;
}) {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50">
      <div className="absolute inset-0 bg-black/50 backdrop-blur-sm" onClick={onClose} />
      <div className="absolute bottom-0 left-0 right-0 bg-white rounded-t-3xl shadow-2xl overflow-hidden animate-slide-up max-h-[90vh]">
        {/* Handle bar */}
        <div className="flex justify-center pt-3 pb-1">
          <div className="w-10 h-1 bg-slate-300 rounded-full" />
        </div>
        {title && (
          <div className="px-5 py-3 border-b border-slate-100">
            <h3 className="text-lg font-semibold text-slate-800 text-center">{title}</h3>
          </div>
        )}
        <div className="p-5 overflow-y-auto max-h-[70vh] pb-safe">
          {children}
        </div>
      </div>
    </div>
  );
}

// ==================== COMPONENTS ====================

function CircularProgress({ value, size = 120, strokeWidth = 10, color = '#f97316' }: {
  value: number;
  size?: number;
  strokeWidth?: number;
  color?: string;
}) {
  const radius = (size - strokeWidth) / 2;
  const circumference = radius * 2 * Math.PI;
  const offset = circumference - (value / 100) * circumference;

  return (
    <div className="relative" style={{ width: size, height: size }}>
      <svg className="transform -rotate-90" width={size} height={size}>
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="currentColor"
          strokeWidth={strokeWidth}
          className="text-slate-100"
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth={strokeWidth}
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
          className="transition-all duration-500"
        />
      </svg>
      <div className="absolute inset-0 flex items-center justify-center">
        <span className="text-2xl font-bold text-slate-700">{Math.round(value)}</span>
      </div>
    </div>
  );
}

function ProgressBar({ value, max, color = 'orange', showLabel = false }: { value: number; max: number; color?: string; showLabel?: boolean }) {
  const percentage = Math.min((value / max) * 100, 100);
  const colorClasses: Record<string, string> = {
    orange: 'bg-orange-500',
    green: 'bg-emerald-500',
    red: 'bg-red-500',
    blue: 'bg-blue-500',
    amber: 'bg-amber-500',
    purple: 'bg-purple-500',
  };

  return (
    <div className="w-full">
      <div className="w-full bg-slate-100 rounded-full h-2.5">
        <div
          className={`h-2.5 rounded-full transition-all duration-500 ${colorClasses[color] || colorClasses.orange}`}
          style={{ width: `${percentage}%` }}
        />
      </div>
      {showLabel && (
        <p className="text-xs text-slate-500 mt-1 text-right">{Math.round(percentage)}%</p>
      )}
    </div>
  );
}

function QuickStatCard({
  icon: Icon,
  label,
  value,
  subValue,
  trend,
  color = 'slate',
  onClick,
}: {
  icon: React.ElementType;
  label: string;
  value: string | number;
  subValue?: string;
  trend?: 'up' | 'down';
  color?: string;
  onClick?: () => void;
}) {
  const colorMap: Record<string, { bg: string; icon: string; border: string }> = {
    orange: { bg: 'bg-orange-50', icon: 'text-orange-500', border: 'border-orange-200' },
    green: { bg: 'bg-emerald-50', icon: 'text-emerald-500', border: 'border-emerald-200' },
    red: { bg: 'bg-red-50', icon: 'text-red-500', border: 'border-red-200' },
    blue: { bg: 'bg-blue-50', icon: 'text-blue-500', border: 'border-blue-200' },
    amber: { bg: 'bg-amber-50', icon: 'text-amber-500', border: 'border-amber-200' },
    purple: { bg: 'bg-purple-50', icon: 'text-purple-500', border: 'border-purple-200' },
    slate: { bg: 'bg-slate-50', icon: 'text-slate-500', border: 'border-slate-200' },
  };

  const colors = colorMap[color] || colorMap.slate;

  return (
    <button
      onClick={onClick}
      className={`${colors.bg} ${colors.border} border rounded-2xl p-4 flex items-center gap-3 w-full text-left active:scale-[0.98] transition-all touch-manipulation`}
    >
      <div className={`w-11 h-11 sm:w-12 sm:h-12 rounded-xl bg-white border ${colors.border} flex items-center justify-center flex-shrink-0`}>
        <Icon className={`w-5 h-5 sm:w-6 sm:h-6 ${colors.icon}`} />
      </div>
      <div className="min-w-0 flex-1">
        <p className="text-xl sm:text-2xl font-bold text-slate-700 truncate flex items-center gap-1">
          {value}
          {trend && (
            trend === 'up'
              ? <ArrowUpRight className="w-4 h-4 text-emerald-500" />
              : <ArrowDownRight className="w-4 h-4 text-red-500" />
          )}
        </p>
        <p className="text-[10px] sm:text-xs text-slate-500 font-medium uppercase tracking-wide truncate">{label}</p>
        {subValue && <p className="text-[10px] sm:text-xs text-slate-400 truncate">{subValue}</p>}
      </div>
      <ChevronRight className="w-4 h-4 text-slate-300 flex-shrink-0" />
    </button>
  );
}

// ==================== MAIN PAGE ====================

export default function WorkerProfilePage() {
  const [activeTab, setActiveTab] = useState<'overview' | 'attendance' | 'compliance' | 'health'>('overview');

  // Modal states
  const [selectedShift, setSelectedShift] = useState<typeof mockShiftSchedule[0] | null>(null);
  const [selectedBadge, setSelectedBadge] = useState<typeof mockBadges[0] | null>(null);
  const [selectedViolation, setSelectedViolation] = useState<typeof mockRecentViolations[0] | null>(null);
  const [selectedAnnouncement, setSelectedAnnouncement] = useState<typeof mockAnnouncements[0] | null>(null);
  const [showLeaderboard, setShowLeaderboard] = useState(false);
  const [showNotifications, setShowNotifications] = useState(false);
  const [showAllViolations, setShowAllViolations] = useState(false);
  const [showHealthDetails, setShowHealthDetails] = useState(false);
  const [showAttendanceDetails, setShowAttendanceDetails] = useState(false);

  // Calculate total XP
  const totalXP = mockBadges.filter(b => b.earned).reduce((sum, b) => sum + b.xp, 0);
  const unreadNotifications = mockNotifications.filter(n => !n.read).length;

  return (
    <AppLayout>
      <div className="max-w-7xl mx-auto pb-24 sm:pb-8">

        {/* Profile Header - Mobile Optimized */}
        <div className="bg-gradient-to-br from-orange-100 via-amber-50 to-orange-100 border border-orange-200 rounded-2xl sm:rounded-3xl p-4 sm:p-6 shadow-sm mb-4">
          <div className="flex flex-col sm:flex-row sm:items-start gap-4">
            {/* Avatar */}
            <div className="flex items-center gap-4 sm:block">
              <button className="w-16 h-16 sm:w-20 sm:h-20 rounded-2xl bg-gradient-to-br from-orange-400 to-amber-500 flex items-center justify-center text-white text-2xl sm:text-3xl font-bold shadow-lg flex-shrink-0 active:scale-95 transition-transform">
                {mockWorkerProfile.name.split(' ').map(n => n[0]).join('')}
              </button>

              {/* Mobile: Name beside avatar */}
              <div className="sm:hidden flex-1">
                <h1 className="text-xl font-bold text-slate-800">{mockWorkerProfile.name}</h1>
                <p className="text-slate-600 text-sm">{mockWorkerProfile.designation}</p>
                <div className="flex items-center gap-2 mt-1">
                  <span className="text-xs px-2 py-0.5 bg-orange-200 text-orange-800 rounded-full font-medium">
                    {mockWorkerProfile.employee_id}
                  </span>
                  <span className="text-xs px-2 py-0.5 bg-purple-100 text-purple-700 rounded-full font-medium">
                    {totalXP} XP
                  </span>
                </div>
              </div>

              {/* Notification bell - Mobile */}
              <button
                onClick={() => setShowNotifications(true)}
                className="sm:hidden relative p-2 bg-white rounded-xl border border-orange-200 active:scale-95 transition-transform"
              >
                <Bell className="w-5 h-5 text-orange-600" />
                {unreadNotifications > 0 && (
                  <span className="absolute -top-1 -right-1 w-5 h-5 bg-red-500 text-white text-xs rounded-full flex items-center justify-center font-bold">
                    {unreadNotifications}
                  </span>
                )}
              </button>
            </div>

            {/* Desktop: Info beside avatar */}
            <div className="flex-1 min-w-0">
              <div className="hidden sm:block">
                <div className="flex items-start justify-between gap-4">
                  <div>
                    <h1 className="text-2xl font-bold text-slate-800">{mockWorkerProfile.name}</h1>
                    <p className="text-slate-600">{mockWorkerProfile.designation}</p>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="px-3 py-1 bg-purple-100 text-purple-700 rounded-full text-sm font-semibold">
                      {totalXP} XP
                    </span>
                    <span className="px-3 py-1 bg-orange-200 text-orange-800 rounded-full text-sm font-semibold">
                      {mockWorkerProfile.employee_id}
                    </span>
                    <button
                      onClick={() => setShowNotifications(true)}
                      className="relative p-2 bg-white rounded-xl border border-orange-200 hover:bg-orange-50 transition-colors"
                    >
                      <Bell className="w-5 h-5 text-orange-600" />
                      {unreadNotifications > 0 && (
                        <span className="absolute -top-1 -right-1 w-5 h-5 bg-red-500 text-white text-xs rounded-full flex items-center justify-center font-bold">
                          {unreadNotifications}
                        </span>
                      )}
                    </button>
                  </div>
                </div>
              </div>

              {/* Info Grid */}
              <div className="grid grid-cols-2 gap-2 sm:gap-4 mt-3 sm:mt-4">
                <button className="flex items-center gap-2 text-sm text-slate-600 hover:text-orange-600 transition-colors text-left">
                  <MapPin className="w-4 h-4 text-orange-500 flex-shrink-0" />
                  <span className="truncate">{mockWorkerProfile.mine_name}</span>
                </button>
                <button className="flex items-center gap-2 text-sm text-slate-600 hover:text-orange-600 transition-colors text-left">
                  <Briefcase className="w-4 h-4 text-orange-500 flex-shrink-0" />
                  <span className="truncate">{mockWorkerProfile.department}</span>
                </button>
                <button
                  onClick={() => setSelectedShift(mockShiftSchedule[0])}
                  className="flex items-center gap-2 text-sm text-slate-600 hover:text-orange-600 transition-colors text-left"
                >
                  {getShiftIcon(mockWorkerProfile.shift)}
                  <span className="truncate">{getShiftLabel(mockWorkerProfile.shift)}</span>
                  <ChevronRight className="w-3 h-3 text-slate-400" />
                </button>
                <a href={`tel:${mockWorkerProfile.phone}`} className="flex items-center gap-2 text-sm text-slate-600 hover:text-orange-600 transition-colors">
                  <Phone className="w-4 h-4 text-orange-500 flex-shrink-0" />
                  <span className="truncate">{mockWorkerProfile.phone}</span>
                </a>
              </div>
            </div>

            {/* Compliance Score Circle - Right side on desktop */}
            <button
              onClick={() => setShowLeaderboard(true)}
              className="flex justify-center sm:justify-end active:scale-95 transition-transform"
            >
              <div className="text-center">
                <CircularProgress value={mockComplianceData.score} color="#f97316" />
                <p className="text-sm font-semibold text-slate-600 mt-2">Compliance Score</p>
                <div className="flex items-center justify-center gap-1 mt-1">
                  {mockComplianceData.trend === 'up' ? (
                    <TrendingUp className="w-4 h-4 text-emerald-500" />
                  ) : (
                    <TrendingDown className="w-4 h-4 text-red-500" />
                  )}
                  <span className={`text-xs font-medium ${mockComplianceData.trend === 'up' ? 'text-emerald-600' : 'text-red-600'}`}>
                    {mockComplianceData.trendValue}% this week
                  </span>
                </div>
                <p className="text-xs text-slate-400 mt-1">Tap to see leaderboard</p>
              </div>
            </button>
          </div>
        </div>

        {/* Tab Navigation - Scrollable on mobile */}
        <div className="flex gap-2 overflow-x-auto pb-2 mb-4 -mx-4 px-4 sm:mx-0 sm:px-0 scrollbar-hide">
          {[
            { id: 'overview', label: 'Overview', icon: BarChart3 },
            { id: 'attendance', label: 'Attendance', icon: CalendarDays },
            { id: 'compliance', label: 'Compliance', icon: ShieldCheck },
            { id: 'health', label: 'Health', icon: Heart },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as typeof activeTab)}
              className={`flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm font-medium whitespace-nowrap transition-all flex-shrink-0 active:scale-95 touch-manipulation ${
                activeTab === tab.id
                  ? 'bg-orange-500 text-white shadow-md'
                  : 'bg-white border border-slate-200 text-slate-600 hover:bg-slate-50'
              }`}
            >
              <tab.icon className="w-4 h-4" />
              {tab.label}
            </button>
          ))}
        </div>

        {/* Quick Stats Row */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4 mb-4 sm:mb-6">
          <QuickStatCard
            icon={Calendar}
            label="Days Present"
            value={mockAttendanceData.present_days}
            subValue={`of ${mockAttendanceData.total_working_days} days`}
            color="blue"
            onClick={() => setShowAttendanceDetails(true)}
          />
          <QuickStatCard
            icon={AlertTriangle}
            label="Violations"
            value={mockComplianceData.total_violations}
            subValue="this month"
            color="red"
            onClick={() => setShowAllViolations(true)}
          />
          <QuickStatCard
            icon={Flame}
            label="Current Streak"
            value={`${mockComplianceData.current_streak_days}d`}
            subValue={`Best: ${mockComplianceData.best_streak_days}d`}
            trend="up"
            color="orange"
          />
          <QuickStatCard
            icon={Trophy}
            label="Rank"
            value={`#${mockComplianceData.rank}`}
            subValue={`Top ${100 - mockComplianceData.percentile}%`}
            color="purple"
            onClick={() => setShowLeaderboard(true)}
          />
        </div>

        {/* Main Content - Conditional based on tab */}
        {activeTab === 'overview' && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 sm:gap-6">
            {/* Left Column */}
            <div className="lg:col-span-2 space-y-4 sm:space-y-6">

              {/* Weekly Progress Chart */}
              <Card title="Weekly Progress" icon={<TrendingUp className="w-5 h-5 text-orange-600" />}>
                <div className="h-56 sm:h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={mockMonthlyProgress}>
                      <defs>
                        <linearGradient id="colorCompliance" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#f97316" stopOpacity={0.3}/>
                          <stop offset="95%" stopColor="#f97316" stopOpacity={0}/>
                        </linearGradient>
                        <linearGradient id="colorAttendance" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
                          <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                      <XAxis dataKey="week" tick={{ fontSize: 11 }} stroke="#94a3b8" />
                      <YAxis tick={{ fontSize: 11 }} stroke="#94a3b8" domain={[0, 100]} />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: 'white',
                          border: '1px solid #e2e8f0',
                          borderRadius: '12px',
                          boxShadow: '0 4px 6px -1px rgba(0,0,0,0.1)',
                          fontSize: '12px'
                        }}
                      />
                      <Area
                        type="monotone"
                        dataKey="compliance"
                        stroke="#f97316"
                        fillOpacity={1}
                        fill="url(#colorCompliance)"
                        strokeWidth={2}
                        name="Compliance %"
                      />
                      <Area
                        type="monotone"
                        dataKey="attendance"
                        stroke="#3b82f6"
                        fillOpacity={1}
                        fill="url(#colorAttendance)"
                        strokeWidth={2}
                        name="Attendance %"
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
                <div className="flex items-center justify-center gap-6 mt-4">
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-orange-500"></div>
                    <span className="text-xs sm:text-sm text-slate-600">Compliance</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-blue-500"></div>
                    <span className="text-xs sm:text-sm text-slate-600">Attendance</span>
                  </div>
                </div>
              </Card>

              {/* Shift Schedule */}
              <Card title="Upcoming Shifts" icon={<Clock className="w-5 h-5 text-orange-600" />}>
                <div className="grid grid-cols-7 gap-1.5 sm:gap-2">
                  {mockShiftSchedule.map((schedule, idx) => (
                    <button
                      key={idx}
                      onClick={() => setSelectedShift(schedule)}
                      className={`text-center p-2 sm:p-3 rounded-xl border transition-all active:scale-95 touch-manipulation ${
                        schedule.status === 'current'
                          ? 'bg-orange-100 border-orange-300 ring-2 ring-orange-400 ring-offset-1'
                          : schedule.status === 'off'
                          ? 'bg-slate-50 border-slate-200'
                          : 'bg-white border-slate-200 hover:border-orange-200 hover:bg-orange-50'
                      }`}
                    >
                      <p className="text-[10px] sm:text-xs text-slate-500 mb-0.5">{getDayName(schedule.date)}</p>
                      <p className="text-xs sm:text-sm font-semibold text-slate-700">{formatDate(schedule.date)}</p>
                      <div className="mt-1.5 flex justify-center">
                        {schedule.shift === 'off' ? (
                          <span className="text-[10px] text-slate-400">Off</span>
                        ) : (
                          getShiftIcon(schedule.shift)
                        )}
                      </div>
                    </button>
                  ))}
                </div>
              </Card>

              {/* Recent Violations */}
              <Card
                title="Recent Violations"
                icon={<AlertTriangle className="w-5 h-5 text-orange-600" />}
                action={
                  <button
                    onClick={() => setShowAllViolations(true)}
                    className="text-xs text-orange-600 font-medium hover:text-orange-700"
                  >
                    View All
                  </button>
                }
              >
                {mockRecentViolations.length === 0 ? (
                  <div className="text-center py-8">
                    <ShieldCheck className="w-16 h-16 text-emerald-300 mx-auto mb-3" />
                    <p className="text-slate-600 font-medium">No recent violations!</p>
                    <p className="text-sm text-slate-400">Keep up the great work</p>
                  </div>
                ) : (
                  <div className="space-y-3">
                    {mockRecentViolations.slice(0, 3).map((violation) => (
                      <button
                        key={violation.id}
                        onClick={() => setSelectedViolation(violation)}
                        className="w-full flex items-center gap-3 p-3 bg-red-50 border border-red-100 rounded-xl hover:bg-red-100 transition-colors active:scale-[0.98] text-left"
                      >
                        <div className="w-10 h-10 bg-red-100 rounded-xl flex items-center justify-center flex-shrink-0">
                          <AlertTriangle className="w-5 h-5 text-red-500" />
                        </div>
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2">
                            <p className="font-medium text-slate-700">{violation.type.replace('NO-', 'Missing ')}</p>
                            <span className={`text-[10px] px-1.5 py-0.5 rounded-full border ${getSeverityColor(violation.severity)}`}>
                              {violation.severity}
                            </span>
                          </div>
                          <p className="text-sm text-slate-500">{violation.date} at {violation.time}</p>
                        </div>
                        <ChevronRight className="w-4 h-4 text-slate-400 flex-shrink-0" />
                      </button>
                    ))}
                  </div>
                )}
              </Card>

              {/* Emergency Response History */}
              <Card
                title="Emergency Response History"
                icon={<AlertTriangle className="w-5 h-5 text-red-600" />}
              >
                {mockEmergencyHistory.length === 0 ? (
                  <div className="text-center py-8">
                    <Shield className="w-16 h-16 text-green-300 mx-auto mb-3" />
                    <p className="text-slate-600 font-medium">No emergency events</p>
                    <p className="text-sm text-slate-400">This worker has not been involved in any emergencies</p>
                  </div>
                ) : (
                  <div className="space-y-3">
                    {mockEmergencyHistory.map((emergency) => (
                      <div
                        key={emergency.id}
                        className="p-4 bg-gradient-to-r from-orange-50 to-red-50 border border-orange-200 rounded-xl"
                      >
                        <div className="flex items-start gap-3">
                          <div className="w-12 h-12 bg-orange-100 rounded-xl flex items-center justify-center flex-shrink-0">
                            <AlertTriangle className="w-6 h-6 text-orange-600" />
                          </div>
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2 mb-1">
                              <p className="font-semibold text-slate-800">{emergency.type}</p>
                              <span className="text-xs px-2 py-0.5 rounded-full bg-green-100 text-green-700 border border-green-200">
                                Safely Evacuated
                              </span>
                            </div>
                            <p className="text-sm text-slate-600">{emergency.zone}</p>
                            <p className="text-xs text-slate-500 mt-1">{emergency.date} at {emergency.time}</p>
                          </div>
                        </div>
                        <div className="mt-3 pt-3 border-t border-orange-200">
                          <div className="grid grid-cols-2 gap-3 text-sm">
                            <div>
                              <span className="text-slate-500">Gas Type:</span>
                              <span className="ml-2 font-medium text-red-600">{emergency.gas_type}</span>
                            </div>
                            <div>
                              <span className="text-slate-500">Level:</span>
                              <span className="ml-2 font-medium text-red-600">{emergency.gas_level}</span>
                            </div>
                            <div>
                              <span className="text-slate-500">Response Time:</span>
                              <span className="ml-2 font-medium text-green-600">{emergency.response_time}</span>
                            </div>
                            <div>
                              <span className="text-slate-500">Role:</span>
                              <span className="ml-2 font-medium text-slate-700">{emergency.role}</span>
                            </div>
                          </div>
                          <p className="mt-3 text-sm text-slate-600 bg-white/50 p-2 rounded-lg">
                            {emergency.details}
                          </p>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </Card>
            </div>

            {/* Right Column */}
            <div className="space-y-4 sm:space-y-6">

              {/* Monthly Targets */}
              <Card title="Monthly Targets" icon={<Target className="w-5 h-5 text-orange-600" />}>
                <div className="space-y-4">
                  <div>
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-sm text-slate-600">Compliance</span>
                      <span className="text-sm font-medium text-slate-700">
                        {mockTargets.current_compliance}% / {mockTargets.monthly_compliance_target}%
                      </span>
                    </div>
                    <ProgressBar
                      value={mockTargets.current_compliance}
                      max={mockTargets.monthly_compliance_target}
                      color={mockTargets.current_compliance >= mockTargets.monthly_compliance_target ? 'green' : 'orange'}
                    />
                  </div>
                  <div>
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-sm text-slate-600">Attendance</span>
                      <span className="text-sm font-medium text-slate-700">
                        {mockTargets.current_attendance}% / {mockTargets.attendance_target}%
                      </span>
                    </div>
                    <ProgressBar
                      value={mockTargets.current_attendance}
                      max={mockTargets.attendance_target}
                      color={mockTargets.current_attendance >= mockTargets.attendance_target ? 'green' : 'blue'}
                    />
                  </div>
                  <div>
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-sm text-slate-600">Zero Violations Goal</span>
                      <span className="text-sm font-medium text-slate-700">
                        {mockTargets.current_violations} violations
                      </span>
                    </div>
                    <ProgressBar
                      value={Math.max(0, 5 - mockTargets.current_violations)}
                      max={5}
                      color={mockTargets.current_violations === 0 ? 'green' : 'red'}
                    />
                  </div>
                </div>
              </Card>

              {/* Badges & Achievements */}
              <Card
                title="Badges & Achievements"
                icon={<Award className="w-5 h-5 text-orange-600" />}
                action={
                  <span className="text-xs text-purple-600 font-medium">{totalXP} XP</span>
                }
              >
                <div className="grid grid-cols-4 gap-2">
                  {mockBadges.map((badge) => {
                    const Icon = badge.icon;
                    const colorClasses: Record<string, string> = {
                      amber: badge.earned ? 'bg-amber-100 border-amber-300 text-amber-600' : 'bg-slate-50 border-slate-200 text-slate-300',
                      emerald: badge.earned ? 'bg-emerald-100 border-emerald-300 text-emerald-600' : 'bg-slate-50 border-slate-200 text-slate-300',
                      orange: badge.earned ? 'bg-orange-100 border-orange-300 text-orange-600' : 'bg-slate-50 border-slate-200 text-slate-300',
                      red: badge.earned ? 'bg-red-100 border-red-300 text-red-600' : 'bg-slate-50 border-slate-200 text-slate-300',
                      purple: badge.earned ? 'bg-purple-100 border-purple-300 text-purple-600' : 'bg-slate-50 border-slate-200 text-slate-300',
                      cyan: badge.earned ? 'bg-cyan-100 border-cyan-300 text-cyan-600' : 'bg-slate-50 border-slate-200 text-slate-300',
                      yellow: badge.earned ? 'bg-yellow-100 border-yellow-300 text-yellow-600' : 'bg-slate-50 border-slate-200 text-slate-300',
                      blue: badge.earned ? 'bg-blue-100 border-blue-300 text-blue-600' : 'bg-slate-50 border-slate-200 text-slate-300',
                    };

                    return (
                      <button
                        key={badge.id}
                        onClick={() => setSelectedBadge(badge)}
                        className={`relative text-center p-2 sm:p-3 rounded-xl border transition-all active:scale-95 touch-manipulation ${colorClasses[badge.color]} ${badge.earned ? '' : 'opacity-60'}`}
                      >
                        <Icon className="w-5 h-5 sm:w-6 sm:h-6 mx-auto" />
                        <p className="text-[9px] sm:text-[10px] font-medium mt-1 truncate leading-tight">{badge.name}</p>
                        {badge.earned && (
                          <CheckCircle className="absolute -top-1 -right-1 w-3.5 h-3.5 text-emerald-500 bg-white rounded-full" />
                        )}
                        {!badge.earned && (
                          <Lock className="absolute -top-1 -right-1 w-3.5 h-3.5 text-slate-400 bg-white rounded-full" />
                        )}
                      </button>
                    );
                  })}
                </div>
              </Card>

              {/* Announcements */}
              <Card title="Announcements" icon={<Bell className="w-5 h-5 text-orange-600" />}>
                <div className="space-y-3">
                  {mockAnnouncements.slice(0, 2).map((announcement) => (
                    <button
                      key={announcement.id}
                      onClick={() => setSelectedAnnouncement(announcement)}
                      className={`w-full p-3 rounded-xl border-l-4 text-left transition-all active:scale-[0.98] ${
                        announcement.priority === 'high'
                          ? 'bg-red-50 border-red-400 hover:bg-red-100'
                          : announcement.priority === 'medium'
                          ? 'bg-amber-50 border-amber-400 hover:bg-amber-100'
                          : 'bg-blue-50 border-blue-400 hover:bg-blue-100'
                      }`}
                    >
                      <p className="font-medium text-slate-700 text-sm">{announcement.title}</p>
                      <p className="text-xs text-slate-500 mt-1 line-clamp-2">{announcement.message}</p>
                      <p className="text-[10px] text-slate-400 mt-2">{formatDate(announcement.date)}</p>
                    </button>
                  ))}
                </div>
              </Card>

              {/* Emergency Contacts */}
              <Card title="Emergency Contacts" icon={<Phone className="w-5 h-5 text-orange-600" />}>
                <div className="space-y-2">
                  <a href="tel:100" className="flex items-center gap-3 p-3 bg-red-50 border border-red-100 rounded-xl hover:bg-red-100 transition-colors active:scale-[0.98]">
                    <div className="w-10 h-10 bg-red-100 rounded-xl flex items-center justify-center">
                      <Phone className="w-5 h-5 text-red-500" />
                    </div>
                    <div className="flex-1">
                      <p className="font-medium text-slate-700 text-sm">Emergency (100)</p>
                      <p className="text-xs text-slate-500">Police Emergency</p>
                    </div>
                  </a>
                  <a href={`tel:${mockWorkerProfile.emergency_contact}`} className="flex items-center gap-3 p-3 bg-orange-50 border border-orange-100 rounded-xl hover:bg-orange-100 transition-colors active:scale-[0.98]">
                    <div className="w-10 h-10 bg-orange-100 rounded-xl flex items-center justify-center">
                      <User className="w-5 h-5 text-orange-500" />
                    </div>
                    <div className="flex-1">
                      <p className="font-medium text-slate-700 text-sm">Personal Emergency</p>
                      <p className="text-xs text-slate-500">{mockWorkerProfile.emergency_contact}</p>
                    </div>
                  </a>
                </div>
              </Card>
            </div>
          </div>
        )}

        {activeTab === 'attendance' && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 sm:gap-6">
            <div className="lg:col-span-2 space-y-4 sm:space-y-6">
              {/* Attendance Stats */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <button onClick={() => setShowAttendanceDetails(true)} className="bg-emerald-50 border border-emerald-200 rounded-xl p-4 text-center active:scale-95 transition-transform">
                  <p className="text-2xl font-bold text-emerald-600">{mockAttendanceData.attendance_percentage}%</p>
                  <p className="text-xs text-slate-500 mt-1">Attendance Rate</p>
                </button>
                <button className="bg-amber-50 border border-amber-200 rounded-xl p-4 text-center active:scale-95 transition-transform">
                  <p className="text-2xl font-bold text-amber-600">{mockAttendanceData.late_arrivals}</p>
                  <p className="text-xs text-slate-500 mt-1">Late Arrivals</p>
                </button>
                <button className="bg-blue-50 border border-blue-200 rounded-xl p-4 text-center active:scale-95 transition-transform">
                  <p className="text-2xl font-bold text-blue-600">{mockAttendanceData.overtime_hours}h</p>
                  <p className="text-xs text-slate-500 mt-1">Overtime</p>
                </button>
                <button className="bg-purple-50 border border-purple-200 rounded-xl p-4 text-center active:scale-95 transition-transform">
                  <p className="text-2xl font-bold text-purple-600">{mockAttendanceData.leaves_remaining}</p>
                  <p className="text-xs text-slate-500 mt-1">Leaves Left</p>
                </button>
              </div>

              {/* Weekly Attendance Chart */}
              <Card title="This Week's Hours" icon={<CalendarDays className="w-5 h-5 text-orange-600" />}>
                <div className="h-56 sm:h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={mockWeeklyAttendance}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                      <XAxis dataKey="day" tick={{ fontSize: 11 }} stroke="#94a3b8" />
                      <YAxis tick={{ fontSize: 11 }} stroke="#94a3b8" domain={[0, 10]} />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: 'white',
                          border: '1px solid #e2e8f0',
                          borderRadius: '12px',
                          fontSize: '12px'
                        }}
                      />
                      <Bar
                        dataKey="hours"
                        fill="#f97316"
                        radius={[4, 4, 0, 0]}
                        name="Hours Worked"
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </Card>

              {/* Attendance Calendar Preview */}
              <Card title="December 2024" icon={<Calendar className="w-5 h-5 text-orange-600" />}>
                <div className="grid grid-cols-7 gap-1 text-center">
                  {['S', 'M', 'T', 'W', 'T', 'F', 'S'].map((day, i) => (
                    <div key={i} className="text-xs font-medium text-slate-400 py-2">{day}</div>
                  ))}
                  {Array.from({ length: 31 }, (_, i) => {
                    const isPresent = i < 11 ? Math.random() > 0.1 : false;
                    const isToday = i === 10;
                    const isFuture = i > 10;
                    return (
                      <button
                        key={i}
                        className={`aspect-square flex items-center justify-center rounded-lg text-xs sm:text-sm font-medium active:scale-90 transition-transform ${
                          isToday
                            ? 'bg-orange-500 text-white font-bold ring-2 ring-orange-300 ring-offset-1'
                            : isFuture
                            ? 'bg-slate-50 text-slate-300'
                            : isPresent
                            ? 'bg-emerald-100 text-emerald-700 hover:bg-emerald-200'
                            : 'bg-red-100 text-red-700 hover:bg-red-200'
                        }`}
                      >
                        {i + 1}
                      </button>
                    );
                  })}
                </div>
                <div className="flex items-center justify-center gap-4 mt-4 text-xs sm:text-sm">
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded bg-emerald-100 border border-emerald-300"></div>
                    <span className="text-slate-500">Present</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded bg-red-100 border border-red-300"></div>
                    <span className="text-slate-500">Absent</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded bg-orange-500"></div>
                    <span className="text-slate-500">Today</span>
                  </div>
                </div>
              </Card>
            </div>

            <div className="space-y-4 sm:space-y-6">
              {/* Leave Balance */}
              <Card title="Leave Balance" icon={<FileText className="w-5 h-5 text-orange-600" />}>
                <div className="space-y-4">
                  <div className="text-center py-4">
                    <CircularProgress value={(mockAttendanceData.leaves_remaining / 12) * 100} size={100} color="#8b5cf6" />
                    <p className="text-sm text-slate-500 mt-2">
                      {mockAttendanceData.leaves_remaining} of 12 leaves remaining
                    </p>
                  </div>
                  <div className="space-y-3">
                    <button className="w-full flex justify-between items-center p-3 bg-slate-50 rounded-xl hover:bg-slate-100 transition-colors">
                      <span className="text-sm text-slate-600">Casual Leave</span>
                      <span className="font-medium text-slate-700">4 / 6</span>
                    </button>
                    <button className="w-full flex justify-between items-center p-3 bg-slate-50 rounded-xl hover:bg-slate-100 transition-colors">
                      <span className="text-sm text-slate-600">Sick Leave</span>
                      <span className="font-medium text-slate-700">4 / 4</span>
                    </button>
                    <button className="w-full flex justify-between items-center p-3 bg-slate-50 rounded-xl hover:bg-slate-100 transition-colors">
                      <span className="text-sm text-slate-600">Earned Leave</span>
                      <span className="font-medium text-slate-700">2 / 2</span>
                    </button>
                  </div>
                  <button className="w-full py-3 bg-orange-500 text-white rounded-xl font-medium hover:bg-orange-600 transition-colors active:scale-[0.98]">
                    Apply for Leave
                  </button>
                </div>
              </Card>

              {/* Shift Info */}
              <Card title="Current Shift" icon={<Clock className="w-5 h-5 text-orange-600" />}>
                <button
                  onClick={() => setSelectedShift(mockShiftSchedule[0])}
                  className="w-full text-center py-4 active:scale-95 transition-transform"
                >
                  <div className="w-16 h-16 mx-auto bg-amber-100 rounded-2xl flex items-center justify-center mb-3">
                    <Sun className="w-8 h-8 text-amber-500" />
                  </div>
                  <p className="text-lg font-semibold text-slate-700">Day Shift</p>
                  <p className="text-slate-500">6:00 AM - 2:00 PM</p>
                  <div className="mt-4 px-4 py-2 bg-emerald-100 border border-emerald-200 rounded-xl inline-block">
                    <p className="text-sm font-medium text-emerald-700">Currently Active</p>
                  </div>
                  <p className="text-xs text-slate-400 mt-2">Tap for details</p>
                </button>
              </Card>
            </div>
          </div>
        )}

        {activeTab === 'compliance' && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 sm:gap-6">
            <div className="lg:col-span-2 space-y-4 sm:space-y-6">
              {/* Compliance Trend */}
              <Card title="Compliance Score Trend" icon={<TrendingUp className="w-5 h-5 text-orange-600" />}>
                <div className="h-56 sm:h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={mockComplianceTrend}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                      <XAxis dataKey="date" tick={{ fontSize: 11 }} stroke="#94a3b8" />
                      <YAxis tick={{ fontSize: 11 }} stroke="#94a3b8" domain={[80, 100]} />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: 'white',
                          border: '1px solid #e2e8f0',
                          borderRadius: '12px',
                          fontSize: '12px'
                        }}
                      />
                      <Line
                        type="monotone"
                        dataKey="score"
                        stroke="#f97316"
                        strokeWidth={3}
                        dot={{ fill: '#f97316', strokeWidth: 2, r: 4 }}
                        activeDot={{ r: 6 }}
                        name="Score"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </Card>

              {/* Violation History */}
              <Card
                title="Violation History"
                icon={<AlertTriangle className="w-5 h-5 text-orange-600" />}
                action={
                  <button
                    onClick={() => setShowAllViolations(true)}
                    className="text-xs text-orange-600 font-medium"
                  >
                    View All ({mockRecentViolations.length})
                  </button>
                }
              >
                <div className="space-y-3">
                  {mockRecentViolations.map((v) => (
                    <button
                      key={v.id}
                      onClick={() => setSelectedViolation(v)}
                      className="w-full flex items-center gap-4 p-4 bg-slate-50 rounded-xl hover:bg-slate-100 transition-colors active:scale-[0.98] text-left"
                    >
                      <div className="w-12 h-12 bg-red-100 rounded-xl flex items-center justify-center flex-shrink-0">
                        <AlertTriangle className="w-6 h-6 text-red-500" />
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1">
                          <p className="font-medium text-slate-700">{v.type.replace('NO-', 'Missing ')}</p>
                          <span className={`text-[10px] px-1.5 py-0.5 rounded-full border ${getSeverityColor(v.severity)}`}>
                            {v.severity}
                          </span>
                        </div>
                        <p className="text-sm text-slate-500">{v.date} • {v.time}</p>
                        <p className="text-xs text-slate-400">{v.gate} • {v.shift} shift</p>
                      </div>
                      <ChevronRight className="w-5 h-5 text-slate-400 flex-shrink-0" />
                    </button>
                  ))}
                </div>
              </Card>
            </div>

            <div className="space-y-4 sm:space-y-6">
              {/* Violation Breakdown */}
              <Card title="Violation Breakdown" icon={<PieChart className="w-5 h-5 text-orange-600" />}>
                <div className="h-44">
                  <ResponsiveContainer width="100%" height="100%">
                    <RechartsPie>
                      <Pie
                        data={mockViolationBreakdown.filter(v => v.value > 0)}
                        cx="50%"
                        cy="50%"
                        innerRadius={35}
                        outerRadius={65}
                        paddingAngle={5}
                        dataKey="value"
                      >
                        {mockViolationBreakdown.filter(v => v.value > 0).map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                      </Pie>
                      <Tooltip />
                    </RechartsPie>
                  </ResponsiveContainer>
                </div>
                <div className="grid grid-cols-2 gap-2 mt-4">
                  {mockViolationBreakdown.map((item) => (
                    <div key={item.name} className="flex items-center gap-2 p-2 bg-slate-50 rounded-lg">
                      <div className="w-3 h-3 rounded-full flex-shrink-0" style={{ backgroundColor: item.color }}></div>
                      <span className="text-sm text-slate-600 truncate">{item.name}</span>
                      <span className="text-sm font-medium text-slate-700 ml-auto">{item.value}</span>
                    </div>
                  ))}
                </div>
              </Card>

              {/* PPE Checklist */}
              <Card title="Required PPE" icon={<ClipboardCheck className="w-5 h-5 text-orange-600" />}>
                <div className="space-y-2">
                  {[
                    { name: 'Safety Helmet', icon: '🪖', status: 'ok' },
                    { name: 'High-Vis Vest', icon: '🦺', status: 'ok' },
                    { name: 'Safety Gloves', icon: '🧤', status: 'ok' },
                    { name: 'Safety Boots', icon: '👢', status: 'ok' },
                    { name: 'Eye Protection', icon: '🥽', status: 'optional' },
                  ].map((item, idx) => (
                    <div key={idx} className="flex items-center gap-3 p-3 bg-slate-50 rounded-xl">
                      <span className="text-xl">{item.icon}</span>
                      <span className="flex-1 text-slate-700 text-sm">{item.name}</span>
                      {item.status === 'ok' ? (
                        <CheckCircle className="w-5 h-5 text-emerald-500" />
                      ) : (
                        <span className="text-xs px-2 py-1 bg-slate-200 text-slate-600 rounded-full">Optional</span>
                      )}
                    </div>
                  ))}
                </div>
              </Card>

              {/* Smart Helmet Status */}
              <Card title="Smart Helmet Status" icon={<Shield className="w-5 h-5 text-orange-600" />}>
                <div className="space-y-3">
                  <div className="flex items-center justify-between p-3 bg-green-50 rounded-xl border border-green-200">
                    <div className="flex items-center gap-3">
                      <span className="text-2xl">🪖</span>
                      <div>
                        <p className="font-medium text-green-800">Helmet Connected</p>
                        <p className="text-xs text-green-600">Real-time monitoring active</p>
                      </div>
                    </div>
                    <CheckCircle className="w-6 h-6 text-green-500" />
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    <div className="p-3 bg-slate-50 rounded-xl text-center">
                      <p className="text-xs text-slate-500">Battery</p>
                      <p className="text-lg font-bold text-slate-700">3.8V</p>
                    </div>
                    <div className="p-3 bg-slate-50 rounded-xl text-center">
                      <p className="text-xs text-slate-500">Signal</p>
                      <p className="text-lg font-bold text-green-600">Strong</p>
                    </div>
                  </div>
                  <a
                    href="/helmet-monitoring"
                    className="block text-center text-sm text-orange-600 hover:text-orange-700 font-medium py-2 border-t border-slate-100"
                  >
                    View Detailed Helmet Data →
                  </a>
                </div>
              </Card>
            </div>
          </div>
        )}

        {activeTab === 'health' && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 sm:gap-6">
            <div className="lg:col-span-2 space-y-4 sm:space-y-6">
              {/* Health Status */}
              <Card title="Health & Safety Status" icon={<Heart className="w-5 h-5 text-orange-600" />}>
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                  <button
                    onClick={() => setShowHealthDetails(true)}
                    className={`p-4 rounded-xl border transition-all active:scale-95 ${getRiskColor(mockHealthAnalysis.risk_level)}`}
                  >
                    <div className="flex items-center gap-2 mb-2">
                      <Activity className="w-5 h-5" />
                      <span className="font-medium text-sm">Risk Level</span>
                    </div>
                    <p className="text-2xl font-bold capitalize">{mockHealthAnalysis.risk_level}</p>
                    <p className="text-xs opacity-70">Score: {mockHealthAnalysis.risk_score}/100</p>
                  </button>
                  <button className="p-4 rounded-xl border bg-orange-50 border-orange-200 text-orange-600 transition-all active:scale-95">
                    <div className="flex items-center gap-2 mb-2">
                      <AlertTriangle className="w-5 h-5" />
                      <span className="font-medium text-sm">Emergency Response</span>
                    </div>
                    <p className="text-2xl font-bold">1</p>
                    <p className="text-xs opacity-70">Safely evacuated today</p>
                  </button>
                  <button className="p-4 rounded-xl border bg-emerald-50 border-emerald-200 text-emerald-600 transition-all active:scale-95">
                    <div className="flex items-center gap-2 mb-2">
                      <Heart className="w-5 h-5" />
                      <span className="font-medium text-sm">Stress Level</span>
                    </div>
                    <p className="text-2xl font-bold capitalize">{mockHealthAnalysis.stress_indicator}</p>
                    <p className="text-xs opacity-70">Keep it up!</p>
                  </button>
                </div>
              </Card>

              {/* Vitals */}
              <Card title="Latest Vitals" icon={<Activity className="w-5 h-5 text-orange-600" />}>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                  <div className="text-center p-4 bg-red-50 border border-red-100 rounded-xl">
                    <Heart className="w-6 h-6 text-red-500 mx-auto mb-2" />
                    <p className="text-2xl font-bold text-slate-700">{mockHealthAnalysis.vitals.heart_rate}</p>
                    <p className="text-xs text-slate-500">Heart Rate (bpm)</p>
                  </div>
                  <div className="text-center p-4 bg-blue-50 border border-blue-100 rounded-xl">
                    <Activity className="w-6 h-6 text-blue-500 mx-auto mb-2" />
                    <p className="text-2xl font-bold text-slate-700">{mockHealthAnalysis.vitals.spo2}%</p>
                    <p className="text-xs text-slate-500">SpO2 Level</p>
                  </div>
                  <div className="text-center p-4 bg-purple-50 border border-purple-100 rounded-xl col-span-2 sm:col-span-2">
                    <Zap className="w-6 h-6 text-purple-500 mx-auto mb-2" />
                    <p className="text-2xl font-bold text-slate-700">{mockHealthAnalysis.vitals.bp_systolic}/{mockHealthAnalysis.vitals.bp_diastolic}</p>
                    <p className="text-xs text-slate-500">Blood Pressure (mmHg)</p>
                  </div>
                </div>
              </Card>

              {/* Safety Tips */}
              <Card title="Safety Tips" icon={<Shield className="w-5 h-5 text-orange-600" />}>
                <div className="grid grid-cols-2 gap-3">
                  {[
                    { title: 'Stay Hydrated', desc: 'Drink water every 30 minutes', icon: '💧' },
                    { title: 'Regular Breaks', desc: 'Take a 5-min break every hour', icon: '⏰' },
                    { title: 'Proper Posture', desc: 'Maintain correct posture', icon: '🧘' },
                    { title: 'Report Issues', desc: 'Report safety concerns', icon: '📢' },
                  ].map((tip, idx) => (
                    <button key={idx} className="p-4 bg-gradient-to-br from-orange-50 to-amber-50 border border-orange-100 rounded-xl text-left hover:shadow-md transition-all active:scale-[0.98]">
                      <span className="text-2xl">{tip.icon}</span>
                      <p className="font-medium text-slate-700 mt-2 text-sm">{tip.title}</p>
                      <p className="text-xs text-slate-500 mt-1">{tip.desc}</p>
                    </button>
                  ))}
                </div>
              </Card>
            </div>

            <div className="space-y-4 sm:space-y-6">
              {/* Health Check Schedule */}
              <Card title="Health Check Schedule" icon={<CalendarDays className="w-5 h-5 text-orange-600" />}>
                <div className="space-y-3">
                  <div className="p-4 bg-emerald-50 border border-emerald-200 rounded-xl">
                    <p className="text-sm text-emerald-600 font-medium">Last Check-up</p>
                    <p className="text-lg font-bold text-emerald-700">{formatDate(mockHealthAnalysis.last_health_check)}</p>
                    <p className="text-xs text-emerald-600 flex items-center gap-1 mt-1">
                      <CheckCircle className="w-3 h-3" /> All clear
                    </p>
                  </div>
                  <div className="p-4 bg-amber-50 border border-amber-200 rounded-xl">
                    <p className="text-sm text-amber-600 font-medium">Next Check-up</p>
                    <p className="text-lg font-bold text-amber-700">{formatDate(mockHealthAnalysis.next_health_check)}</p>
                    <p className="text-xs text-amber-600">In 20 days</p>
                  </div>
                  <button className="w-full py-3 bg-orange-500 text-white rounded-xl font-medium hover:bg-orange-600 transition-colors active:scale-[0.98]">
                    Schedule Check-up
                  </button>
                </div>
              </Card>

              {/* Quick Actions */}
              <Card title="Quick Actions" icon={<Zap className="w-5 h-5 text-orange-600" />}>
                <div className="space-y-2">
                  <button className="w-full flex items-center gap-3 p-3 bg-red-50 border border-red-200 rounded-xl hover:bg-red-100 transition-colors active:scale-[0.98] text-left">
                    <AlertCircle className="w-5 h-5 text-red-500" />
                    <div>
                      <p className="font-medium text-red-700 text-sm">Report Health Issue</p>
                      <p className="text-[10px] text-red-500">Notify supervisor</p>
                    </div>
                  </button>
                  <button className="w-full flex items-center gap-3 p-3 bg-blue-50 border border-blue-200 rounded-xl hover:bg-blue-100 transition-colors active:scale-[0.98] text-left">
                    <FileText className="w-5 h-5 text-blue-500" />
                    <div>
                      <p className="font-medium text-blue-700 text-sm">Request Sick Leave</p>
                      <p className="text-[10px] text-blue-500">Apply for medical leave</p>
                    </div>
                  </button>
                  <button className="w-full flex items-center gap-3 p-3 bg-emerald-50 border border-emerald-200 rounded-xl hover:bg-emerald-100 transition-colors active:scale-[0.98] text-left">
                    <Download className="w-5 h-5 text-emerald-500" />
                    <div>
                      <p className="font-medium text-emerald-700 text-sm">Download Health Report</p>
                      <p className="text-[10px] text-emerald-500">Get PDF summary</p>
                    </div>
                  </button>
                </div>
              </Card>
            </div>
          </div>
        )}

        {/* Safety Tip Footer */}
        <div className="mt-6 bg-gradient-to-r from-blue-50 to-cyan-50 rounded-2xl p-4 border border-blue-100">
          <div className="flex items-start gap-3">
            <Shield className="w-6 h-6 text-blue-600 flex-shrink-0" />
            <div>
              <p className="font-medium text-blue-900 text-sm">Daily Safety Tip</p>
              <p className="text-xs sm:text-sm text-blue-700 mt-1">
                Always ensure your helmet chin strap is properly fastened. A loose helmet provides no protection during an accident.
              </p>
            </div>
          </div>
        </div>

      </div>

      {/* ==================== MODALS ==================== */}

      {/* Shift Detail Modal */}
      <Modal
        isOpen={!!selectedShift}
        onClose={() => setSelectedShift(null)}
        title="Shift Details"
      >
        {selectedShift && (
          <div className="space-y-4">
            <div className="text-center py-4">
              <div className="w-20 h-20 mx-auto bg-gradient-to-br from-orange-100 to-amber-100 rounded-2xl flex items-center justify-center mb-3">
                {selectedShift.shift === 'day' && <Sun className="w-10 h-10 text-amber-500" />}
                {selectedShift.shift === 'afternoon' && <Sunset className="w-10 h-10 text-orange-500" />}
                {selectedShift.shift === 'night' && <Moon className="w-10 h-10 text-indigo-500" />}
                {selectedShift.shift === 'off' && <Calendar className="w-10 h-10 text-slate-400" />}
              </div>
              <h3 className="text-xl font-bold text-slate-800">{getShiftLabel(selectedShift.shift)}</h3>
              <p className="text-slate-500">{formatFullDate(selectedShift.date)}</p>
            </div>

            {selectedShift.shift !== 'off' && (
              <div className="grid grid-cols-2 gap-3">
                <div className="p-4 bg-emerald-50 rounded-xl text-center">
                  <p className="text-sm text-emerald-600">Start Time</p>
                  <p className="text-lg font-bold text-emerald-700">{selectedShift.start}</p>
                </div>
                <div className="p-4 bg-red-50 rounded-xl text-center">
                  <p className="text-sm text-red-600">End Time</p>
                  <p className="text-lg font-bold text-red-700">{selectedShift.end}</p>
                </div>
              </div>
            )}

            <div className="p-4 bg-slate-50 rounded-xl">
              <p className="text-sm text-slate-600 mb-2">Location</p>
              <p className="font-medium text-slate-800">{mockWorkerProfile.mine_name}</p>
              <p className="text-sm text-slate-500">{mockWorkerProfile.zone_name}</p>
            </div>

            {selectedShift.status === 'current' && (
              <div className="p-4 bg-emerald-100 border border-emerald-200 rounded-xl text-center">
                <p className="text-emerald-700 font-medium">This is your current shift</p>
              </div>
            )}
          </div>
        )}
      </Modal>

      {/* Badge Detail Modal */}
      <Modal
        isOpen={!!selectedBadge}
        onClose={() => setSelectedBadge(null)}
        title="Badge Details"
      >
        {selectedBadge && (
          <div className="space-y-4">
            <div className="text-center py-4">
              <div className={`w-24 h-24 mx-auto rounded-2xl flex items-center justify-center mb-3 ${
                selectedBadge.earned
                  ? `bg-${selectedBadge.color}-100`
                  : 'bg-slate-100'
              }`}>
                <selectedBadge.icon className={`w-12 h-12 ${
                  selectedBadge.earned ? `text-${selectedBadge.color}-500` : 'text-slate-300'
                }`} />
              </div>
              <h3 className="text-xl font-bold text-slate-800">{selectedBadge.name}</h3>
              <p className="text-slate-500 mt-1">{selectedBadge.description}</p>
              <div className="flex items-center justify-center gap-2 mt-3">
                <Sparkles className="w-4 h-4 text-purple-500" />
                <span className="text-purple-600 font-semibold">{selectedBadge.xp} XP</span>
              </div>
            </div>

            {selectedBadge.earned ? (
              <div className="p-4 bg-emerald-50 border border-emerald-200 rounded-xl">
                <div className="flex items-center gap-2 text-emerald-700">
                  <CheckCircle className="w-5 h-5" />
                  <span className="font-medium">Earned on {formatDate(selectedBadge.date!)}</span>
                </div>
              </div>
            ) : (
              <div className="space-y-3">
                <div className="p-4 bg-slate-50 border border-slate-200 rounded-xl">
                  <div className="flex items-center gap-2 text-slate-600 mb-2">
                    <Lock className="w-4 h-4" />
                    <span className="font-medium">Not yet earned</span>
                  </div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm text-slate-500">Progress</span>
                    <span className="text-sm font-medium text-slate-700">{selectedBadge.progress}%</span>
                  </div>
                  <ProgressBar value={selectedBadge.progress || 0} max={100} color="orange" />
                </div>
              </div>
            )}

            <button
              onClick={() => setSelectedBadge(null)}
              className="w-full py-3 bg-orange-500 text-white rounded-xl font-medium hover:bg-orange-600 transition-colors"
            >
              {selectedBadge.earned ? 'Share Badge' : 'Keep Going!'}
            </button>
          </div>
        )}
      </Modal>

      {/* Violation Detail Modal */}
      <Modal
        isOpen={!!selectedViolation}
        onClose={() => setSelectedViolation(null)}
        title="Violation Details"
      >
        {selectedViolation && (
          <div className="space-y-4">
            <div className="text-center py-4">
              <div className="w-20 h-20 mx-auto bg-red-100 rounded-2xl flex items-center justify-center mb-3">
                <AlertTriangle className="w-10 h-10 text-red-500" />
              </div>
              <h3 className="text-xl font-bold text-slate-800">{selectedViolation.type.replace('NO-', 'Missing ')}</h3>
              <span className={`inline-block mt-2 text-sm px-3 py-1 rounded-full border ${getSeverityColor(selectedViolation.severity)}`}>
                {selectedViolation.severity.charAt(0).toUpperCase() + selectedViolation.severity.slice(1)} Severity
              </span>
            </div>

            <div className="space-y-3">
              <div className="flex justify-between items-center p-3 bg-slate-50 rounded-xl">
                <span className="text-slate-600">Date</span>
                <span className="font-medium text-slate-800">{selectedViolation.date}</span>
              </div>
              <div className="flex justify-between items-center p-3 bg-slate-50 rounded-xl">
                <span className="text-slate-600">Time</span>
                <span className="font-medium text-slate-800">{selectedViolation.time}</span>
              </div>
              <div className="flex justify-between items-center p-3 bg-slate-50 rounded-xl">
                <span className="text-slate-600">Location</span>
                <span className="font-medium text-slate-800">{selectedViolation.gate}</span>
              </div>
              <div className="flex justify-between items-center p-3 bg-slate-50 rounded-xl">
                <span className="text-slate-600">Shift</span>
                <span className="font-medium text-slate-800 capitalize">{selectedViolation.shift}</span>
              </div>
            </div>

            {selectedViolation.acknowledged && (
              <div className="p-4 bg-emerald-50 border border-emerald-200 rounded-xl">
                <div className="flex items-center gap-2 text-emerald-700">
                  <CheckCircle className="w-5 h-5" />
                  <span className="font-medium">Acknowledged</span>
                </div>
              </div>
            )}

            <div className="p-4 bg-blue-50 border border-blue-100 rounded-xl">
              <p className="text-sm text-blue-700">
                <strong>Tip:</strong> Always double-check your PPE before entering the work area. Missing {selectedViolation.type.replace('NO-', '').toLowerCase()} can result in serious injuries.
              </p>
            </div>
          </div>
        )}
      </Modal>

      {/* Announcement Detail Modal */}
      <Modal
        isOpen={!!selectedAnnouncement}
        onClose={() => setSelectedAnnouncement(null)}
        title="Announcement"
      >
        {selectedAnnouncement && (
          <div className="space-y-4">
            <div className={`p-4 rounded-xl border-l-4 ${
              selectedAnnouncement.priority === 'high'
                ? 'bg-red-50 border-red-400'
                : selectedAnnouncement.priority === 'medium'
                ? 'bg-amber-50 border-amber-400'
                : 'bg-blue-50 border-blue-400'
            }`}>
              <h3 className="text-lg font-bold text-slate-800">{selectedAnnouncement.title}</h3>
              <p className="text-slate-600 mt-2">{selectedAnnouncement.message}</p>
            </div>

            <div className="flex justify-between items-center p-3 bg-slate-50 rounded-xl">
              <span className="text-slate-600">Posted by</span>
              <span className="font-medium text-slate-800">{selectedAnnouncement.author}</span>
            </div>
            <div className="flex justify-between items-center p-3 bg-slate-50 rounded-xl">
              <span className="text-slate-600">Date</span>
              <span className="font-medium text-slate-800">{formatFullDate(selectedAnnouncement.date)}</span>
            </div>

            <button className="w-full py-3 bg-orange-500 text-white rounded-xl font-medium hover:bg-orange-600 transition-colors">
              Mark as Read
            </button>
          </div>
        )}
      </Modal>

      {/* Leaderboard Bottom Sheet */}
      <BottomSheet
        isOpen={showLeaderboard}
        onClose={() => setShowLeaderboard(false)}
        title="Compliance Leaderboard"
      >
        <div className="space-y-2">
          {mockLeaderboard.map((worker) => (
            <div
              key={worker.rank}
              className={`flex items-center gap-3 p-3 rounded-xl transition-colors ${
                worker.isCurrentUser
                  ? 'bg-orange-100 border-2 border-orange-300'
                  : 'bg-slate-50 hover:bg-slate-100'
              }`}
            >
              <div className={`w-8 h-8 rounded-full flex items-center justify-center font-bold text-sm ${
                worker.rank === 1 ? 'bg-amber-400 text-amber-900' :
                worker.rank === 2 ? 'bg-slate-300 text-slate-700' :
                worker.rank === 3 ? 'bg-orange-300 text-orange-800' :
                'bg-slate-200 text-slate-600'
              }`}>
                {worker.rank <= 3 ? (
                  worker.rank === 1 ? '🥇' : worker.rank === 2 ? '🥈' : '🥉'
                ) : (
                  worker.rank
                )}
              </div>
              <div className="flex-1 min-w-0">
                <p className="font-medium text-slate-800 truncate">
                  {worker.name}
                  {worker.isCurrentUser && <span className="text-orange-600 ml-1">(You)</span>}
                </p>
                <p className="text-xs text-slate-500">{worker.department}</p>
              </div>
              <div className="text-right">
                <p className="font-bold text-slate-800">{worker.score}%</p>
              </div>
            </div>
          ))}
        </div>
      </BottomSheet>

      {/* Notifications Bottom Sheet */}
      <BottomSheet
        isOpen={showNotifications}
        onClose={() => setShowNotifications(false)}
        title="Notifications"
      >
        <div className="space-y-3">
          {mockNotifications.map((notif) => (
            <div
              key={notif.id}
              className={`p-4 rounded-xl border-l-4 ${
                notif.type === 'warning'
                  ? 'bg-amber-50 border-amber-400'
                  : notif.type === 'success'
                  ? 'bg-emerald-50 border-emerald-400'
                  : 'bg-blue-50 border-blue-400'
              } ${!notif.read ? 'ring-2 ring-offset-1 ring-orange-200' : ''}`}
            >
              <div className="flex items-start justify-between">
                <div>
                  <p className="font-medium text-slate-800">{notif.title}</p>
                  <p className="text-sm text-slate-600 mt-1">{notif.message}</p>
                  <p className="text-xs text-slate-400 mt-2">{formatDate(notif.date)}</p>
                </div>
                {!notif.read && (
                  <div className="w-2 h-2 bg-orange-500 rounded-full flex-shrink-0" />
                )}
              </div>
            </div>
          ))}
        </div>
      </BottomSheet>

      {/* All Violations Bottom Sheet */}
      <BottomSheet
        isOpen={showAllViolations}
        onClose={() => setShowAllViolations(false)}
        title={`All Violations (${mockRecentViolations.length})`}
      >
        <div className="space-y-3">
          {mockRecentViolations.map((v) => (
            <button
              key={v.id}
              onClick={() => {
                setShowAllViolations(false);
                setSelectedViolation(v);
              }}
              className="w-full flex items-center gap-3 p-4 bg-slate-50 rounded-xl hover:bg-slate-100 transition-colors text-left"
            >
              <div className="w-10 h-10 bg-red-100 rounded-xl flex items-center justify-center flex-shrink-0">
                <AlertTriangle className="w-5 h-5 text-red-500" />
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <p className="font-medium text-slate-700">{v.type.replace('NO-', 'Missing ')}</p>
                  <span className={`text-[10px] px-1.5 py-0.5 rounded-full border ${getSeverityColor(v.severity)}`}>
                    {v.severity}
                  </span>
                </div>
                <p className="text-sm text-slate-500">{v.date} • {v.time}</p>
              </div>
              <ChevronRight className="w-4 h-4 text-slate-400" />
            </button>
          ))}
        </div>
      </BottomSheet>

      {/* Mobile Bottom Navigation */}
      <nav className="fixed bottom-0 left-0 right-0 bg-white border-t border-slate-200 px-2 py-2 sm:hidden z-40 safe-area-inset">
        <div className="flex items-center justify-around">
          {[
            { id: 'overview', icon: Home, label: 'Home' },
            { id: 'attendance', icon: CalendarDays, label: 'Attendance' },
            { id: 'compliance', icon: ShieldCheck, label: 'Compliance' },
            { id: 'health', icon: Heart, label: 'Health' },
          ].map((item) => (
            <button
              key={item.id}
              onClick={() => setActiveTab(item.id as typeof activeTab)}
              className={`flex flex-col items-center gap-1 px-4 py-2 rounded-xl transition-all ${
                activeTab === item.id
                  ? 'text-orange-600 bg-orange-50'
                  : 'text-slate-400'
              }`}
            >
              <item.icon className="w-5 h-5" />
              <span className="text-[10px] font-medium">{item.label}</span>
            </button>
          ))}
        </div>
      </nav>

      {/* Animation styles */}
      <style jsx global>{`
        @keyframes slide-up {
          from {
            transform: translateY(100%);
            opacity: 0;
          }
          to {
            transform: translateY(0);
            opacity: 1;
          }
        }

        @keyframes scale-in {
          from {
            transform: scale(0.95);
            opacity: 0;
          }
          to {
            transform: scale(1);
            opacity: 1;
          }
        }

        .animate-slide-up {
          animation: slide-up 0.3s ease-out;
        }

        .animate-scale-in {
          animation: scale-in 0.2s ease-out;
        }

        .pb-safe {
          padding-bottom: env(safe-area-inset-bottom, 16px);
        }

        .safe-area-inset {
          padding-bottom: env(safe-area-inset-bottom, 8px);
        }

        .line-clamp-2 {
          display: -webkit-box;
          -webkit-line-clamp: 2;
          -webkit-box-orient: vertical;
          overflow: hidden;
        }
      `}</style>
    </AppLayout>
  );
}
