'use client';

import { useEffect, useState } from 'react';
import {
  Shield,
  Award,
  AlertTriangle,
  Clock,
  Calendar,
  TrendingUp,
  Bell,
  CheckCircle,
  Star,
  Flame,
} from 'lucide-react';
import { Card } from '@/components/Card';
import { Spinner } from '@/components/Loading';
import type { WorkerDashboard as DashboardType } from '@/types';

// Badge icons and colors - Brighter pastel theme
const badgeConfig: Record<string, { icon: React.ReactNode; color: string; label: string }> = {
  safety_star: { icon: <Star className="w-4 h-4" />, color: 'bg-amber-100 text-amber-700 border border-amber-300', label: 'Safety Star' },
  perfect_record: { icon: <CheckCircle className="w-4 h-4" />, color: 'bg-emerald-100 text-emerald-700 border border-emerald-300', label: 'Perfect Record' },
  streak_7: { icon: <Flame className="w-4 h-4" />, color: 'bg-orange-100 text-orange-700 border border-orange-300', label: '7 Day Streak' },
  streak_30: { icon: <Flame className="w-4 h-4" />, color: 'bg-red-100 text-red-700 border border-red-300', label: '30 Day Streak' },
};

// Mock data for Rushabh Savla
const mockDashboardData: DashboardType = {
  worker: {
    id: 'w1',
    name: 'Rushabh Savla',
    employee_id: 'EMP-2024-001',
    mine_name: 'Central Mining Complex',
    zone_name: 'Zone A - Mining Area',
    department: 'Operations',
  },
  compliance: {
    score: 87,
    total_violations: 6,
    current_streak_days: 12,
  },
  statistics: {
    total_entries: 156,
  },
  shift_info: {
    assigned_shift: 'day',
    shift_name: 'Day Shift',
    start_time: '06:00 AM',
    end_time: '02:00 PM',
    is_current_shift: true,
  },
  badges: ['safety_star', 'streak_7'],
  notifications: [
    {
      type: 'warning',
      id: 'n1',
      message: 'Missing helmet detected at Gate A on Dec 9. Please ensure all PPE is worn.',
      severity: 'moderate',
      date: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString(),
    },
  ],
  recent_violations: [
    {
      date: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toLocaleDateString(),
      time: '08:45 AM',
      violations: ['NO-HELMET'],
    },
    {
      date: new Date(Date.now() - 5 * 24 * 60 * 60 * 1000).toLocaleDateString(),
      time: '02:30 PM',
      violations: ['NO-VEST', 'NO-GLOVES'],
    },
  ],
};

export default function WorkerDashboard() {
  const [data, setData] = useState<DashboardType | null>(null);
  const [loading, setLoading] = useState(true);
  const [acknowledgedNotifications, setAcknowledgedNotifications] = useState<Set<string>>(new Set());

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 500));
    setData(mockDashboardData);
    setLoading(false);
  };

  const handleAcknowledgeNotification = async (id: string) => {
    // Mark notification as acknowledged locally
    setAcknowledgedNotifications(prev => new Set([...prev, id]));
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Spinner size="lg" />
      </div>
    );
  }

  if (!data) {
    return (
      <div className="text-center py-12">
        <p className="text-stone-500">Loading dashboard...</p>
      </div>
    );
  }

  // Filter out acknowledged notifications
  const activeNotifications = data.notifications.filter(n => !acknowledgedNotifications.has(n.id));

  const getScoreColor = (score: number) => {
    if (score >= 90) return 'text-emerald-600';
    if (score >= 70) return 'text-amber-600';
    return 'text-rose-600';
  };

  const getScoreBgColor = (score: number) => {
    if (score >= 90) return 'bg-emerald-400';
    if (score >= 70) return 'bg-amber-400';
    return 'bg-rose-400';
  };

  return (
    <div className="space-y-6 max-w-2xl mx-auto">
      {/* Welcome Header */}
      <div className="bg-gradient-to-br from-orange-200 via-amber-100 to-orange-200 border border-orange-300 rounded-2xl p-6 text-slate-800 shadow-sm">
        <h1 className="text-2xl font-bold text-slate-800">Welcome, {data.worker.name}!</h1>
        <p className="text-slate-700 mt-1 font-medium">
          {data.worker.mine_name} • {data.worker.zone_name || 'General'}
        </p>

        {/* Compliance Score Circle */}
        <div className="flex items-center gap-6 mt-6">
          <div className="relative">
            <svg className="w-24 h-24 transform -rotate-90">
              <circle
                cx="48"
                cy="48"
                r="40"
                fill="none"
                stroke="rgba(251, 146, 60, 0.3)"
                strokeWidth="8"
              />
              <circle
                cx="48"
                cy="48"
                r="40"
                fill="none"
                stroke="rgb(251, 146, 60)"
                strokeWidth="8"
                strokeDasharray={`${data.compliance.score * 2.51} 251`}
                strokeLinecap="round"
              />
            </svg>
            <div className="absolute inset-0 flex items-center justify-center">
              <span className="text-2xl font-bold text-slate-800">{Math.round(data.compliance.score)}</span>
            </div>
          </div>
          <div>
            <p className="text-slate-700 text-sm font-semibold">Compliance Score</p>
            <p className="text-3xl font-bold text-slate-800">{data.compliance.score >= 90 ? 'Excellent!' : data.compliance.score >= 70 ? 'Good' : 'Needs Improvement'}</p>
            <p className="text-slate-700 text-sm mt-1">
              {data.compliance.current_streak_days} day streak
            </p>
          </div>
        </div>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-3 gap-4">
        <div className="bg-white border border-slate-200 rounded-2xl p-4 text-center shadow-sm">
          <div className="w-12 h-12 mx-auto mb-3 bg-gradient-to-br from-blue-50 to-sky-100 border border-blue-200 rounded-xl flex items-center justify-center">
            <Calendar className="w-6 h-6 text-blue-500" strokeWidth={2.5} />
          </div>
          <p className="text-2xl font-bold text-slate-700">{data.statistics.total_entries}</p>
          <p className="text-xs text-slate-500 font-semibold uppercase tracking-wide mt-1">Total Entries</p>
        </div>
        <div className="bg-white border border-slate-200 rounded-2xl p-4 text-center shadow-sm">
          <div className="w-12 h-12 mx-auto mb-3 bg-gradient-to-br from-red-50 to-rose-100 border border-red-200 rounded-xl flex items-center justify-center">
            <AlertTriangle className="w-6 h-6 text-red-500" strokeWidth={2.5} />
          </div>
          <p className="text-2xl font-bold text-slate-700">{data.compliance.total_violations}</p>
          <p className="text-xs text-slate-500 font-semibold uppercase tracking-wide mt-1">Violations</p>
        </div>
        <div className="bg-white border border-slate-200 rounded-2xl p-4 text-center shadow-sm">
          <div className="w-12 h-12 mx-auto mb-3 bg-gradient-to-br from-orange-50 to-amber-100 border border-orange-200 rounded-xl flex items-center justify-center">
            <Flame className="w-6 h-6 text-orange-500" strokeWidth={2.5} />
          </div>
          <p className="text-2xl font-bold text-slate-700">{data.compliance.current_streak_days}</p>
          <p className="text-xs text-slate-500 font-semibold uppercase tracking-wide mt-1">Day Streak</p>
        </div>
      </div>

      {/* Shift Info Card */}
      <Card title="Your Shift" className="bg-white">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className={`w-12 h-12 rounded-xl flex items-center justify-center border ${
              data.shift_info.is_current_shift ? 'bg-emerald-50 border-emerald-200' : 'bg-slate-50 border-slate-200'
            }`}>
              <Clock className={`w-6 h-6 ${data.shift_info.is_current_shift ? 'text-emerald-600' : 'text-slate-500'}`} strokeWidth={2.5} />
            </div>
            <div>
              <p className="font-semibold text-slate-700">{data.shift_info.shift_name}</p>
              <p className="text-sm text-slate-500">
                {data.shift_info.start_time} - {data.shift_info.end_time}
              </p>
            </div>
          </div>
          {data.shift_info.is_current_shift && (
            <span className="px-3 py-1.5 bg-emerald-50 text-emerald-700 border border-emerald-200 rounded-full text-sm font-semibold">
              Active Now
            </span>
          )}
        </div>
      </Card>

      {/* Badges */}
      {data.badges.length > 0 && (
        <Card title="Badges Earned" className="bg-white">
          <div className="flex flex-wrap gap-2">
            {data.badges.map((badge) => {
              const config = badgeConfig[badge] || {
                icon: <Award className="w-4 h-4" />,
                color: 'bg-slate-100 text-slate-700 border border-slate-300',
                label: badge
              };
              return (
                <span
                  key={badge}
                  className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-sm font-medium ${config.color}`}
                >
                  {config.icon}
                  {config.label}
                </span>
              );
            })}
          </div>
        </Card>
      )}

      {/* Notifications */}
      {activeNotifications.length > 0 && (
        <Card title="Notifications" className="bg-white">
          <div className="space-y-3">
            {activeNotifications.map((notif) => (
              <div
                key={notif.id}
                className={`p-3 rounded-xl border-l-4 ${
                  notif.severity === 'severe'
                    ? 'bg-red-50 border-red-400'
                    : notif.severity === 'moderate'
                    ? 'bg-amber-50 border-amber-400'
                    : 'bg-blue-50 border-blue-400'
                }`}
              >
                <div className="flex items-start justify-between">
                  <div>
                    <p className="text-sm font-medium text-slate-700">{notif.message}</p>
                    <p className="text-xs text-slate-500 mt-1">
                      {new Date(notif.date).toLocaleDateString()}
                    </p>
                  </div>
                  <button
                    onClick={() => handleAcknowledgeNotification(notif.id)}
                    className="text-xs px-3 py-1.5 bg-white border border-slate-200 rounded-lg hover:bg-slate-50 font-medium transition-colors"
                  >
                    OK
                  </button>
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Recent Violations */}
      {data.recent_violations.length > 0 && (
        <Card title="Recent Violations" className="bg-white">
          <div className="space-y-3">
            {data.recent_violations.map((violation, idx) => (
              <div key={idx} className="flex items-center gap-3 p-3 bg-red-50 border border-red-200 rounded-xl">
                <div className="w-10 h-10 bg-red-100 border border-red-200 rounded-xl flex items-center justify-center">
                  <AlertTriangle className="w-5 h-5 text-red-600" strokeWidth={2.5} />
                </div>
                <div className="flex-1">
                  <div className="flex flex-wrap gap-1">
                    {violation.violations.map((v, i) => (
                      <span key={i} className="text-sm text-red-700">
                        {v.replace('NO-', 'Missing ')}
                        {i < violation.violations.length - 1 && ', '}
                      </span>
                    ))}
                  </div>
                  <p className="text-xs text-stone-500">
                    {violation.date} at {violation.time}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Safety Tip */}
      <div className="bg-blue-50 rounded-xl p-4 border border-blue-100">
        <div className="flex items-start gap-3">
          <Shield className="w-6 h-6 text-blue-600 flex-shrink-0" />
          <div>
            <p className="font-medium text-blue-900">Safety Tip</p>
            <p className="text-sm text-blue-700 mt-1">
              Always ensure your helmet is securely fastened before entering the work area.
              A properly worn helmet can save your life!
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
