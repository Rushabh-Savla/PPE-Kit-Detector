'use client';

import { useEffect, useState } from 'react';
import {
  Sun,
  Sunset,
  Moon,
  Users,
  Shield,
  AlertTriangle,
  Clock,
  TrendingUp,
  TrendingDown,
  Calendar,
  BarChart3,
  RefreshCw,
} from 'lucide-react';
import AppLayout from '@/components/AppLayout';
import { Card, StatCard } from '@/components/Card';
import { Spinner } from '@/components/Loading';
import { dashboardApi, gateEntryApi } from '@/lib/api';
import { useAuthStore } from '@/lib/store';
import type { ManagerDashboard, GateEntry } from '@/types';

interface ShiftStats {
  shift: string;
  workers: number;
  entries: number;
  violations: number;
  compliance: number;
}

export default function ShiftsPage() {
  const { getMineIds } = useAuthStore();
  const [data, setData] = useState<ManagerDashboard | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedDate, setSelectedDate] = useState(new Date().toISOString().split('T')[0]);
  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => {
    loadData();
  }, [selectedDate]);

  const loadData = async () => {
    try {
      setLoading(true);
      const dashboardData = await dashboardApi.getManager();
      setData(dashboardData);
      setError(null);
    } catch (err) {
      setError('Failed to load shift performance data');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleRefresh = async () => {
    setRefreshing(true);
    await loadData();
    setRefreshing(false);
  };

  if (loading) {
    return (
      <AppLayout>
        <div className="flex items-center justify-center h-64">
          <Spinner size="lg" />
        </div>
      </AppLayout>
    );
  }

  if (error || !data) {
    return (
      <AppLayout>
        <div className="text-center py-12">
          <AlertTriangle className="w-12 h-12 text-red-500 mx-auto mb-4" />
          <p className="text-red-500 mb-4">{error || 'Failed to load data'}</p>
          <button onClick={loadData} className="btn btn-primary">
            Retry
          </button>
        </div>
      </AppLayout>
    );
  }

  const shiftConfig = {
    day: {
      icon: <Sun className="w-8 h-8" />,
      label: 'Day Shift',
      time: '6:00 AM - 2:00 PM',
      gradient: 'from-yellow-400 to-orange-400',
      lightBg: 'bg-yellow-50',
      iconColor: 'text-yellow-500',
    },
    afternoon: {
      icon: <Sunset className="w-8 h-8" />,
      label: 'Afternoon Shift',
      time: '2:00 PM - 10:00 PM',
      gradient: 'from-orange-400 to-red-400',
      lightBg: 'bg-orange-50',
      iconColor: 'text-orange-500',
    },
    night: {
      icon: <Moon className="w-8 h-8" />,
      label: 'Night Shift',
      time: '10:00 PM - 6:00 AM',
      gradient: 'from-blue-600 to-indigo-600',
      lightBg: 'bg-blue-50',
      iconColor: 'text-blue-500',
    },
  };

  const shifts = Object.entries(data.shift_performance).map(([shift, compliance]) => ({
    shift,
    compliance,
    config: shiftConfig[shift as keyof typeof shiftConfig],
  }));

  const avgCompliance = shifts.reduce((sum, s) => sum + s.compliance, 0) / shifts.length;
  const bestShift = shifts.reduce((best, s) => s.compliance > best.compliance ? s : best, shifts[0]);
  const worstShift = shifts.reduce((worst, s) => s.compliance < worst.compliance ? s : worst, shifts[0]);

  return (
    <AppLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold text-stone-800">Shift Performance</h1>
            <p className="text-stone-500 mt-1">{data.mine_name} • Performance metrics by shift</p>
          </div>
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2">
              <Calendar className="w-5 h-5 text-stone-400" />
              <input
                type="date"
                value={selectedDate}
                onChange={(e) => setSelectedDate(e.target.value)}
                className="px-3 py-2 border border-stone-300 rounded-lg focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <button
              onClick={handleRefresh}
              disabled={refreshing}
              className="flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700"
            >
              <RefreshCw className={`w-4 h-4 ${refreshing ? 'animate-spin' : ''}`} />
              Refresh
            </button>
          </div>
        </div>

        {/* Summary Stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <StatCard
            title="Total Workers"
            value={data.overview.total_workers}
            icon={<Users size={24} />}
            color="blue"
          />
          <StatCard
            title="Active Today"
            value={data.overview.active_workers_today}
            icon={<Users size={24} />}
            color="green"
          />
          <StatCard
            title="Avg Compliance"
            value={`${avgCompliance.toFixed(1)}%`}
            icon={<Shield size={24} />}
            color={avgCompliance >= 90 ? 'green' : avgCompliance >= 80 ? 'yellow' : 'red'}
          />
          <StatCard
            title="Pending Issues"
            value={data.overview.pending_escalations}
            icon={<AlertTriangle size={24} />}
            color={data.overview.pending_escalations > 0 ? 'red' : 'green'}
          />
        </div>

        {/* Shift Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {shifts.map(({ shift, compliance, config }) => (
            <div
              key={shift}
              className={`rounded-2xl overflow-hidden shadow-lg bg-gradient-to-br ${config.gradient}`}
            >
              <div className="p-6 text-white">
                <div className="flex items-center justify-between mb-4">
                  <div>
                    <p className="text-white/80 text-sm">{config.time}</p>
                    <h3 className="text-xl font-bold">{config.label}</h3>
                  </div>
                  {config.icon}
                </div>
                <div className="mt-6">
                  <p className="text-white/80 text-sm">Compliance Rate</p>
                  <div className="flex items-end gap-2">
                    <span className="text-5xl font-bold">{compliance}</span>
                    <span className="text-2xl font-bold mb-1">%</span>
                  </div>
                </div>
                <div className="mt-4">
                  <div className="w-full bg-white/30 rounded-full h-2">
                    <div
                      className="bg-white h-2 rounded-full"
                      style={{ width: `${compliance}%` }}
                    />
                  </div>
                </div>
                <div className="flex items-center justify-between mt-4 text-sm text-white/80">
                  <span>Target: 95%</span>
                  {compliance >= 95 ? (
                    <span className="flex items-center gap-1 text-white">
                      <TrendingUp className="w-4 h-4" /> Above Target
                    </span>
                  ) : (
                    <span className="flex items-center gap-1 text-white">
                      <TrendingDown className="w-4 h-4" /> Below Target
                    </span>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Comparison Analysis */}
        <Card title="Shift Comparison" description="Side-by-side analysis of shift performance">
          <div className="space-y-6">
            {/* Best and Worst */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="p-4 bg-green-50 rounded-xl border border-green-200">
                <div className="flex items-center gap-2 text-green-600 mb-2">
                  <TrendingUp className="w-5 h-5" />
                  <span className="font-medium">Best Performing Shift</span>
                </div>
                <div className="flex items-center gap-3">
                  <div className={`p-2 rounded-lg ${bestShift.config.lightBg}`}>
                    <span className={bestShift.config.iconColor}>{bestShift.config.icon}</span>
                  </div>
                  <div>
                    <p className="font-bold text-stone-800">{bestShift.config.label}</p>
                    <p className="text-sm text-stone-500">{bestShift.compliance}% compliance</p>
                  </div>
                </div>
              </div>

              <div className="p-4 bg-red-50 rounded-xl border border-red-200">
                <div className="flex items-center gap-2 text-red-600 mb-2">
                  <TrendingDown className="w-5 h-5" />
                  <span className="font-medium">Needs Improvement</span>
                </div>
                <div className="flex items-center gap-3">
                  <div className={`p-2 rounded-lg ${worstShift.config.lightBg}`}>
                    <span className={worstShift.config.iconColor}>{worstShift.config.icon}</span>
                  </div>
                  <div>
                    <p className="font-bold text-stone-800">{worstShift.config.label}</p>
                    <p className="text-sm text-stone-500">{worstShift.compliance}% compliance</p>
                  </div>
                </div>
              </div>
            </div>

            {/* Bar Chart Style Comparison */}
            <div className="space-y-4">
              {shifts.map(({ shift, compliance, config }) => (
                <div key={shift} className="flex items-center gap-4">
                  <div className={`w-12 h-12 rounded-lg ${config.lightBg} flex items-center justify-center`}>
                    <span className={config.iconColor}>{config.icon}</span>
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-medium text-stone-800">{config.label}</span>
                      <span className={`font-bold ${
                        compliance >= 95 ? 'text-green-600' :
                        compliance >= 85 ? 'text-yellow-600' : 'text-red-600'
                      }`}>
                        {compliance}%
                      </span>
                    </div>
                    <div className="w-full bg-stone-200 rounded-full h-3">
                      <div
                        className={`h-3 rounded-full ${
                          compliance >= 95 ? 'bg-green-500' :
                          compliance >= 85 ? 'bg-yellow-500' : 'bg-red-500'
                        }`}
                        style={{ width: `${compliance}%` }}
                      />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </Card>

        {/* Top Workers */}
        <Card title="Top Compliant Workers" description="Best performing workers across all shifts">
          <div className="space-y-3">
            {data.top_compliant_workers.length === 0 ? (
              <div className="text-center py-8 text-stone-500">
                <Users className="w-12 h-12 mx-auto mb-4 text-stone-400" />
                <p>No worker data available</p>
              </div>
            ) : (
              data.top_compliant_workers.map((worker, index) => (
                <div
                  key={worker.id}
                  className="flex items-center justify-between p-4 bg-stone-50 rounded-lg"
                >
                  <div className="flex items-center gap-4">
                    <div className={`w-10 h-10 rounded-full flex items-center justify-center text-white font-bold ${
                      index === 0 ? 'bg-yellow-500' :
                      index === 1 ? 'bg-stone-400' :
                      index === 2 ? 'bg-orange-400' : 'bg-stone-300'
                    }`}>
                      {index + 1}
                    </div>
                    <div>
                      <p className="font-medium text-stone-800">{worker.name}</p>
                      <div className="flex items-center gap-2 text-sm text-stone-500">
                        <span>ID: {worker.employee_id}</span>
                        <span>•</span>
                        <span className="capitalize">{worker.assigned_shift} Shift</span>
                      </div>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="text-2xl font-bold text-green-600">{worker.compliance_score}%</p>
                    <p className="text-xs text-stone-500">{worker.total_violations} violations</p>
                  </div>
                </div>
              ))
            )}
          </div>
        </Card>

        {/* Recommendations */}
        {worstShift.compliance < 90 && (
          <div className="bg-yellow-50 border border-yellow-200 rounded-xl p-4">
            <div className="flex items-start gap-3">
              <AlertTriangle className="w-6 h-6 text-yellow-600 flex-shrink-0" />
              <div>
                <h3 className="font-semibold text-yellow-800">Improvement Recommendations</h3>
                <ul className="mt-2 space-y-1 text-sm text-yellow-700">
                  <li>• {worstShift.config.label} shows lower compliance ({worstShift.compliance}%). Consider additional PPE checks.</li>
                  <li>• Schedule a safety briefing for workers on the {worstShift.shift} shift.</li>
                  <li>• Review gate entry procedures during shift transitions.</li>
                </ul>
              </div>
            </div>
          </div>
        )}
      </div>
    </AppLayout>
  );
}
