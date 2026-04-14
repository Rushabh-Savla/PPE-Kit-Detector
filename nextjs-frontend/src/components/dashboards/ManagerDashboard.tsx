'use client';

import { useEffect, useState } from 'react';
import {
  Users,
  TrendingUp,
  Clock,
  AlertTriangle,
  Award,
  Sun,
  Sunset,
  Moon,
  ShieldCheck,
} from 'lucide-react';
import { Card, StatCard } from '@/components/Card';
import { Spinner } from '@/components/Loading';
import AtRiskWorkersWidget from '@/components/AtRiskWorkersWidget';
import { dashboardApi } from '@/lib/api';
import type { ManagerDashboard as DashboardType } from '@/types';

export default function ManagerDashboard() {
  const [data, setData] = useState<DashboardType | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      setLoading(true);
      const dashboardData = await dashboardApi.getManager();
      setData(dashboardData);
    } catch (err) {
      setError('Failed to load dashboard');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Spinner size="lg" />
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="text-center py-12">
        <p className="text-red-500">{error || 'Failed to load data'}</p>
        <button onClick={loadData} className="mt-4 btn btn-primary">
          Retry
        </button>
      </div>
    );
  }

  const shiftIcons: Record<string, React.ReactNode> = {
    day: <Sun className="w-5 h-5 text-yellow-500" />,
    afternoon: <Sunset className="w-5 h-5 text-orange-500" />,
    night: <Moon className="w-5 h-5 text-blue-500" />,
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-stone-800">Manager Dashboard</h1>
        <p className="text-stone-500 mt-1">{data.mine_name} • Operations Overview</p>
      </div>

      {/* Stats Grid */}
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
          title="Compliance Rate"
          value={`${data.overview.compliance_rate}%`}
          icon={<ShieldCheck size={24} />}
          color={data.overview.compliance_rate >= 90 ? 'green' : 'yellow'}
        />
        <StatCard
          title="Pending Escalations"
          value={data.overview.pending_escalations}
          icon={<AlertTriangle size={24} />}
          color={data.overview.pending_escalations > 0 ? 'red' : 'green'}
        />
      </div>

      {/* Shift Performance */}
      <Card title="Shift Performance" description="Compliance rate by shift this week">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {Object.entries(data.shift_performance).map(([shift, rate]) => (
            <div
              key={shift}
              className={`p-4 rounded-xl border-2 ${
                rate >= 95
                  ? 'border-green-200 bg-green-50'
                  : rate >= 85
                  ? 'border-yellow-200 bg-yellow-50'
                  : 'border-red-200 bg-red-50'
              }`}
            >
              <div className="flex items-center gap-3 mb-3">
                {shiftIcons[shift]}
                <span className="font-semibold text-stone-800 capitalize">
                  {shift} Shift
                </span>
              </div>
              <div className="flex items-end gap-2">
                <span className={`text-3xl font-bold ${
                  rate >= 95 ? 'text-green-600' : rate >= 85 ? 'text-yellow-600' : 'text-red-600'
                }`}>
                  {rate}%
                </span>
                <span className="text-stone-500 text-sm mb-1">compliance</span>
              </div>
              <div className="mt-3 w-full bg-stone-200 rounded-full h-2">
                <div
                  className={`h-2 rounded-full ${
                    rate >= 95 ? 'bg-green-500' : rate >= 85 ? 'bg-yellow-500' : 'bg-red-500'
                  }`}
                  style={{ width: `${rate}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </Card>

      {/* Two Column Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Top Compliant Workers */}
        <Card title="Top Compliant Workers" description="Best performing workers">
          <div className="space-y-3">
            {data.top_compliant_workers.length === 0 ? (
              <p className="text-stone-500 text-center py-4">No data available</p>
            ) : (
              data.top_compliant_workers.map((worker, index) => (
                <div
                  key={worker.id}
                  className="flex items-center justify-between p-3 bg-stone-50 rounded-lg"
                >
                  <div className="flex items-center gap-3">
                    <div className={`w-8 h-8 rounded-full flex items-center justify-center text-white font-bold ${
                      index === 0
                        ? 'bg-yellow-500'
                        : index === 1
                        ? 'bg-stone-400'
                        : index === 2
                        ? 'bg-orange-400'
                        : 'bg-stone-300'
                    }`}>
                      {index + 1}
                    </div>
                    <div>
                      <p className="font-medium text-stone-800">{worker.name}</p>
                      <p className="text-xs text-stone-500">ID: {worker.employee_id}</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    {worker.badges.length > 0 && (
                      <Award className="w-5 h-5 text-yellow-500" />
                    )}
                    <span className="font-bold text-green-600">
                      {worker.compliance_score}%
                    </span>
                  </div>
                </div>
              ))
            )}
          </div>
        </Card>

        {/* Workforce Summary */}
        <Card title="Workforce Summary" description="Today's worker status">
          <div className="space-y-4">
            <div className="flex items-center justify-between p-4 bg-green-50 rounded-lg">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-green-100 rounded-lg flex items-center justify-center">
                  <Users className="w-5 h-5 text-green-600" />
                </div>
                <div>
                  <p className="text-sm text-stone-600">Active Today</p>
                  <p className="font-semibold text-stone-800">{data.overview.active_workers_today} workers</p>
                </div>
              </div>
              <span className="text-2xl font-bold text-green-600">
                {data.overview.total_workers > 0
                  ? Math.round((data.overview.active_workers_today / data.overview.total_workers) * 100)
                  : 0}%
              </span>
            </div>

            <div className="flex items-center justify-between p-4 bg-stone-50 rounded-lg">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-stone-200 rounded-lg flex items-center justify-center">
                  <Users className="w-5 h-5 text-stone-600" />
                </div>
                <div>
                  <p className="text-sm text-stone-600">Absent Today</p>
                  <p className="font-semibold text-stone-800">
                    {data.overview.total_workers - data.overview.active_workers_today} workers
                  </p>
                </div>
              </div>
            </div>

            {data.overview.pending_escalations > 0 && (
              <div className="flex items-center justify-between p-4 bg-red-50 rounded-lg border border-red-100">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 bg-red-100 rounded-lg flex items-center justify-center">
                    <AlertTriangle className="w-5 h-5 text-red-600" />
                  </div>
                  <div>
                    <p className="text-sm text-red-600">Pending Escalations</p>
                    <p className="font-semibold text-red-700">Requires attention</p>
                  </div>
                </div>
                <span className="text-2xl font-bold text-red-600">
                  {data.overview.pending_escalations}
                </span>
              </div>
            )}
          </div>
        </Card>
      </div>

      {/* Predictive Analysis - At-Risk Workers */}
      <AtRiskWorkersWidget limit={5} showTitle={true} />
    </div>
  );
}
