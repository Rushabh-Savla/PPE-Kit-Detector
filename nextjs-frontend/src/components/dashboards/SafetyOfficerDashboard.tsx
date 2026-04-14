'use client';

import { useEffect, useState } from 'react';
import {
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  Users,
  Shield,
  MapPin,
  Clock,
  BarChart3,
} from 'lucide-react';
import { Card, StatCard } from '@/components/Card';
import { Spinner } from '@/components/Loading';
import { dashboardApi } from '@/lib/api';
import { formatTime } from '@/lib/utils';
import type { SafetyOfficerDashboard as DashboardType } from '@/types';

export default function SafetyOfficerDashboard() {
  const [data, setData] = useState<DashboardType | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      setLoading(true);
      const dashboardData = await dashboardApi.getSafetyOfficer();
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

  const getTrendIcon = (current: number, previous: number) => {
    if (current > previous) return <TrendingUp className="w-4 h-4 text-green-500" />;
    if (current < previous) return <TrendingDown className="w-4 h-4 text-red-500" />;
    return null;
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-stone-800">Safety Analytics</h1>
        <p className="text-stone-500 mt-1">{data.mine_name} • Compliance & Risk Analysis</p>
      </div>

      {/* Compliance Rate Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-gradient-to-br from-green-500 to-green-600 rounded-xl p-6 text-white">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-green-100 text-sm">Today&apos;s Compliance</p>
              <p className="text-4xl font-bold mt-1">{data.compliance_rates.today}%</p>
            </div>
            <Shield className="w-12 h-12 text-green-200" />
          </div>
        </div>
        <div className="bg-gradient-to-br from-blue-500 to-blue-600 rounded-xl p-6 text-white">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-blue-100 text-sm">This Week</p>
              <p className="text-4xl font-bold mt-1">{data.compliance_rates.this_week}%</p>
            </div>
            <BarChart3 className="w-12 h-12 text-blue-200" />
          </div>
        </div>
        <div className="bg-gradient-to-br from-purple-500 to-purple-600 rounded-xl p-6 text-white">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-purple-100 text-sm">This Month</p>
              <p className="text-4xl font-bold mt-1">{data.compliance_rates.this_month}%</p>
            </div>
            <TrendingUp className="w-12 h-12 text-purple-200" />
          </div>
        </div>
      </div>

      {/* Violation Stats */}
      <div className="grid grid-cols-2 gap-4">
        <StatCard
          title="Violations Today"
          value={data.violations.today}
          icon={<AlertTriangle size={24} />}
          color="red"
        />
        <StatCard
          title="Violations This Week"
          value={data.violations.this_week}
          icon={<AlertTriangle size={24} />}
          color="yellow"
        />
      </div>

      {/* Main Content */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Violation Trends */}
        <Card title="Violation Breakdown" description="Types of violations this week">
          <div className="space-y-4">
            {Object.entries(data.violation_trends).length === 0 ? (
              <p className="text-stone-500 text-center py-4">No violations recorded</p>
            ) : (
              Object.entries(data.violation_trends)
                .sort(([, a], [, b]) => b - a)
                .map(([type, count]) => (
                  <div key={type}>
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-sm font-medium text-stone-700">
                        {type.replace('NO-', 'Missing ')}
                      </span>
                      <span className="text-sm font-semibold text-stone-800">{count}</span>
                    </div>
                    <div className="w-full bg-stone-200 rounded-full h-2">
                      <div
                        className="bg-red-500 h-2 rounded-full"
                        style={{
                          width: `${Math.min((count / Math.max(...Object.values(data.violation_trends))) * 100, 100)}%`
                        }}
                      />
                    </div>
                  </div>
                ))
            )}
          </div>
        </Card>

        {/* Zone Risk Analysis */}
        <Card title="Zone Risk Analysis" description="Safety status by zone">
          <div className="space-y-3">
            {data.zone_risk_analysis.length === 0 ? (
              <p className="text-stone-500 text-center py-4">No zones configured</p>
            ) : (
              data.zone_risk_analysis.map((zone) => (
                <div
                  key={zone.zone_id}
                  className={`p-3 rounded-lg border-l-4 ${
                    zone.risk_level === 'critical'
                      ? 'bg-red-50 border-red-500'
                      : zone.risk_level === 'high'
                      ? 'bg-orange-50 border-orange-500'
                      : zone.risk_level === 'normal'
                      ? 'bg-yellow-50 border-yellow-500'
                      : 'bg-green-50 border-green-500'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <MapPin className="w-4 h-4 text-stone-500" />
                      <span className="font-medium">{zone.zone_name}</span>
                    </div>
                    <span className={`text-xs px-2 py-1 rounded-full font-medium ${
                      zone.risk_level === 'critical'
                        ? 'bg-red-100 text-red-700'
                        : zone.risk_level === 'high'
                        ? 'bg-orange-100 text-orange-700'
                        : zone.risk_level === 'normal'
                        ? 'bg-yellow-100 text-yellow-700'
                        : 'bg-green-100 text-green-700'
                    }`}>
                      {zone.risk_level.toUpperCase()}
                    </span>
                  </div>
                  <div className="flex items-center gap-4 mt-2 text-sm text-stone-500">
                    <span className="flex items-center gap-1">
                      <Users className="w-3 h-3" />
                      {zone.worker_count} workers
                    </span>
                    <span className="flex items-center gap-1">
                      <AlertTriangle className="w-3 h-3" />
                      {zone.violations_this_week} violations
                    </span>
                  </div>
                </div>
              ))
            )}
          </div>
        </Card>
      </div>

      {/* High Risk Workers & Recent Alerts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* High Risk Workers */}
        <Card title="High Risk Workers" description="Workers with low compliance scores">
          <div className="space-y-3">
            {data.high_risk_workers.length === 0 ? (
              <div className="text-center py-6 text-stone-500">
                <Shield className="w-8 h-8 mx-auto mb-2 text-green-500" />
                <p>All workers are compliant!</p>
              </div>
            ) : (
              data.high_risk_workers.map((worker) => (
                <div key={worker.id} className="flex items-center justify-between p-3 bg-red-50 rounded-lg">
                  <div>
                    <p className="font-medium text-stone-800">{worker.name}</p>
                    <p className="text-xs text-stone-500">ID: {worker.employee_id}</p>
                  </div>
                  <div className="text-right">
                    <p className={`font-bold ${worker.compliance_score < 50 ? 'text-red-600' : 'text-yellow-600'}`}>
                      {worker.compliance_score}%
                    </p>
                    <p className="text-xs text-stone-500">{worker.total_violations} violations</p>
                  </div>
                </div>
              ))
            )}
          </div>
        </Card>

        {/* Recent Alerts */}
        <Card title="Recent Alerts" description="Latest safety alerts">
          <div className="space-y-3">
            {data.recent_alerts.length === 0 ? (
              <p className="text-stone-500 text-center py-4">No recent alerts</p>
            ) : (
              data.recent_alerts.slice(0, 5).map((alert) => (
                <div
                  key={alert.id}
                  className={`p-3 rounded-lg ${
                    alert.severity === 'critical'
                      ? 'bg-red-50'
                      : alert.severity === 'high'
                      ? 'bg-orange-50'
                      : 'bg-yellow-50'
                  }`}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <p className="text-sm font-medium text-stone-800">{alert.message}</p>
                      <div className="flex items-center gap-2 mt-1">
                        <span className={`text-xs px-1.5 py-0.5 rounded ${
                          alert.status === 'active'
                            ? 'bg-red-100 text-red-700'
                            : alert.status === 'acknowledged'
                            ? 'bg-yellow-100 text-yellow-700'
                            : 'bg-green-100 text-green-700'
                        }`}>
                          {alert.status}
                        </span>
                        <span className="text-xs text-stone-500">
                          {formatTime(alert.created_at)}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </Card>
      </div>
    </div>
  );
}
