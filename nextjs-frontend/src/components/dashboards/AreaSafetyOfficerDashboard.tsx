'use client';

import { useEffect, useState } from 'react';
import {
  Mountain,
  TrendingUp,
  AlertTriangle,
  Users,
  Shield,
  MapPin,
} from 'lucide-react';
import { Card, StatCard } from '@/components/Card';
import { Spinner } from '@/components/Loading';
import { dashboardApi } from '@/lib/api';
import { formatTime } from '@/lib/utils';
import type { AreaSafetyOfficerDashboard as DashboardType } from '@/types';

export default function AreaSafetyOfficerDashboard() {
  const [data, setData] = useState<DashboardType | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      setLoading(true);
      const dashboardData = await dashboardApi.getAreaSafetyOfficer();
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

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-stone-800">Area Safety Overview</h1>
        <p className="text-stone-500 mt-1">Multi-Mine Comparison & Risk Analysis</p>
      </div>

      {/* Overall Stats */}
      <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
        <div className="bg-gradient-to-br from-cyan-500 to-cyan-600 rounded-xl p-6 text-white">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-cyan-100 text-sm">Overall Compliance</p>
              <p className="text-4xl font-bold mt-1">{data.overall_compliance_rate}%</p>
            </div>
            <Shield className="w-12 h-12 text-cyan-200" />
          </div>
        </div>
        <StatCard
          title="Total Mines"
          value={data.total_mines}
          icon={<Mountain size={24} />}
          color="blue"
        />
        <StatCard
          title="Critical Alerts"
          value={data.critical_alerts.length}
          icon={<AlertTriangle size={24} />}
          color={data.critical_alerts.length > 0 ? 'red' : 'green'}
        />
      </div>

      {/* Mines Comparison */}
      <Card title="Mines Comparison" description="Performance across all assigned mines">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-stone-100">
                <th className="text-left py-3 px-4 text-xs font-medium text-stone-500 uppercase">Mine</th>
                <th className="text-left py-3 px-4 text-xs font-medium text-stone-500 uppercase">Location</th>
                <th className="text-left py-3 px-4 text-xs font-medium text-stone-500 uppercase">Workers</th>
                <th className="text-left py-3 px-4 text-xs font-medium text-stone-500 uppercase">Compliance</th>
                <th className="text-left py-3 px-4 text-xs font-medium text-stone-500 uppercase">Violations</th>
                <th className="text-left py-3 px-4 text-xs font-medium text-stone-500 uppercase">Alerts</th>
              </tr>
            </thead>
            <tbody>
              {data.mines_overview.map((mine) => (
                <tr key={mine.mine_id} className="border-b border-stone-50 hover:bg-stone-50">
                  <td className="py-3 px-4">
                    <div className="flex items-center gap-2">
                      <Mountain className="w-4 h-4 text-stone-400" />
                      <span className="font-medium text-stone-800">{mine.mine_name}</span>
                    </div>
                  </td>
                  <td className="py-3 px-4 text-sm text-stone-500">{mine.location || '-'}</td>
                  <td className="py-3 px-4">
                    <div className="flex items-center gap-1">
                      <Users className="w-4 h-4 text-stone-400" />
                      <span>{mine.worker_count}</span>
                    </div>
                  </td>
                  <td className="py-3 px-4">
                    <div className="flex items-center gap-2">
                      <div className="w-20 bg-stone-200 rounded-full h-2">
                        <div
                          className={`h-2 rounded-full ${
                            mine.compliance_rate >= 90 ? 'bg-green-500' :
                            mine.compliance_rate >= 70 ? 'bg-yellow-500' : 'bg-red-500'
                          }`}
                          style={{ width: `${mine.compliance_rate}%` }}
                        />
                      </div>
                      <span className={`font-medium ${
                        mine.compliance_rate >= 90 ? 'text-green-600' :
                        mine.compliance_rate >= 70 ? 'text-yellow-600' : 'text-red-600'
                      }`}>
                        {mine.compliance_rate}%
                      </span>
                    </div>
                  </td>
                  <td className="py-3 px-4">
                    <span className={`inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium ${
                      mine.violations_this_week > 10
                        ? 'bg-red-100 text-red-700'
                        : mine.violations_this_week > 5
                        ? 'bg-yellow-100 text-yellow-700'
                        : 'bg-green-100 text-green-700'
                    }`}>
                      {mine.violations_this_week}
                    </span>
                  </td>
                  <td className="py-3 px-4">
                    {mine.active_alerts > 0 ? (
                      <span className="inline-flex items-center gap-1 px-2 py-1 bg-red-100 text-red-700 rounded-full text-xs font-medium">
                        <AlertTriangle className="w-3 h-3" />
                        {mine.active_alerts}
                      </span>
                    ) : (
                      <span className="text-green-600 text-sm">None</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Risk Heatmap and Critical Alerts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Risk Heatmap */}
        <Card title="Risk Heatmap" description="Zones with highest violations">
          <div className="space-y-3">
            {data.risk_heatmap.length === 0 ? (
              <div className="text-center py-6 text-stone-500">
                <Shield className="w-8 h-8 mx-auto mb-2 text-green-500" />
                <p>No high-risk zones detected</p>
              </div>
            ) : (
              data.risk_heatmap.map((zone) => (
                <div
                  key={`${zone.mine_id}-${zone.zone_id}`}
                  className={`p-3 rounded-lg border-l-4 ${
                    zone.risk_level === 'high'
                      ? 'bg-red-50 border-red-500'
                      : zone.risk_level === 'medium'
                      ? 'bg-orange-50 border-orange-500'
                      : 'bg-yellow-50 border-yellow-500'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="flex items-center gap-2">
                        <MapPin className="w-4 h-4 text-stone-500" />
                        <span className="font-medium">{zone.zone_name}</span>
                      </div>
                      <p className="text-xs text-stone-500 mt-1">{zone.mine_name}</p>
                    </div>
                    <div className="text-right">
                      <span className={`text-lg font-bold ${
                        zone.violations > 10 ? 'text-red-600' : 'text-orange-600'
                      }`}>
                        {zone.violations}
                      </span>
                      <p className="text-xs text-stone-500">violations</p>
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </Card>

        {/* Critical Alerts */}
        <Card title="Critical Alerts" description="High priority alerts across mines">
          <div className="space-y-3">
            {data.critical_alerts.length === 0 ? (
              <div className="text-center py-6 text-stone-500">
                <Shield className="w-8 h-8 mx-auto mb-2 text-green-500" />
                <p>No critical alerts</p>
              </div>
            ) : (
              data.critical_alerts.map((alert) => (
                <div
                  key={alert.id}
                  className="p-3 bg-red-50 rounded-lg border border-red-100"
                >
                  <div className="flex items-start gap-3">
                    <div className="w-8 h-8 bg-red-100 rounded-lg flex items-center justify-center flex-shrink-0">
                      <AlertTriangle className="w-4 h-4 text-red-600" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-stone-800">{alert.message}</p>
                      <div className="flex items-center gap-2 mt-1">
                        <span className="text-xs text-stone-500">{alert.mine_name}</span>
                        <span className="text-xs text-stone-400">•</span>
                        <span className="text-xs text-stone-500">{formatTime(alert.created_at)}</span>
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
