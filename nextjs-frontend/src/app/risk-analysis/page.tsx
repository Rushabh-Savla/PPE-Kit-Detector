'use client';

import { useEffect, useState } from 'react';
import {
  AlertTriangle,
  MapPin,
  Shield,
  TrendingUp,
  TrendingDown,
  Mountain,
  Users,
  Filter,
  RefreshCw,
  BarChart3,
} from 'lucide-react';
import AppLayout from '@/components/AppLayout';
import { Card, StatCard } from '@/components/Card';
import { Spinner } from '@/components/Loading';
import { dashboardApi } from '@/lib/api';
import type { AreaSafetyOfficerDashboard } from '@/types';

export default function RiskAnalysisPage() {
  const [data, setData] = useState<AreaSafetyOfficerDashboard | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [riskFilter, setRiskFilter] = useState<'all' | 'high' | 'medium' | 'low'>('all');

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      setLoading(true);
      const dashboardData = await dashboardApi.getAreaSafetyOfficer();
      setData(dashboardData);
      setError(null);
    } catch (err) {
      setError('Failed to load risk analysis data');
      console.error(err);
    } finally {
      setLoading(false);
    }
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

  const filteredRiskZones = riskFilter === 'all'
    ? data.risk_heatmap
    : data.risk_heatmap.filter(z => z.risk_level === riskFilter);

  const highRiskCount = data.risk_heatmap.filter(z => z.risk_level === 'high').length;
  const mediumRiskCount = data.risk_heatmap.filter(z => z.risk_level === 'medium').length;
  const lowRiskCount = data.risk_heatmap.filter(z => z.risk_level === 'low' || z.risk_level === 'normal').length;
  const totalViolations = data.risk_heatmap.reduce((sum, z) => sum + z.violations, 0);

  return (
    <AppLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold text-stone-900">Risk Analysis</h1>
            <p className="text-stone-500 mt-1">Zone-level risk assessment across all mines</p>
          </div>
          <button
            onClick={loadData}
            className="flex items-center gap-2 px-4 py-2 bg-cyan-600 text-white rounded-lg hover:bg-cyan-700"
          >
            <RefreshCw className="w-4 h-4" />
            Refresh Analysis
          </button>
        </div>

        {/* Risk Summary */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <StatCard
            title="High Risk Zones"
            value={highRiskCount}
            icon={<AlertTriangle size={24} />}
            color="red"
          />
          <StatCard
            title="Medium Risk Zones"
            value={mediumRiskCount}
            icon={<AlertTriangle size={24} />}
            color="yellow"
          />
          <StatCard
            title="Low Risk Zones"
            value={lowRiskCount}
            icon={<Shield size={24} />}
            color="green"
          />
          <StatCard
            title="Total Violations"
            value={totalViolations}
            icon={<BarChart3 size={24} />}
            color="blue"
          />
        </div>

        {/* Risk Distribution */}
        <Card title="Risk Distribution Overview">
          <div className="flex items-center gap-4">
            <div className="flex-1">
              <div className="flex h-8 rounded-full overflow-hidden">
                {highRiskCount > 0 && (
                  <div
                    className="bg-red-500 flex items-center justify-center text-white text-xs font-medium"
                    style={{ width: `${(highRiskCount / (highRiskCount + mediumRiskCount + lowRiskCount)) * 100}%` }}
                  >
                    {highRiskCount}
                  </div>
                )}
                {mediumRiskCount > 0 && (
                  <div
                    className="bg-yellow-500 flex items-center justify-center text-white text-xs font-medium"
                    style={{ width: `${(mediumRiskCount / (highRiskCount + mediumRiskCount + lowRiskCount)) * 100}%` }}
                  >
                    {mediumRiskCount}
                  </div>
                )}
                {lowRiskCount > 0 && (
                  <div
                    className="bg-green-500 flex items-center justify-center text-white text-xs font-medium"
                    style={{ width: `${(lowRiskCount / (highRiskCount + mediumRiskCount + lowRiskCount)) * 100}%` }}
                  >
                    {lowRiskCount}
                  </div>
                )}
              </div>
            </div>
            <div className="flex items-center gap-4 text-sm">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                <span>High</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
                <span>Medium</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                <span>Low</span>
              </div>
            </div>
          </div>
        </Card>

        {/* Mine Risk Summary */}
        <Card title="Mine Risk Summary" description="Overall risk assessment by mine">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {data.mines_overview.map((mine) => {
              const mineZones = data.risk_heatmap.filter(z => z.mine_id === mine.mine_id);
              const mineHighRisk = mineZones.filter(z => z.risk_level === 'high').length;
              const riskLevel = mineHighRisk > 0 ? 'high' : mine.compliance_rate < 85 ? 'medium' : 'low';

              return (
                <div
                  key={mine.mine_id}
                  className={`p-4 rounded-xl border-l-4 ${
                    riskLevel === 'high'
                      ? 'bg-red-50 border-red-500'
                      : riskLevel === 'medium'
                      ? 'bg-yellow-50 border-yellow-500'
                      : 'bg-green-50 border-green-500'
                  }`}
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <Mountain className="w-5 h-5 text-stone-500" />
                      <span className="font-medium text-stone-900">{mine.mine_name}</span>
                    </div>
                    <span className={`text-xs px-2 py-1 rounded-full font-medium ${
                      riskLevel === 'high'
                        ? 'bg-red-100 text-red-700'
                        : riskLevel === 'medium'
                        ? 'bg-yellow-100 text-yellow-700'
                        : 'bg-green-100 text-green-700'
                    }`}>
                      {riskLevel.toUpperCase()}
                    </span>
                  </div>
                  <div className="space-y-2 text-sm">
                    <div className="flex items-center justify-between">
                      <span className="text-stone-500">Compliance Rate</span>
                      <span className="font-medium">{mine.compliance_rate}%</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-stone-500">Violations (Week)</span>
                      <span className="font-medium">{mine.violations_this_week}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-stone-500">Active Alerts</span>
                      <span className={`font-medium ${mine.active_alerts > 0 ? 'text-red-600' : 'text-green-600'}`}>
                        {mine.active_alerts}
                      </span>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </Card>

        {/* Risk Heatmap */}
        <Card
          title="Zone Risk Heatmap"
          description="Detailed risk assessment by zone"
          action={
            <div className="flex items-center gap-2">
              <Filter className="w-4 h-4 text-stone-400" />
              <select
                value={riskFilter}
                onChange={(e) => setRiskFilter(e.target.value as any)}
                className="text-sm border border-stone-300 rounded-lg px-3 py-1 focus:ring-2 focus:ring-cyan-500"
              >
                <option value="all">All Risks</option>
                <option value="high">High Risk</option>
                <option value="medium">Medium Risk</option>
                <option value="low">Low Risk</option>
              </select>
            </div>
          }
        >
          {filteredRiskZones.length === 0 ? (
            <div className="text-center py-12 text-stone-500">
              <Shield className="w-12 h-12 mx-auto mb-4 text-green-500" />
              <p>No zones match the selected risk level</p>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {filteredRiskZones
                .sort((a, b) => {
                  const riskOrder = { high: 0, medium: 1, low: 2, normal: 2 };
                  return (riskOrder[a.risk_level as keyof typeof riskOrder] || 2) -
                         (riskOrder[b.risk_level as keyof typeof riskOrder] || 2);
                })
                .map((zone) => (
                  <div
                    key={`${zone.mine_id}-${zone.zone_id}`}
                    className={`p-4 rounded-xl border ${
                      zone.risk_level === 'high'
                        ? 'bg-red-50 border-red-200'
                        : zone.risk_level === 'medium'
                        ? 'bg-yellow-50 border-yellow-200'
                        : 'bg-green-50 border-green-200'
                    }`}
                  >
                    <div className="flex items-start justify-between mb-3">
                      <div>
                        <div className="flex items-center gap-2">
                          <MapPin className={`w-4 h-4 ${
                            zone.risk_level === 'high' ? 'text-red-500' :
                            zone.risk_level === 'medium' ? 'text-yellow-500' : 'text-green-500'
                          }`} />
                          <span className="font-medium text-stone-900">{zone.zone_name}</span>
                        </div>
                        <p className="text-xs text-stone-500 mt-1">{zone.mine_name}</p>
                      </div>
                      <div className={`w-10 h-10 rounded-full flex items-center justify-center ${
                        zone.risk_level === 'high' ? 'bg-red-100' :
                        zone.risk_level === 'medium' ? 'bg-yellow-100' : 'bg-green-100'
                      }`}>
                        {zone.risk_level === 'high' ? (
                          <TrendingUp className="w-5 h-5 text-red-600" />
                        ) : zone.risk_level === 'medium' ? (
                          <AlertTriangle className="w-5 h-5 text-yellow-600" />
                        ) : (
                          <Shield className="w-5 h-5 text-green-600" />
                        )}
                      </div>
                    </div>
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-stone-500">Violations</span>
                      <span className={`font-bold ${
                        zone.violations > 10 ? 'text-red-600' :
                        zone.violations > 5 ? 'text-yellow-600' : 'text-green-600'
                      }`}>
                        {zone.violations}
                      </span>
                    </div>
                    <div className="mt-2">
                      <div className="w-full bg-stone-200 rounded-full h-2">
                        <div
                          className={`h-2 rounded-full ${
                            zone.risk_level === 'high' ? 'bg-red-500' :
                            zone.risk_level === 'medium' ? 'bg-yellow-500' : 'bg-green-500'
                          }`}
                          style={{ width: `${Math.min(100, zone.violations * 5)}%` }}
                        />
                      </div>
                    </div>
                  </div>
                ))}
            </div>
          )}
        </Card>

        {/* Critical Alerts */}
        {data.critical_alerts.length > 0 && (
          <Card title="Critical Alerts" description="High-priority alerts requiring immediate attention">
            <div className="space-y-3">
              {data.critical_alerts.map((alert) => (
                <div
                  key={alert.id}
                  className="p-4 bg-red-50 rounded-lg border border-red-200"
                >
                  <div className="flex items-start gap-3">
                    <div className="w-10 h-10 bg-red-100 rounded-lg flex items-center justify-center flex-shrink-0">
                      <AlertTriangle className="w-5 h-5 text-red-600" />
                    </div>
                    <div className="flex-1">
                      <p className="font-medium text-stone-900">{alert.message}</p>
                      <div className="flex items-center gap-2 mt-1 text-sm text-stone-500">
                        <span>{alert.mine_name}</span>
                        <span>•</span>
                        <span>{new Date(alert.created_at).toLocaleString()}</span>
                      </div>
                    </div>
                    <span className="px-2 py-1 bg-red-100 text-red-700 rounded-full text-xs font-medium">
                      {alert.severity.toUpperCase()}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </Card>
        )}
      </div>
    </AppLayout>
  );
}
