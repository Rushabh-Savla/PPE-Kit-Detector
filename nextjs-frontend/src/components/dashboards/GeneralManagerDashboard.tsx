'use client';

import { useEffect, useState } from 'react';
import {
  Building2,
  Users,
  TrendingUp,
  TrendingDown,
  Shield,
  DollarSign,
  AlertTriangle,
  CheckCircle,
  BarChart3,
  PieChart,
  Target,
  Award,
  FileCheck,
  Mountain,
} from 'lucide-react';
import { Card, StatCard } from '@/components/Card';
import { Spinner } from '@/components/Loading';
import { dashboardApi } from '@/lib/api';
import { formatTime } from '@/lib/utils';
import type { GeneralManagerDashboard as DashboardType } from '@/types';

export default function GeneralManagerDashboard() {
  const [data, setData] = useState<DashboardType | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      setLoading(true);
      const dashboardData = await dashboardApi.getGeneralManager();
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

  const getComplianceStatus = () => {
    const current = data.regulatory_status.current_compliance;
    const threshold = data.regulatory_status.compliance_threshold;
    if (current >= threshold) return { color: 'green', text: 'Compliant' };
    if (current >= threshold - 5) return { color: 'yellow', text: 'At Risk' };
    return { color: 'red', text: 'Non-Compliant' };
  };

  const complianceStatus = getComplianceStatus();

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-stone-800">Executive Dashboard</h1>
          <p className="text-stone-500 mt-1">Organization-wide KPIs & Strategic Overview</p>
        </div>
        <div className="flex items-center gap-2">
          <span className={`inline-flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium ${
            complianceStatus.color === 'green'
              ? 'bg-green-100 text-green-700'
              : complianceStatus.color === 'yellow'
              ? 'bg-yellow-100 text-yellow-700'
              : 'bg-red-100 text-red-700'
          }`}>
            <FileCheck className="w-4 h-4" />
            Regulatory: {complianceStatus.text}
          </span>
        </div>
      </div>

      {/* KPI Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-gradient-to-br from-blue-600 to-blue-700 rounded-xl p-6 text-white">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-blue-100 text-sm">Total Mines</p>
              <p className="text-4xl font-bold mt-1">{data.organization_overview.total_mines}</p>
            </div>
            <Building2 className="w-12 h-12 text-blue-200" />
          </div>
        </div>

        <div className="bg-gradient-to-br from-purple-600 to-purple-700 rounded-xl p-6 text-white">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-purple-100 text-sm">Total Workforce</p>
              <p className="text-4xl font-bold mt-1">{data.organization_overview.total_workers}</p>
            </div>
            <Users className="w-12 h-12 text-purple-200" />
          </div>
        </div>

        <div className="bg-gradient-to-br from-green-600 to-green-700 rounded-xl p-6 text-white">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-green-100 text-sm">Overall Compliance</p>
              <p className="text-4xl font-bold mt-1">{data.organization_overview.compliance_rate}%</p>
            </div>
            <Shield className="w-12 h-12 text-green-200" />
          </div>
        </div>

        <div className="bg-gradient-to-br from-amber-500 to-amber-600 rounded-xl p-6 text-white">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-amber-100 text-sm">Est. Cost Savings</p>
              <p className="text-4xl font-bold mt-1">
                ${(data.financial_insights.estimated_cost_savings / 1000).toFixed(0)}k
              </p>
            </div>
            <DollarSign className="w-12 h-12 text-amber-200" />
          </div>
        </div>
      </div>

      {/* Monthly KPIs */}
      <Card title="Monthly Performance Summary" description="Current month metrics">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="text-center p-4 bg-stone-50 rounded-xl">
            <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-3">
              <BarChart3 className="w-8 h-8 text-blue-600" />
            </div>
            <p className="text-3xl font-bold text-stone-800">{data.kpi_summary.monthly_entries.toLocaleString()}</p>
            <p className="text-sm text-stone-500 mt-1">Total Entries</p>
          </div>
          <div className="text-center p-4 bg-stone-50 rounded-xl">
            <div className="w-16 h-16 bg-red-100 rounded-full flex items-center justify-center mx-auto mb-3">
              <AlertTriangle className="w-8 h-8 text-red-600" />
            </div>
            <p className="text-3xl font-bold text-stone-800">{data.kpi_summary.monthly_violations}</p>
            <p className="text-sm text-stone-500 mt-1">Total Violations</p>
          </div>
          <div className="text-center p-4 bg-stone-50 rounded-xl">
            <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-3">
              <Target className="w-8 h-8 text-green-600" />
            </div>
            <p className="text-3xl font-bold text-stone-800">{data.kpi_summary.monthly_compliance_rate}%</p>
            <p className="text-sm text-stone-500 mt-1">Compliance Rate</p>
          </div>
        </div>
      </Card>

      {/* Mine Performance Table */}
      <Card title="Mine Performance Comparison" description="Performance metrics across all mines">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-stone-100">
                <th className="text-left py-3 px-4 text-xs font-medium text-stone-500 uppercase">Mine</th>
                <th className="text-left py-3 px-4 text-xs font-medium text-stone-500 uppercase">Entries</th>
                <th className="text-left py-3 px-4 text-xs font-medium text-stone-500 uppercase">Violations</th>
                <th className="text-left py-3 px-4 text-xs font-medium text-stone-500 uppercase">Compliance</th>
                <th className="text-left py-3 px-4 text-xs font-medium text-stone-500 uppercase">Trend</th>
              </tr>
            </thead>
            <tbody>
              {data.mine_performance.map((mine) => {
                const avgCompliance = data.organization_overview.compliance_rate;
                const trend = mine.compliance_rate >= avgCompliance ? 'up' : 'down';
                return (
                  <tr key={mine.mine_id} className="border-b border-stone-50 hover:bg-stone-50">
                    <td className="py-3 px-4">
                      <div className="flex items-center gap-2">
                        <Mountain className="w-4 h-4 text-stone-400" />
                        <span className="font-medium text-stone-800">{mine.mine_name}</span>
                      </div>
                    </td>
                    <td className="py-3 px-4">
                      <span className="text-stone-700">{mine.total_entries.toLocaleString()}</span>
                    </td>
                    <td className="py-3 px-4">
                      <span className={`inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium ${
                        mine.violations > 10
                          ? 'bg-red-100 text-red-700'
                          : mine.violations > 5
                          ? 'bg-yellow-100 text-yellow-700'
                          : 'bg-green-100 text-green-700'
                      }`}>
                        {mine.violations}
                      </span>
                    </td>
                    <td className="py-3 px-4">
                      <div className="flex items-center gap-2">
                        <div className="w-24 bg-stone-200 rounded-full h-2">
                          <div
                            className={`h-2 rounded-full ${
                              mine.compliance_rate >= 95 ? 'bg-green-500' :
                              mine.compliance_rate >= 85 ? 'bg-yellow-500' : 'bg-red-500'
                            }`}
                            style={{ width: `${mine.compliance_rate}%` }}
                          />
                        </div>
                        <span className={`font-semibold text-sm ${
                          mine.compliance_rate >= 95 ? 'text-green-600' :
                          mine.compliance_rate >= 85 ? 'text-yellow-600' : 'text-red-600'
                        }`}>
                          {mine.compliance_rate}%
                        </span>
                      </div>
                    </td>
                    <td className="py-3 px-4">
                      {trend === 'up' ? (
                        <span className="flex items-center gap-1 text-green-600">
                          <TrendingUp className="w-4 h-4" />
                          Above Avg
                        </span>
                      ) : (
                        <span className="flex items-center gap-1 text-red-600">
                          <TrendingDown className="w-4 h-4" />
                          Below Avg
                        </span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Regulatory Compliance Status */}
        <Card title="Regulatory Compliance" description="Current compliance status vs threshold">
          <div className="space-y-6">
            <div className="flex items-center justify-between p-4 rounded-xl bg-stone-50">
              <div>
                <p className="text-sm text-stone-500">Regulatory Threshold</p>
                <p className="text-2xl font-bold text-stone-800">{data.regulatory_status.compliance_threshold}%</p>
              </div>
              <div className="w-12 h-12 bg-stone-200 rounded-full flex items-center justify-center">
                <Target className="w-6 h-6 text-stone-600" />
              </div>
            </div>

            <div className={`flex items-center justify-between p-4 rounded-xl ${
              complianceStatus.color === 'green'
                ? 'bg-green-50'
                : complianceStatus.color === 'yellow'
                ? 'bg-yellow-50'
                : 'bg-red-50'
            }`}>
              <div>
                <p className="text-sm text-stone-500">Current Compliance</p>
                <p className={`text-2xl font-bold ${
                  complianceStatus.color === 'green'
                    ? 'text-green-700'
                    : complianceStatus.color === 'yellow'
                    ? 'text-yellow-700'
                    : 'text-red-700'
                }`}>
                  {data.regulatory_status.current_compliance}%
                </p>
              </div>
              <div className={`w-12 h-12 rounded-full flex items-center justify-center ${
                complianceStatus.color === 'green'
                  ? 'bg-green-100'
                  : complianceStatus.color === 'yellow'
                  ? 'bg-yellow-100'
                  : 'bg-red-100'
              }`}>
                {complianceStatus.color === 'green' ? (
                  <CheckCircle className="w-6 h-6 text-green-600" />
                ) : (
                  <AlertTriangle className={`w-6 h-6 ${
                    complianceStatus.color === 'yellow' ? 'text-yellow-600' : 'text-red-600'
                  }`} />
                )}
              </div>
            </div>

            <div className="relative pt-2">
              <div className="flex mb-2 items-center justify-between">
                <span className="text-xs font-semibold text-stone-600">Progress to Target</span>
                <span className="text-xs font-semibold text-stone-600">
                  {Math.min(100, Math.round((data.regulatory_status.current_compliance / data.regulatory_status.compliance_threshold) * 100))}%
                </span>
              </div>
              <div className="overflow-hidden h-3 text-xs flex rounded-full bg-stone-200">
                <div
                  style={{ width: `${Math.min(100, (data.regulatory_status.current_compliance / data.regulatory_status.compliance_threshold) * 100)}%` }}
                  className={`shadow-none flex flex-col text-center whitespace-nowrap text-white justify-center ${
                    complianceStatus.color === 'green'
                      ? 'bg-green-500'
                      : complianceStatus.color === 'yellow'
                      ? 'bg-yellow-500'
                      : 'bg-red-500'
                  }`}
                />
              </div>
            </div>
          </div>
        </Card>

        {/* Strategic Alerts */}
        <Card title="Strategic Alerts" description="High-priority alerts requiring executive attention">
          <div className="space-y-3">
            {data.strategic_alerts.length === 0 ? (
              <div className="text-center py-8 text-stone-500">
                <CheckCircle className="w-12 h-12 mx-auto mb-3 text-green-500" />
                <p className="font-medium">All Clear</p>
                <p className="text-sm">No strategic alerts at this time</p>
              </div>
            ) : (
              data.strategic_alerts.map((alert) => (
                <div
                  key={alert.id}
                  className={`p-4 rounded-lg border-l-4 ${
                    alert.severity === 'critical'
                      ? 'bg-red-50 border-red-500'
                      : alert.severity === 'high'
                      ? 'bg-orange-50 border-orange-500'
                      : 'bg-yellow-50 border-yellow-500'
                  }`}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <AlertTriangle className={`w-4 h-4 ${
                          alert.severity === 'critical'
                            ? 'text-red-600'
                            : alert.severity === 'high'
                            ? 'text-orange-600'
                            : 'text-yellow-600'
                        }`} />
                        <span className={`text-xs font-medium px-2 py-0.5 rounded ${
                          alert.severity === 'critical'
                            ? 'bg-red-100 text-red-700'
                            : alert.severity === 'high'
                            ? 'bg-orange-100 text-orange-700'
                            : 'bg-yellow-100 text-yellow-700'
                        }`}>
                          {alert.severity.toUpperCase()}
                        </span>
                      </div>
                      <p className="text-sm font-medium text-stone-800">{alert.message}</p>
                      <div className="flex items-center gap-2 mt-2 text-xs text-stone-500">
                        <span>{alert.mine_name}</span>
                        <span>•</span>
                        <span>{formatTime(alert.created_at)}</span>
                      </div>
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </Card>
      </div>

      {/* Financial Insights */}
      <Card title="Financial Impact" description="Safety-related financial metrics">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-gradient-to-br from-green-50 to-green-100 rounded-xl p-6">
            <div className="flex items-center gap-4">
              <div className="w-14 h-14 bg-green-500 rounded-xl flex items-center justify-center">
                <DollarSign className="w-7 h-7 text-white" />
              </div>
              <div>
                <p className="text-sm text-green-700">Estimated Cost Savings</p>
                <p className="text-3xl font-bold text-green-800">
                  ${data.financial_insights.estimated_cost_savings.toLocaleString()}
                </p>
                <p className="text-xs text-green-600 mt-1">from prevented incidents</p>
              </div>
            </div>
          </div>

          <div className="bg-gradient-to-br from-blue-50 to-blue-100 rounded-xl p-6">
            <div className="flex items-center gap-4">
              <div className="w-14 h-14 bg-blue-500 rounded-xl flex items-center justify-center">
                <Shield className="w-7 h-7 text-white" />
              </div>
              <div>
                <p className="text-sm text-blue-700">Violations Prevented</p>
                <p className="text-3xl font-bold text-blue-800">
                  {data.financial_insights.violations_prevented_this_week}
                </p>
                <p className="text-xs text-blue-600 mt-1">this week through early intervention</p>
              </div>
            </div>
          </div>
        </div>
      </Card>
    </div>
  );
}
