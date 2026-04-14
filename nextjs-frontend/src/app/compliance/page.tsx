'use client';

import { useEffect, useState } from 'react';
import {
  Shield,
  AlertTriangle,
  TrendingUp,
  TrendingDown,
  Users,
  MapPin,
  Calendar,
  BarChart3,
  PieChart,
  Download,
  RefreshCw,
  Search,
} from 'lucide-react';
import AppLayout from '@/components/AppLayout';
import { Card, StatCard } from '@/components/Card';
import { Spinner } from '@/components/Loading';
import { dashboardApi, workerApi } from '@/lib/api';
import type { SafetyOfficerDashboard, Worker } from '@/types';

export default function CompliancePage() {
  const [data, setData] = useState<SafetyOfficerDashboard | null>(null);
  const [workers, setWorkers] = useState<Worker[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [complianceFilter, setComplianceFilter] = useState<'all' | 'high' | 'medium' | 'low'>('all');

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      setLoading(true);
      const [dashboardData, workersResult] = await Promise.all([
        dashboardApi.getSafetyOfficer(),
        workerApi.list({ limit: 100 }),
      ]);
      setData(dashboardData);
      setWorkers(workersResult.workers);
      setError(null);
    } catch (err) {
      setError('Failed to load compliance data');
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

  // Filter workers
  const filteredWorkers = workers
    .filter((w) => {
      if (searchTerm) {
        const search = searchTerm.toLowerCase();
        return (
          w.name.toLowerCase().includes(search) ||
          w.employee_id.toLowerCase().includes(search)
        );
      }
      return true;
    })
    .filter((w) => {
      if (complianceFilter === 'all') return true;
      if (complianceFilter === 'high') return w.compliance_score >= 90;
      if (complianceFilter === 'medium') return w.compliance_score >= 70 && w.compliance_score < 90;
      if (complianceFilter === 'low') return w.compliance_score < 70;
      return true;
    })
    .sort((a, b) => a.compliance_score - b.compliance_score);

  // Calculate compliance distribution
  const highCompliance = workers.filter((w) => w.compliance_score >= 90).length;
  const mediumCompliance = workers.filter((w) => w.compliance_score >= 70 && w.compliance_score < 90).length;
  const lowCompliance = workers.filter((w) => w.compliance_score < 70).length;

  return (
    <AppLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold text-stone-800">Compliance Overview</h1>
            <p className="text-stone-500 mt-1">{data.mine_name} • Worker compliance analysis</p>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={loadData}
              className="flex items-center gap-2 px-4 py-2 bg-stone-100 text-stone-700 rounded-lg hover:bg-stone-200"
            >
              <RefreshCw className="w-4 h-4" />
              Refresh
            </button>
            <button className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700">
              <Download className="w-4 h-4" />
              Export Report
            </button>
          </div>
        </div>

        {/* Compliance Rate Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-gradient-to-br from-green-500 to-green-600 rounded-xl p-6 text-white">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-green-100 text-sm">Today&apos;s Rate</p>
                <p className="text-4xl font-bold mt-1">{data.compliance_rates.today}%</p>
                <p className="text-green-200 text-xs mt-2">{data.violations.today} violations</p>
              </div>
              <Shield className="w-12 h-12 text-green-200" />
            </div>
          </div>
          <div className="bg-gradient-to-br from-blue-500 to-blue-600 rounded-xl p-6 text-white">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-blue-100 text-sm">This Week</p>
                <p className="text-4xl font-bold mt-1">{data.compliance_rates.this_week}%</p>
                <p className="text-blue-200 text-xs mt-2">{data.violations.this_week} violations</p>
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

        {/* Compliance Distribution */}
        <Card title="Worker Compliance Distribution">
          <div className="flex items-center gap-4 mb-6">
            <div className="flex-1">
              <div className="flex h-8 rounded-full overflow-hidden">
                <div
                  className="bg-green-500 flex items-center justify-center text-white text-xs font-medium"
                  style={{ width: `${(highCompliance / workers.length) * 100}%` }}
                >
                  {highCompliance}
                </div>
                <div
                  className="bg-yellow-500 flex items-center justify-center text-white text-xs font-medium"
                  style={{ width: `${(mediumCompliance / workers.length) * 100}%` }}
                >
                  {mediumCompliance}
                </div>
                <div
                  className="bg-red-500 flex items-center justify-center text-white text-xs font-medium"
                  style={{ width: `${(lowCompliance / workers.length) * 100}%` }}
                >
                  {lowCompliance}
                </div>
              </div>
            </div>
          </div>
          <div className="grid grid-cols-3 gap-4">
            <div className="text-center p-4 bg-green-50 rounded-lg">
              <div className="w-4 h-4 bg-green-500 rounded-full mx-auto mb-2"></div>
              <p className="text-2xl font-bold text-green-600">{highCompliance}</p>
              <p className="text-sm text-stone-600">High (&gt;90%)</p>
            </div>
            <div className="text-center p-4 bg-yellow-50 rounded-lg">
              <div className="w-4 h-4 bg-yellow-500 rounded-full mx-auto mb-2"></div>
              <p className="text-2xl font-bold text-yellow-600">{mediumCompliance}</p>
              <p className="text-sm text-stone-600">Medium (70-90%)</p>
            </div>
            <div className="text-center p-4 bg-red-50 rounded-lg">
              <div className="w-4 h-4 bg-red-500 rounded-full mx-auto mb-2"></div>
              <p className="text-2xl font-bold text-red-600">{lowCompliance}</p>
              <p className="text-sm text-stone-600">Low (&lt;70%)</p>
            </div>
          </div>
        </Card>

        {/* Violation Breakdown */}
        <Card title="Violation Types" description="Breakdown of violation categories this week">
          <div className="space-y-4">
            {Object.entries(data.violation_trends).length === 0 ? (
              <div className="text-center py-8 text-stone-500">
                <Shield className="w-12 h-12 mx-auto mb-4 text-green-500" />
                <p>No violations recorded this week!</p>
              </div>
            ) : (
              Object.entries(data.violation_trends)
                .sort(([, a], [, b]) => b - a)
                .map(([type, count]) => {
                  const maxCount = Math.max(...Object.values(data.violation_trends));
                  return (
                    <div key={type}>
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-sm font-medium text-stone-700">
                          {type.replace('NO-', 'Missing ')}
                        </span>
                        <span className="text-sm font-bold text-stone-800">{count}</span>
                      </div>
                      <div className="w-full bg-stone-200 rounded-full h-3">
                        <div
                          className="bg-red-500 h-3 rounded-full"
                          style={{ width: `${(count / maxCount) * 100}%` }}
                        />
                      </div>
                    </div>
                  );
                })
            )}
          </div>
        </Card>

        {/* Worker List */}
        <Card
          title="Worker Compliance Details"
          description="Individual worker compliance scores"
          action={
            <div className="flex items-center gap-2">
              <select
                value={complianceFilter}
                onChange={(e) => setComplianceFilter(e.target.value as any)}
                className="text-sm border border-stone-300 rounded-lg px-3 py-1.5"
              >
                <option value="all">All Workers</option>
                <option value="high">High Compliance</option>
                <option value="medium">Medium Compliance</option>
                <option value="low">Low Compliance</option>
              </select>
            </div>
          }
        >
          <div className="mb-4">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-stone-400 w-5 h-5" />
              <input
                type="text"
                placeholder="Search workers..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full pl-10 pr-4 py-2 border border-stone-300 rounded-lg focus:ring-2 focus:ring-blue-500"
              />
            </div>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-stone-100">
                  <th className="text-left py-3 px-4 text-xs font-medium text-stone-500 uppercase">Worker</th>
                  <th className="text-left py-3 px-4 text-xs font-medium text-stone-500 uppercase">Department</th>
                  <th className="text-left py-3 px-4 text-xs font-medium text-stone-500 uppercase">Shift</th>
                  <th className="text-left py-3 px-4 text-xs font-medium text-stone-500 uppercase">Compliance</th>
                  <th className="text-left py-3 px-4 text-xs font-medium text-stone-500 uppercase">Violations</th>
                </tr>
              </thead>
              <tbody>
                {filteredWorkers.length === 0 ? (
                  <tr>
                    <td colSpan={5} className="text-center py-8 text-stone-500">
                      No workers found
                    </td>
                  </tr>
                ) : (
                  filteredWorkers.map((worker) => (
                    <tr key={worker.id} className="border-b border-stone-50 hover:bg-stone-50">
                      <td className="py-3 px-4">
                        <div>
                          <p className="font-medium text-stone-800">{worker.name}</p>
                          <p className="text-xs text-stone-500">{worker.employee_id}</p>
                        </div>
                      </td>
                      <td className="py-3 px-4 text-sm text-stone-600">
                        {worker.department || '-'}
                      </td>
                      <td className="py-3 px-4">
                        <span className="text-sm capitalize">{worker.assigned_shift}</span>
                      </td>
                      <td className="py-3 px-4">
                        <div className="flex items-center gap-2">
                          <div className="w-16 bg-stone-200 rounded-full h-2">
                            <div
                              className={`h-2 rounded-full ${
                                worker.compliance_score >= 90 ? 'bg-green-500' :
                                worker.compliance_score >= 70 ? 'bg-yellow-500' : 'bg-red-500'
                              }`}
                              style={{ width: `${worker.compliance_score}%` }}
                            />
                          </div>
                          <span className={`font-medium text-sm ${
                            worker.compliance_score >= 90 ? 'text-green-600' :
                            worker.compliance_score >= 70 ? 'text-yellow-600' : 'text-red-600'
                          }`}>
                            {worker.compliance_score}%
                          </span>
                        </div>
                      </td>
                      <td className="py-3 px-4">
                        <span className={`inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium ${
                          worker.total_violations === 0
                            ? 'bg-green-100 text-green-700'
                            : worker.total_violations <= 3
                            ? 'bg-yellow-100 text-yellow-700'
                            : 'bg-red-100 text-red-700'
                        }`}>
                          {worker.total_violations}
                        </span>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </Card>

        {/* Zone Risk Analysis */}
        <Card title="Zone Risk Analysis" description="Compliance status by zone">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {data.zone_risk_analysis.map((zone) => (
              <div
                key={zone.zone_id}
                className={`p-4 rounded-lg border-l-4 ${
                  zone.risk_level === 'critical' ? 'bg-red-50 border-red-500' :
                  zone.risk_level === 'high' ? 'bg-orange-50 border-orange-500' :
                  zone.risk_level === 'normal' ? 'bg-yellow-50 border-yellow-500' :
                  'bg-green-50 border-green-500'
                }`}
              >
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <MapPin className="w-4 h-4 text-stone-500" />
                    <span className="font-medium text-stone-800">{zone.zone_name}</span>
                  </div>
                  <span className={`text-xs px-2 py-1 rounded-full font-medium ${
                    zone.risk_level === 'critical' ? 'bg-red-100 text-red-700' :
                    zone.risk_level === 'high' ? 'bg-orange-100 text-orange-700' :
                    zone.risk_level === 'normal' ? 'bg-yellow-100 text-yellow-700' :
                    'bg-green-100 text-green-700'
                  }`}>
                    {zone.risk_level.toUpperCase()}
                  </span>
                </div>
                <div className="flex items-center justify-between text-sm text-stone-600">
                  <span className="flex items-center gap-1">
                    <Users className="w-4 h-4" />
                    {zone.worker_count} workers
                  </span>
                  <span className="flex items-center gap-1">
                    <AlertTriangle className="w-4 h-4" />
                    {zone.violations_this_week} violations
                  </span>
                </div>
              </div>
            ))}
          </div>
        </Card>
      </div>
    </AppLayout>
  );
}
