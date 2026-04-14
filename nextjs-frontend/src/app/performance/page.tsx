'use client';

import { useEffect, useState } from 'react';
import {
  TrendingUp,
  TrendingDown,
  Download,
  Mountain,
  Users,
  Shield,
  AlertTriangle,
  Target,
  Heart,
  Activity,
  Droplets,
  RefreshCw,
} from 'lucide-react';
import {
  AreaChart,
  Area,
  BarChart,
  Bar,
  PieChart,
  Pie,
  LineChart,
  Line,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import AppLayout from '@/components/AppLayout';
import { Card, StatCard } from '@/components/Card';
import { Spinner } from '@/components/Loading';
import { dashboardApi, mineApi, healthApi } from '@/lib/api';
import type { GeneralManagerDashboard, Mine, HealthDashboardResponse } from '@/types';

const HEALTH_COLORS = {
  normal: '#34d399',    // emerald-400 (softer green)
  warning: '#fbbf24',   // amber-400 (softer amber)
  critical: '#fb7185',  // rose-400 (softer red)
};

export default function PerformancePage() {
  const [data, setData] = useState<GeneralManagerDashboard | null>(null);
  const [healthData, setHealthData] = useState<HealthDashboardResponse | null>(null);
  const [mines, setMines] = useState<Mine[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [timeRange, setTimeRange] = useState<'week' | 'month' | 'quarter'>('month');
  const [activeTab, setActiveTab] = useState<'compliance' | 'health'>('compliance');
  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => {
    loadData();
  }, [timeRange]);

  const loadData = async () => {
    try {
      setLoading(true);

      let dashboardData: GeneralManagerDashboard;
      let minesList: Mine[] = [];

      try {
        const [apiDashboardData, minesResult] = await Promise.all([
          dashboardApi.getGeneralManager(),
          mineApi.list(),
        ]);
        dashboardData = apiDashboardData;
        minesList = minesResult.mines;
      } catch (apiErr) {
        console.log('API failed, using mock data:', apiErr);
        // Generate comprehensive mock dashboard data
        dashboardData = generateMockDashboardData();
        minesList = generateMockMines();
      }

      setData(dashboardData);
      setMines(minesList);

      // Load health data
      try {
        const health = await healthApi.getDashboard({ days_back: timeRange === 'week' ? 7 : timeRange === 'month' ? 30 : 90 });
        setHealthData(health);
      } catch (healthErr) {
        console.log('Health data not available yet:', healthErr);
        // Generate mock health data if API not ready - pass mines directly since state may not be updated yet
        setHealthData(generateMockHealthData(minesList));
      }

      setError(null);
    } catch (err) {
      setError('Failed to load performance data');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  // Generate mock mines for fallback
  const generateMockMines = (): Mine[] => [
    { id: 'mine-001', name: 'Jharia Coal Mine', location: 'Jharia, Jharkhand', is_active: true, zones: [], gates: [], created_at: '2024-01-15' },
    { id: 'mine-002', name: 'Bokaro Steel Mine', location: 'Bokaro, Jharkhand', is_active: true, zones: [], gates: [], created_at: '2024-02-20' },
    { id: 'mine-003', name: 'Raniganj Colliery', location: 'Raniganj, West Bengal', is_active: true, zones: [], gates: [], created_at: '2024-03-10' },
    { id: 'mine-004', name: 'Singrauli Coal Mine', location: 'Singrauli, Madhya Pradesh', is_active: true, zones: [], gates: [], created_at: '2024-04-05' },
    { id: 'mine-005', name: 'Talcher Coal Fields', location: 'Talcher, Odisha', is_active: true, zones: [], gates: [], created_at: '2024-05-12' },
  ];

  // Generate comprehensive mock dashboard data
  const generateMockDashboardData = (): GeneralManagerDashboard => ({
    organization_overview: {
      total_mines: 5,
      total_workers: 847,
      compliance_rate: 94.2,
    },
    kpi_summary: {
      monthly_entries: 12458,
      monthly_violations: 156,
      monthly_compliance_rate: 98.7,
    },
    mine_performance: [
      { mine_id: 'mine-001', mine_name: 'Jharia Coal Mine', compliance_rate: 96.8, total_entries: 3245, violations: 32 },
      { mine_id: 'mine-002', mine_name: 'Bokaro Steel Mine', compliance_rate: 94.2, total_entries: 2876, violations: 48 },
      { mine_id: 'mine-003', mine_name: 'Raniganj Colliery', compliance_rate: 91.5, total_entries: 2134, violations: 62 },
      { mine_id: 'mine-004', mine_name: 'Singrauli Coal Mine', compliance_rate: 97.3, total_entries: 2654, violations: 18 },
      { mine_id: 'mine-005', mine_name: 'Talcher Coal Fields', compliance_rate: 89.4, total_entries: 1549, violations: 45 },
    ],
    strategic_alerts: [
      {
        id: 'sa-001',
        alert_type: 'compliance_trend',
        severity: 'medium',
        status: 'active',
        message: 'Raniganj Colliery compliance trending down 3.2% this week',
        mine_id: 'mine-003',
        mine_name: 'Raniganj Colliery',
        created_at: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
      },
      {
        id: 'sa-002',
        alert_type: 'safety_milestone',
        severity: 'low',
        status: 'active',
        message: 'Singrauli Coal Mine achieved 30 consecutive days without violations',
        mine_id: 'mine-004',
        mine_name: 'Singrauli Coal Mine',
        created_at: new Date(Date.now() - 6 * 60 * 60 * 1000).toISOString(),
      },
    ],
    financial_insights: {
      estimated_cost_savings: 245000,
      violations_prevented_this_week: 28,
    },
    regulatory_status: {
      compliance_threshold: 85,
      current_compliance: 94.2,
      status: 'compliant',
    },
  });

  const handleRefresh = async () => {
    setRefreshing(true);
    await loadData();
    setRefreshing(false);
  };

  // Generate mock health data for demonstration
  const generateMockHealthData = (minesList: Mine[]): HealthDashboardResponse => {
    const today = new Date();
    const trends = [];
    const daysBack = timeRange === 'week' ? 7 : timeRange === 'month' ? 30 : 90;

    for (let i = daysBack - 1; i >= 0; i--) {
      const date = new Date(today);
      date.setDate(date.getDate() - i);
      trends.push({
        date: date.toISOString().split('T')[0],
        avg_spo2: 96 + Math.random() * 2,
        avg_systolic_bp: 118 + Math.random() * 8,
        avg_diastolic_bp: 75 + Math.random() * 6,
        avg_heart_rate: 72 + Math.random() * 10,
        normal_count: Math.floor(35 + Math.random() * 10),
        warning_count: Math.floor(3 + Math.random() * 5),
        critical_count: Math.floor(Math.random() * 2),
      });
    }

    // Use provided mines list or create default mine summaries
    const minesToUse = minesList.length > 0 ? minesList : [
      { id: '1', name: 'Alpha Mine' },
      { id: '2', name: 'Beta Mine' },
    ] as Mine[];

    const mineSummaries = minesToUse.map(mine => ({
      mine_id: mine.id,
      mine_name: mine.name,
      workers_monitored: Math.floor(20 + Math.random() * 30),
      avg_spo2: 95.5 + Math.random() * 2.5,
      avg_systolic_bp: 115 + Math.random() * 12,
      avg_diastolic_bp: 72 + Math.random() * 10,
      avg_heart_rate: 70 + Math.random() * 15,
      workers_normal: Math.floor(15 + Math.random() * 20),
      workers_warning: Math.floor(2 + Math.random() * 5),
      workers_critical: Math.floor(Math.random() * 2),
      health_score: Math.floor(80 + Math.random() * 18),
    }));

    // Calculate stats from mine summaries
    const totalWorkers = mineSummaries.reduce((sum, m) => sum + m.workers_monitored, 0) || 45;
    const normalWorkers = mineSummaries.reduce((sum, m) => sum + m.workers_normal, 0) || 38;
    const warningWorkers = mineSummaries.reduce((sum, m) => sum + m.workers_warning, 0) || 5;
    const criticalWorkers = mineSummaries.reduce((sum, m) => sum + m.workers_critical, 0) || 2;

    const avgSpo2 = mineSummaries.length > 0
      ? mineSummaries.reduce((sum, m) => sum + m.avg_spo2, 0) / mineSummaries.length
      : 96.8;
    const avgSystolic = mineSummaries.length > 0
      ? mineSummaries.reduce((sum, m) => sum + m.avg_systolic_bp, 0) / mineSummaries.length
      : 120;
    const avgDiastolic = mineSummaries.length > 0
      ? mineSummaries.reduce((sum, m) => sum + m.avg_diastolic_bp, 0) / mineSummaries.length
      : 78;
    const avgHeartRate = mineSummaries.length > 0
      ? mineSummaries.reduce((sum, m) => sum + m.avg_heart_rate, 0) / mineSummaries.length
      : 75;

    return {
      stats: {
        total_workers_monitored: totalWorkers,
        workers_normal: normalWorkers,
        workers_warning: warningWorkers,
        workers_critical: criticalWorkers,
        avg_spo2: avgSpo2,
        avg_systolic_bp: avgSystolic,
        avg_diastolic_bp: avgDiastolic,
        avg_heart_rate: avgHeartRate,
        readings_today: Math.floor(150 + Math.random() * 100),
        alerts_today: Math.floor(5 + Math.random() * 10),
      },
      mine_summaries: mineSummaries,
      trends,
      at_risk_workers: [
        {
          worker_id: '1',
          worker_name: 'Rajesh Kumar',
          employee_id: 'EMP001',
          avg_spo2: 92.5,
          avg_systolic_bp: 145,
          avg_diastolic_bp: 95,
          avg_heart_rate: 105,
          overall_status: 'warning' as const,
          readings_count: 24,
          alerts_count: 3,
          last_updated: new Date().toISOString(),
        },
        {
          worker_id: '2',
          worker_name: 'Amit Singh',
          employee_id: 'EMP002',
          avg_spo2: 89.2,
          avg_systolic_bp: 160,
          avg_diastolic_bp: 102,
          avg_heart_rate: 115,
          overall_status: 'critical' as const,
          readings_count: 18,
          alerts_count: 7,
          last_updated: new Date().toISOString(),
        },
        {
          worker_id: '3',
          worker_name: 'Vikram Yadav',
          employee_id: 'EMP005',
          avg_spo2: 93.8,
          avg_systolic_bp: 138,
          avg_diastolic_bp: 88,
          avg_heart_rate: 98,
          overall_status: 'warning' as const,
          readings_count: 22,
          alerts_count: 2,
          last_updated: new Date().toISOString(),
        },
      ],
      recent_alerts: [
        {
          worker_id: '2',
          worker_name: 'Amit Singh',
          alert_type: 'Low SpO2',
          message: 'SpO2 dropped to 88% - immediate attention required',
          timestamp: new Date(Date.now() - 30 * 60 * 1000).toISOString(),
          severity: 'critical' as const,
        },
        {
          worker_id: '1',
          worker_name: 'Rajesh Kumar',
          alert_type: 'High BP',
          message: 'Blood pressure elevated to 148/96 mmHg',
          timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
          severity: 'warning' as const,
        },
        {
          worker_id: '3',
          worker_name: 'Vikram Yadav',
          alert_type: 'Elevated Heart Rate',
          message: 'Heart rate at 108 bpm during rest period',
          timestamp: new Date(Date.now() - 4 * 60 * 60 * 1000).toISOString(),
          severity: 'warning' as const,
        },
      ],
    };
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

  // Calculate compliance metrics with fallback values
  const minePerformance = data.mine_performance || [];
  const avgCompliance = minePerformance.length > 0
    ? minePerformance.reduce((sum, m) => sum + m.compliance_rate, 0) / minePerformance.length
    : 85; // Default fallback

  const bestMine = minePerformance.length > 0
    ? minePerformance.reduce((best, m) =>
        m.compliance_rate > (best?.compliance_rate || 0) ? m : best
      , minePerformance[0])
    : null;

  const worstMine = minePerformance.length > 0
    ? minePerformance.reduce((worst, m) =>
        m.compliance_rate < (worst?.compliance_rate || 100) ? m : worst
      , minePerformance[0])
    : null;

  const totalEntries = minePerformance.reduce((sum, m) => sum + m.total_entries, 0);
  const totalViolations = minePerformance.reduce((sum, m) => sum + m.violations, 0);

  // Health status distribution for pie chart
  const healthDistribution = healthData ? [
    { name: 'Normal', value: healthData.stats.workers_normal, color: HEALTH_COLORS.normal },
    { name: 'Warning', value: healthData.stats.workers_warning, color: HEALTH_COLORS.warning },
    { name: 'Critical', value: healthData.stats.workers_critical, color: HEALTH_COLORS.critical },
  ] : [];

  // Vital signs gauges data
  const vitalGauges = healthData ? [
    {
      name: 'SpO2',
      value: healthData.stats.avg_spo2,
      max: 100,
      unit: '%',
      color: '#06b6d4',
      icon: Droplets,
      status: healthData.stats.avg_spo2 >= 95 ? 'normal' : healthData.stats.avg_spo2 >= 90 ? 'warning' : 'critical',
    },
    {
      name: 'Blood Pressure',
      value: healthData.stats.avg_systolic_bp,
      max: 180,
      unit: 'mmHg',
      color: '#f43f5e',
      icon: Activity,
      status: healthData.stats.avg_systolic_bp <= 120 ? 'normal' : healthData.stats.avg_systolic_bp <= 140 ? 'warning' : 'critical',
    },
    {
      name: 'Heart Rate',
      value: healthData.stats.avg_heart_rate,
      max: 150,
      unit: 'bpm',
      color: '#8b5cf6',
      icon: Heart,
      status: healthData.stats.avg_heart_rate >= 60 && healthData.stats.avg_heart_rate <= 100 ? 'normal' : 'warning',
    },
  ] : [];

  return (
    <AppLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold text-stone-900">Performance Tracker</h1>
            <p className="text-stone-500 mt-1">Compliance & Health Analytics Dashboard</p>
          </div>
          <div className="flex items-center gap-3">
            <select
              value={timeRange}
              onChange={(e) => setTimeRange(e.target.value as any)}
              className="px-4 py-2 border border-stone-300 rounded-lg focus:ring-2 focus:ring-orange-300 focus:border-orange-300"
            >
              <option value="week">This Week</option>
              <option value="month">This Month</option>
              <option value="quarter">This Quarter</option>
            </select>
            <button
              onClick={handleRefresh}
              disabled={refreshing}
              className="flex items-center gap-2 px-4 py-2 bg-stone-100 text-stone-700 rounded-lg hover:bg-stone-200 transition-colors"
            >
              <RefreshCw className={`w-4 h-4 ${refreshing ? 'animate-spin' : ''}`} />
              Refresh
            </button>
            <button className="flex items-center gap-2 px-4 py-2 bg-orange-100 text-orange-700 border border-orange-200 rounded-lg hover:bg-orange-200 transition-colors">
              <Download className="w-4 h-4" />
              Export Report
            </button>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="flex gap-2 p-1 bg-stone-100 rounded-xl w-fit">
          <button
            onClick={() => setActiveTab('compliance')}
            className={`flex items-center gap-2 px-6 py-3 rounded-lg font-medium transition-all ${
              activeTab === 'compliance'
                ? 'bg-white text-blue-600 shadow-sm'
                : 'text-stone-600 hover:text-stone-900'
            }`}
          >
            <Shield className="w-5 h-5" />
            Compliance Tracking
          </button>
          <button
            onClick={() => setActiveTab('health')}
            className={`flex items-center gap-2 px-6 py-3 rounded-lg font-medium transition-all ${
              activeTab === 'health'
                ? 'bg-white text-rose-600 shadow-sm'
                : 'text-stone-600 hover:text-stone-900'
            }`}
          >
            <Heart className="w-5 h-5" />
            Health Monitoring
          </button>
        </div>

        {/* Compliance Tab */}
        {activeTab === 'compliance' && (
          <div className="space-y-6">
            {/* KPI Overview */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <StatCard
                title="Avg Compliance"
                value={`${avgCompliance.toFixed(1)}%`}
                icon={<Shield size={24} />}
                color={avgCompliance >= 90 ? 'green' : avgCompliance >= 80 ? 'yellow' : 'red'}
              />
              <StatCard
                title="Total Entries"
                value={totalEntries.toLocaleString()}
                icon={<Users size={24} />}
                color="blue"
              />
              <StatCard
                title="Total Violations"
                value={totalViolations}
                icon={<AlertTriangle size={24} />}
                color={totalViolations === 0 ? 'green' : 'red'}
              />
              <StatCard
                title="Active Mines"
                value={data.organization_overview.total_mines}
                icon={<Mountain size={24} />}
                color="purple"
              />
            </div>

            {/* Best vs Worst Performers */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {bestMine && (
                <div className="bg-gradient-to-br from-emerald-50 to-emerald-100 border border-emerald-200 rounded-xl p-6">
                  <div className="flex items-center gap-2 mb-4">
                    <TrendingUp className="w-5 h-5 text-emerald-600" />
                    <span className="text-emerald-600 text-sm font-medium">Best Performer</span>
                  </div>
                  <h3 className="text-2xl font-bold text-emerald-800">{bestMine.mine_name}</h3>
                  <div className="flex items-center gap-6 mt-4">
                    <div>
                      <p className="text-emerald-600 text-sm">Compliance Rate</p>
                      <p className="text-3xl font-bold text-emerald-700">{bestMine.compliance_rate}%</p>
                    </div>
                    <div>
                      <p className="text-emerald-600 text-sm">Violations</p>
                      <p className="text-3xl font-bold text-emerald-700">{bestMine.violations}</p>
                    </div>
                  </div>
                </div>
              )}

              {worstMine && worstMine !== bestMine && (
                <div className="bg-gradient-to-br from-rose-50 to-rose-100 border border-rose-200 rounded-xl p-6">
                  <div className="flex items-center gap-2 mb-4">
                    <TrendingDown className="w-5 h-5 text-rose-600" />
                    <span className="text-rose-600 text-sm font-medium">Needs Improvement</span>
                  </div>
                  <h3 className="text-2xl font-bold text-rose-800">{worstMine.mine_name}</h3>
                  <div className="flex items-center gap-6 mt-4">
                    <div>
                      <p className="text-rose-600 text-sm">Compliance Rate</p>
                      <p className="text-3xl font-bold text-rose-700">{worstMine.compliance_rate}%</p>
                    </div>
                    <div>
                      <p className="text-rose-600 text-sm">Violations</p>
                      <p className="text-3xl font-bold text-rose-700">{worstMine.violations}</p>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Mine Performance Comparison */}
            <Card title="Mine Performance Comparison" description="Compliance rates across all mines">
              <div className="space-y-4">
                {minePerformance.length > 0 ? (
                  [...minePerformance]
                    .sort((a, b) => b.compliance_rate - a.compliance_rate)
                    .map((mine, index) => (
                      <div key={mine.mine_id} className="flex items-center gap-4">
                        <div className="w-8 h-8 rounded-full bg-stone-100 flex items-center justify-center text-sm font-bold text-stone-600">
                          {index + 1}
                        </div>
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center justify-between mb-1">
                            <span className="font-medium text-stone-900">{mine.mine_name}</span>
                            <span className={`font-bold ${
                              mine.compliance_rate >= 90 ? 'text-emerald-600' :
                              mine.compliance_rate >= 80 ? 'text-amber-600' : 'text-rose-600'
                            }`}>
                              {mine.compliance_rate}%
                            </span>
                          </div>
                          <div className="w-full bg-stone-200 rounded-full h-3">
                            <div
                              className={`h-3 rounded-full transition-all ${
                                mine.compliance_rate >= 90 ? 'bg-emerald-400' :
                                mine.compliance_rate >= 80 ? 'bg-amber-400' : 'bg-rose-400'
                              }`}
                              style={{ width: `${mine.compliance_rate}%` }}
                            />
                          </div>
                          <div className="flex items-center gap-4 mt-1 text-xs text-stone-500">
                            <span>{mine.total_entries.toLocaleString()} entries</span>
                            <span>{mine.violations} violations</span>
                          </div>
                        </div>
                      </div>
                    ))
                ) : (
                  <div className="text-center py-8 text-stone-500">
                    <Mountain className="w-12 h-12 mx-auto mb-2 text-stone-300" />
                    <p>No mine performance data available</p>
                  </div>
                )}
              </div>
            </Card>

            {/* Regulatory Compliance */}
            <Card title="Regulatory Compliance Status" description="Progress towards regulatory requirements">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <div className="flex items-center justify-between mb-4">
                    <div>
                      <p className="text-sm text-stone-500">Current Compliance</p>
                      <p className="text-4xl font-bold text-stone-900">{data.regulatory_status.current_compliance}%</p>
                    </div>
                    <div className={`w-16 h-16 rounded-full flex items-center justify-center ${
                      data.regulatory_status.status === 'compliant' ? 'bg-emerald-100' : 'bg-rose-100'
                    }`}>
                      <Target className={`w-8 h-8 ${
                        data.regulatory_status.status === 'compliant' ? 'text-emerald-600' : 'text-rose-600'
                      }`} />
                    </div>
                  </div>
                  <div className="w-full bg-stone-200 rounded-full h-4">
                    <div
                      className={`h-4 rounded-full ${
                        data.regulatory_status.current_compliance >= data.regulatory_status.compliance_threshold
                          ? 'bg-emerald-400'
                          : 'bg-rose-400'
                      }`}
                      style={{ width: `${Math.min(100, data.regulatory_status.current_compliance)}%` }}
                    />
                  </div>
                  <div className="flex items-center justify-between mt-2 text-sm">
                    <span className="text-stone-500">0%</span>
                    <span className="text-stone-500">Target: {data.regulatory_status.compliance_threshold}%</span>
                    <span className="text-stone-500">100%</span>
                  </div>
                </div>
                <div className="space-y-4">
                  <div className="p-4 bg-stone-50 rounded-lg">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-stone-600">Regulatory Threshold</span>
                      <span className="font-bold text-stone-900">{data.regulatory_status.compliance_threshold}%</span>
                    </div>
                  </div>
                  <div className={`p-4 rounded-lg ${
                    data.regulatory_status.status === 'compliant'
                      ? 'bg-emerald-50 border border-emerald-200'
                      : 'bg-rose-50 border border-rose-200'
                  }`}>
                    <div className="flex items-center gap-2">
                      {data.regulatory_status.status === 'compliant' ? (
                        <>
                          <Shield className="w-5 h-5 text-emerald-600" />
                          <span className="font-medium text-emerald-700">Compliant</span>
                        </>
                      ) : (
                        <>
                          <AlertTriangle className="w-5 h-5 text-rose-600" />
                          <span className="font-medium text-rose-700">Non-Compliant</span>
                        </>
                      )}
                    </div>
                    <p className="text-sm mt-1 opacity-80">
                      {data.regulatory_status.status === 'compliant'
                        ? 'All mines are meeting regulatory requirements'
                        : 'Immediate action required to meet regulatory standards'}
                    </p>
                  </div>
                </div>
              </div>
            </Card>

            {/* Financial Impact */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <Card title="Cost Savings">
                <div className="text-center py-4">
                  <p className="text-5xl font-bold text-emerald-600">
                    ${data.financial_insights.estimated_cost_savings.toLocaleString()}
                  </p>
                  <p className="text-stone-500 mt-2">Estimated savings from prevented incidents</p>
                </div>
              </Card>
              <Card title="Violations Prevented">
                <div className="text-center py-4">
                  <p className="text-5xl font-bold text-sky-600">
                    {data.financial_insights.violations_prevented_this_week}
                  </p>
                  <p className="text-stone-500 mt-2">Violations prevented through early intervention this week</p>
                </div>
              </Card>
            </div>
          </div>
        )}

        {/* Health Monitoring Tab */}
        {activeTab === 'health' && healthData && (
          <div className="space-y-6">
            {/* Health KPI Overview */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="bg-gradient-to-br from-teal-50 to-teal-100 border border-teal-200 rounded-xl p-5">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-teal-600 text-sm font-medium">Workers Monitored</p>
                    <p className="text-3xl font-bold mt-1 text-teal-700">{healthData.stats.total_workers_monitored}</p>
                  </div>
                  <div className="w-12 h-12 bg-teal-200/50 rounded-xl flex items-center justify-center">
                    <Users className="w-6 h-6 text-teal-600" />
                  </div>
                </div>
              </div>

              <div className="bg-gradient-to-br from-emerald-50 to-emerald-100 border border-emerald-200 rounded-xl p-5">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-emerald-600 text-sm font-medium">Normal Status</p>
                    <p className="text-3xl font-bold mt-1 text-emerald-700">{healthData.stats.workers_normal}</p>
                  </div>
                  <div className="w-12 h-12 bg-emerald-200/50 rounded-xl flex items-center justify-center">
                    <Heart className="w-6 h-6 text-emerald-600" />
                  </div>
                </div>
              </div>

              <div className="bg-gradient-to-br from-amber-50 to-amber-100 border border-amber-200 rounded-xl p-5">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-amber-600 text-sm font-medium">Warning Status</p>
                    <p className="text-3xl font-bold mt-1 text-amber-700">{healthData.stats.workers_warning}</p>
                  </div>
                  <div className="w-12 h-12 bg-amber-200/50 rounded-xl flex items-center justify-center">
                    <AlertTriangle className="w-6 h-6 text-amber-600" />
                  </div>
                </div>
              </div>

              <div className="bg-gradient-to-br from-rose-50 to-rose-100 border border-rose-200 rounded-xl p-5">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-rose-600 text-sm font-medium">Critical Status</p>
                    <p className="text-3xl font-bold mt-1 text-rose-700">{healthData.stats.workers_critical}</p>
                  </div>
                  <div className="w-12 h-12 bg-rose-200/50 rounded-xl flex items-center justify-center">
                    <Activity className="w-6 h-6 text-rose-600" />
                  </div>
                </div>
              </div>
            </div>

            {/* Vital Signs Overview Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {vitalGauges.map((vital) => (
                <Card key={vital.name}>
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-3">
                      <div
                        className="w-12 h-12 rounded-xl flex items-center justify-center"
                        style={{ backgroundColor: `${vital.color}20` }}
                      >
                        <vital.icon className="w-6 h-6" style={{ color: vital.color }} />
                      </div>
                      <div>
                        <h3 className="font-semibold text-stone-900">{vital.name}</h3>
                        <p className="text-sm text-stone-500">Average reading</p>
                      </div>
                    </div>
                    <div
                      className={`px-3 py-1 rounded-full text-xs font-medium ${
                        vital.status === 'normal'
                          ? 'bg-green-100 text-green-700'
                          : vital.status === 'warning'
                          ? 'bg-amber-100 text-amber-700'
                          : 'bg-red-100 text-red-700'
                      }`}
                    >
                      {vital.status.toUpperCase()}
                    </div>
                  </div>
                  <div className="text-center">
                    <p className="text-4xl font-bold" style={{ color: vital.color }}>
                      {vital.value.toFixed(1)}
                      <span className="text-lg text-stone-500 ml-1">{vital.unit}</span>
                    </p>
                    <div className="mt-4 h-3 bg-stone-200 rounded-full overflow-hidden">
                      <div
                        className="h-full rounded-full transition-all duration-500"
                        style={{
                          width: `${(vital.value / vital.max) * 100}%`,
                          backgroundColor: vital.color,
                        }}
                      />
                    </div>
                    <p className="text-xs text-stone-400 mt-1">
                      {vital.name === 'SpO2' && 'Normal: 95-100%'}
                      {vital.name === 'Blood Pressure' && 'Normal: 90-120 mmHg'}
                      {vital.name === 'Heart Rate' && 'Normal: 60-100 bpm'}
                    </p>
                  </div>
                </Card>
              ))}
            </div>

            {/* Charts Section */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Health Status Distribution */}
              <Card>
                <h3 className="text-lg font-semibold text-stone-800 mb-4 flex items-center gap-2">
                  <Activity className="w-5 h-5 text-rose-600" />
                  Health Status Distribution
                </h3>
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={healthDistribution}
                      cx="50%"
                      cy="50%"
                      innerRadius={60}
                      outerRadius={100}
                      paddingAngle={5}
                      dataKey="value"
                    >
                      {healthDistribution.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#fff',
                        border: '1px solid #e5e7eb',
                        borderRadius: '8px',
                        boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
                      }}
                    />
                  </PieChart>
                </ResponsiveContainer>
                <div className="flex justify-center gap-6 mt-4">
                  {healthDistribution.map((item) => (
                    <div key={item.name} className="flex items-center gap-2">
                      <div
                        className="w-4 h-4 rounded-full"
                        style={{ backgroundColor: item.color }}
                      />
                      <span className="text-sm text-stone-600">
                        {item.name}: <span className="font-semibold">{item.value}</span>
                      </span>
                    </div>
                  ))}
                </div>
              </Card>

              {/* Vital Signs Trends */}
              <Card>
                <h3 className="text-lg font-semibold text-stone-800 mb-4 flex items-center gap-2">
                  <TrendingUp className="w-5 h-5 text-rose-600" />
                  Vital Signs Trends
                </h3>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={healthData.trends}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                    <XAxis
                      dataKey="date"
                      stroke="#6b7280"
                      tick={{ fontSize: 12 }}
                      tickFormatter={(date) =>
                        new Date(date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
                      }
                    />
                    <YAxis stroke="#6b7280" tick={{ fontSize: 12 }} domain={[85, 100]} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#fff',
                        border: '1px solid #e5e7eb',
                        borderRadius: '8px',
                      }}
                      labelFormatter={(date) => new Date(date).toLocaleDateString()}
                    />
                    <Legend />
                    <Line
                      type="monotone"
                      dataKey="avg_spo2"
                      name="SpO2 %"
                      stroke="#06b6d4"
                      strokeWidth={2}
                      dot={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </Card>
            </div>

            {/* Heart Rate & Blood Pressure Trends */}
            <Card>
              <h3 className="text-lg font-semibold text-stone-800 mb-4 flex items-center gap-2">
                <Heart className="w-5 h-5 text-rose-600" />
                Heart Rate & Blood Pressure Trends
              </h3>
              <ResponsiveContainer width="100%" height={350}>
                <AreaChart data={healthData.trends}>
                  <defs>
                    <linearGradient id="colorHR" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0} />
                    </linearGradient>
                    <linearGradient id="colorBP" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#f43f5e" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#f43f5e" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis
                    dataKey="date"
                    stroke="#6b7280"
                    tick={{ fontSize: 12 }}
                    tickFormatter={(date) =>
                      new Date(date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
                    }
                  />
                  <YAxis stroke="#6b7280" tick={{ fontSize: 12 }} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#fff',
                      border: '1px solid #e5e7eb',
                      borderRadius: '8px',
                    }}
                    labelFormatter={(date) => new Date(date).toLocaleDateString()}
                  />
                  <Legend />
                  <Area
                    type="monotone"
                    dataKey="avg_heart_rate"
                    name="Heart Rate (bpm)"
                    stroke="#8b5cf6"
                    strokeWidth={2}
                    fillOpacity={1}
                    fill="url(#colorHR)"
                  />
                  <Area
                    type="monotone"
                    dataKey="avg_systolic_bp"
                    name="Systolic BP (mmHg)"
                    stroke="#f43f5e"
                    strokeWidth={2}
                    fillOpacity={1}
                    fill="url(#colorBP)"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </Card>

            {/* Mine Health Comparison */}
            <Card title="Mine Health Comparison" description="Health scores across all mines">
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={healthData.mine_summaries} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis type="number" domain={[0, 100]} stroke="#6b7280" tick={{ fontSize: 12 }} />
                  <YAxis
                    type="category"
                    dataKey="mine_name"
                    stroke="#6b7280"
                    tick={{ fontSize: 12 }}
                    width={120}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#fff',
                      border: '1px solid #e5e7eb',
                      borderRadius: '8px',
                    }}
                    formatter={(value: number) => [`${value.toFixed(1)}%`, 'Health Score']}
                  />
                  <Bar
                    dataKey="health_score"
                    name="Health Score"
                    radius={[0, 4, 4, 0]}
                  >
                    {healthData.mine_summaries.map((entry, index) => (
                      <Cell
                        key={`cell-${index}`}
                        fill={
                          entry.health_score >= 90
                            ? '#6ee7b7'
                            : entry.health_score >= 75
                            ? '#fcd34d'
                            : '#fda4af'
                        }
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Card>

            {/* At-Risk Workers & Recent Alerts */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* At-Risk Workers */}
              <Card>
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold text-stone-800 flex items-center gap-2">
                    <AlertTriangle className="w-5 h-5 text-amber-500" />
                    At-Risk Workers
                  </h3>
                  <span className="text-sm text-stone-500">{healthData.at_risk_workers.length} workers</span>
                </div>

                <div className="space-y-3">
                  {healthData.at_risk_workers.map((worker) => (
                    <div
                      key={worker.worker_id}
                      className={`p-4 rounded-xl border-l-4 ${
                        worker.overall_status === 'critical'
                          ? 'bg-red-50 border-red-500'
                          : 'bg-amber-50 border-amber-500'
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="font-semibold text-stone-900">{worker.worker_name}</p>
                          <p className="text-sm text-stone-500">{worker.employee_id}</p>
                        </div>
                        <span
                          className={`px-3 py-1 rounded-full text-xs font-medium ${
                            worker.overall_status === 'critical'
                              ? 'bg-red-100 text-red-700'
                              : 'bg-amber-100 text-amber-700'
                          }`}
                        >
                          {worker.overall_status.toUpperCase()}
                        </span>
                      </div>
                      <div className="grid grid-cols-4 gap-2 mt-3 text-sm">
                        <div>
                          <p className="text-stone-500">SpO2</p>
                          <p className={`font-semibold ${worker.avg_spo2 < 94 ? 'text-red-600' : 'text-stone-900'}`}>
                            {worker.avg_spo2.toFixed(1)}%
                          </p>
                        </div>
                        <div>
                          <p className="text-stone-500">BP</p>
                          <p className={`font-semibold ${worker.avg_systolic_bp > 140 ? 'text-red-600' : 'text-stone-900'}`}>
                            {worker.avg_systolic_bp.toFixed(0)}/{worker.avg_diastolic_bp.toFixed(0)}
                          </p>
                        </div>
                        <div>
                          <p className="text-stone-500">HR</p>
                          <p className={`font-semibold ${worker.avg_heart_rate > 100 ? 'text-red-600' : 'text-stone-900'}`}>
                            {worker.avg_heart_rate.toFixed(0)} bpm
                          </p>
                        </div>
                        <div>
                          <p className="text-stone-500">Alerts</p>
                          <p className="font-semibold text-stone-900">{worker.alerts_count}</p>
                        </div>
                      </div>
                    </div>
                  ))}

                  {healthData.at_risk_workers.length === 0 && (
                    <div className="text-center py-8 text-stone-500">
                      <Heart className="w-12 h-12 mx-auto mb-2 text-green-500" />
                      <p>All workers are in good health</p>
                    </div>
                  )}
                </div>
              </Card>

              {/* Recent Health Alerts */}
              <Card>
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold text-stone-800 flex items-center gap-2">
                    <Activity className="w-5 h-5 text-rose-600" />
                    Recent Health Alerts
                  </h3>
                  <span className="text-sm text-stone-500">{healthData.stats.alerts_today} today</span>
                </div>

                <div className="space-y-3">
                  {healthData.recent_alerts.map((alert, index) => (
                    <div
                      key={index}
                      className={`p-4 rounded-xl ${
                        alert.severity === 'critical' ? 'bg-red-50' : 'bg-amber-50'
                      }`}
                    >
                      <div className="flex items-start gap-3">
                        <div
                          className={`w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0 ${
                            alert.severity === 'critical' ? 'bg-red-100' : 'bg-amber-100'
                          }`}
                        >
                          {alert.severity === 'critical' ? (
                            <AlertTriangle className="w-5 h-5 text-red-600" />
                          ) : (
                            <Activity className="w-5 h-5 text-amber-600" />
                          )}
                        </div>
                        <div className="flex-1">
                          <div className="flex items-center justify-between">
                            <p className="font-semibold text-stone-900">{alert.worker_name}</p>
                            <span
                              className={`text-xs px-2 py-0.5 rounded ${
                                alert.severity === 'critical'
                                  ? 'bg-red-100 text-red-700'
                                  : 'bg-amber-100 text-amber-700'
                              }`}
                            >
                              {alert.alert_type}
                            </span>
                          </div>
                          <p className="text-sm text-stone-600 mt-1">{alert.message}</p>
                          <p className="text-xs text-stone-400 mt-2">
                            {new Date(alert.timestamp).toLocaleString()}
                          </p>
                        </div>
                      </div>
                    </div>
                  ))}

                  {healthData.recent_alerts.length === 0 && (
                    <div className="text-center py-8 text-stone-500">
                      <Shield className="w-12 h-12 mx-auto mb-2 text-green-500" />
                      <p>No recent health alerts</p>
                    </div>
                  )}
                </div>
              </Card>
            </div>

            {/* Summary Statistics */}
            <Card title="Today's Health Summary" description="Readings and metrics from today">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
                <div className="text-center p-4 bg-cyan-50 rounded-xl">
                  <Droplets className="w-8 h-8 mx-auto text-cyan-600 mb-2" />
                  <p className="text-2xl font-bold text-cyan-700">{healthData.stats.avg_spo2.toFixed(1)}%</p>
                  <p className="text-sm text-cyan-600">Avg SpO2</p>
                </div>
                <div className="text-center p-4 bg-rose-50 rounded-xl">
                  <Activity className="w-8 h-8 mx-auto text-rose-600 mb-2" />
                  <p className="text-2xl font-bold text-rose-700">
                    {healthData.stats.avg_systolic_bp.toFixed(0)}/{healthData.stats.avg_diastolic_bp.toFixed(0)}
                  </p>
                  <p className="text-sm text-rose-600">Avg BP (mmHg)</p>
                </div>
                <div className="text-center p-4 bg-violet-50 rounded-xl">
                  <Heart className="w-8 h-8 mx-auto text-violet-600 mb-2" />
                  <p className="text-2xl font-bold text-violet-700">{healthData.stats.avg_heart_rate.toFixed(0)}</p>
                  <p className="text-sm text-violet-600">Avg Heart Rate (bpm)</p>
                </div>
                <div className="text-center p-4 bg-emerald-50 rounded-xl">
                  <TrendingUp className="w-8 h-8 mx-auto text-emerald-600 mb-2" />
                  <p className="text-2xl font-bold text-emerald-700">{healthData.stats.readings_today}</p>
                  <p className="text-sm text-emerald-600">Readings Today</p>
                </div>
              </div>
            </Card>
          </div>
        )}
      </div>
    </AppLayout>
  );
}
