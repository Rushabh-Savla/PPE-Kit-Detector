'use client';

import { useEffect, useState } from 'react';
import {
  TrendingUp,
  AlertTriangle,
  Users,
  ShieldAlert,
  Calendar,
  Target,
  Activity,
  Brain,
  RefreshCw,
} from 'lucide-react';
import {
  AreaChart,
  Area,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  LineChart,
  Line,
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
import { predictionApi } from '@/lib/api';
import type {
  AtRiskWorkersResponse,
  PredictionTrendsResponse,
  RiskCategory,
} from '@/types';

const RISK_COLORS = {
  low: '#10b981', // green
  medium: '#f59e0b', // amber
  high: '#f97316', // orange
  critical: '#ef4444', // red
};

export default function PredictiveAnalysisPage() {
  const [atRiskWorkers, setAtRiskWorkers] = useState<AtRiskWorkersResponse | null>(null);
  const [trends, setTrends] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedCategory, setSelectedCategory] = useState<RiskCategory | undefined>();
  const [generating, setGenerating] = useState(false);

  useEffect(() => {
    loadData();
  }, [selectedCategory]);

  const loadData = async () => {
    try {
      setLoading(true);
      setError(null);

      // Load at-risk workers
      const workersData = await predictionApi.getAtRiskWorkers({
        limit: 50,
      });
      setAtRiskWorkers(workersData);

      // Load trends
      const trendsData = await predictionApi.getTrends({ days_back: 30 });

      // Transform trends for charting
      const trendArray = Object.entries(trendsData.trends).map(([date, categories]) => {
        const dataPoint: any = { date };
        let total = 0;
        Object.entries(categories).forEach(([category, data]: [string, any]) => {
          dataPoint[category] = data.count;
          total += data.count;
        });
        dataPoint.total = total;
        return dataPoint;
      }).sort((a, b) => a.date.localeCompare(b.date));

      setTrends(trendArray);
    } catch (err) {
      console.error('Failed to load prediction data:', err);
      setError('Failed to load prediction data');
    } finally {
      setLoading(false);
    }
  };

  const handleGenerateAll = async () => {
    try {
      setGenerating(true);
      // Refresh predictions data
      await loadData();
    } catch (err) {
      console.error('Failed to refresh predictions:', err);
      alert('Failed to refresh predictions');
    } finally {
      setGenerating(false);
    }
  };

  if (loading) {
    return (
      <AppLayout>
        <div className="flex items-center justify-center h-screen">
          <Spinner size="lg" />
        </div>
      </AppLayout>
    );
  }

  if (error || !atRiskWorkers) {
    return (
      <AppLayout>
        <div className="text-center py-12">
          <p className="text-red-500">{error || 'Failed to load data'}</p>
          <button onClick={loadData} className="mt-4 px-4 py-2 bg-blue-600 text-white rounded-lg">
            Retry
          </button>
        </div>
      </AppLayout>
    );
  }

  // Prepare risk distribution data for pie chart
  const riskDistribution = [
    {
      name: 'Low Risk',
      value: atRiskWorkers.by_category.low || 0,
      color: RISK_COLORS.low,
    },
    {
      name: 'Medium Risk',
      value: atRiskWorkers.by_category.medium || 0,
      color: RISK_COLORS.medium,
    },
    {
      name: 'High Risk',
      value: atRiskWorkers.by_category.high || 0,
      color: RISK_COLORS.high,
    },
    {
      name: 'Critical Risk',
      value: atRiskWorkers.by_category.critical || 0,
      color: RISK_COLORS.critical,
    },
  ];

  const totalWorkers = riskDistribution.reduce((sum, item) => sum + item.value, 0);
  const atRiskCount = atRiskWorkers.total_at_risk;
  const criticalCount = atRiskWorkers.by_category.critical || 0;
  const interventionNeeded = atRiskWorkers.workers.filter((w) => w.requires_intervention).length;

  return (
    <AppLayout>
      <div className="space-y-6 pb-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-stone-800 flex items-center gap-3">
            <Brain className="w-8 h-8 text-purple-600" />
            Predictive Analysis
          </h1>
          <p className="text-stone-500 mt-1">AI-powered worker risk prediction & insights</p>
        </div>
        <button
          onClick={handleGenerateAll}
          disabled={generating}
          className="flex items-center gap-2 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:opacity-50 transition-colors"
        >
          <RefreshCw className={`w-4 h-4 ${generating ? 'animate-spin' : ''}`} />
          {generating ? 'Generating...' : 'Refresh Predictions'}
        </button>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard
          title="Total Workers"
          value={totalWorkers}
          icon={<Users size={24} />}
          color="blue"
        />
        <StatCard
          title="At Risk"
          value={atRiskCount}
          subtitle={`${((atRiskCount / totalWorkers) * 100).toFixed(1)}% of total`}
          icon={<AlertTriangle size={24} />}
          color="yellow"
        />
        <StatCard
          title="Critical Risk"
          value={criticalCount}
          subtitle="Immediate action needed"
          icon={<ShieldAlert size={24} />}
          color="red"
        />
        <StatCard
          title="Intervention Needed"
          value={interventionNeeded}
          subtitle="Requires follow-up"
          icon={<Target size={24} />}
          color="orange"
        />
      </div>

      {/* Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Risk Distribution Pie Chart */}
        <Card>
          <h3 className="text-lg font-semibold text-stone-800 mb-4 flex items-center gap-2">
            <Activity className="w-5 h-5 text-purple-600" />
            Risk Distribution
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={riskDistribution}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({name, percent}) => `${name}: ${((percent ?? 0) * 100).toFixed(0)}%`}
                outerRadius={100}
                fill="#8884d8"
                dataKey="value"
              >
                {riskDistribution.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
          <div className="grid grid-cols-2 gap-2 mt-4">
            {riskDistribution.map((item) => (
              <div key={item.name} className="flex items-center gap-2">
                <div
                  className="w-3 h-3 rounded-full"
                  style={{ backgroundColor: item.color }}
                />
                <span className="text-sm text-stone-600">
                  {item.name}: {item.value}
                </span>
              </div>
            ))}
          </div>
        </Card>

        {/* Risk Trends Over Time */}
        <Card>
          <h3 className="text-lg font-semibold text-stone-800 mb-4 flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-purple-600" />
            Risk Trends (30 Days)
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={trends}>
              <defs>
                <linearGradient id="colorCritical" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={RISK_COLORS.critical} stopOpacity={0.8}/>
                  <stop offset="95%" stopColor={RISK_COLORS.critical} stopOpacity={0}/>
                </linearGradient>
                <linearGradient id="colorHigh" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={RISK_COLORS.high} stopOpacity={0.8}/>
                  <stop offset="95%" stopColor={RISK_COLORS.high} stopOpacity={0}/>
                </linearGradient>
                <linearGradient id="colorMedium" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={RISK_COLORS.medium} stopOpacity={0.8}/>
                  <stop offset="95%" stopColor={RISK_COLORS.medium} stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis
                dataKey="date"
                stroke="#6b7280"
                tick={{ fontSize: 12 }}
                tickFormatter={(date) => new Date(date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
              />
              <YAxis stroke="#6b7280" tick={{ fontSize: 12 }} />
              <Tooltip
                contentStyle={{ backgroundColor: '#fff', border: '1px solid #e5e7eb', borderRadius: '8px' }}
                labelFormatter={(date) => new Date(date).toLocaleDateString()}
              />
              <Legend />
              <Area
                type="monotone"
                dataKey="critical"
                stackId="1"
                stroke={RISK_COLORS.critical}
                fillOpacity={1}
                fill="url(#colorCritical)"
                name="Critical"
              />
              <Area
                type="monotone"
                dataKey="high"
                stackId="1"
                stroke={RISK_COLORS.high}
                fillOpacity={1}
                fill="url(#colorHigh)"
                name="High"
              />
              <Area
                type="monotone"
                dataKey="medium"
                stackId="1"
                stroke={RISK_COLORS.medium}
                fillOpacity={1}
                fill="url(#colorMedium)"
                name="Medium"
              />
            </AreaChart>
          </ResponsiveContainer>
        </Card>
      </div>

      {/* Category Filter */}
      <div className="flex gap-2">
        <button
          onClick={() => setSelectedCategory(undefined)}
          className={`px-4 py-2 rounded-lg font-medium transition-colors ${
            !selectedCategory
              ? 'bg-purple-600 text-white'
              : 'bg-stone-100 text-stone-600 hover:bg-stone-200'
          }`}
        >
          All At-Risk
        </button>
        <button
          onClick={() => setSelectedCategory('medium')}
          className={`px-4 py-2 rounded-lg font-medium transition-colors ${
            selectedCategory === 'medium'
              ? 'bg-amber-500 text-white'
              : 'bg-stone-100 text-stone-600 hover:bg-stone-200'
          }`}
        >
          Medium
        </button>
        <button
          onClick={() => setSelectedCategory('high')}
          className={`px-4 py-2 rounded-lg font-medium transition-colors ${
            selectedCategory === 'high'
              ? 'bg-orange-500 text-white'
              : 'bg-stone-100 text-stone-600 hover:bg-stone-200'
          }`}
        >
          High
        </button>
        <button
          onClick={() => setSelectedCategory('critical')}
          className={`px-4 py-2 rounded-lg font-medium transition-colors ${
            selectedCategory === 'critical'
              ? 'bg-red-600 text-white'
              : 'bg-stone-100 text-stone-600 hover:bg-stone-200'
          }`}
        >
          Critical
        </button>
      </div>

      {/* At-Risk Workers Table */}
      <Card>
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-stone-800">
            At-Risk Workers
            {selectedCategory && ` (${selectedCategory.charAt(0).toUpperCase() + selectedCategory.slice(1)})`}
          </h3>
          <span className="text-sm text-stone-500">{atRiskWorkers.workers.length} workers</span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-stone-50 border-b-2 border-stone-200">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-medium text-stone-500 uppercase tracking-wider">
                  Worker
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-stone-500 uppercase tracking-wider">
                  Risk Score
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-stone-500 uppercase tracking-wider">
                  Category
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-stone-500 uppercase tracking-wider">
                  Main Issue
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-stone-500 uppercase tracking-wider">
                  Action
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-stone-200">
              {atRiskWorkers.workers.map((worker) => {
                const riskColor = RISK_COLORS[worker.risk_category];

                return (
                  <tr key={worker.worker_id} className="hover:bg-stone-50 transition-colors">
                    <td className="px-4 py-4 whitespace-nowrap">
                      <div>
                        <div className="text-sm font-medium text-stone-900">{worker.worker_name}</div>
                        <div className="text-sm text-stone-500">{worker.employee_id}</div>
                      </div>
                    </td>
                    <td className="px-4 py-4 whitespace-nowrap">
                      <div className="flex items-center gap-2">
                        <div className="flex-1 bg-stone-200 rounded-full h-2 max-w-[100px]">
                          <div
                            className="h-2 rounded-full transition-all"
                            style={{
                              width: `${worker.risk_score}%`,
                              backgroundColor: riskColor,
                            }}
                          />
                        </div>
                        <span className="text-sm font-semibold" style={{ color: riskColor }}>
                          {worker.risk_score.toFixed(0)}
                        </span>
                      </div>
                    </td>
                    <td className="px-4 py-4 whitespace-nowrap">
                      <span
                        className="px-3 py-1 inline-flex text-xs leading-5 font-semibold rounded-full text-white"
                        style={{ backgroundColor: riskColor }}
                      >
                        {worker.risk_category.toUpperCase()}
                      </span>
                    </td>
                    <td className="px-4 py-4">
                      <p className="text-sm text-stone-600 max-w-md">{worker.main_issue}</p>
                    </td>
                    <td className="px-4 py-4 whitespace-nowrap text-sm">
                      {worker.requires_intervention ? (
                        <span className="flex items-center gap-1 text-red-600 font-medium">
                          <AlertTriangle className="w-4 h-4" />
                          Intervention Needed
                        </span>
                      ) : (
                        <span className="text-stone-500">Monitor</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>

          {atRiskWorkers.workers.length === 0 && (
            <div className="text-center py-12 text-stone-500">
              <ShieldAlert className="w-16 h-16 mx-auto mb-4 text-stone-300" />
              <p>No at-risk workers in this category</p>
            </div>
          )}
        </div>
      </Card>
      </div>
    </AppLayout>
  );
}
