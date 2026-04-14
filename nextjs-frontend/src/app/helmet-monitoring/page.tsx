'use client';

import { useEffect, useState, useCallback, useRef } from 'react';
import {
  HardHat,
  Battery,
  BatteryLow,
  BatteryWarning,
  Heart,
  Wind,
  Activity,
  AlertTriangle,
  RefreshCw,
  Users,
  Shield,
  XCircle,
  CheckCircle,
  Volume2,
  VolumeX,
  Clock,
  BarChart3,
  History,
  AlertCircle,
  Zap,
  ThermometerSun,
  Gauge,
  User,
  Siren,
  TrendingUp,
} from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  BarChart,
  Bar,
} from 'recharts';
import AppLayout from '@/components/AppLayout';
import { Card, StatCard } from '@/components/Card';
import { Spinner } from '@/components/Loading';
import { helmetApi, HelmetReading, HelmetStats, HelmetAlert } from '@/lib/api';
import { useAuthStore } from '@/lib/store';

const SEVERITY_COLORS = {
  normal: '#10B981',
  medium: '#F59E0B',
  high: '#F97316',
  critical: '#EF4444',
};

const PIE_COLORS = ['#10B981', '#F59E0B', '#F97316', '#EF4444'];

// Battery level indicator component
function BatteryIndicator({ voltage, isLow }: { voltage: number; isLow: boolean }) {
  const percent = Math.min(100, Math.max(0, ((voltage - 3.0) / (4.2 - 3.0)) * 100));
  const color = isLow ? 'text-red-500' : percent < 30 ? 'text-yellow-500' : 'text-green-500';

  return (
    <div className="flex items-center gap-2">
      {isLow ? (
        <BatteryLow className={`w-5 h-5 ${color}`} />
      ) : percent < 30 ? (
        <BatteryWarning className={`w-5 h-5 ${color}`} />
      ) : (
        <Battery className={`w-5 h-5 ${color}`} />
      )}
      <span className={`font-semibold ${color}`}>{voltage.toFixed(2)}V</span>
      <span className="text-xs text-slate-500">({percent.toFixed(0)}%)</span>
    </div>
  );
}

// Health gauge component for heart rate and SpO2
function HealthGauge({ value, label, unit, min, max, dangerLow, dangerHigh }: {
  value: number;
  label: string;
  unit: string;
  min: number;
  max: number;
  dangerLow?: number;
  dangerHigh?: number;
}) {
  const percent = ((value - min) / (max - min)) * 100;
  const isLow = dangerLow !== undefined && value < dangerLow;
  const isHigh = dangerHigh !== undefined && value > dangerHigh;
  const color = isLow || isHigh ? 'text-red-500' : 'text-green-500';
  const bgColor = isLow || isHigh ? 'bg-red-500' : 'bg-green-500';

  return (
    <div className="flex flex-col gap-1">
      <div className="flex justify-between items-center">
        <span className="text-xs text-slate-500">{label}</span>
        <span className={`font-bold ${color}`}>{value}{unit}</span>
      </div>
      <div className="w-full h-2 bg-slate-200 rounded-full overflow-hidden">
        <div
          className={`h-full ${bgColor} rounded-full transition-all duration-300`}
          style={{ width: `${Math.min(100, Math.max(0, percent))}%` }}
        />
      </div>
    </div>
  );
}

// Worker helmet card component
function WorkerHelmetCard({ reading }: { reading: HelmetReading }) {
  const severityBorder = {
    normal: 'border-l-green-500',
    medium: 'border-l-yellow-500',
    high: 'border-l-orange-500',
    critical: 'border-l-red-500',
  };

  return (
    <div className={`bg-white rounded-xl border-l-4 ${severityBorder[reading.severity]} shadow-sm p-4 hover:shadow-md transition-shadow`}>
      <div className="flex justify-between items-start mb-4">
        <div className="flex items-center gap-3">
          <div className={`w-10 h-10 rounded-full flex items-center justify-center ${
            reading.severity === 'critical' ? 'bg-red-100' :
            reading.severity === 'high' ? 'bg-orange-100' :
            reading.severity === 'medium' ? 'bg-yellow-100' : 'bg-green-100'
          }`}>
            <User className={`w-5 h-5 ${
              reading.severity === 'critical' ? 'text-red-600' :
              reading.severity === 'high' ? 'text-orange-600' :
              reading.severity === 'medium' ? 'text-yellow-600' : 'text-green-600'
            }`} />
          </div>
          <div>
            <h3 className="font-semibold text-slate-800">{reading.worker_name}</h3>
            <p className="text-xs text-slate-500">ID: {reading.worker_id}</p>
          </div>
        </div>
        {reading.sos_active && (
          <div className="flex items-center gap-1 bg-red-100 text-red-700 px-2 py-1 rounded-full animate-pulse">
            <Siren className="w-4 h-4" />
            <span className="text-xs font-bold">SOS</span>
          </div>
        )}
      </div>

      <div className="grid grid-cols-2 gap-4 mb-4">
        <HealthGauge
          value={reading.heart_rate}
          label="Heart Rate"
          unit=" BPM"
          min={40}
          max={150}
          dangerLow={50}
          dangerHigh={120}
        />
        <HealthGauge
          value={reading.spo2}
          label="SpO2"
          unit="%"
          min={85}
          max={100}
          dangerLow={90}
        />
      </div>

      <div className="grid grid-cols-2 gap-3 text-sm">
        <div className="flex items-center gap-2">
          <Wind className="w-4 h-4 text-orange-500" />
          <span className="text-slate-600">CH4:</span>
          <span className={`font-semibold ${reading.methane_ppm > 4000 ? 'text-red-600' : 'text-slate-800'}`}>
            {reading.methane_ppm} PPM
          </span>
        </div>
        <div className="flex items-center gap-2">
          <Gauge className="w-4 h-4 text-blue-500" />
          <span className="text-slate-600">CO:</span>
          <span className="font-semibold text-slate-800">{reading.co_raw}</span>
        </div>
      </div>

      <div className="mt-4 pt-4 border-t border-slate-100 flex justify-between items-center">
        <BatteryIndicator voltage={reading.battery_voltage} isLow={reading.battery_low} />
        <span className="text-xs text-slate-400">
          {new Date(reading.timestamp).toLocaleTimeString()}
        </span>
      </div>
    </div>
  );
}

// Alert card component
function AlertCard({ alert, onAcknowledge, onResolve }: {
  alert: HelmetAlert;
  onAcknowledge: (id: string) => void;
  onResolve: (id: string) => void;
}) {
  const severityColors = {
    medium: { bg: 'bg-yellow-50', border: 'border-yellow-200', text: 'text-yellow-700', icon: 'text-yellow-500' },
    high: { bg: 'bg-orange-50', border: 'border-orange-200', text: 'text-orange-700', icon: 'text-orange-500' },
    critical: { bg: 'bg-red-50', border: 'border-red-200', text: 'text-red-700', icon: 'text-red-500' },
  };
  const colors = severityColors[alert.severity];

  return (
    <div className={`${colors.bg} border ${colors.border} rounded-xl p-4`}>
      <div className="flex justify-between items-start mb-2">
        <div className="flex items-center gap-2">
          <AlertTriangle className={`w-5 h-5 ${colors.icon}`} />
          <span className={`font-semibold ${colors.text} capitalize`}>{alert.severity}</span>
        </div>
        <span className={`text-xs px-2 py-1 rounded-full ${
          alert.status === 'active' ? 'bg-red-100 text-red-700' :
          alert.status === 'acknowledged' ? 'bg-yellow-100 text-yellow-700' :
          'bg-green-100 text-green-700'
        }`}>
          {alert.status}
        </span>
      </div>

      <p className="text-sm text-slate-700 mb-2">{alert.message}</p>

      <div className="flex items-center gap-2 text-xs text-slate-500 mb-3">
        <User className="w-3 h-3" />
        <span>{alert.worker_name}</span>
        <span className="mx-1">|</span>
        <Clock className="w-3 h-3" />
        <span>{new Date(alert.created_at).toLocaleString()}</span>
      </div>

      {alert.status === 'active' && (
        <div className="flex gap-2">
          <button
            onClick={() => onAcknowledge(alert.id)}
            className="flex-1 px-3 py-1.5 text-xs font-medium bg-yellow-500 text-white rounded-lg hover:bg-yellow-600 transition-colors"
          >
            Acknowledge
          </button>
          <button
            onClick={() => onResolve(alert.id)}
            className="flex-1 px-3 py-1.5 text-xs font-medium bg-green-500 text-white rounded-lg hover:bg-green-600 transition-colors"
          >
            Resolve
          </button>
        </div>
      )}
      {alert.status === 'acknowledged' && (
        <button
          onClick={() => onResolve(alert.id)}
          className="w-full px-3 py-1.5 text-xs font-medium bg-green-500 text-white rounded-lg hover:bg-green-600 transition-colors"
        >
          Resolve
        </button>
      )}
    </div>
  );
}

export default function HelmetMonitoringPage() {
  const { getMineId } = useAuthStore();
  const [readings, setReadings] = useState<HelmetReading[]>([]);
  const [stats, setStats] = useState<HelmetStats | null>(null);
  const [alerts, setAlerts] = useState<HelmetAlert[]>([]);
  const [historicalReadings, setHistoricalReadings] = useState<HelmetReading[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [activeTab, setActiveTab] = useState<'overview' | 'workers' | 'alerts' | 'history'>('overview');
  const [audioEnabled, setAudioEnabled] = useState(true);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const lastAlertRef = useRef<string | null>(null);

  const loadData = useCallback(async () => {
    try {
      setError(null);
      const mineId = getMineId() || undefined;

      // Fetch all data in parallel
      const [latestRes, statsRes, alertsRes] = await Promise.all([
        helmetApi.getLatest(mineId).catch(() => ({ readings: [] })),
        helmetApi.getStats({ mine_id: mineId }).catch(() => null),
        helmetApi.getAlerts({ status: 'active', limit: 20 }).catch(() => ({ alerts: [] })),
      ]);

      setReadings(latestRes.readings || []);
      setStats(statsRes);
      setAlerts(alertsRes.alerts || []);

      // Check for critical alerts and play sound
      const criticalAlerts = (alertsRes.alerts || []).filter(a => a.severity === 'critical' && a.status === 'active');
      if (criticalAlerts.length > 0 && audioEnabled) {
        const latestCritical = criticalAlerts[0];
        if (lastAlertRef.current !== latestCritical.id) {
          lastAlertRef.current = latestCritical.id;
          // Play alert sound
          if (audioRef.current) {
            audioRef.current.play().catch(console.error);
          }
        }
      }

    } catch (err) {
      console.error('Error loading helmet data:', err);
      setError('Failed to load helmet data');
    } finally {
      setLoading(false);
    }
  }, [getMineId, audioEnabled]);

  const loadHistory = useCallback(async () => {
    try {
      const res = await helmetApi.getReadings({ limit: 100 });
      setHistoricalReadings(res.readings || []);
    } catch (err) {
      console.error('Error loading history:', err);
    }
  }, []);

  useEffect(() => {
    loadData();

    // Load history when switching to history tab
    if (activeTab === 'history') {
      loadHistory();
    }
  }, [loadData, activeTab, loadHistory]);

  useEffect(() => {
    if (!autoRefresh) return;

    const interval = setInterval(loadData, 3000); // Refresh every 3 seconds
    return () => clearInterval(interval);
  }, [autoRefresh, loadData]);

  const handleAcknowledge = async (alertId: string) => {
    try {
      await helmetApi.acknowledgeAlert(alertId);
      loadData();
    } catch (err) {
      console.error('Error acknowledging alert:', err);
    }
  };

  const handleResolve = async (alertId: string) => {
    try {
      await helmetApi.resolveAlert(alertId);
      loadData();
    } catch (err) {
      console.error('Error resolving alert:', err);
    }
  };

  // Prepare chart data
  const severityData = stats ? [
    { name: 'Normal', value: stats.severity_distribution.normal, color: SEVERITY_COLORS.normal },
    { name: 'Medium', value: stats.severity_distribution.medium, color: SEVERITY_COLORS.medium },
    { name: 'High', value: stats.severity_distribution.high, color: SEVERITY_COLORS.high },
    { name: 'Critical', value: stats.severity_distribution.critical, color: SEVERITY_COLORS.critical },
  ].filter(d => d.value > 0) : [];

  if (loading) {
    return (
      <AppLayout>
        <div className="flex items-center justify-center h-96">
          <Spinner size="lg" />
        </div>
      </AppLayout>
    );
  }

  return (
    <AppLayout>
      {/* Hidden audio element for alerts */}
      <audio ref={audioRef} src="/alert-sound.mp3" preload="auto" />

      <div className="space-y-6">
        {/* Header */}
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold text-slate-800">Smart Helmet Monitoring</h1>
            <p className="text-slate-500 mt-1">Real-time helmet sensor data and worker safety</p>
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={() => setAudioEnabled(!audioEnabled)}
              className={`p-2 rounded-lg border ${audioEnabled ? 'bg-orange-50 border-orange-200 text-orange-600' : 'bg-slate-50 border-slate-200 text-slate-400'}`}
              title={audioEnabled ? 'Disable audio alerts' : 'Enable audio alerts'}
            >
              {audioEnabled ? <Volume2 className="w-5 h-5" /> : <VolumeX className="w-5 h-5" />}
            </button>

            <button
              onClick={() => setAutoRefresh(!autoRefresh)}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg border ${
                autoRefresh ? 'bg-green-50 border-green-200 text-green-700' : 'bg-slate-50 border-slate-200 text-slate-600'
              }`}
            >
              <RefreshCw className={`w-4 h-4 ${autoRefresh ? 'animate-spin' : ''}`} />
              {autoRefresh ? 'Live' : 'Paused'}
            </button>

            <button
              onClick={loadData}
              className="p-2 rounded-lg border border-slate-200 hover:bg-slate-50 text-slate-600"
            >
              <RefreshCw className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Error Banner */}
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-xl p-4 flex items-center gap-3">
            <XCircle className="w-5 h-5 text-red-500" />
            <p className="text-red-700">{error}</p>
          </div>
        )}

        {/* Tab Navigation */}
        <div className="border-b border-slate-200">
          <nav className="flex gap-6">
            {[
              { id: 'overview', label: 'Overview', icon: BarChart3 },
              { id: 'workers', label: 'Workers', icon: Users },
              { id: 'alerts', label: 'Alerts', icon: AlertTriangle },
              { id: 'history', label: 'History', icon: History },
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as typeof activeTab)}
                className={`flex items-center gap-2 px-1 py-3 border-b-2 transition-colors ${
                  activeTab === tab.id
                    ? 'border-orange-500 text-orange-600'
                    : 'border-transparent text-slate-500 hover:text-slate-700'
                }`}
              >
                <tab.icon className="w-4 h-4" />
                {tab.label}
                {tab.id === 'alerts' && alerts.filter(a => a.status === 'active').length > 0 && (
                  <span className="bg-red-500 text-white text-xs px-2 py-0.5 rounded-full">
                    {alerts.filter(a => a.status === 'active').length}
                  </span>
                )}
              </button>
            ))}
          </nav>
        </div>

        {/* Overview Tab */}
        {activeTab === 'overview' && (
          <div className="space-y-6">
            {/* Stats Cards */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <StatCard
                title="Active Helmets"
                value={stats?.active_helmets || readings.length}
                subtitle={`of ${stats?.total_workers || 0} workers`}
                icon={<HardHat className="w-6 h-6" />}
                color="orange"
              />
              <StatCard
                title="Avg Battery"
                value={`${stats?.avg_battery_voltage?.toFixed(2) || '0.00'}V`}
                subtitle={stats?.low_battery_count ? `${stats.low_battery_count} low` : 'All good'}
                icon={<Battery className="w-6 h-6" />}
                color={stats?.low_battery_count ? 'red' : 'green'}
              />
              <StatCard
                title="Avg Heart Rate"
                value={`${stats?.avg_heart_rate || 0}`}
                subtitle="BPM"
                icon={<Heart className="w-6 h-6" />}
                color="red"
              />
              <StatCard
                title="SOS Alerts"
                value={stats?.sos_active_count || 0}
                subtitle={stats?.is_safe ? 'All safe' : 'Needs attention'}
                icon={<Siren className="w-6 h-6" />}
                color={stats?.sos_active_count ? 'red' : 'green'}
              />
            </div>

            {/* Active Danger Zones */}
            <Card
              title="Active Danger Zones"
              icon={<AlertTriangle className="w-5 h-5 text-red-500" />}
              className="border-red-200 bg-gradient-to-r from-red-50 to-orange-50"
            >
              <div className="space-y-3">
                {/* Zone A - Extraction - Active Danger */}
                <div className="p-4 bg-white border-2 border-red-300 rounded-xl animate-pulse">
                  <div className="flex items-start justify-between">
                    <div className="flex items-center gap-3">
                      <div className="w-12 h-12 bg-red-100 rounded-xl flex items-center justify-center">
                        <AlertTriangle className="w-6 h-6 text-red-600" />
                      </div>
                      <div>
                        <div className="flex items-center gap-2">
                          <h3 className="font-semibold text-slate-800">Zone A - Extraction</h3>
                          <span className="px-2 py-0.5 bg-red-600 text-white text-xs font-bold rounded-full animate-pulse">
                            CRITICAL
                          </span>
                        </div>
                        <p className="text-sm text-red-600 font-medium">High Methane Detected</p>
                      </div>
                    </div>
                    <div className="text-right">
                      <p className="text-2xl font-bold text-red-600">15,200</p>
                      <p className="text-xs text-slate-500">PPM</p>
                    </div>
                  </div>
                  <div className="mt-3 pt-3 border-t border-red-100 grid grid-cols-3 gap-4 text-sm">
                    <div>
                      <span className="text-slate-500">Threshold:</span>
                      <span className="ml-2 font-medium text-slate-700">5,000 PPM</span>
                    </div>
                    <div>
                      <span className="text-slate-500">Workers in Zone:</span>
                      <span className="ml-2 font-medium text-red-600">8</span>
                    </div>
                    <div>
                      <span className="text-slate-500">Status:</span>
                      <span className="ml-2 font-medium text-orange-600">Evacuating</span>
                    </div>
                  </div>
                </div>

                {/* Safe Zones */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  <div className="p-3 bg-white border border-green-200 rounded-xl">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <div className="w-8 h-8 bg-green-100 rounded-lg flex items-center justify-center">
                          <Shield className="w-4 h-4 text-green-600" />
                        </div>
                        <div>
                          <h4 className="font-medium text-slate-700 text-sm">Zone B - Storage</h4>
                          <p className="text-xs text-green-600">Safe - 320 PPM</p>
                        </div>
                      </div>
                      <span className="px-2 py-0.5 bg-green-100 text-green-700 text-xs font-medium rounded-full">
                        SAFE
                      </span>
                    </div>
                  </div>
                  <div className="p-3 bg-white border border-green-200 rounded-xl">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <div className="w-8 h-8 bg-green-100 rounded-lg flex items-center justify-center">
                          <Shield className="w-4 h-4 text-green-600" />
                        </div>
                        <div>
                          <h4 className="font-medium text-slate-700 text-sm">Zone C - Ventilation</h4>
                          <p className="text-xs text-green-600">Safe - 180 PPM</p>
                        </div>
                      </div>
                      <span className="px-2 py-0.5 bg-green-100 text-green-700 text-xs font-medium rounded-full">
                        SAFE
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </Card>

            {/* Charts Row */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Sensor Readings Table */}
              <Card title="Live Sensor Readings" icon={<Activity className="w-5 h-5 text-orange-500" />}>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="text-left text-slate-500 border-b">
                        <th className="pb-2 font-medium">Worker</th>
                        <th className="pb-2 font-medium">CH4 (PPM)</th>
                        <th className="pb-2 font-medium">HR (BPM)</th>
                        <th className="pb-2 font-medium">SpO2</th>
                        <th className="pb-2 font-medium">Battery</th>
                        <th className="pb-2 font-medium">Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      {readings.length === 0 ? (
                        <tr>
                          <td colSpan={6} className="py-8 text-center text-slate-400">
                            No active helmets detected
                          </td>
                        </tr>
                      ) : (
                        readings.map((reading, idx) => (
                          <tr key={idx} className="border-b border-slate-100 last:border-0">
                            <td className="py-3 font-medium text-slate-700">{reading.worker_name}</td>
                            <td className={`py-3 ${reading.methane_ppm > 4000 ? 'text-red-600 font-bold' : ''}`}>
                              {reading.methane_ppm}
                            </td>
                            <td className="py-3">{reading.heart_rate}</td>
                            <td className="py-3">{reading.spo2}%</td>
                            <td className="py-3">
                              <span className={reading.battery_low ? 'text-red-600' : ''}>
                                {reading.battery_voltage.toFixed(2)}V
                              </span>
                            </td>
                            <td className="py-3">
                              <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium ${
                                reading.severity === 'critical' ? 'bg-red-100 text-red-700' :
                                reading.severity === 'high' ? 'bg-orange-100 text-orange-700' :
                                reading.severity === 'medium' ? 'bg-yellow-100 text-yellow-700' :
                                'bg-green-100 text-green-700'
                              }`}>
                                {reading.sos_active && <Siren className="w-3 h-3" />}
                                {reading.severity}
                              </span>
                            </td>
                          </tr>
                        ))
                      )}
                    </tbody>
                  </table>
                </div>
              </Card>

              {/* Severity Distribution */}
              <Card title="Alert Distribution" icon={<AlertCircle className="w-5 h-5 text-orange-500" />}>
                {severityData.length > 0 ? (
                  <div className="flex items-center justify-center h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={severityData}
                          cx="50%"
                          cy="50%"
                          innerRadius={60}
                          outerRadius={80}
                          paddingAngle={5}
                          dataKey="value"
                        >
                          {severityData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.color} />
                          ))}
                        </Pie>
                        <Tooltip />
                        <Legend />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                ) : (
                  <div className="flex flex-col items-center justify-center h-64 text-slate-400">
                    <CheckCircle className="w-12 h-12 mb-2 text-green-400" />
                    <p>No alerts recorded</p>
                  </div>
                )}
              </Card>
            </div>

            {/* Motion Data */}
            {readings.length > 0 && (
              <Card title="Motion & Orientation Data" icon={<TrendingUp className="w-5 h-5 text-orange-500" />}>
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
                  {readings[0] && (
                    <>
                      <div className="text-center p-3 bg-slate-50 rounded-lg">
                        <p className="text-xs text-slate-500 mb-1">Accel X</p>
                        <p className="text-lg font-bold text-slate-700">{readings[0].accel_x}</p>
                      </div>
                      <div className="text-center p-3 bg-slate-50 rounded-lg">
                        <p className="text-xs text-slate-500 mb-1">Accel Y</p>
                        <p className="text-lg font-bold text-slate-700">{readings[0].accel_y}</p>
                      </div>
                      <div className="text-center p-3 bg-slate-50 rounded-lg">
                        <p className="text-xs text-slate-500 mb-1">Accel Z</p>
                        <p className="text-lg font-bold text-slate-700">{readings[0].accel_z}</p>
                      </div>
                      <div className="text-center p-3 bg-slate-50 rounded-lg">
                        <p className="text-xs text-slate-500 mb-1">Roll</p>
                        <p className="text-lg font-bold text-slate-700">{readings[0].roll}°</p>
                      </div>
                      <div className="text-center p-3 bg-slate-50 rounded-lg">
                        <p className="text-xs text-slate-500 mb-1">Pitch</p>
                        <p className="text-lg font-bold text-slate-700">{readings[0].pitch}°</p>
                      </div>
                      <div className="text-center p-3 bg-slate-50 rounded-lg">
                        <p className="text-xs text-slate-500 mb-1">Yaw</p>
                        <p className="text-lg font-bold text-slate-700">{readings[0].yaw}°</p>
                      </div>
                    </>
                  )}
                </div>
              </Card>
            )}
          </div>
        )}

        {/* Workers Tab */}
        {activeTab === 'workers' && (
          <div className="space-y-6">
            {readings.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-16 text-slate-400">
                <HardHat className="w-16 h-16 mb-4" />
                <p className="text-lg">No active helmet connections</p>
                <p className="text-sm">Waiting for helmet data...</p>
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {readings.map((reading, idx) => (
                  <WorkerHelmetCard key={idx} reading={reading} />
                ))}
              </div>
            )}
          </div>
        )}

        {/* Alerts Tab */}
        {activeTab === 'alerts' && (
          <div className="space-y-6">
            {alerts.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-16 text-slate-400">
                <CheckCircle className="w-16 h-16 mb-4 text-green-400" />
                <p className="text-lg">No active alerts</p>
                <p className="text-sm">All workers are within safe parameters</p>
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {alerts.map((alert) => (
                  <AlertCard
                    key={alert.id}
                    alert={alert}
                    onAcknowledge={handleAcknowledge}
                    onResolve={handleResolve}
                  />
                ))}
              </div>
            )}
          </div>
        )}

        {/* History Tab */}
        {activeTab === 'history' && (
          <div className="space-y-6">
            <Card title="Historical Readings (Threshold Breaches)" icon={<History className="w-5 h-5 text-orange-500" />}>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left text-slate-500 border-b">
                      <th className="pb-2 font-medium">Time</th>
                      <th className="pb-2 font-medium">Worker</th>
                      <th className="pb-2 font-medium">CH4 (PPM)</th>
                      <th className="pb-2 font-medium">HR (BPM)</th>
                      <th className="pb-2 font-medium">SpO2</th>
                      <th className="pb-2 font-medium">Battery</th>
                      <th className="pb-2 font-medium">SOS</th>
                      <th className="pb-2 font-medium">Severity</th>
                    </tr>
                  </thead>
                  <tbody>
                    {historicalReadings.length === 0 ? (
                      <tr>
                        <td colSpan={8} className="py-8 text-center text-slate-400">
                          No historical data available (only threshold breaches are stored)
                        </td>
                      </tr>
                    ) : (
                      historicalReadings.map((reading, idx) => (
                        <tr key={idx} className="border-b border-slate-100 last:border-0">
                          <td className="py-3 text-slate-500">
                            {new Date(reading.timestamp).toLocaleString()}
                          </td>
                          <td className="py-3 font-medium text-slate-700">{reading.worker_name}</td>
                          <td className={`py-3 ${reading.methane_ppm > 4000 ? 'text-red-600 font-bold' : ''}`}>
                            {reading.methane_ppm}
                          </td>
                          <td className="py-3">{reading.heart_rate}</td>
                          <td className="py-3">{reading.spo2}%</td>
                          <td className="py-3">
                            <span className={reading.battery_low ? 'text-red-600' : ''}>
                              {reading.battery_voltage?.toFixed(2)}V
                            </span>
                          </td>
                          <td className="py-3">
                            {reading.sos_active ? (
                              <span className="text-red-600 font-bold">YES</span>
                            ) : (
                              <span className="text-slate-400">No</span>
                            )}
                          </td>
                          <td className="py-3">
                            <span className={`inline-flex px-2 py-0.5 rounded-full text-xs font-medium ${
                              reading.severity === 'critical' ? 'bg-red-100 text-red-700' :
                              reading.severity === 'high' ? 'bg-orange-100 text-orange-700' :
                              reading.severity === 'medium' ? 'bg-yellow-100 text-yellow-700' :
                              'bg-green-100 text-green-700'
                            }`}>
                              {reading.severity}
                            </span>
                          </td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
            </Card>
          </div>
        )}
      </div>
    </AppLayout>
  );
}
