'use client';

import { useEffect, useState, useCallback, useRef } from 'react';
import {
  Flame,
  Wind,
  Gauge,
  TrendingUp,
  AlertTriangle,
  RefreshCw,
  Activity,
  MapPin,
  ThermometerSun,
  Droplets,
  Mountain,
  Shield,
  XCircle,
  CheckCircle,
  Volume2,
  VolumeX,
  Clock,
  Map,
  BarChart3,
  History,
  AlertCircle,
  Zap,
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
  AreaChart,
  Area,
  BarChart,
  Bar,
} from 'recharts';
import AppLayout from '@/components/AppLayout';
import { Card, StatCard } from '@/components/Card';
import { Spinner } from '@/components/Loading';
import { gasSensorApi, apiClient } from '@/lib/api';
import { useAuthStore } from '@/lib/store';

interface GasReading {
  id: string;
  mine_id: string;
  mine_name: string;
  zone_id?: string;
  zone_name?: string;
  gate_id?: string;
  gate_name?: string;
  sensor_id?: string;
  methane_ppm: number;
  co_ppm: number;
  pressure_hpa: number;
  altitude_m: number;
  temperature_c?: number;
  humidity?: number;
  severity: 'normal' | 'medium' | 'high' | 'critical';
  timestamp: string;
}

interface GasStats {
  total_readings: number;
  time_range_hours: number;
  methane: {
    current: number;
    avg: number;
    max: number;
    min: number;
  };
  co: {
    current: number;
    avg: number;
    max: number;
    min: number;
  };
  severity_distribution: {
    normal: number;
    medium: number;
    high: number;
    critical: number;
  };
  total_alerts: number;
  is_safe: boolean;
}

interface DangerZone {
  id: string;
  zone_id: string;
  zone_name: string;
  coordinates: { x: number; y: number; width: number; height: number };
  danger_type: string;
  severity: string;
  detected_at: string;
  resolved_at?: string;
  status: string;
  peak_methane_ppm?: number;
  peak_co_ppm?: number;
  affected_workers: number;
  evacuation_ordered: boolean;
}

interface TrendDataPoint {
  time: string;
  methane: number;
  co: number;
  timestamp: Date;
}

interface ZoneWithReading {
  id: string;
  name: string;
  risk_level: string;
  coordinates: { x: number; y: number; width: number; height: number };
  depth_m: number;
  latest_reading?: GasReading;
  is_danger: boolean;
}

const SEVERITY_COLORS = {
  normal: '#10B981',
  medium: '#F59E0B',
  high: '#F97316',
  critical: '#EF4444',
};

const PIE_COLORS = ['#10B981', '#F59E0B', '#F97316', '#EF4444'];

export default function GasMonitoringPage() {
  const { getMineIds } = useAuthStore();
  const [latestReadings, setLatestReadings] = useState<GasReading[]>([]);
  const [stats, setStats] = useState<GasStats | null>(null);
  const [trendData, setTrendData] = useState<TrendDataPoint[]>([]);
  const [dangerZones, setDangerZones] = useState<DangerZone[]>([]);
  const [zones, setZones] = useState<ZoneWithReading[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [timeRange, setTimeRange] = useState(24);
  const [activeTab, setActiveTab] = useState<'overview' | 'map' | 'history' | 'alerts'>('overview');
  const [audioEnabled, setAudioEnabled] = useState(true);
  const [showAlertModal, setShowAlertModal] = useState(false);
  const [selectedZone, setSelectedZone] = useState<ZoneWithReading | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const lastAlertRef = useRef<string | null>(null);

  // Initialize audio
  useEffect(() => {
    audioRef.current = new Audio('/alert-sound.mp3');
    audioRef.current.loop = false;

    return () => {
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current = null;
      }
    };
  }, []);

  // Play alert sound for critical situations
  const playAlertSound = useCallback(() => {
    if (audioEnabled && audioRef.current) {
      audioRef.current.currentTime = 0;
      audioRef.current.play().catch(console.error);
    }
  }, [audioEnabled]);

  const loadData = useCallback(async () => {
    try {
      setLoading(true);
      const mineIds = getMineIds();
      const mineId = mineIds.length > 0 ? mineIds[0] : undefined;

      const [latestData, statsData] = await Promise.all([
        gasSensorApi.getLatest(mineId),
        gasSensorApi.getStats(mineId, timeRange).catch(() => null),
      ]);

      // Transform latest readings to expected format
      const readings: GasReading[] = (Array.isArray(latestData) ? latestData : []).map((sensor: { sensor_id: string; mine_id: string; latest: { timestamp: string; gas_type: string; value: number; unit: string; status: string } }, index: number) => {
        const isMethane = sensor.latest?.gas_type?.toLowerCase().includes('methane') || sensor.latest?.gas_type === 'CH4';
        const status = sensor.latest?.status || 'normal';
        const severity = status === 'critical' ? 'critical' : status === 'danger' || status === 'high' ? 'high' : status === 'warning' || status === 'medium' ? 'medium' : 'normal';

        return {
          id: `${sensor.sensor_id}-${index}`,
          mine_id: sensor.mine_id,
          mine_name: `Mine ${sensor.mine_id}`,
          sensor_id: sensor.sensor_id,
          methane_ppm: isMethane ? (sensor.latest?.value || 0) : 0,
          co_ppm: !isMethane ? (sensor.latest?.value || 0) : 0,
          pressure_hpa: 1013,
          altitude_m: 0,
          severity: severity as 'normal' | 'medium' | 'high' | 'critical',
          timestamp: sensor.latest?.timestamp || new Date().toISOString(),
        };
      });

      setLatestReadings(readings);
      if (statsData) {
        setStats(statsData);
      }

      // Check for critical alerts and play sound
      const hasCritical = readings.some(
        (r: GasReading) => r.severity === 'critical' || r.severity === 'high'
      );

      if (hasCritical) {
        const criticalIds = readings
          .filter((r: GasReading) => r.severity === 'critical')
          .map((r: GasReading) => r.id)
          .join(',');

        if (criticalIds && criticalIds !== lastAlertRef.current) {
          playAlertSound();
          setShowAlertModal(true);
          lastAlertRef.current = criticalIds;
        }
      }

      // Load trend data (historical readings)
      try {
        const historyData = await gasSensorApi.getReadings({
          mine_id: mineId || undefined,
          limit: 200,
        });

        const trends: TrendDataPoint[] = (historyData.readings || [])
          .reverse()
          .map((reading: GasReading) => ({
            time: new Date(reading.timestamp).toLocaleTimeString('en-US', {
              hour: '2-digit',
              minute: '2-digit',
            }),
            methane: reading.methane_ppm,
            co: reading.co_ppm * 100, // Scale for visibility
            timestamp: new Date(reading.timestamp),
          }));

        setTrendData(trends);
      } catch (err) {
        console.error('Failed to load trend data:', err);
      }

      // Load danger zones
      try {
        const dangerData = await apiClient.get(`/api/danger-zones?mine_id=${mineId}&status=active`);
        setDangerZones(dangerData.danger_zones || []);
      } catch (err) {
        // Use mock data if endpoint not available
        setDangerZones([]);
      }

      // Load zones with readings
      try {
        const zonesData = await apiClient.get(`/mines/${mineId}/zones`);
        const zonesWithReadings: ZoneWithReading[] = (zonesData.zones || zonesData || []).map((zone: any) => {
          const reading = (latestData.readings || []).find(
            (r: GasReading) => r.zone_id === zone.id || r.zone_name === zone.name
          );
          return {
            ...zone,
            latest_reading: reading,
            is_danger: reading ? ['high', 'critical'].includes(reading.severity) : false,
          };
        });
        setZones(zonesWithReadings);
      } catch (err) {
        console.error('Failed to load zones:', err);
      }

      setError(null);
    } catch (err) {
      setError('Failed to load gas monitoring data');
      console.error(err);
    } finally {
      setLoading(false);
    }
  }, [getMineIds, timeRange, playAlertSound]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  // Auto-refresh every 30 seconds
  useEffect(() => {
    if (!autoRefresh) return;

    const interval = setInterval(() => {
      loadData();
    }, 30000);

    return () => clearInterval(interval);
  }, [autoRefresh, loadData]);

  if (loading && !stats) {
    return (
      <AppLayout>
        <div className="flex items-center justify-center h-64">
          <Spinner size="lg" />
        </div>
      </AppLayout>
    );
  }

  if (error && !stats) {
    return (
      <AppLayout>
        <div className="text-center py-12">
          <AlertTriangle className="w-12 h-12 text-red-500 mx-auto mb-4" />
          <p className="text-red-500 mb-4">{error}</p>
          <button onClick={loadData} className="btn btn-primary">
            Retry
          </button>
        </div>
      </AppLayout>
    );
  }

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical':
        return 'red';
      case 'high':
        return 'red';
      case 'medium':
        return 'yellow';
      default:
        return 'green';
    }
  };

  const getSeverityBadge = (severity: string) => {
    const colors = {
      critical: 'bg-red-100 text-red-700 border-red-200',
      high: 'bg-orange-100 text-orange-700 border-orange-200',
      medium: 'bg-yellow-100 text-yellow-700 border-yellow-200',
      normal: 'bg-green-100 text-green-700 border-green-200',
    };
    return colors[severity as keyof typeof colors] || colors.normal;
  };

  const getGasLevelStatus = (ppm: number, type: 'methane' | 'co') => {
    if (type === 'methane') {
      if (ppm > 12500) return { status: 'DANGER', color: 'text-red-600' };
      if (ppm > 10000) return { status: 'WARNING', color: 'text-orange-600' };
      if (ppm > 5000) return { status: 'ELEVATED', color: 'text-yellow-600' };
      return { status: 'SAFE', color: 'text-green-600' };
    } else {
      if (ppm > 50) return { status: 'DANGER', color: 'text-red-600' };
      if (ppm > 35) return { status: 'WARNING', color: 'text-orange-600' };
      if (ppm > 25) return { status: 'ELEVATED', color: 'text-yellow-600' };
      return { status: 'SAFE', color: 'text-green-600' };
    }
  };

  const pieData = stats
    ? [
        { name: 'Normal', value: stats.severity_distribution.normal, color: SEVERITY_COLORS.normal },
        { name: 'Medium', value: stats.severity_distribution.medium, color: SEVERITY_COLORS.medium },
        { name: 'High', value: stats.severity_distribution.high, color: SEVERITY_COLORS.high },
        { name: 'Critical', value: stats.severity_distribution.critical, color: SEVERITY_COLORS.critical },
      ].filter((item) => item.value > 0)
    : [];

  const getZoneColor = (zone: ZoneWithReading) => {
    if (zone.is_danger) return 'rgba(239, 68, 68, 0.5)';
    if (zone.risk_level === 'critical') return 'rgba(239, 68, 68, 0.3)';
    if (zone.risk_level === 'high') return 'rgba(249, 115, 22, 0.3)';
    if (zone.risk_level === 'medium') return 'rgba(245, 158, 11, 0.3)';
    return 'rgba(16, 185, 129, 0.3)';
  };

  return (
    <AppLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold text-stone-900 flex items-center gap-2">
              <Wind className="w-7 h-7 text-cyan-600" />
              Gas Monitoring System
            </h1>
            <p className="text-stone-500 mt-1">Real-time methane and carbon monoxide detection</p>
          </div>
          <div className="flex items-center gap-3 flex-wrap">
            <select
              value={timeRange}
              onChange={(e) => setTimeRange(Number(e.target.value))}
              className="px-4 py-2 border border-stone-300 rounded-lg focus:ring-2 focus:ring-cyan-500"
            >
              <option value={1}>Last Hour</option>
              <option value={6}>Last 6 Hours</option>
              <option value={24}>Last 24 Hours</option>
              <option value={72}>Last 3 Days</option>
              <option value={168}>Last Week</option>
            </select>
            <button
              onClick={() => setAudioEnabled(!audioEnabled)}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${
                audioEnabled ? 'bg-red-600 text-white' : 'bg-stone-200 text-stone-600'
              }`}
              title={audioEnabled ? 'Disable Audio Alerts' : 'Enable Audio Alerts'}
            >
              {audioEnabled ? <Volume2 className="w-4 h-4" /> : <VolumeX className="w-4 h-4" />}
              {audioEnabled ? 'Audio On' : 'Audio Off'}
            </button>
            <button
              onClick={() => setAutoRefresh(!autoRefresh)}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg ${
                autoRefresh ? 'bg-cyan-600 text-white' : 'bg-stone-100 text-stone-700'
              }`}
            >
              <Activity className="w-4 h-4" />
              {autoRefresh ? 'Live' : 'Paused'}
            </button>
            <button
              onClick={loadData}
              className="flex items-center gap-2 px-4 py-2 bg-stone-100 text-stone-700 rounded-lg hover:bg-stone-200"
            >
              <RefreshCw className="w-4 h-4" />
              Refresh
            </button>
          </div>
        </div>

        {/* Overall Status Banner */}
        {stats && (
          <div
            className={`rounded-xl p-6 ${
              stats.is_safe
                ? 'bg-gradient-to-r from-green-500 to-green-600 text-white'
                : 'bg-gradient-to-r from-red-500 to-red-600 text-white animate-pulse'
            }`}
          >
            <div className="flex items-center justify-between">
              <div>
                <div className="flex items-center gap-2 mb-2">
                  {stats.is_safe ? <CheckCircle className="w-6 h-6" /> : <XCircle className="w-6 h-6" />}
                  <span className="text-sm font-medium opacity-90">System Status</span>
                </div>
                <h2 className="text-3xl font-bold">{stats.is_safe ? 'ALL CLEAR' : 'ALERT - ACTION REQUIRED'}</h2>
                <p className="mt-1 opacity-90">
                  {stats.is_safe
                    ? 'All gas levels within safe limits'
                    : `${stats.total_alerts} active alerts requiring attention`}
                </p>
              </div>
              <Shield className="w-16 h-16 opacity-20" />
            </div>
          </div>
        )}

        {/* Tab Navigation */}
        <div className="flex gap-2 border-b border-stone-200 pb-1">
          {[
            { id: 'overview', label: 'Overview', icon: BarChart3 },
            { id: 'map', label: 'Danger Zones Map', icon: Map },
            { id: 'history', label: 'History & Trends', icon: History },
            { id: 'alerts', label: 'Active Alerts', icon: AlertCircle },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`flex items-center gap-2 px-4 py-2 rounded-t-lg font-medium transition-colors ${
                activeTab === tab.id
                  ? 'bg-cyan-50 text-cyan-700 border-b-2 border-cyan-600'
                  : 'text-stone-500 hover:text-stone-700 hover:bg-stone-50'
              }`}
            >
              <tab.icon className="w-4 h-4" />
              {tab.label}
            </button>
          ))}
        </div>

        {/* Overview Tab */}
        {activeTab === 'overview' && stats && (
          <>
            {/* Statistics Overview */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <StatCard
                title="Methane (CH4)"
                value={`${Math.round(stats.methane.current)} PPM`}
                icon={<Flame size={24} />}
                color={getSeverityColor(getGasLevelStatus(stats.methane.current, 'methane').status)}
                subtitle={getGasLevelStatus(stats.methane.current, 'methane').status}
              />
              <StatCard
                title="Carbon Monoxide"
                value={`${Math.round(stats.co.current)} PPM`}
                icon={<Wind size={24} />}
                color={getSeverityColor(getGasLevelStatus(stats.co.current, 'co').status)}
                subtitle={getGasLevelStatus(stats.co.current, 'co').status}
              />
              <StatCard
                title="Active Alerts"
                value={stats.total_alerts}
                icon={<AlertTriangle size={24} />}
                color={stats.total_alerts > 0 ? 'red' : 'green'}
              />
              <StatCard
                title="Total Readings"
                value={stats.total_readings}
                icon={<Activity size={24} />}
                color="blue"
                subtitle={`Last ${timeRange}h`}
              />
            </div>

            {/* Charts Row */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Real-time Trend Chart */}
              <div className="lg:col-span-2">
                <Card title="Gas Level Trends" description="Real-time monitoring data">
                  <div className="h-80">
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={trendData.slice(-50)}>
                        <defs>
                          <linearGradient id="methaneGradient" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#F97316" stopOpacity={0.3} />
                            <stop offset="95%" stopColor="#F97316" stopOpacity={0} />
                          </linearGradient>
                          <linearGradient id="coGradient" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#06B6D4" stopOpacity={0.3} />
                            <stop offset="95%" stopColor="#06B6D4" stopOpacity={0} />
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
                        <XAxis dataKey="time" stroke="#9CA3AF" fontSize={12} />
                        <YAxis stroke="#9CA3AF" fontSize={12} />
                        <Tooltip
                          contentStyle={{
                            backgroundColor: '#1F2937',
                            border: 'none',
                            borderRadius: '8px',
                            color: '#F9FAFB',
                          }}
                          formatter={(value: number, name: string) => [
                            name === 'methane' ? `${value} PPM` : `${(value / 100).toFixed(1)} PPM`,
                            name === 'methane' ? 'Methane (CH4)' : 'CO',
                          ]}
                        />
                        <Legend />
                        <Area
                          type="monotone"
                          dataKey="methane"
                          stroke="#F97316"
                          fill="url(#methaneGradient)"
                          strokeWidth={2}
                          name="Methane (PPM)"
                        />
                        <Area
                          type="monotone"
                          dataKey="co"
                          stroke="#06B6D4"
                          fill="url(#coGradient)"
                          strokeWidth={2}
                          name="CO (x100 PPM)"
                        />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </Card>
              </div>

              {/* Severity Distribution Pie Chart */}
              <Card title="Severity Distribution" description={`Last ${timeRange} hours`}>
                <div className="h-80 flex flex-col items-center justify-center">
                  {pieData.length > 0 ? (
                    <>
                      <ResponsiveContainer width="100%" height="70%">
                        <PieChart>
                          <Pie
                            data={pieData}
                            cx="50%"
                            cy="50%"
                            innerRadius={50}
                            outerRadius={80}
                            paddingAngle={5}
                            dataKey="value"
                          >
                            {pieData.map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={entry.color} />
                            ))}
                          </Pie>
                          <Tooltip />
                        </PieChart>
                      </ResponsiveContainer>
                      <div className="flex flex-wrap justify-center gap-3 mt-4">
                        {pieData.map((item) => (
                          <div key={item.name} className="flex items-center gap-2">
                            <div className="w-3 h-3 rounded-full" style={{ backgroundColor: item.color }} />
                            <span className="text-sm text-stone-600">
                              {item.name}: {item.value}
                            </span>
                          </div>
                        ))}
                      </div>
                    </>
                  ) : (
                    <p className="text-stone-500">No data available</p>
                  )}
                </div>
              </Card>
            </div>

            {/* Gas Level Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Methane Trends */}
              <Card title="Methane (CH4) Levels" description={`Statistics for last ${timeRange} hours`}>
                <div className="space-y-4">
                  <div className="flex items-center justify-between p-4 bg-stone-50 rounded-lg">
                    <div>
                      <p className="text-sm text-stone-500">Current</p>
                      <p className={`text-2xl font-bold ${getGasLevelStatus(stats.methane.current, 'methane').color}`}>
                        {Math.round(stats.methane.current)} PPM
                      </p>
                    </div>
                    <Flame className={`w-12 h-12 ${getGasLevelStatus(stats.methane.current, 'methane').color}`} />
                  </div>
                  <div className="grid grid-cols-3 gap-4 text-center">
                    <div className="p-3 bg-blue-50 rounded-lg">
                      <p className="text-xs text-stone-500">Average</p>
                      <p className="text-lg font-bold text-blue-600">{Math.round(stats.methane.avg)}</p>
                    </div>
                    <div className="p-3 bg-red-50 rounded-lg">
                      <p className="text-xs text-stone-500">Max</p>
                      <p className="text-lg font-bold text-red-600">{Math.round(stats.methane.max)}</p>
                    </div>
                    <div className="p-3 bg-green-50 rounded-lg">
                      <p className="text-xs text-stone-500">Min</p>
                      <p className="text-lg font-bold text-green-600">{Math.round(stats.methane.min)}</p>
                    </div>
                  </div>
                  {/* Gauge visualization */}
                  <div className="pt-4">
                    <div className="relative h-4 bg-gradient-to-r from-green-500 via-yellow-500 via-orange-500 to-red-500 rounded-full">
                      <div
                        className="absolute top-1/2 -translate-y-1/2 w-4 h-6 bg-stone-800 rounded shadow-lg border-2 border-white"
                        style={{
                          left: `${Math.min((stats.methane.current / 15000) * 100, 100)}%`,
                          transform: 'translateX(-50%) translateY(-50%)',
                        }}
                      />
                    </div>
                    <div className="flex justify-between mt-1 text-xs text-stone-500">
                      <span>0</span>
                      <span>5,000</span>
                      <span>10,000</span>
                      <span>15,000 PPM</span>
                    </div>
                  </div>
                </div>
              </Card>

              {/* CO Trends */}
              <Card title="Carbon Monoxide (CO) Levels" description={`Statistics for last ${timeRange} hours`}>
                <div className="space-y-4">
                  <div className="flex items-center justify-between p-4 bg-stone-50 rounded-lg">
                    <div>
                      <p className="text-sm text-stone-500">Current</p>
                      <p className={`text-2xl font-bold ${getGasLevelStatus(stats.co.current, 'co').color}`}>
                        {Math.round(stats.co.current)} PPM
                      </p>
                    </div>
                    <Wind className={`w-12 h-12 ${getGasLevelStatus(stats.co.current, 'co').color}`} />
                  </div>
                  <div className="grid grid-cols-3 gap-4 text-center">
                    <div className="p-3 bg-blue-50 rounded-lg">
                      <p className="text-xs text-stone-500">Average</p>
                      <p className="text-lg font-bold text-blue-600">{Math.round(stats.co.avg)}</p>
                    </div>
                    <div className="p-3 bg-red-50 rounded-lg">
                      <p className="text-xs text-stone-500">Max</p>
                      <p className="text-lg font-bold text-red-600">{Math.round(stats.co.max)}</p>
                    </div>
                    <div className="p-3 bg-green-50 rounded-lg">
                      <p className="text-xs text-stone-500">Min</p>
                      <p className="text-lg font-bold text-green-600">{Math.round(stats.co.min)}</p>
                    </div>
                  </div>
                  {/* Gauge visualization */}
                  <div className="pt-4">
                    <div className="relative h-4 bg-gradient-to-r from-green-500 via-yellow-500 via-orange-500 to-red-500 rounded-full">
                      <div
                        className="absolute top-1/2 -translate-y-1/2 w-4 h-6 bg-stone-800 rounded shadow-lg border-2 border-white"
                        style={{
                          left: `${Math.min((stats.co.current / 60) * 100, 100)}%`,
                          transform: 'translateX(-50%) translateY(-50%)',
                        }}
                      />
                    </div>
                    <div className="flex justify-between mt-1 text-xs text-stone-500">
                      <span>0</span>
                      <span>25</span>
                      <span>35</span>
                      <span>50+ PPM</span>
                    </div>
                  </div>
                </div>
              </Card>
            </div>
          </>
        )}

        {/* Map Tab - Danger Zone Visualization */}
        {activeTab === 'map' && (
          <Card title="Mine Danger Zone Map" description="Real-time visualization of gas hazard zones">
            <div className="relative bg-stone-900 rounded-xl overflow-hidden" style={{ height: '600px' }}>
              {/* Legend */}
              <div className="absolute top-4 right-4 bg-white/90 rounded-lg p-3 shadow-lg z-10">
                <h4 className="font-medium text-sm mb-2">Risk Levels</h4>
                <div className="space-y-1 text-xs">
                  <div className="flex items-center gap-2">
                    <div className="w-4 h-4 rounded" style={{ backgroundColor: 'rgba(16, 185, 129, 0.5)' }} />
                    <span>Low Risk</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-4 h-4 rounded" style={{ backgroundColor: 'rgba(245, 158, 11, 0.5)' }} />
                    <span>Medium Risk</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-4 h-4 rounded" style={{ backgroundColor: 'rgba(249, 115, 22, 0.5)' }} />
                    <span>High Risk</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-4 h-4 rounded animate-pulse" style={{ backgroundColor: 'rgba(239, 68, 68, 0.7)' }} />
                    <span>Active Danger</span>
                  </div>
                </div>
              </div>

              {/* Mine map grid */}
              <svg viewBox="0 0 100 80" className="w-full h-full">
                {/* Grid lines */}
                <defs>
                  <pattern id="grid" width="10" height="10" patternUnits="userSpaceOnUse">
                    <path d="M 10 0 L 0 0 0 10" fill="none" stroke="#374151" strokeWidth="0.2" />
                  </pattern>
                </defs>
                <rect width="100" height="80" fill="url(#grid)" />

                {/* Zones */}
                {zones.map((zone) => (
                  <g key={zone.id} className="cursor-pointer" onClick={() => setSelectedZone(zone)}>
                    <rect
                      x={zone.coordinates?.x || 0}
                      y={zone.coordinates?.y || 0}
                      width={zone.coordinates?.width || 20}
                      height={zone.coordinates?.height || 15}
                      fill={getZoneColor(zone)}
                      stroke={zone.is_danger ? '#EF4444' : '#6B7280'}
                      strokeWidth={zone.is_danger ? '0.5' : '0.2'}
                      className={zone.is_danger ? 'animate-pulse' : ''}
                      rx="1"
                    />
                    <text
                      x={(zone.coordinates?.x || 0) + (zone.coordinates?.width || 20) / 2}
                      y={(zone.coordinates?.y || 0) + (zone.coordinates?.height || 15) / 2}
                      textAnchor="middle"
                      dominantBaseline="middle"
                      fill="white"
                      fontSize="2.5"
                      fontWeight="bold"
                    >
                      {zone.name.split(' - ')[0]}
                    </text>
                    {zone.latest_reading && (
                      <text
                        x={(zone.coordinates?.x || 0) + (zone.coordinates?.width || 20) / 2}
                        y={(zone.coordinates?.y || 0) + (zone.coordinates?.height || 15) / 2 + 4}
                        textAnchor="middle"
                        dominantBaseline="middle"
                        fill="#9CA3AF"
                        fontSize="1.8"
                      >
                        CH4: {Math.round(zone.latest_reading.methane_ppm)} | CO: {Math.round(zone.latest_reading.co_ppm)}
                      </text>
                    )}
                    {zone.is_danger && (
                      <circle
                        cx={(zone.coordinates?.x || 0) + 3}
                        cy={(zone.coordinates?.y || 0) + 3}
                        r="2"
                        fill="#EF4444"
                        className="animate-ping"
                      />
                    )}
                  </g>
                ))}

                {/* Mine entrance */}
                <rect x="45" y="72" width="10" height="6" fill="#10B981" rx="1" />
                <text x="50" y="76" textAnchor="middle" fill="white" fontSize="2" fontWeight="bold">
                  ENTRY
                </text>
              </svg>

              {/* Zone Detail Popup */}
              {selectedZone && (
                <div className="absolute bottom-4 left-4 bg-white rounded-lg p-4 shadow-xl max-w-sm z-10">
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="font-bold text-stone-800">{selectedZone.name}</h4>
                    <button onClick={() => setSelectedZone(null)} className="text-stone-400 hover:text-stone-600">
                      <XCircle className="w-5 h-5" />
                    </button>
                  </div>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-stone-500">Risk Level:</span>
                      <span className={`font-medium capitalize ${selectedZone.risk_level === 'critical' ? 'text-red-600' : selectedZone.risk_level === 'high' ? 'text-orange-600' : selectedZone.risk_level === 'medium' ? 'text-yellow-600' : 'text-green-600'}`}>
                        {selectedZone.risk_level}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-stone-500">Depth:</span>
                      <span className="font-medium">{selectedZone.depth_m}m</span>
                    </div>
                    {selectedZone.latest_reading && (
                      <>
                        <div className="flex justify-between">
                          <span className="text-stone-500">Methane:</span>
                          <span className={`font-medium ${getGasLevelStatus(selectedZone.latest_reading.methane_ppm, 'methane').color}`}>
                            {Math.round(selectedZone.latest_reading.methane_ppm)} PPM
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-stone-500">CO:</span>
                          <span className={`font-medium ${getGasLevelStatus(selectedZone.latest_reading.co_ppm, 'co').color}`}>
                            {Math.round(selectedZone.latest_reading.co_ppm)} PPM
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-stone-500">Temperature:</span>
                          <span className="font-medium">{selectedZone.latest_reading.temperature_c?.toFixed(1)}C</span>
                        </div>
                      </>
                    )}
                    {selectedZone.is_danger && (
                      <div className="mt-2 p-2 bg-red-50 rounded-lg text-red-700 text-xs flex items-center gap-2">
                        <AlertTriangle className="w-4 h-4" />
                        <span>Active danger zone - Evacuation may be required</span>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          </Card>
        )}

        {/* History Tab */}
        {activeTab === 'history' && (
          <div className="space-y-6">
            {/* Historical Trend Line Chart */}
            <Card title="Historical Gas Levels" description="Detailed trend analysis">
              <div className="h-96">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={trendData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
                    <XAxis dataKey="time" stroke="#9CA3AF" fontSize={11} />
                    <YAxis yAxisId="left" stroke="#F97316" fontSize={11} />
                    <YAxis yAxisId="right" orientation="right" stroke="#06B6D4" fontSize={11} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#1F2937',
                        border: 'none',
                        borderRadius: '8px',
                        color: '#F9FAFB',
                      }}
                    />
                    <Legend />
                    <Line
                      yAxisId="left"
                      type="monotone"
                      dataKey="methane"
                      stroke="#F97316"
                      strokeWidth={2}
                      dot={false}
                      name="Methane (PPM)"
                    />
                    <Line
                      yAxisId="right"
                      type="monotone"
                      dataKey="co"
                      stroke="#06B6D4"
                      strokeWidth={2}
                      dot={false}
                      name="CO (x100 PPM)"
                    />
                    {/* Threshold lines */}
                    <Line
                      yAxisId="left"
                      type="monotone"
                      dataKey={() => 10000}
                      stroke="#EF4444"
                      strokeDasharray="5 5"
                      dot={false}
                      name="Methane Warning"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </Card>

            {/* Hourly Distribution Bar Chart */}
            <Card title="Reading Distribution by Hour" description="When gas levels are highest">
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={Array.from({ length: 24 }, (_, i) => ({
                      hour: `${i}:00`,
                      readings: trendData.filter((d) => d.timestamp.getHours() === i).length,
                      avgMethane:
                        trendData.filter((d) => d.timestamp.getHours() === i).reduce((sum, d) => sum + d.methane, 0) /
                          (trendData.filter((d) => d.timestamp.getHours() === i).length || 1) || 0,
                    }))}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
                    <XAxis dataKey="hour" stroke="#9CA3AF" fontSize={10} />
                    <YAxis stroke="#9CA3AF" fontSize={11} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#1F2937',
                        border: 'none',
                        borderRadius: '8px',
                        color: '#F9FAFB',
                      }}
                    />
                    <Bar dataKey="avgMethane" fill="#F97316" name="Avg Methane (PPM)" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </Card>

            {/* Danger Zone History */}
            <Card title="Danger Zone Incident History" description="Past 30 days">
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="bg-stone-50">
                      <th className="px-4 py-3 text-left font-medium text-stone-600">Date/Time</th>
                      <th className="px-4 py-3 text-left font-medium text-stone-600">Zone</th>
                      <th className="px-4 py-3 text-left font-medium text-stone-600">Type</th>
                      <th className="px-4 py-3 text-left font-medium text-stone-600">Severity</th>
                      <th className="px-4 py-3 text-left font-medium text-stone-600">Peak Levels</th>
                      <th className="px-4 py-3 text-left font-medium text-stone-600">Status</th>
                      <th className="px-4 py-3 text-left font-medium text-stone-600">Duration</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-stone-200">
                    {dangerZones.length === 0 ? (
                      <tr>
                        <td colSpan={7} className="px-4 py-8 text-center text-stone-500">
                          No danger zone incidents recorded
                        </td>
                      </tr>
                    ) : (
                      dangerZones.map((zone) => (
                        <tr key={zone.id} className="hover:bg-stone-50">
                          <td className="px-4 py-3">
                            {new Date(zone.detected_at).toLocaleString()}
                          </td>
                          <td className="px-4 py-3 font-medium">{zone.zone_name}</td>
                          <td className="px-4 py-3 capitalize">{zone.danger_type.replace('_', ' ')}</td>
                          <td className="px-4 py-3">
                            <span className={`px-2 py-1 rounded-full text-xs font-medium ${getSeverityBadge(zone.severity)}`}>
                              {zone.severity.toUpperCase()}
                            </span>
                          </td>
                          <td className="px-4 py-3 text-xs">
                            {zone.peak_methane_ppm && <div>CH4: {zone.peak_methane_ppm} PPM</div>}
                            {zone.peak_co_ppm && <div>CO: {zone.peak_co_ppm} PPM</div>}
                          </td>
                          <td className="px-4 py-3">
                            <span
                              className={`px-2 py-1 rounded-full text-xs font-medium ${
                                zone.status === 'resolved'
                                  ? 'bg-green-100 text-green-700'
                                  : 'bg-red-100 text-red-700 animate-pulse'
                              }`}
                            >
                              {zone.status.toUpperCase()}
                            </span>
                          </td>
                          <td className="px-4 py-3">
                            {zone.resolved_at
                              ? `${Math.round((new Date(zone.resolved_at).getTime() - new Date(zone.detected_at).getTime()) / 60000)} mins`
                              : 'Ongoing'}
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

        {/* Alerts Tab */}
        {activeTab === 'alerts' && (
          <div className="space-y-4">
            {/* Active Critical Alerts */}
            {latestReadings.filter((r) => r.severity === 'critical' || r.severity === 'high').length > 0 ? (
              latestReadings
                .filter((r) => r.severity === 'critical' || r.severity === 'high')
                .map((reading) => (
                  <div
                    key={reading.id}
                    className={`rounded-xl p-6 ${
                      reading.severity === 'critical'
                        ? 'bg-red-50 border-2 border-red-200 animate-pulse'
                        : 'bg-orange-50 border-2 border-orange-200'
                    }`}
                  >
                    <div className="flex items-start gap-4">
                      <div
                        className={`p-3 rounded-full ${
                          reading.severity === 'critical' ? 'bg-red-100' : 'bg-orange-100'
                        }`}
                      >
                        <AlertTriangle
                          className={`w-8 h-8 ${
                            reading.severity === 'critical' ? 'text-red-600' : 'text-orange-600'
                          }`}
                        />
                      </div>
                      <div className="flex-1">
                        <div className="flex items-center gap-3 mb-2">
                          <h3
                            className={`text-xl font-bold ${
                              reading.severity === 'critical' ? 'text-red-800' : 'text-orange-800'
                            }`}
                          >
                            {reading.severity === 'critical' ? 'CRITICAL GAS ALERT' : 'HIGH GAS WARNING'}
                          </h3>
                          <span className={`px-3 py-1 rounded-full text-sm font-medium ${getSeverityBadge(reading.severity)}`}>
                            {reading.severity.toUpperCase()}
                          </span>
                        </div>
                        <p className="text-stone-700 mb-4">
                          Dangerous gas levels detected in <strong>{reading.zone_name || reading.gate_name || reading.mine_name}</strong>
                        </p>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                          <div className="bg-white/70 p-3 rounded-lg">
                            <p className="text-xs text-stone-500">Methane (CH4)</p>
                            <p className="text-xl font-bold text-red-600">{Math.round(reading.methane_ppm)} PPM</p>
                          </div>
                          <div className="bg-white/70 p-3 rounded-lg">
                            <p className="text-xs text-stone-500">Carbon Monoxide</p>
                            <p className="text-xl font-bold text-red-600">{Math.round(reading.co_ppm)} PPM</p>
                          </div>
                          <div className="bg-white/70 p-3 rounded-lg">
                            <p className="text-xs text-stone-500">Location</p>
                            <p className="font-medium">{reading.zone_name || 'Unknown'}</p>
                          </div>
                          <div className="bg-white/70 p-3 rounded-lg">
                            <p className="text-xs text-stone-500">Detected At</p>
                            <p className="font-medium">{new Date(reading.timestamp).toLocaleTimeString()}</p>
                          </div>
                        </div>
                        <div className="flex gap-3">
                          <button className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 font-medium flex items-center gap-2">
                            <Zap className="w-4 h-4" />
                            Initiate Evacuation
                          </button>
                          <button className="px-4 py-2 bg-white text-stone-700 rounded-lg hover:bg-stone-100 font-medium">
                            Acknowledge Alert
                          </button>
                          <button className="px-4 py-2 bg-white text-stone-700 rounded-lg hover:bg-stone-100 font-medium">
                            View Zone Details
                          </button>
                        </div>
                      </div>
                    </div>
                  </div>
                ))
            ) : (
              <Card>
                <div className="text-center py-12">
                  <CheckCircle className="w-16 h-16 text-green-500 mx-auto mb-4" />
                  <h3 className="text-xl font-bold text-stone-800 mb-2">All Clear</h3>
                  <p className="text-stone-500">No active gas alerts at this time</p>
                </div>
              </Card>
            )}

            {/* Live Sensor Readings */}
            <Card title="Live Sensor Readings" description="Real-time data from all active sensors">
              {latestReadings.length === 0 ? (
                <div className="text-center py-12 text-stone-500">
                  <Wind className="w-12 h-12 mx-auto mb-4 text-stone-300" />
                  <p>No sensor data available</p>
                  <p className="text-sm mt-2">Waiting for sensor readings...</p>
                </div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {latestReadings.map((reading) => (
                    <div
                      key={reading.id}
                      className={`p-4 rounded-xl border-2 ${
                        reading.severity === 'critical'
                          ? 'bg-red-50 border-red-200'
                          : reading.severity === 'high'
                          ? 'bg-orange-50 border-orange-200'
                          : reading.severity === 'medium'
                          ? 'bg-yellow-50 border-yellow-200'
                          : 'bg-green-50 border-green-200'
                      }`}
                    >
                      <div className="flex items-start justify-between mb-3">
                        <div className="flex-1">
                          <div className="flex items-center gap-2">
                            <MapPin className="w-4 h-4 text-stone-500" />
                            <span className="font-medium text-stone-900">
                              {reading.zone_name || reading.gate_name || reading.mine_name}
                            </span>
                          </div>
                          {reading.sensor_id && (
                            <p className="text-xs text-stone-500 mt-1">Sensor {reading.sensor_id}</p>
                          )}
                        </div>
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${getSeverityBadge(reading.severity)}`}>
                          {reading.severity.toUpperCase()}
                        </span>
                      </div>

                      <div className="space-y-3">
                        {/* Methane */}
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <Flame className="w-4 h-4 text-orange-500" />
                            <span className="text-sm text-stone-600">CH4</span>
                          </div>
                          <span className={`font-bold ${getGasLevelStatus(reading.methane_ppm, 'methane').color}`}>
                            {Math.round(reading.methane_ppm)} PPM
                          </span>
                        </div>

                        {/* CO */}
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <Wind className="w-4 h-4 text-cyan-500" />
                            <span className="text-sm text-stone-600">CO</span>
                          </div>
                          <span className={`font-bold ${getGasLevelStatus(reading.co_ppm, 'co').color}`}>
                            {Math.round(reading.co_ppm)} PPM
                          </span>
                        </div>

                        {/* Environmental Data */}
                        <div className="pt-3 border-t border-stone-200 grid grid-cols-2 gap-2 text-xs">
                          {reading.temperature_c !== undefined && (
                            <div className="flex items-center gap-1">
                              <ThermometerSun className="w-3 h-3 text-stone-400" />
                              <span className="text-stone-600">{reading.temperature_c.toFixed(1)}C</span>
                            </div>
                          )}
                          {reading.humidity !== undefined && (
                            <div className="flex items-center gap-1">
                              <Droplets className="w-3 h-3 text-stone-400" />
                              <span className="text-stone-600">{reading.humidity.toFixed(0)}%</span>
                            </div>
                          )}
                          <div className="flex items-center gap-1">
                            <Gauge className="w-3 h-3 text-stone-400" />
                            <span className="text-stone-600">{reading.pressure_hpa.toFixed(0)} hPa</span>
                          </div>
                          <div className="flex items-center gap-1">
                            <Mountain className="w-3 h-3 text-stone-400" />
                            <span className="text-stone-600">{reading.altitude_m.toFixed(1)} m</span>
                          </div>
                        </div>

                        {/* Timestamp */}
                        <div className="text-xs text-stone-400 text-right">
                          {new Date(reading.timestamp).toLocaleTimeString()}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </Card>
          </div>
        )}

        {/* Safety Information */}
        <Card title="Safety Guidelines" description="Emergency procedures and contact information">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-medium text-stone-900 mb-3 flex items-center gap-2">
                <AlertTriangle className="w-5 h-5 text-red-600" />
                Emergency Response
              </h4>
              <ul className="space-y-2 text-sm text-stone-600">
                <li className="flex items-start gap-2">
                  <span className="text-red-600">1.</span>
                  <span>If gas levels exceed danger thresholds, evacuate immediately</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-red-600">2.</span>
                  <span>Alert shift incharge and safety officer</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-red-600">3.</span>
                  <span>Do not re-enter until levels return to safe limits</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-red-600">4.</span>
                  <span>Report all gas alerts to management</span>
                </li>
              </ul>
            </div>
            <div>
              <h4 className="font-medium text-stone-900 mb-3 flex items-center gap-2">
                <Shield className="w-5 h-5 text-green-600" />
                Prevention Measures
              </h4>
              <ul className="space-y-2 text-sm text-stone-600">
                <li className="flex items-start gap-2">
                  <span className="text-green-600">1.</span>
                  <span>Ensure proper ventilation in all zones</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-green-600">2.</span>
                  <span>Regular sensor calibration and maintenance</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-green-600">3.</span>
                  <span>Monitor readings continuously during shifts</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-green-600">4.</span>
                  <span>Report any unusual odors or symptoms immediately</span>
                </li>
              </ul>
            </div>
          </div>
        </Card>
      </div>

      {/* Critical Alert Modal */}
      {showAlertModal && (
        <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-2xl shadow-2xl max-w-lg w-full overflow-hidden animate-bounce">
            <div className="bg-red-600 p-6 text-white">
              <div className="flex items-center gap-4">
                <div className="p-3 bg-white/20 rounded-full">
                  <AlertTriangle className="w-10 h-10" />
                </div>
                <div>
                  <h2 className="text-2xl font-bold">CRITICAL GAS ALERT</h2>
                  <p className="opacity-90">Immediate action required</p>
                </div>
              </div>
            </div>
            <div className="p-6">
              <p className="text-stone-700 mb-6">
                Dangerous gas levels have been detected. Please review the situation and take appropriate action.
              </p>
              <div className="flex gap-3">
                <button
                  onClick={() => setShowAlertModal(false)}
                  className="flex-1 px-4 py-3 bg-red-600 text-white rounded-lg hover:bg-red-700 font-medium"
                >
                  View Details
                </button>
                <button
                  onClick={() => {
                    setShowAlertModal(false);
                    if (audioRef.current) audioRef.current.pause();
                  }}
                  className="flex-1 px-4 py-3 bg-stone-200 text-stone-700 rounded-lg hover:bg-stone-300 font-medium"
                >
                  Acknowledge
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </AppLayout>
  );
}
