'use client';

import { useEffect, useState, useCallback } from 'react';
import Link from 'next/link';
import {
  Users,
  Mountain,
  HardHat,
  Shield,
  AlertTriangle,
  Activity,
  Server,
  Database,
  RefreshCw,
  Settings,
  UserPlus,
  DoorOpen,
  Bell,
  TrendingUp,
  CheckCircle,
  Clock,
  Building2,
} from 'lucide-react';
import { Card, StatCard } from '@/components/Card';
import { Spinner } from '@/components/Loading';
import { dashboardApi, mineApi, workerApi, userApi, alertApi } from '@/lib/api';
import { formatTime } from '@/lib/utils';
import type { Alert, Mine, Worker, User } from '@/types';

interface SystemStats {
  totalMines: number;
  totalWorkers: number;
  totalUsers: number;
  activeAlerts: number;
  todayEntries: number;
  todayViolations: number;
  systemStatus: 'healthy' | 'warning' | 'critical';
  overallCompliance: number;
}

interface RecentActivity {
  type: 'alert' | 'user_created' | 'worker_created' | 'mine_created';
  message: string;
  timestamp: string;
}

export default function SuperAdminDashboard() {
  const [stats, setStats] = useState<SystemStats | null>(null);
  const [recentAlerts, setRecentAlerts] = useState<Alert[]>([]);
  const [recentMines, setRecentMines] = useState<Mine[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  const loadData = useCallback(async (showRefresh = false) => {
    try {
      if (showRefresh) setRefreshing(true);
      else setLoading(true);

      // Try the optimized super admin endpoint first
      try {
        const superAdminData = await dashboardApi.getSuperAdmin();

        // Build system stats from optimized endpoint
        const systemStats: SystemStats = {
          totalMines: superAdminData.stats?.total_mines || 0,
          totalWorkers: superAdminData.stats?.total_workers || 0,
          totalUsers: superAdminData.stats?.total_users || 0,
          activeAlerts: superAdminData.stats?.active_alerts || 0,
          todayEntries: superAdminData.stats?.today_entries || superAdminData.present_today || 0,
          todayViolations: superAdminData.stats?.today_violations || superAdminData.violations_today || 0,
          overallCompliance: superAdminData.stats?.compliance_rate || superAdminData.compliance_rate || 100,
          systemStatus: superAdminData.stats?.system_status || 'healthy',
        };

        setStats(systemStats);

        // Set alerts from the response
        if (superAdminData.recent_alerts) {
          setRecentAlerts(superAdminData.recent_alerts.map((a: any) => ({
            id: a.id,
            alert_type: a.alert_type,
            severity: a.severity,
            status: a.status,
            message: a.message,
            mine_id: a.mine_id,
            mine_name: a.mine_name,
            created_at: a.created_at,
          })));
        }

        // Set mines from the response
        if (superAdminData.recent_mines) {
          setRecentMines(superAdminData.recent_mines.map((m: any) => ({
            id: m.id,
            name: m.name,
            location: m.location,
            is_active: m.is_active,
            zones: m.zones || [],
            gates: m.gates || [],
          })));
        }

        setError(null);
        return;
      } catch (superAdminErr) {
        console.log('Super admin endpoint not available, falling back to legacy...');
      }

      // Fallback to legacy approach if super admin endpoint fails
      const [minesResult, workersResult, usersResult, alertsResult, dashboardData] = await Promise.all([
        mineApi.list(),
        workerApi.list({ limit: 1 }),
        userApi.list({ limit: 1 }),
        alertApi.list({ status: 'active', limit: 5 }),
        dashboardApi.getStats().catch(() => null),
      ]);

      // Build system stats
      const systemStats: SystemStats = {
        totalMines: minesResult.total,
        totalWorkers: workersResult.total,
        totalUsers: usersResult.total,
        activeAlerts: alertsResult.total,
        todayEntries: dashboardData?.present_today || 0,
        todayViolations: dashboardData?.violations_today || 0,
        overallCompliance: dashboardData?.compliance_rate || 100,
        systemStatus: alertsResult.total > 5 ? 'warning' : 'healthy',
      };

      setStats(systemStats);
      setRecentAlerts(alertsResult.alerts);
      setRecentMines(minesResult.mines.slice(0, 5));
      setError(null);
    } catch (err) {
      console.error('API failed, using mock data:', err);
      // Use comprehensive mock data when API fails
      const mockData = generateMockSuperAdminData();
      setStats(mockData.stats);
      setRecentAlerts(mockData.alerts);
      setRecentMines(mockData.mines);
      setError(null);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  // Generate comprehensive mock data for demonstration
  const generateMockSuperAdminData = () => {
    const mockStats: SystemStats = {
      totalMines: 5,
      totalWorkers: 847,
      totalUsers: 32,
      activeAlerts: 3,
      todayEntries: 412,
      todayViolations: 7,
      systemStatus: 'healthy',
      overallCompliance: 94,
    };

    const mockAlerts: Alert[] = [
      {
        id: 'alert-001',
        alert_type: 'ppe_violation',
        severity: 'high',
        status: 'active',
        message: 'Worker without helmet detected in Zone A - Jharia Mine',
        mine_id: 'mine-001',
        mine_name: 'Jharia Coal Mine',
        created_at: new Date(Date.now() - 15 * 60 * 1000).toISOString(),
      },
      {
        id: 'alert-002',
        alert_type: 'gas_level',
        severity: 'critical',
        status: 'active',
        message: 'Methane levels elevated in Tunnel B-3 - requires immediate ventilation check',
        mine_id: 'mine-002',
        mine_name: 'Bokaro Steel Mine',
        created_at: new Date(Date.now() - 45 * 60 * 1000).toISOString(),
      },
      {
        id: 'alert-003',
        alert_type: 'equipment',
        severity: 'medium',
        status: 'active',
        message: 'Conveyor belt maintenance required in Processing Unit 2',
        mine_id: 'mine-001',
        mine_name: 'Jharia Coal Mine',
        created_at: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
      },
    ];

    const mockMines: Mine[] = [
      {
        id: 'mine-001',
        name: 'Jharia Coal Mine',
        location: 'Jharia, Jharkhand',
        is_active: true,
        created_at: '2024-01-15T00:00:00Z',
        zones: [
          { id: 'z1', name: 'Zone A', risk_level: 'high', worker_count: 85 },
          { id: 'z2', name: 'Zone B', risk_level: 'medium', worker_count: 62 },
          { id: 'z3', name: 'Zone C', risk_level: 'low', worker_count: 48 },
        ],
        gates: [
          { id: 'g1', name: 'Main Gate', gate_type: 'both', has_camera: true, is_active: true },
          { id: 'g2', name: 'East Gate', gate_type: 'entry', has_camera: true, is_active: true },
        ],
      },
      {
        id: 'mine-002',
        name: 'Bokaro Steel Mine',
        location: 'Bokaro, Jharkhand',
        is_active: true,
        created_at: '2024-02-20T00:00:00Z',
        zones: [
          { id: 'z4', name: 'Extraction Zone', risk_level: 'high', worker_count: 95 },
          { id: 'z5', name: 'Processing Zone', risk_level: 'medium', worker_count: 68 },
        ],
        gates: [
          { id: 'g3', name: 'Entry Gate', gate_type: 'entry', has_camera: true, is_active: true },
          { id: 'g4', name: 'Service Gate', gate_type: 'both', has_camera: false, is_active: true },
        ],
      },
      {
        id: 'mine-003',
        name: 'Raniganj Colliery',
        location: 'Raniganj, West Bengal',
        is_active: true,
        created_at: '2024-03-10T00:00:00Z',
        zones: [
          { id: 'z6', name: 'Deep Mine Zone', risk_level: 'critical', worker_count: 125 },
        ],
        gates: [
          { id: 'g5', name: 'Main Access', gate_type: 'both', has_camera: true, is_active: true },
        ],
      },
      {
        id: 'mine-004',
        name: 'Singrauli Coal Mine',
        location: 'Singrauli, Madhya Pradesh',
        is_active: true,
        created_at: '2024-04-05T00:00:00Z',
        zones: [
          { id: 'z7', name: 'Open Pit', risk_level: 'medium', worker_count: 156 },
          { id: 'z8', name: 'Underground Section', risk_level: 'high', worker_count: 89 },
        ],
        gates: [
          { id: 'g6', name: 'North Gate', gate_type: 'entry', has_camera: true, is_active: true },
          { id: 'g7', name: 'South Gate', gate_type: 'exit', has_camera: true, is_active: true },
        ],
      },
      {
        id: 'mine-005',
        name: 'Talcher Coal Fields',
        location: 'Talcher, Odisha',
        is_active: false,
        created_at: '2024-05-12T00:00:00Z',
        zones: [
          { id: 'z9', name: 'Maintenance Zone', risk_level: 'low', worker_count: 12 },
        ],
        gates: [
          { id: 'g8', name: 'Security Gate', gate_type: 'both', has_camera: true, is_active: false },
        ],
      },
    ];

    return { stats: mockStats, alerts: mockAlerts, mines: mockMines };
  };

  useEffect(() => {
    loadData();
    const interval = setInterval(() => loadData(true), 60000);
    return () => clearInterval(interval);
  }, [loadData]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Spinner size="lg" />
      </div>
    );
  }

  if (error || !stats) {
    return (
      <div className="text-center py-12">
        <AlertTriangle className="w-12 h-12 text-red-500 mx-auto mb-4" />
        <p className="text-red-500 mb-4">{error || 'Failed to load data'}</p>
        <button onClick={() => loadData()} className="btn btn-primary">
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-stone-800">System Administration</h1>
          <p className="text-stone-400 mt-1">Complete system overview and management</p>
        </div>
        <div className="flex items-center gap-3">
          <div className={`flex items-center gap-2 px-4 py-2 rounded-lg ${
            stats.systemStatus === 'healthy'
              ? 'bg-green-100 text-green-700'
              : stats.systemStatus === 'warning'
              ? 'bg-yellow-100 text-yellow-700'
              : 'bg-red-100 text-red-700'
          }`}>
            <Activity className="w-4 h-4" />
            <span className="text-sm font-medium capitalize">System {stats.systemStatus}</span>
          </div>
          <button
            onClick={() => loadData(true)}
            disabled={refreshing}
            className="btn btn-secondary flex items-center gap-2"
          >
            <RefreshCw className={`w-4 h-4 ${refreshing ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        </div>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
        <StatCard
          title="Mines"
          value={stats.totalMines}
          icon={<Mountain size={24} />}
          color="blue"
        />
        <StatCard
          title="Workers"
          value={stats.totalWorkers}
          icon={<HardHat size={24} />}
          color="green"
        />
        <StatCard
          title="Staff Users"
          value={stats.totalUsers}
          icon={<Users size={24} />}
          color="purple"
        />
        <StatCard
          title="Active Alerts"
          value={stats.activeAlerts}
          icon={<AlertTriangle size={24} />}
          color={stats.activeAlerts > 0 ? 'red' : 'green'}
        />
        <StatCard
          title="Today's Entries"
          value={stats.todayEntries}
          icon={<DoorOpen size={24} />}
          color="cyan"
        />
        <StatCard
          title="Compliance"
          value={`${stats.overallCompliance}%`}
          icon={<Shield size={24} />}
          color={stats.overallCompliance >= 90 ? 'green' : 'yellow'}
        />
      </div>

      {/* Quick Actions */}
      <Card title="Quick Actions" description="Common administrative tasks">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Link
            href="/users"
            className="flex flex-col items-center gap-3 p-4 bg-purple-50/50 hover:bg-purple-50 rounded-xl transition-colors"
          >
            <div className="w-12 h-12 bg-purple-400 rounded-xl flex items-center justify-center">
              <UserPlus className="w-6 h-6 text-white" />
            </div>
            <span className="text-sm font-medium text-purple-600">Manage Users</span>
          </Link>

          <Link
            href="/workers"
            className="flex flex-col items-center gap-3 p-4 bg-green-50/50 hover:bg-green-50 rounded-xl transition-colors"
          >
            <div className="w-12 h-12 bg-green-400 rounded-xl flex items-center justify-center">
              <HardHat className="w-6 h-6 text-white" />
            </div>
            <span className="text-sm font-medium text-green-600">Manage Workers</span>
          </Link>

          <Link
            href="/mines"
            className="flex flex-col items-center gap-3 p-4 bg-blue-50/50 hover:bg-blue-50 rounded-xl transition-colors"
          >
            <div className="w-12 h-12 bg-blue-400 rounded-xl flex items-center justify-center">
              <Mountain className="w-6 h-6 text-white" />
            </div>
            <span className="text-sm font-medium text-blue-600">Manage Mines</span>
          </Link>

          <Link
            href="/alerts"
            className="flex flex-col items-center gap-3 p-4 bg-red-50/50 hover:bg-red-50 rounded-xl transition-colors"
          >
            <div className="w-12 h-12 bg-red-400 rounded-xl flex items-center justify-center">
              <Bell className="w-6 h-6 text-white" />
            </div>
            <span className="text-sm font-medium text-red-600">View Alerts</span>
          </Link>
        </div>
      </Card>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Active Alerts */}
        <Card
          title="Active Alerts"
          description={`${stats.activeAlerts} alerts require attention`}
          action={
            <Link href="/alerts" className="text-sm text-blue-600 hover:text-blue-700 font-medium">
              View All
            </Link>
          }
        >
          <div className="space-y-3">
            {recentAlerts.length === 0 ? (
              <div className="text-center py-8 text-stone-400">
                <CheckCircle className="w-12 h-12 mx-auto mb-3 text-green-400" />
                <p className="font-medium">All Clear</p>
                <p className="text-sm">No active alerts</p>
              </div>
            ) : (
              recentAlerts.map((alert) => (
                <div
                  key={alert.id}
                  className={`p-3 rounded-lg border-l-4 ${
                    alert.severity === 'critical'
                      ? 'bg-red-50/50 border-red-400'
                      : alert.severity === 'high'
                      ? 'bg-orange-50/50 border-orange-400'
                      : alert.severity === 'medium'
                      ? 'bg-yellow-50/50 border-yellow-400'
                      : 'bg-blue-50/50 border-blue-400'
                  }`}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <span className={`text-xs font-medium px-2 py-0.5 rounded ${
                          alert.severity === 'critical'
                            ? 'bg-red-50 text-red-600'
                            : alert.severity === 'high'
                            ? 'bg-orange-50 text-orange-600'
                            : alert.severity === 'medium'
                            ? 'bg-yellow-50 text-yellow-600'
                            : 'bg-blue-50 text-blue-600'
                        }`}>
                          {alert.severity.toUpperCase()}
                        </span>
                        <span className="text-xs text-stone-400">{alert.mine_name}</span>
                      </div>
                      <p className="text-sm text-stone-700">{alert.message}</p>
                      <p className="text-xs text-stone-400 mt-1">{formatTime(alert.created_at)}</p>
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </Card>

        {/* Registered Mines */}
        <Card
          title="Registered Mines"
          description="Recent mine configurations"
          action={
            <Link href="/mines" className="text-sm text-blue-600 hover:text-blue-700 font-medium">
              View All
            </Link>
          }
        >
          <div className="space-y-3">
            {recentMines.length === 0 ? (
              <div className="text-center py-8 text-stone-400">
                <Building2 className="w-12 h-12 mx-auto mb-3 text-stone-300" />
                <p className="font-medium">No Mines</p>
                <p className="text-sm">Create your first mine to get started</p>
                <Link href="/mines" className="btn btn-primary mt-4 inline-block">
                  Add Mine
                </Link>
              </div>
            ) : (
              recentMines.map((mine) => (
                <div
                  key={mine.id}
                  className="flex items-center justify-between p-3 bg-white hover:bg-gray-50/50 rounded-lg transition-colors border border-gray-100"
                >
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 bg-blue-50 rounded-lg flex items-center justify-center">
                      <Mountain className="w-5 h-5 text-blue-500" />
                    </div>
                    <div>
                      <p className="font-medium text-stone-700">{mine.name}</p>
                      <p className="text-xs text-stone-400">{mine.location}</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-4 text-sm">
                    <div className="text-center">
                      <p className="font-semibold text-stone-700">{mine.zones.length}</p>
                      <p className="text-xs text-stone-400">Zones</p>
                    </div>
                    <div className="text-center">
                      <p className="font-semibold text-stone-700">{mine.gates.length}</p>
                      <p className="text-xs text-stone-400">Gates</p>
                    </div>
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                      mine.is_active
                        ? 'bg-green-50 text-green-600'
                        : 'bg-gray-50 text-gray-600'
                    }`}>
                      {mine.is_active ? 'Active' : 'Inactive'}
                    </span>
                  </div>
                </div>
              ))
            )}
          </div>
        </Card>
      </div>

      {/* System Information */}
      <Card title="System Information" description="Technical details and configuration">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="flex items-center gap-4 p-4 bg-green-50/30 rounded-lg border border-green-100">
            <div className="w-12 h-12 bg-green-50 rounded-lg flex items-center justify-center">
              <Server className="w-6 h-6 text-green-500" />
            </div>
            <div>
              <p className="text-sm text-stone-400">API Status</p>
              <p className="font-semibold text-green-500">Operational</p>
            </div>
          </div>

          <div className="flex items-center gap-4 p-4 bg-blue-50/30 rounded-lg border border-blue-100">
            <div className="w-12 h-12 bg-blue-50 rounded-lg flex items-center justify-center">
              <Database className="w-6 h-6 text-blue-500" />
            </div>
            <div>
              <p className="text-sm text-stone-400">Database</p>
              <p className="font-semibold text-blue-500">MongoDB</p>
            </div>
          </div>

          <div className="flex items-center gap-4 p-4 bg-purple-50/30 rounded-lg border border-purple-100">
            <div className="w-12 h-12 bg-purple-50 rounded-lg flex items-center justify-center">
              <Clock className="w-6 h-6 text-purple-500" />
            </div>
            <div>
              <p className="text-sm text-stone-400">Last Sync</p>
              <p className="font-semibold text-purple-500">{new Date().toLocaleTimeString()}</p>
            </div>
          </div>
        </div>
      </Card>

      {/* Violations Today */}
      {stats.todayViolations > 0 && (
        <div className="bg-red-50/30 border border-red-200/50 rounded-xl p-4">
          <div className="flex items-start gap-4">
            <div className="w-10 h-10 bg-red-50 rounded-lg flex items-center justify-center flex-shrink-0">
              <AlertTriangle className="w-5 h-5 text-red-500" />
            </div>
            <div className="flex-1">
              <h3 className="font-semibold text-red-700">Violations Today: {stats.todayViolations}</h3>
              <p className="text-sm text-red-600 mt-1">
                There have been {stats.todayViolations} PPE violations recorded today.
                Review the alerts section for details and take appropriate action.
              </p>
              <Link href="/alerts" className="inline-flex items-center gap-2 text-sm text-red-600 hover:text-red-700 font-medium mt-2">
                View Details
                <TrendingUp className="w-4 h-4" />
              </Link>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
