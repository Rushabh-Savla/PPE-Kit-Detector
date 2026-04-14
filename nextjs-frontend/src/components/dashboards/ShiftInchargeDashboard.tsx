'use client';

import { useEffect, useState, useCallback } from 'react';
import {
  Users,
  UserCheck,
  UserX,
  AlertTriangle,
  ShieldCheck,
  Clock,
  DoorOpen,
  RefreshCw,
  CheckCircle,
  XCircle,
  AlertCircle,
  HardHat,
  Battery,
  Heart,
  Siren,
} from 'lucide-react';
import { Card, StatCard } from '@/components/Card';
import { Spinner } from '@/components/Loading';
import { dashboardApi, gateEntryApi, alertApi, helmetApi, HelmetStats, HelmetReading } from '@/lib/api';
import { formatTime } from '@/lib/utils';
import Link from 'next/link';
import type { ShiftInchargeDashboard as DashboardType, GateEntry } from '@/types';

export default function ShiftInchargeDashboard() {
  const [data, setData] = useState<DashboardType | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [helmetStats, setHelmetStats] = useState<HelmetStats | null>(null);
  const [helmetReadings, setHelmetReadings] = useState<HelmetReading[]>([]);

  const loadData = useCallback(async (showRefresh = false) => {
    try {
      if (showRefresh) setRefreshing(true);
      else setLoading(true);

      // Load dashboard data and helmet data in parallel
      const [dashboardData, helmetStatsData, helmetLatest] = await Promise.all([
        dashboardApi.getShiftIncharge(),
        helmetApi.getStats().catch(() => null),
        helmetApi.getLatest().catch(() => ({ readings: [] })),
      ]);

      setData(dashboardData);
      setHelmetStats(helmetStatsData);
      setHelmetReadings(helmetLatest.readings || []);
      setError(null);
    } catch (err) {
      setError('Failed to load dashboard data');
      console.error(err);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    loadData();
    // Auto-refresh every 30 seconds
    const interval = setInterval(() => loadData(true), 30000);
    return () => clearInterval(interval);
  }, [loadData]);

  const handleAcknowledgeAlert = async (alertId: string) => {
    try {
      await alertApi.acknowledge(alertId);
      loadData(true);
    } catch (err) {
      console.error('Failed to acknowledge alert:', err);
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
        <AlertCircle className="w-12 h-12 text-red-500 mx-auto mb-4" />
        <p className="text-red-500 mb-4">{error || 'Failed to load data'}</p>
        <button onClick={() => loadData()} className="btn btn-primary">
          Retry
        </button>
      </div>
    );
  }

  const stats = data.statistics;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-stone-800">Shift Dashboard</h1>
          <p className="text-stone-500 mt-1">
            {data.mine_name} • {data.shift_name} ({data.shift_start} - {data.shift_end})
          </p>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 px-3 py-2 bg-blue-50 rounded-lg">
            <Clock className="w-4 h-4 text-blue-600" />
            <span className="text-sm font-medium text-blue-700">
              Live Monitoring
            </span>
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

      {/* Stats Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-4">
        <StatCard
          title="Expected"
          value={stats.expected_workers}
          icon={<Users size={24} />}
          color="blue"
        />
        <StatCard
          title="Entered"
          value={stats.workers_entered}
          icon={<UserCheck size={24} />}
          color="green"
        />
        <StatCard
          title="Inside Now"
          value={stats.currently_inside}
          icon={<DoorOpen size={24} />}
          color="purple"
        />
        <StatCard
          title="Violations"
          value={stats.violations_this_shift}
          icon={<AlertTriangle size={24} />}
          color="red"
        />
        <StatCard
          title="Compliance"
          value={`${stats.compliance_rate}%`}
          icon={<ShieldCheck size={24} />}
          color={stats.compliance_rate >= 90 ? 'green' : stats.compliance_rate >= 70 ? 'yellow' : 'red'}
        />
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Recent Entries - Takes 2 columns */}
        <div className="lg:col-span-2">
          <Card
            title="Recent Gate Entries"
            description="Real-time worker entries and exits"
            action={
              <span className="text-xs text-stone-500">
                Last updated: {new Date().toLocaleTimeString()}
              </span>
            }
          >
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-stone-100">
                    <th className="text-left py-3 px-4 text-xs font-medium text-stone-500 uppercase">Worker</th>
                    <th className="text-left py-3 px-4 text-xs font-medium text-stone-500 uppercase">Type</th>
                    <th className="text-left py-3 px-4 text-xs font-medium text-stone-500 uppercase">Status</th>
                    <th className="text-left py-3 px-4 text-xs font-medium text-stone-500 uppercase">PPE</th>
                    <th className="text-left py-3 px-4 text-xs font-medium text-stone-500 uppercase">Time</th>
                  </tr>
                </thead>
                <tbody>
                  {data.recent_entries.length === 0 ? (
                    <tr>
                      <td colSpan={5} className="text-center text-stone-500 py-8">
                        No entries this shift yet
                      </td>
                    </tr>
                  ) : (
                    data.recent_entries.map((entry) => (
                      <tr key={entry.id} className="border-b border-stone-50 hover:bg-stone-50">
                        <td className="py-3 px-4">
                          <div>
                            <p className="font-medium text-stone-800">{entry.worker_name || 'Unknown'}</p>
                            <p className="text-xs text-stone-500">{entry.employee_id}</p>
                          </div>
                        </td>
                        <td className="py-3 px-4">
                          <span className={`inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium ${
                            entry.entry_type === 'entry'
                              ? 'bg-green-100 text-green-700'
                              : 'bg-blue-100 text-blue-700'
                          }`}>
                            {entry.entry_type === 'entry' ? (
                              <>
                                <UserCheck size={12} />
                                Entry
                              </>
                            ) : (
                              <>
                                <UserX size={12} />
                                Exit
                              </>
                            )}
                          </span>
                        </td>
                        <td className="py-3 px-4">
                          <span className={`inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium ${
                            entry.status === 'approved'
                              ? 'bg-green-100 text-green-700'
                              : entry.status === 'override'
                              ? 'bg-yellow-100 text-yellow-700'
                              : 'bg-red-100 text-red-700'
                          }`}>
                            {entry.status === 'approved' && <CheckCircle size={12} />}
                            {entry.status === 'denied' && <XCircle size={12} />}
                            {entry.status === 'override' && <AlertCircle size={12} />}
                            {entry.status.charAt(0).toUpperCase() + entry.status.slice(1)}
                          </span>
                        </td>
                        <td className="py-3 px-4">
                          {entry.violations.length === 0 ? (
                            <span className="text-green-600 text-sm">All OK</span>
                          ) : (
                            <div className="flex flex-wrap gap-1">
                              {entry.violations.map((v, i) => (
                                <span key={i} className="px-1.5 py-0.5 bg-red-100 text-red-700 rounded text-xs">
                                  {v.replace('NO-', '')}
                                </span>
                              ))}
                            </div>
                          )}
                        </td>
                        <td className="py-3 px-4 text-sm text-stone-500">
                          {formatTime(entry.timestamp)}
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </Card>
        </div>

        {/* Active Alerts */}
        <div>
          <Card
            title="Active Alerts"
            description={`${stats.pending_alerts} pending`}
          >
            <div className="space-y-3">
              {data.active_alerts.length === 0 ? (
                <div className="text-center py-6 text-stone-500">
                  <CheckCircle className="w-8 h-8 mx-auto mb-2 text-green-500" />
                  <p>No active alerts</p>
                </div>
              ) : (
                data.active_alerts.map((alert) => (
                  <div
                    key={alert.id}
                    className={`p-3 rounded-lg border-l-4 ${
                      alert.severity === 'critical'
                        ? 'bg-red-50 border-red-500'
                        : alert.severity === 'high'
                        ? 'bg-orange-50 border-orange-500'
                        : alert.severity === 'medium'
                        ? 'bg-yellow-50 border-yellow-500'
                        : 'bg-blue-50 border-blue-500'
                    }`}
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium text-stone-800 truncate">
                          {alert.message}
                        </p>
                        <p className="text-xs text-stone-500 mt-1">
                          {formatTime(alert.created_at)}
                        </p>
                      </div>
                      <button
                        onClick={() => handleAcknowledgeAlert(alert.id)}
                        className="text-xs px-2 py-1 bg-white border border-stone-200 rounded hover:bg-stone-50"
                      >
                        Acknowledge
                      </button>
                    </div>
                  </div>
                ))
              )}
            </div>
          </Card>

          {/* Compliance Overview */}
          <Card title="PPE Compliance" className="mt-4">
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm text-stone-600">Compliant</span>
                <span className="font-semibold text-green-600">{stats.ppe_compliant}</span>
              </div>
              <div className="w-full bg-stone-200 rounded-full h-2">
                <div
                  className="bg-green-500 h-2 rounded-full transition-all"
                  style={{
                    width: `${stats.workers_entered > 0 ? (stats.ppe_compliant / stats.workers_entered) * 100 : 0}%`
                  }}
                />
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-stone-600">Non-Compliant</span>
                <span className="font-semibold text-red-600">{stats.ppe_non_compliant}</span>
              </div>
              <div className="w-full bg-stone-200 rounded-full h-2">
                <div
                  className="bg-red-500 h-2 rounded-full transition-all"
                  style={{
                    width: `${stats.workers_entered > 0 ? (stats.ppe_non_compliant / stats.workers_entered) * 100 : 0}%`
                  }}
                />
              </div>
            </div>
          </Card>

          {/* Helmet Status Widget */}
          <Card title="Smart Helmet Status" className="mt-4">
            <div className="space-y-3">
              {/* Summary Stats */}
              <div className="grid grid-cols-2 gap-3">
                <div className="flex items-center gap-2 p-2 bg-orange-50 rounded-lg">
                  <HardHat className="w-5 h-5 text-orange-600" />
                  <div>
                    <p className="text-xs text-stone-500">Active</p>
                    <p className="font-semibold text-stone-800">{helmetStats?.active_helmets || helmetReadings.length}</p>
                  </div>
                </div>
                <div className="flex items-center gap-2 p-2 bg-green-50 rounded-lg">
                  <Battery className="w-5 h-5 text-green-600" />
                  <div>
                    <p className="text-xs text-stone-500">Avg Battery</p>
                    <p className="font-semibold text-stone-800">{helmetStats?.avg_battery_voltage?.toFixed(2) || '0.00'}V</p>
                  </div>
                </div>
              </div>

              {/* Active Helmets List */}
              {helmetReadings.length === 0 ? (
                <div className="text-center py-4 text-stone-500">
                  <HardHat className="w-8 h-8 mx-auto mb-2 text-stone-300" />
                  <p className="text-sm">No active helmets</p>
                </div>
              ) : (
                <div className="space-y-2">
                  {helmetReadings.slice(0, 3).map((reading, idx) => (
                    <div key={idx} className={`flex items-center justify-between p-2 rounded-lg border-l-2 ${
                      reading.sos_active ? 'bg-red-50 border-red-500' :
                      reading.severity === 'critical' ? 'bg-red-50 border-red-500' :
                      reading.severity === 'high' ? 'bg-orange-50 border-orange-500' :
                      reading.battery_low ? 'bg-yellow-50 border-yellow-500' :
                      'bg-green-50 border-green-500'
                    }`}>
                      <div className="flex items-center gap-2">
                        <span className="font-medium text-sm text-stone-700">{reading.worker_name}</span>
                        {reading.sos_active && (
                          <Siren className="w-4 h-4 text-red-500 animate-pulse" />
                        )}
                      </div>
                      <div className="flex items-center gap-3 text-xs text-stone-500">
                        <span className="flex items-center gap-1">
                          <Heart className="w-3 h-3" />
                          {reading.heart_rate}
                        </span>
                        <span className={reading.battery_low ? 'text-red-600' : ''}>
                          {reading.battery_voltage.toFixed(1)}V
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {/* SOS Alert Banner */}
              {helmetStats?.sos_active_count && helmetStats.sos_active_count > 0 && (
                <div className="flex items-center gap-2 p-2 bg-red-100 rounded-lg animate-pulse">
                  <Siren className="w-5 h-5 text-red-600" />
                  <span className="text-sm font-medium text-red-700">
                    {helmetStats.sos_active_count} SOS Alert{helmetStats.sos_active_count > 1 ? 's' : ''} Active!
                  </span>
                </div>
              )}

              {/* View More Link */}
              <Link
                href="/helmet-monitoring"
                className="block text-center text-sm text-orange-600 hover:text-orange-700 font-medium py-2 border-t border-stone-100"
              >
                View All Helmet Data →
              </Link>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
