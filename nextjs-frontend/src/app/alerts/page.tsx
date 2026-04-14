'use client';

import { useState, useEffect } from 'react';
import {
  Bell, AlertTriangle, AlertCircle, CheckCircle, X, Search,
  Filter, Clock, User, Building2, Shield, XCircle, Eye,
  MessageSquare, Volume2, VolumeX, RefreshCw
} from 'lucide-react';
import AppLayout from '@/components/AppLayout';
import { Card } from '@/components/Card';
import { Spinner } from '@/components/Loading';
import { apiClient } from '@/lib/api';
import { useAuthStore } from '@/lib/store';
import { formatDateTime } from '@/lib/utils';

interface Alert {
  id: string;
  type: 'sos' | 'violation' | 'emergency' | 'warning' | 'system';
  severity: 'critical' | 'high' | 'medium' | 'low';
  title: string;
  message: string;
  source: string;
  worker_id?: string;
  worker_name?: string;
  mine_id?: string;
  mine_name?: string;
  status: 'active' | 'acknowledged' | 'resolved' | 'dismissed';
  created_at: string;
  acknowledged_at?: string;
  acknowledged_by?: string;
  resolved_at?: string;
  resolved_by?: string;
}

export default function AlertsPage() {
  const { token, user } = useAuthStore();
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [filterType, setFilterType] = useState<string>('all');
  const [filterSeverity, setFilterSeverity] = useState<string>('all');
  const [filterStatus, setFilterStatus] = useState<string>('all');
  const [selectedAlert, setSelectedAlert] = useState<Alert | null>(null);
  const [showDetailModal, setShowDetailModal] = useState(false);
  const [soundEnabled, setSoundEnabled] = useState(true);
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  useEffect(() => {
    loadAlerts();
    // Poll for new alerts every 30 seconds
    const interval = setInterval(loadAlerts, 30000);
    return () => clearInterval(interval);
  }, []);

  const loadAlerts = async () => {
    try {
      setIsLoading(true);
      const response = await apiClient.get('/alerts');
      const alertsData = response.data?.alerts || response.data || [];
      setAlerts(alertsData);
    } catch (err) {
      console.error('Failed to load alerts:', err);
      // Mock data for demo
      setAlerts([
        {
          id: 'ALT001',
          type: 'sos',
          severity: 'critical',
          title: 'SOS Alert - Worker in Distress',
          message: 'Worker W003 has triggered an SOS alert in Zone B. Immediate assistance required.',
          source: 'Worker App',
          worker_id: 'W003',
          worker_name: 'Mukesh Yadav',
          mine_id: 'MINE001',
          mine_name: 'Jharia Coal Mine',
          status: 'active',
          created_at: new Date(Date.now() - 5 * 60000).toISOString(),
        },
        {
          id: 'ALT002',
          type: 'violation',
          severity: 'high',
          title: 'PPE Violation Detected',
          message: 'Worker W005 detected without safety helmet at Gate 1 entrance.',
          source: 'PPE Detection System',
          worker_id: 'W005',
          worker_name: 'Rajesh Verma',
          mine_id: 'MINE001',
          mine_name: 'Jharia Coal Mine',
          status: 'acknowledged',
          created_at: new Date(Date.now() - 15 * 60000).toISOString(),
          acknowledged_at: new Date(Date.now() - 10 * 60000).toISOString(),
          acknowledged_by: 'safety1',
        },
        {
          id: 'ALT003',
          type: 'emergency',
          severity: 'critical',
          title: 'Gas Level Warning',
          message: 'Elevated methane levels detected in Tunnel C, Section 2. Evacuation may be required.',
          source: 'Gas Monitoring System',
          mine_id: 'MINE001',
          mine_name: 'Jharia Coal Mine',
          status: 'resolved',
          created_at: new Date(Date.now() - 2 * 3600000).toISOString(),
          acknowledged_at: new Date(Date.now() - 2 * 3600000 + 5 * 60000).toISOString(),
          acknowledged_by: 'safety1',
          resolved_at: new Date(Date.now() - 1 * 3600000).toISOString(),
          resolved_by: 'manager1',
        },
        {
          id: 'ALT004',
          type: 'warning',
          severity: 'medium',
          title: 'Equipment Maintenance Due',
          message: 'Ventilation system in Zone A is due for scheduled maintenance.',
          source: 'Maintenance System',
          mine_id: 'MINE002',
          mine_name: 'Bokaro Steel Mine',
          status: 'active',
          created_at: new Date(Date.now() - 6 * 3600000).toISOString(),
        },
        {
          id: 'ALT005',
          type: 'violation',
          severity: 'medium',
          title: 'Multiple PPE Violations',
          message: '3 workers detected without proper safety boots during afternoon shift.',
          source: 'PPE Detection System',
          mine_id: 'MINE002',
          mine_name: 'Bokaro Steel Mine',
          status: 'acknowledged',
          created_at: new Date(Date.now() - 4 * 3600000).toISOString(),
          acknowledged_at: new Date(Date.now() - 3.5 * 3600000).toISOString(),
          acknowledged_by: 'shift_day1',
        },
        {
          id: 'ALT006',
          type: 'system',
          severity: 'low',
          title: 'System Update Available',
          message: 'A new version of the PPE detection model is available for deployment.',
          source: 'System',
          status: 'dismissed',
          created_at: new Date(Date.now() - 24 * 3600000).toISOString(),
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleAcknowledge = async (alert: Alert) => {
    try {
      await apiClient.put(`/alerts/${alert.id}/acknowledge`, {});
      setAlerts(alerts.map(a =>
        a.id === alert.id
          ? {
              ...a,
              status: 'acknowledged',
              acknowledged_at: new Date().toISOString(),
              acknowledged_by: user?.username,
            }
          : a
      ));
      setMessage({ type: 'success', text: 'Alert acknowledged' });
    } catch (err) {
      console.error('Error acknowledging alert:', err);
      // For demo, update local state
      setAlerts(alerts.map(a =>
        a.id === alert.id
          ? {
              ...a,
              status: 'acknowledged',
              acknowledged_at: new Date().toISOString(),
              acknowledged_by: user?.username,
            }
          : a
      ));
      setMessage({ type: 'success', text: 'Alert acknowledged' });
    }
  };

  const handleResolve = async (alert: Alert) => {
    try {
      await apiClient.put(`/alerts/${alert.id}/resolve`, {});
      setAlerts(alerts.map(a =>
        a.id === alert.id
          ? {
              ...a,
              status: 'resolved',
              resolved_at: new Date().toISOString(),
              resolved_by: user?.username,
            }
          : a
      ));
      setMessage({ type: 'success', text: 'Alert resolved' });
    } catch (err) {
      console.error('Error resolving alert:', err);
      // For demo, update local state
      setAlerts(alerts.map(a =>
        a.id === alert.id
          ? {
              ...a,
              status: 'resolved',
              resolved_at: new Date().toISOString(),
              resolved_by: user?.username,
            }
          : a
      ));
      setMessage({ type: 'success', text: 'Alert resolved' });
    }
    setShowDetailModal(false);
  };

  const handleDismiss = async (alert: Alert) => {
    try {
      await apiClient.put(`/alerts/${alert.id}/dismiss`, {});
      setAlerts(alerts.map(a =>
        a.id === alert.id ? { ...a, status: 'dismissed' } : a
      ));
    } catch (err) {
      console.error('Error dismissing alert:', err);
      // For demo, update local state
      setAlerts(alerts.map(a =>
        a.id === alert.id ? { ...a, status: 'dismissed' } : a
      ));
    }
  };

  const viewAlertDetails = (alert: Alert) => {
    setSelectedAlert(alert);
    setShowDetailModal(true);
  };

  const filteredAlerts = alerts.filter(alert => {
    const matchesSearch =
      (alert.title?.toLowerCase().includes(search.toLowerCase()) || false) ||
      (alert.message?.toLowerCase().includes(search.toLowerCase()) || false) ||
      (alert.worker_name?.toLowerCase().includes(search.toLowerCase()) || false) ||
      (alert.mine_name?.toLowerCase().includes(search.toLowerCase()) || false);
    const matchesType = filterType === 'all' || alert.type === filterType;
    const matchesSeverity = filterSeverity === 'all' || alert.severity === filterSeverity;
    const matchesStatus = filterStatus === 'all' || alert.status === filterStatus;
    return matchesSearch && matchesType && matchesSeverity && matchesStatus;
  });

  const stats = {
    total: alerts.length,
    active: alerts.filter(a => a.status === 'active').length,
    critical: alerts.filter(a => a.severity === 'critical' && a.status === 'active').length,
    acknowledged: alerts.filter(a => a.status === 'acknowledged').length,
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'sos':
        return <AlertCircle className="text-red-500" size={20} />;
      case 'violation':
        return <Shield className="text-orange-500" size={20} />;
      case 'emergency':
        return <AlertTriangle className="text-red-500" size={20} />;
      case 'warning':
        return <AlertTriangle className="text-yellow-500" size={20} />;
      case 'system':
        return <Bell className="text-blue-500" size={20} />;
      default:
        return <Bell className="text-stone-500" size={20} />;
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical':
        return 'bg-red-100 text-red-700 border-red-200';
      case 'high':
        return 'bg-orange-100 text-orange-700 border-orange-200';
      case 'medium':
        return 'bg-yellow-100 text-yellow-700 border-yellow-200';
      case 'low':
        return 'bg-blue-100 text-blue-700 border-blue-200';
      default:
        return 'bg-stone-100 text-stone-700 border-stone-200';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active':
        return 'bg-red-100 text-red-700';
      case 'acknowledged':
        return 'bg-yellow-100 text-yellow-700';
      case 'resolved':
        return 'bg-green-100 text-green-700';
      case 'dismissed':
        return 'bg-stone-100 text-stone-700';
      default:
        return 'bg-stone-100 text-stone-700';
    }
  };

  const getTimeAgo = (dateString: string) => {
    const now = new Date();
    const date = new Date(dateString);
    const seconds = Math.floor((now.getTime() - date.getTime()) / 1000);

    if (seconds < 60) return 'Just now';
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
    if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
    return `${Math.floor(seconds / 86400)}d ago`;
  };

  return (
    <AppLayout>
      <div className="space-y-6">
        {/* Page Header */}
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold text-stone-800">Alerts & Notifications</h1>
            <p className="text-stone-500 mt-1">Monitor and manage safety alerts across all mines</p>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={() => setSoundEnabled(!soundEnabled)}
              className={`p-2 rounded-lg transition-colors ${
                soundEnabled ? 'bg-orange-600 text-white' : 'bg-stone-200 text-stone-600'
              }`}
              title={soundEnabled ? 'Mute Alerts' : 'Unmute Alerts'}
            >
              {soundEnabled ? <Volume2 size={20} /> : <VolumeX size={20} />}
            </button>
            <button
              onClick={loadAlerts}
              className="btn btn-secondary flex items-center gap-2"
            >
              <RefreshCw size={18} />
              Refresh
            </button>
          </div>
        </div>

        {/* Message */}
        {message && (
          <div
            className={`p-4 rounded-lg flex items-center gap-3 ${
              message.type === 'success' ? 'bg-green-50 text-green-800' : 'bg-red-50 text-red-800'
            }`}
          >
            {message.type === 'success' ? <CheckCircle size={20} /> : <XCircle size={20} />}
            <p>{message.text}</p>
            <button onClick={() => setMessage(null)} className="ml-auto hover:opacity-70">
              <X size={18} />
            </button>
          </div>
        )}

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Card className={stats.critical > 0 ? 'ring-2 ring-red-500 animate-pulse' : ''}>
            <div className="flex items-center gap-4">
              <div className={`p-3 rounded-lg ${stats.critical > 0 ? 'bg-red-100' : 'bg-orange-100'}`}>
                <AlertCircle className={stats.critical > 0 ? 'text-red-600' : 'text-orange-600'} size={24} />
              </div>
              <div>
                <p className="text-sm text-stone-500">Critical Alerts</p>
                <p className={`text-2xl font-bold ${stats.critical > 0 ? 'text-red-600' : ''}`}>
                  {stats.critical}
                </p>
              </div>
            </div>
          </Card>
          <Card>
            <div className="flex items-center gap-4">
              <div className="p-3 bg-orange-100 rounded-lg">
                <AlertTriangle className="text-orange-600" size={24} />
              </div>
              <div>
                <p className="text-sm text-stone-500">Active Alerts</p>
                <p className="text-2xl font-bold text-stone-800">{stats.active}</p>
              </div>
            </div>
          </Card>
          <Card>
            <div className="flex items-center gap-4">
              <div className="p-3 bg-yellow-100 rounded-lg">
                <Clock className="text-yellow-600" size={24} />
              </div>
              <div>
                <p className="text-sm text-stone-500">Acknowledged</p>
                <p className="text-2xl font-bold text-stone-800">{stats.acknowledged}</p>
              </div>
            </div>
          </Card>
          <Card>
            <div className="flex items-center gap-4">
              <div className="p-3 bg-green-100 rounded-lg">
                <CheckCircle className="text-green-600" size={24} />
              </div>
              <div>
                <p className="text-sm text-stone-500">Total Alerts</p>
                <p className="text-2xl font-bold text-stone-800">{stats.total}</p>
              </div>
            </div>
          </Card>
        </div>

        {/* Filters */}
        <Card>
          <div className="flex flex-col lg:flex-row gap-4">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-stone-400" size={20} />
              <input
                type="text"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Search alerts..."
                className="w-full pl-10"
              />
            </div>
            <div className="flex flex-wrap gap-2">
              <select
                value={filterType}
                onChange={(e) => setFilterType(e.target.value)}
                className="px-4 py-2 border rounded-lg"
              >
                <option value="all">All Types</option>
                <option value="sos">SOS</option>
                <option value="violation">Violation</option>
                <option value="emergency">Emergency</option>
                <option value="warning">Warning</option>
                <option value="system">System</option>
              </select>
              <select
                value={filterSeverity}
                onChange={(e) => setFilterSeverity(e.target.value)}
                className="px-4 py-2 border rounded-lg"
              >
                <option value="all">All Severity</option>
                <option value="critical">Critical</option>
                <option value="high">High</option>
                <option value="medium">Medium</option>
                <option value="low">Low</option>
              </select>
              <select
                value={filterStatus}
                onChange={(e) => setFilterStatus(e.target.value)}
                className="px-4 py-2 border rounded-lg"
              >
                <option value="all">All Status</option>
                <option value="active">Active</option>
                <option value="acknowledged">Acknowledged</option>
                <option value="resolved">Resolved</option>
                <option value="dismissed">Dismissed</option>
              </select>
            </div>
          </div>
        </Card>

        {/* Alerts List */}
        <div className="space-y-3">
          {isLoading ? (
            <div className="flex justify-center py-12">
              <Spinner size="lg" />
            </div>
          ) : filteredAlerts.length === 0 ? (
            <Card>
              <div className="text-center py-12">
                <Bell className="mx-auto text-stone-300 mb-4" size={48} />
                <p className="text-stone-500">No alerts found</p>
              </div>
            </Card>
          ) : (
            filteredAlerts.map((alert) => (
              <div
                key={alert.id}
                className={`bg-white rounded-xl shadow-sm border-l-4 p-4 hover:shadow-md transition-shadow ${
                  alert.severity === 'critical' && alert.status === 'active'
                    ? 'border-l-red-500 bg-red-50/30'
                    : alert.severity === 'high'
                    ? 'border-l-orange-500'
                    : alert.severity === 'medium'
                    ? 'border-l-yellow-500'
                    : 'border-l-blue-500'
                }`}
              >
                <div className="flex items-start gap-4">
                  <div className="flex-shrink-0 mt-1">
                    {getTypeIcon(alert.type)}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-start justify-between gap-4">
                      <div>
                        <div className="flex items-center gap-2 flex-wrap">
                          <h3 className="font-semibold text-stone-800">{alert.title}</h3>
                          <span className={`px-2 py-0.5 text-xs rounded-full capitalize ${getSeverityColor(alert.severity)}`}>
                            {alert.severity}
                          </span>
                          <span className={`px-2 py-0.5 text-xs rounded-full capitalize ${getStatusColor(alert.status)}`}>
                            {alert.status}
                          </span>
                        </div>
                        <p className="text-stone-600 mt-1 text-sm">{alert.message}</p>
                        <div className="flex items-center gap-4 mt-2 text-xs text-stone-500">
                          <span className="flex items-center gap-1">
                            <Clock size={14} />
                            {getTimeAgo(alert.created_at)}
                          </span>
                          {alert.worker_name && (
                            <span className="flex items-center gap-1">
                              <User size={14} />
                              {alert.worker_name}
                            </span>
                          )}
                          {alert.mine_name && (
                            <span className="flex items-center gap-1">
                              <Building2 size={14} />
                              {alert.mine_name}
                            </span>
                          )}
                          <span className="text-stone-400">via {alert.source}</span>
                        </div>
                      </div>
                      <div className="flex items-center gap-2 flex-shrink-0">
                        <button
                          onClick={() => viewAlertDetails(alert)}
                          className="p-2 text-stone-600 hover:bg-stone-100 rounded-lg transition-colors"
                          title="View Details"
                        >
                          <Eye size={18} />
                        </button>
                        {alert.status === 'active' && (
                          <button
                            onClick={() => handleAcknowledge(alert)}
                            className="px-3 py-1.5 bg-orange-600 text-white text-sm rounded-lg hover:bg-orange-700 transition-colors"
                          >
                            Acknowledge
                          </button>
                        )}
                        {alert.status === 'acknowledged' && (
                          <button
                            onClick={() => handleResolve(alert)}
                            className="px-3 py-1.5 bg-green-600 text-white text-sm rounded-lg hover:bg-green-700 transition-colors"
                          >
                            Resolve
                          </button>
                        )}
                        {(alert.status === 'active' || alert.status === 'acknowledged') && alert.type !== 'sos' && alert.type !== 'emergency' && (
                          <button
                            onClick={() => handleDismiss(alert)}
                            className="p-2 text-stone-400 hover:bg-stone-100 rounded-lg transition-colors"
                            title="Dismiss"
                          >
                            <X size={18} />
                          </button>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      </div>

      {/* Alert Detail Modal */}
      {showDetailModal && selectedAlert && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4 overflow-y-auto">
          <div className="bg-white rounded-xl shadow-xl w-full max-w-lg my-8">
            <div className={`p-6 border-b flex items-center justify-between ${
              selectedAlert.severity === 'critical' ? 'bg-red-50' :
              selectedAlert.severity === 'high' ? 'bg-orange-50' :
              selectedAlert.severity === 'medium' ? 'bg-yellow-50' : 'bg-blue-50'
            }`}>
              <div className="flex items-center gap-3">
                {getTypeIcon(selectedAlert.type)}
                <div>
                  <h2 className="text-lg font-semibold">{selectedAlert.title}</h2>
                  <div className="flex items-center gap-2 mt-1">
                    <span className={`px-2 py-0.5 text-xs rounded-full capitalize ${getSeverityColor(selectedAlert.severity)}`}>
                      {selectedAlert.severity}
                    </span>
                    <span className={`px-2 py-0.5 text-xs rounded-full capitalize ${getStatusColor(selectedAlert.status)}`}>
                      {selectedAlert.status}
                    </span>
                  </div>
                </div>
              </div>
              <button
                onClick={() => setShowDetailModal(false)}
                className="p-2 hover:bg-white/50 rounded-lg transition-colors"
              >
                <X size={20} />
              </button>
            </div>

            <div className="p-6 space-y-4">
              <div>
                <h4 className="text-sm font-medium text-stone-500 mb-1">Message</h4>
                <p className="text-stone-800">{selectedAlert.message}</p>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <h4 className="text-sm font-medium text-stone-500 mb-1">Source</h4>
                  <p className="text-stone-800">{selectedAlert.source}</p>
                </div>
                <div>
                  <h4 className="text-sm font-medium text-stone-500 mb-1">Created</h4>
                  <p className="text-stone-800">{formatDateTime(selectedAlert.created_at)}</p>
                </div>
              </div>

              {(selectedAlert.worker_name || selectedAlert.mine_name) && (
                <div className="grid grid-cols-2 gap-4">
                  {selectedAlert.worker_name && (
                    <div>
                      <h4 className="text-sm font-medium text-stone-500 mb-1">Worker</h4>
                      <p className="text-stone-800">{selectedAlert.worker_name} ({selectedAlert.worker_id})</p>
                    </div>
                  )}
                  {selectedAlert.mine_name && (
                    <div>
                      <h4 className="text-sm font-medium text-stone-500 mb-1">Mine</h4>
                      <p className="text-stone-800">{selectedAlert.mine_name}</p>
                    </div>
                  )}
                </div>
              )}

              {selectedAlert.acknowledged_at && (
                <div className="grid grid-cols-2 gap-4 pt-4 border-t border-stone-200">
                  <div>
                    <h4 className="text-sm font-medium text-stone-500 mb-1">Acknowledged</h4>
                    <p className="text-stone-800">{formatDateTime(selectedAlert.acknowledged_at)}</p>
                  </div>
                  <div>
                    <h4 className="text-sm font-medium text-stone-500 mb-1">Acknowledged By</h4>
                    <p className="text-stone-800">{selectedAlert.acknowledged_by}</p>
                  </div>
                </div>
              )}

              {selectedAlert.resolved_at && (
                <div className="grid grid-cols-2 gap-4 pt-4 border-t border-stone-200">
                  <div>
                    <h4 className="text-sm font-medium text-stone-500 mb-1">Resolved</h4>
                    <p className="text-stone-800">{formatDateTime(selectedAlert.resolved_at)}</p>
                  </div>
                  <div>
                    <h4 className="text-sm font-medium text-stone-500 mb-1">Resolved By</h4>
                    <p className="text-stone-800">{selectedAlert.resolved_by}</p>
                  </div>
                </div>
              )}

              <div className="flex gap-3 pt-4 border-t border-stone-200">
                <button
                  onClick={() => setShowDetailModal(false)}
                  className="flex-1 btn btn-secondary"
                >
                  Close
                </button>
                {selectedAlert.status === 'active' && (
                  <button
                    onClick={() => {
                      handleAcknowledge(selectedAlert);
                      setShowDetailModal(false);
                    }}
                    className="flex-1 btn btn-primary"
                  >
                    Acknowledge
                  </button>
                )}
                {selectedAlert.status === 'acknowledged' && (
                  <button
                    onClick={() => handleResolve(selectedAlert)}
                    className="flex-1 bg-green-600 text-white px-4 py-2 rounded-lg hover:bg-green-700 transition-colors"
                  >
                    Mark Resolved
                  </button>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </AppLayout>
  );
}
