'use client';

import { useEffect, useState, useCallback } from 'react';
import {
  FileText,
  Download,
  Calendar,
  Clock,
  Users,
  UserCheck,
  UserX,
  Shield,
  AlertTriangle,
  Printer,
  RefreshCw,
  Sun,
  Sunset,
  Moon,
  CheckCircle,
  XCircle,
} from 'lucide-react';
import AppLayout from '@/components/AppLayout';
import { Card, StatCard } from '@/components/Card';
import { Spinner } from '@/components/Loading';
import { dashboardApi, gateEntryApi } from '@/lib/api';
import { formatTime } from '@/lib/utils';
import { useAuthStore } from '@/lib/store';
import type { ShiftInchargeDashboard, GateEntry, ShiftType } from '@/types';

export default function ShiftReportPage() {
  const { user } = useAuthStore();
  const [data, setData] = useState<ShiftInchargeDashboard | null>(null);
  const [entries, setEntries] = useState<GateEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedShift, setSelectedShift] = useState<ShiftType | 'current'>('current');
  const [selectedDate, setSelectedDate] = useState(new Date().toISOString().split('T')[0]);

  const loadData = useCallback(async () => {
    try {
      setLoading(true);
      const [dashboardData, entriesResult] = await Promise.all([
        dashboardApi.getShiftIncharge(),
        gateEntryApi.list({
          start_date: selectedDate,
          end_date: selectedDate,
          limit: 100,
        }),
      ]);
      setData(dashboardData);
      setEntries(entriesResult.entries);
      setError(null);
    } catch (err) {
      setError('Failed to load shift report data');
      console.error(err);
    } finally {
      setLoading(false);
    }
  }, [selectedDate]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const handlePrint = () => {
    window.print();
  };

  const handleExport = () => {
    if (!data) return;

    const reportData = {
      date: selectedDate,
      shift: data.shift_name,
      mine: data.mine_name,
      statistics: data.statistics,
      entries: entries.map(e => ({
        worker: e.worker_name,
        employee_id: e.employee_id,
        type: e.entry_type,
        status: e.status,
        violations: e.violations,
        time: e.timestamp,
      })),
    };

    const blob = new Blob([JSON.stringify(reportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `shift-report-${selectedDate}.json`;
    a.click();
    URL.revokeObjectURL(url);
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

  const shiftIcons: Record<string, React.ReactNode> = {
    day: <Sun className="w-5 h-5 text-yellow-500" />,
    afternoon: <Sunset className="w-5 h-5 text-orange-500" />,
    night: <Moon className="w-5 h-5 text-blue-500" />,
  };

  const stats = data.statistics;
  const violationsEntries = entries.filter(e => e.violations.length > 0);
  const compliantEntries = entries.filter(e => e.violations.length === 0);

  return (
    <AppLayout>
      <div className="space-y-6 print:space-y-4">
        {/* Header - Hide print buttons when printing */}
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 print:hidden">
          <div>
            <h1 className="text-2xl font-bold text-stone-900">Shift Report</h1>
            <p className="text-stone-500 mt-1">Generate and export shift summary reports</p>
          </div>
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2">
              <Calendar className="w-5 h-5 text-stone-400" />
              <input
                type="date"
                value={selectedDate}
                onChange={(e) => setSelectedDate(e.target.value)}
                className="px-3 py-2 border border-stone-300 rounded-lg focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <button
              onClick={loadData}
              className="flex items-center gap-2 px-4 py-2 bg-stone-100 text-stone-700 rounded-lg hover:bg-stone-200"
            >
              <RefreshCw className="w-4 h-4" />
              Refresh
            </button>
            <button
              onClick={handlePrint}
              className="flex items-center gap-2 px-4 py-2 bg-stone-100 text-stone-700 rounded-lg hover:bg-stone-200"
            >
              <Printer className="w-4 h-4" />
              Print
            </button>
            <button
              onClick={handleExport}
              className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
            >
              <Download className="w-4 h-4" />
              Export
            </button>
          </div>
        </div>

        {/* Report Header */}
        <div className="bg-gradient-to-r from-blue-600 to-blue-700 rounded-xl p-6 text-white print:bg-stone-100 print:text-stone-900">
          <div className="flex items-center justify-between">
            <div>
              <div className="flex items-center gap-2 mb-2 print:mb-1">
                <FileText className="w-6 h-6 print:w-5 print:h-5" />
                <span className="text-blue-100 print:text-stone-600">Shift Report</span>
              </div>
              <h2 className="text-2xl font-bold print:text-xl">{data.mine_name}</h2>
              <p className="text-blue-100 print:text-stone-600 mt-1">
                {data.shift_name} • {data.shift_start} - {data.shift_end}
              </p>
            </div>
            <div className="text-right">
              <p className="text-blue-100 print:text-stone-600 text-sm">Report Date</p>
              <p className="text-xl font-bold print:text-lg">{new Date(selectedDate).toLocaleDateString()}</p>
              <p className="text-blue-100 print:text-stone-600 text-sm mt-2">Generated by</p>
              <p className="font-medium">{user?.full_name || 'Unknown'}</p>
            </div>
          </div>
        </div>

        {/* Summary Stats */}
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4 print:grid-cols-5 print:gap-2">
          <div className="bg-blue-50 rounded-xl p-4 text-center print:p-2">
            <Users className="w-8 h-8 text-blue-600 mx-auto mb-2 print:w-6 print:h-6 print:mb-1" />
            <p className="text-2xl font-bold text-stone-900 print:text-lg">{stats.expected_workers}</p>
            <p className="text-xs text-stone-500">Expected</p>
          </div>
          <div className="bg-green-50 rounded-xl p-4 text-center print:p-2">
            <UserCheck className="w-8 h-8 text-green-600 mx-auto mb-2 print:w-6 print:h-6 print:mb-1" />
            <p className="text-2xl font-bold text-stone-900 print:text-lg">{stats.workers_entered}</p>
            <p className="text-xs text-stone-500">Entered</p>
          </div>
          <div className="bg-purple-50 rounded-xl p-4 text-center print:p-2">
            <UserX className="w-8 h-8 text-purple-600 mx-auto mb-2 print:w-6 print:h-6 print:mb-1" />
            <p className="text-2xl font-bold text-stone-900 print:text-lg">{stats.workers_exited}</p>
            <p className="text-xs text-stone-500">Exited</p>
          </div>
          <div className="bg-red-50 rounded-xl p-4 text-center print:p-2">
            <AlertTriangle className="w-8 h-8 text-red-600 mx-auto mb-2 print:w-6 print:h-6 print:mb-1" />
            <p className="text-2xl font-bold text-stone-900 print:text-lg">{stats.violations_this_shift}</p>
            <p className="text-xs text-stone-500">Violations</p>
          </div>
          <div className="bg-cyan-50 rounded-xl p-4 text-center print:p-2">
            <Shield className="w-8 h-8 text-cyan-600 mx-auto mb-2 print:w-6 print:h-6 print:mb-1" />
            <p className="text-2xl font-bold text-stone-900 print:text-lg">{stats.compliance_rate}%</p>
            <p className="text-xs text-stone-500">Compliance</p>
          </div>
        </div>

        {/* PPE Compliance Summary */}
        <Card title="PPE Compliance Summary">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h4 className="text-sm font-medium text-stone-700 mb-4">Compliance Breakdown</h4>
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between mb-1">
                    <span className="text-sm text-stone-600">Compliant Workers</span>
                    <span className="text-sm font-medium text-green-600">{stats.ppe_compliant}</span>
                  </div>
                  <div className="w-full bg-stone-200 rounded-full h-3">
                    <div
                      className="bg-green-500 h-3 rounded-full"
                      style={{ width: `${stats.workers_entered > 0 ? (stats.ppe_compliant / stats.workers_entered) * 100 : 0}%` }}
                    />
                  </div>
                </div>
                <div>
                  <div className="flex justify-between mb-1">
                    <span className="text-sm text-stone-600">Non-Compliant Workers</span>
                    <span className="text-sm font-medium text-red-600">{stats.ppe_non_compliant}</span>
                  </div>
                  <div className="w-full bg-stone-200 rounded-full h-3">
                    <div
                      className="bg-red-500 h-3 rounded-full"
                      style={{ width: `${stats.workers_entered > 0 ? (stats.ppe_non_compliant / stats.workers_entered) * 100 : 0}%` }}
                    />
                  </div>
                </div>
              </div>
            </div>
            <div>
              <h4 className="text-sm font-medium text-stone-700 mb-4">Entry Status</h4>
              <div className="grid grid-cols-3 gap-4 text-center">
                <div className="p-3 bg-green-50 rounded-lg">
                  <CheckCircle className="w-6 h-6 text-green-600 mx-auto mb-1" />
                  <p className="text-lg font-bold text-stone-900">
                    {entries.filter(e => e.status === 'approved').length}
                  </p>
                  <p className="text-xs text-stone-500">Approved</p>
                </div>
                <div className="p-3 bg-red-50 rounded-lg">
                  <XCircle className="w-6 h-6 text-red-600 mx-auto mb-1" />
                  <p className="text-lg font-bold text-stone-900">
                    {entries.filter(e => e.status === 'denied').length}
                  </p>
                  <p className="text-xs text-stone-500">Denied</p>
                </div>
                <div className="p-3 bg-yellow-50 rounded-lg">
                  <AlertTriangle className="w-6 h-6 text-yellow-600 mx-auto mb-1" />
                  <p className="text-lg font-bold text-stone-900">
                    {entries.filter(e => e.status === 'override').length}
                  </p>
                  <p className="text-xs text-stone-500">Override</p>
                </div>
              </div>
            </div>
          </div>
        </Card>

        {/* Violations Log */}
        {violationsEntries.length > 0 && (
          <Card title="Violation Log" description="Workers with PPE violations this shift">
            <div className="overflow-x-auto">
              <table className="w-full print:text-sm">
                <thead>
                  <tr className="border-b border-stone-100">
                    <th className="text-left py-3 px-4 text-xs font-medium text-stone-500 uppercase">Worker</th>
                    <th className="text-left py-3 px-4 text-xs font-medium text-stone-500 uppercase">Violations</th>
                    <th className="text-left py-3 px-4 text-xs font-medium text-stone-500 uppercase">Status</th>
                    <th className="text-left py-3 px-4 text-xs font-medium text-stone-500 uppercase">Time</th>
                  </tr>
                </thead>
                <tbody>
                  {violationsEntries.map((entry) => (
                    <tr key={entry.id} className="border-b border-stone-50">
                      <td className="py-3 px-4">
                        <div>
                          <p className="font-medium text-stone-900">{entry.worker_name || 'Unknown'}</p>
                          <p className="text-xs text-stone-500">{entry.employee_id}</p>
                        </div>
                      </td>
                      <td className="py-3 px-4">
                        <div className="flex flex-wrap gap-1">
                          {entry.violations.map((v, i) => (
                            <span key={i} className="px-2 py-0.5 bg-red-100 text-red-700 rounded text-xs">
                              {v.replace('NO-', '')}
                            </span>
                          ))}
                        </div>
                      </td>
                      <td className="py-3 px-4">
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                          entry.status === 'approved' ? 'bg-green-100 text-green-700' :
                          entry.status === 'override' ? 'bg-yellow-100 text-yellow-700' :
                          'bg-red-100 text-red-700'
                        }`}>
                          {entry.status}
                        </span>
                      </td>
                      <td className="py-3 px-4 text-sm text-stone-500">
                        {formatTime(entry.timestamp)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        )}

        {/* All Entries */}
        <Card title="All Gate Entries" description="Complete log of entries and exits">
          <div className="overflow-x-auto">
            <table className="w-full print:text-sm">
              <thead>
                <tr className="border-b border-stone-100">
                  <th className="text-left py-3 px-4 text-xs font-medium text-stone-500 uppercase">Worker</th>
                  <th className="text-left py-3 px-4 text-xs font-medium text-stone-500 uppercase">Type</th>
                  <th className="text-left py-3 px-4 text-xs font-medium text-stone-500 uppercase">Gate</th>
                  <th className="text-left py-3 px-4 text-xs font-medium text-stone-500 uppercase">PPE Status</th>
                  <th className="text-left py-3 px-4 text-xs font-medium text-stone-500 uppercase">Time</th>
                </tr>
              </thead>
              <tbody>
                {entries.length === 0 ? (
                  <tr>
                    <td colSpan={5} className="text-center py-8 text-stone-500">
                      No entries recorded for this shift
                    </td>
                  </tr>
                ) : (
                  entries.map((entry) => (
                    <tr key={entry.id} className="border-b border-stone-50">
                      <td className="py-3 px-4">
                        <div>
                          <p className="font-medium text-stone-900">{entry.worker_name || 'Unknown'}</p>
                          <p className="text-xs text-stone-500">{entry.employee_id}</p>
                        </div>
                      </td>
                      <td className="py-3 px-4">
                        <span className={`inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium ${
                          entry.entry_type === 'entry'
                            ? 'bg-green-100 text-green-700'
                            : 'bg-blue-100 text-blue-700'
                        }`}>
                          {entry.entry_type === 'entry' ? <UserCheck size={12} /> : <UserX size={12} />}
                          {entry.entry_type.charAt(0).toUpperCase() + entry.entry_type.slice(1)}
                        </span>
                      </td>
                      <td className="py-3 px-4 text-sm text-stone-600">
                        {entry.gate_name || '-'}
                      </td>
                      <td className="py-3 px-4">
                        {entry.violations.length === 0 ? (
                          <span className="text-green-600 text-sm flex items-center gap-1">
                            <CheckCircle size={14} />
                            Compliant
                          </span>
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

        {/* Footer */}
        <div className="text-center text-sm text-stone-500 print:mt-8">
          <p>Report generated on {new Date().toLocaleString()}</p>
          <p className="mt-1">Mine Safety PPE & Attendance System v2.0</p>
        </div>
      </div>
    </AppLayout>
  );
}
