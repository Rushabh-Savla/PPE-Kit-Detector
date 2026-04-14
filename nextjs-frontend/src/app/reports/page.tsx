'use client';

import { useState, useEffect } from 'react';
import { FileText, Calendar, Download, Users, AlertTriangle, Clock, Settings, Plus, Siren } from 'lucide-react';
import AppLayout from '@/components/AppLayout';
import { Card } from '@/components/Card';
import { Spinner } from '@/components/Loading';
import { reportsApi, employeeApi } from '@/lib/api';
import { formatDate, formatTime, getDateString } from '@/lib/utils';
import { ReportGenerator, ScheduleList, ReportScheduleModal } from '@/components/reports';
import type {
  AttendanceReport,
  ViolationReport,
  Employee,
  ReportTypeInfo,
  ReportSchedule,
  CreateScheduleRequest
} from '@/types';
import { useAuthStore } from '@/lib/store';

type TabType = 'generate' | 'schedules' | 'legacy';

export default function ReportsPage() {
  const { user } = useAuthStore();
  const [activeTab, setActiveTab] = useState<TabType>('generate');

  // New report state
  const [reportTypes, setReportTypes] = useState<ReportTypeInfo[]>([]);
  const [schedules, setSchedules] = useState<ReportSchedule[]>([]);
  const [schedulesLoading, setSchedulesLoading] = useState(true);
  const [showScheduleModal, setShowScheduleModal] = useState(false);
  const [editingSchedule, setEditingSchedule] = useState<ReportSchedule | undefined>();

  // Emergency report state
  const [emergencyReportLoading, setEmergencyReportLoading] = useState(false);

  // Legacy report state
  const [legacyReportType, setLegacyReportType] = useState<'attendance' | 'violations'>('attendance');
  const [startDate, setStartDate] = useState(getDateString(new Date(Date.now() - 7 * 24 * 60 * 60 * 1000)));
  const [endDate, setEndDate] = useState(getDateString());
  const [selectedEmployee, setSelectedEmployee] = useState<string>('');
  const [isLoading, setIsLoading] = useState(false);
  const [attendanceReport, setAttendanceReport] = useState<AttendanceReport | null>(null);
  const [violationReport, setViolationReport] = useState<ViolationReport | null>(null);
  const [employees, setEmployees] = useState<Employee[]>([]);

  // Load report types and schedules on mount
  useEffect(() => {
    loadReportTypes();
    loadSchedules();
    loadEmployees();
  }, []);

  const loadReportTypes = async () => {
    try {
      const data = await reportsApi.getReportTypes();
      setReportTypes(data.report_types);
    } catch (err) {
      console.error('Failed to load report types, using mock data:', err);
      // Mock report types when API fails - using valid ReportType values
      setReportTypes([
        {
          id: 'shift_summary',
          name: 'Shift Summary Report',
          description: 'Comprehensive shift summary with attendance and compliance data',
          available_formats: ['pdf', 'excel', 'csv'],
          min_role: 'shift_incharge',
          parameters: ['shift', 'date'],
        },
        {
          id: 'weekly_compliance',
          name: 'Weekly Compliance Report',
          description: 'Weekly PPE compliance summary across all zones and workers',
          available_formats: ['pdf', 'excel'],
          min_role: 'safety_officer',
          parameters: ['week', 'zone'],
        },
        {
          id: 'violation_trends',
          name: 'Violation Trends Report',
          description: 'Analysis of PPE violation patterns and trends over time',
          available_formats: ['pdf', 'excel', 'csv'],
          min_role: 'safety_officer',
          parameters: ['date_range'],
        },
        {
          id: 'daily_operations',
          name: 'Daily Operations Report',
          description: 'Daily operational summary including entries, exits, and alerts',
          available_formats: ['pdf'],
          min_role: 'manager',
          parameters: ['date'],
        },
        {
          id: 'executive_summary',
          name: 'Executive Summary Report',
          description: 'High-level overview for management with KPIs and compliance metrics',
          available_formats: ['pdf', 'excel'],
          min_role: 'general_manager',
          parameters: ['date_range', 'mine'],
        },
        {
          id: 'critical_incidents',
          name: 'Critical Incidents Report',
          description: 'Summary of critical safety incidents and emergency responses',
          available_formats: ['pdf', 'excel'],
          min_role: 'area_safety_officer',
          parameters: ['date_range'],
        },
      ]);
    }
  };

  const loadSchedules = async () => {
    setSchedulesLoading(true);
    try {
      const data = await reportsApi.getSchedules();
      setSchedules(data.schedules);
    } catch (err) {
      console.error('Failed to load schedules, using mock data:', err);
      // Mock schedules when API fails - using correct ReportSchedule structure
      setSchedules([
        {
          id: 'sched-001',
          name: 'Daily Shift Summary',
          report_type: 'shift_summary',
          role_target: 'manager',
          created_by: 'admin',
          mine_id: 'mine-001',
          schedule: {
            frequency: 'daily',
            time: '18:00',
          },
          recipients: [
            { email: 'manager@jhariamine.in', name: 'Mine Manager', type: 'to' },
            { email: 'safety@jhariamine.in', name: 'Safety Officer', type: 'cc' },
          ],
          config: {
            format: ['pdf'],
            date_range: 'previous_shift',
          },
          is_active: true,
          created_at: '2024-01-15T00:00:00Z',
          last_run: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(),
          next_run: new Date(Date.now() + 6 * 60 * 60 * 1000).toISOString(),
        },
        {
          id: 'sched-002',
          name: 'Weekly Compliance Report',
          report_type: 'weekly_compliance',
          role_target: 'safety_officer',
          created_by: 'admin',
          schedule: {
            frequency: 'weekly',
            time: '09:00',
            day_of_week: 1,
          },
          recipients: [
            { email: 'hr@coalmines.gov.in', name: 'HR Department', type: 'to' },
          ],
          config: {
            format: ['excel'],
            date_range: 'previous_week',
          },
          is_active: true,
          created_at: '2024-02-01T00:00:00Z',
          last_run: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString(),
          next_run: new Date(Date.now() + 2 * 24 * 60 * 60 * 1000).toISOString(),
        },
        {
          id: 'sched-003',
          name: 'Monthly Executive Summary',
          report_type: 'executive_summary',
          role_target: 'general_manager',
          created_by: 'admin',
          schedule: {
            frequency: 'monthly',
            time: '10:00',
            day_of_month: 1,
          },
          recipients: [
            { email: 'gm@coalmines.gov.in', name: 'General Manager', type: 'to' },
            { email: 'safety.head@coalmines.gov.in', name: 'Safety Head', type: 'cc' },
          ],
          config: {
            format: ['pdf'],
            date_range: 'previous_month',
          },
          is_active: false,
          created_at: '2024-03-01T00:00:00Z',
          next_run: new Date(Date.now() + 20 * 24 * 60 * 60 * 1000).toISOString(),
        },
      ]);
    } finally {
      setSchedulesLoading(false);
    }
  };

  const loadEmployees = async () => {
    try {
      const data = await employeeApi.list(0, 100);
      setEmployees(data.employees);
    } catch (err) {
      console.error('Failed to load employees, using mock data:', err);
      // Mock employees when API fails
      setEmployees([
        { id: 'emp-001', employee_id: 'WRK001', name: 'Rajesh Kumar', department: 'Underground Mining', face_registered: true, created_at: '2024-01-15T00:00:00Z' },
        { id: 'emp-002', employee_id: 'WRK002', name: 'Suresh Yadav', department: 'Extraction', face_registered: true, created_at: '2024-01-20T00:00:00Z' },
        { id: 'emp-003', employee_id: 'WRK003', name: 'Mukesh Singh', department: 'Underground Mining', face_registered: true, created_at: '2024-01-25T00:00:00Z' },
        { id: 'emp-004', employee_id: 'WRK004', name: 'Anil Verma', department: 'Processing', face_registered: true, created_at: '2024-02-01T00:00:00Z' },
        { id: 'emp-005', employee_id: 'WRK005', name: 'Vikram Tiwari', department: 'Extraction', face_registered: false, created_at: '2024-02-05T00:00:00Z' },
        { id: 'emp-006', employee_id: 'WRK006', name: 'Ramesh Sharma', department: 'Underground Mining', face_registered: true, created_at: '2024-02-10T00:00:00Z' },
        { id: 'emp-007', employee_id: 'WRK007', name: 'Dinesh Patel', department: 'Maintenance', face_registered: true, created_at: '2024-02-15T00:00:00Z' },
        { id: 'emp-008', employee_id: 'WRK008', name: 'Prakash Jha', department: 'Safety', face_registered: true, created_at: '2024-02-20T00:00:00Z' },
      ]);
    }
  };

  const handleCreateSchedule = async (schedule: CreateScheduleRequest) => {
    await reportsApi.createSchedule(schedule);
    await loadSchedules();
  };

  const handleEditSchedule = (schedule: ReportSchedule) => {
    setEditingSchedule(schedule);
    setShowScheduleModal(true);
  };

  const handleDeleteSchedule = async (scheduleId: string) => {
    try {
      await reportsApi.deleteSchedule(scheduleId);
      await loadSchedules();
    } catch (err) {
      console.error('Failed to delete schedule:', err);
    }
  };

  const handleToggleSchedule = async (scheduleId: string, isActive: boolean) => {
    try {
      await reportsApi.updateSchedule(scheduleId, { is_active: isActive });
      await loadSchedules();
    } catch (err) {
      console.error('Failed to toggle schedule:', err);
    }
  };

  const handleTestSchedule = async (scheduleId: string) => {
    try {
      await reportsApi.testSchedule(scheduleId);
      alert('Test report sent successfully!');
    } catch (err) {
      console.error('Failed to test schedule:', err);
      alert('Failed to send test report');
    }
  };

  // Emergency Incident Report download
  const downloadEmergencyReport = async () => {
    setEmergencyReportLoading(true);
    try {
      const response = await fetch('/api/reports/emergency-incident', {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`,
        },
      });

      if (!response.ok) {
        throw new Error('Failed to generate report');
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'Emergency_Incident_Report_Dec_12_2024.pdf';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Failed to download emergency report:', err);
      alert('Failed to download emergency report. Please try again.');
    } finally {
      setEmergencyReportLoading(false);
    }
  };

  // Legacy report functions
  const generateLegacyReport = async () => {
    setIsLoading(true);
    setAttendanceReport(null);
    setViolationReport(null);

    try {
      if (legacyReportType === 'attendance') {
        const report = await reportsApi.getAttendanceReport(
          startDate,
          endDate,
          selectedEmployee || undefined
        );
        setAttendanceReport(report);
      } else {
        const report = await reportsApi.getViolationsReport(
          startDate,
          endDate,
          selectedEmployee || undefined
        );
        setViolationReport(report);
      }
    } catch (err) {
      console.error('Failed to generate report:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const downloadCSV = () => {
    let csvContent = '';

    if (legacyReportType === 'attendance' && attendanceReport) {
      csvContent = 'Employee ID,Employee Name,Date,Check Ins,Check Outs,Total Hours\n';
      attendanceReport.records.forEach((record) => {
        csvContent += `${record.employee_id},${record.employee_name},${record.date},"${record.check_ins.map(t => formatTime(t)).join(', ')}","${record.check_outs.map(t => formatTime(t)).join(', ')}",${record.total_hours || 'N/A'}\n`;
      });
    } else if (legacyReportType === 'violations' && violationReport) {
      csvContent = 'ID,Employee ID,Employee Name,Timestamp,Violations,Location\n';
      violationReport.violations.forEach((v) => {
        csvContent += `${v.id},${v.employee_id || 'Unknown'},${v.employee_name || 'Unknown'},${v.timestamp},"${v.violations.map(viol => viol.label).join(', ')}",${v.location || 'N/A'}\n`;
      });
    }

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${legacyReportType}_report_${startDate}_to_${endDate}.csv`;
    a.click();
  };

  return (
    <AppLayout>
      <div className="space-y-6">
        {/* Page Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-stone-800">Reports</h1>
            <p className="text-stone-500 mt-1">Generate, schedule, and download reports</p>
          </div>
          {activeTab === 'schedules' && (
            <button
              onClick={() => {
                setEditingSchedule(undefined);
                setShowScheduleModal(true);
              }}
              className="flex items-center gap-2 px-4 py-2 bg-orange-600 text-white rounded-lg hover:bg-orange-700 transition-colors"
            >
              <Plus size={18} />
              New Schedule
            </button>
          )}
        </div>

        {/* Tabs */}
        <div className="flex gap-1 bg-stone-100 p-1 rounded-lg w-fit">
          <button
            onClick={() => setActiveTab('generate')}
            className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
              activeTab === 'generate'
                ? 'bg-white text-stone-900 shadow-sm'
                : 'text-stone-600 hover:text-stone-900'
            }`}
          >
            <span className="flex items-center gap-2">
              <FileText size={16} />
              Generate Report
            </span>
          </button>
          <button
            onClick={() => setActiveTab('schedules')}
            className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
              activeTab === 'schedules'
                ? 'bg-white text-stone-900 shadow-sm'
                : 'text-stone-600 hover:text-stone-900'
            }`}
          >
            <span className="flex items-center gap-2">
              <Settings size={16} />
              Scheduled Reports
              {schedules.length > 0 && (
                <span className="px-1.5 py-0.5 text-xs bg-orange-100 text-orange-700 rounded-full">
                  {schedules.length}
                </span>
              )}
            </span>
          </button>
          <button
            onClick={() => setActiveTab('legacy')}
            className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
              activeTab === 'legacy'
                ? 'bg-white text-stone-900 shadow-sm'
                : 'text-stone-600 hover:text-stone-900'
            }`}
          >
            <span className="flex items-center gap-2">
              <Clock size={16} />
              Legacy Reports
            </span>
          </button>
        </div>

        {/* Generate Report Tab */}
        {activeTab === 'generate' && (
          <div className="space-y-6">
            {/* Emergency Incident Report - Quick Access
            <div className="bg-gradient-to-r from-red-600 to-red-700 rounded-xl p-6 text-white shadow-lg">
              <div className="flex items-start justify-between">
                <div className="flex items-start gap-4">
                  <div className="p-3 bg-white/20 rounded-lg">
                    <Siren className="w-8 h-8" />
                  </div>
                  <div>
                    <h3 className="text-xl font-bold">Emergency Incident Report</h3>
                    <p className="text-red-100 mt-1">
                      Gas Emergency - Methane Spike | Zone A - Extraction | December 12, 2024
                    </p>
                    <div className="flex items-center gap-4 mt-3 text-sm">
                      <span className="flex items-center gap-1">
                        <span className="w-2 h-2 bg-green-400 rounded-full"></span>
                        2 Workers Safely Evacuated
                      </span>
                      <span className="flex items-center gap-1">
                        <span className="w-2 h-2 bg-yellow-400 rounded-full"></span>
                        Peak: 15,200 PPM
                      </span>
                      <span className="flex items-center gap-1">
                        <span className="w-2 h-2 bg-blue-400 rounded-full"></span>
                        Response: 3 min 25 sec
                      </span>
                    </div>
                  </div>
                </div>
                <button
                  onClick={downloadEmergencyReport}
                  disabled={emergencyReportLoading}
                  className="flex items-center gap-2 px-4 py-2 bg-white text-red-600 rounded-lg font-semibold hover:bg-red-50 transition-colors disabled:opacity-50"
                >
                  {emergencyReportLoading ? (
                    <>
                      <Spinner size="sm" className="border-red-300 border-t-red-600" />
                      Generating...
                    </>
                  ) : (
                    <>
                      <Download size={18} />
                      Download PDF
                    </>
                  )}
                </button>
              </div>
            </div> */}

            <ReportGenerator
              reportTypes={reportTypes}
              mineId={user?.mine_id}
            />

            {/* Report Types Info */}
            <Card title="Available Report Types">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {reportTypes.map((rt) => (
                  <div
                    key={rt.id}
                    className="p-4 border border-stone-200 rounded-lg hover:border-orange-300 transition-colors"
                  >
                    <h4 className="font-medium text-stone-800">{rt.name}</h4>
                    <p className="text-sm text-stone-500 mt-1">{rt.description}</p>
                    <div className="flex gap-1 mt-2">
                      {rt.available_formats.map((f) => (
                        <span
                          key={f}
                          className={`px-2 py-0.5 text-xs rounded ${
                            f === 'pdf'
                              ? 'bg-red-100 text-red-700'
                              : f === 'excel'
                              ? 'bg-green-100 text-green-700'
                              : 'bg-blue-100 text-blue-700'
                          }`}
                        >
                          {f.toUpperCase()}
                        </span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </Card>
          </div>
        )}

        {/* Scheduled Reports Tab */}
        {activeTab === 'schedules' && (
          <div className="space-y-6">
            <ScheduleList
              schedules={schedules}
              onEdit={handleEditSchedule}
              onDelete={handleDeleteSchedule}
              onToggle={handleToggleSchedule}
              onTest={handleTestSchedule}
              loading={schedulesLoading}
            />
          </div>
        )}

        {/* Legacy Reports Tab */}
        {activeTab === 'legacy' && (
          <div className="space-y-6">
            {/* Report Configuration */}
            <Card title="Generate Legacy Report">
              <div className="space-y-4">
                {/* Report Type */}
                <div className="flex gap-2">
                  <button
                    onClick={() => setLegacyReportType('attendance')}
                    className={`flex-1 py-3 px-4 rounded-lg font-medium transition-all flex items-center justify-center gap-2 ${
                      legacyReportType === 'attendance'
                        ? 'bg-orange-600 text-white'
                        : 'bg-stone-100 text-stone-600 hover:bg-stone-200'
                    }`}
                  >
                    <Users size={20} />
                    Attendance Report
                  </button>
                  <button
                    onClick={() => setLegacyReportType('violations')}
                    className={`flex-1 py-3 px-4 rounded-lg font-medium transition-all flex items-center justify-center gap-2 ${
                      legacyReportType === 'violations'
                        ? 'bg-orange-600 text-white'
                        : 'bg-stone-100 text-stone-600 hover:bg-stone-200'
                    }`}
                  >
                    <AlertTriangle size={20} />
                    Violations Report
                  </button>
                </div>

                {/* Date Range */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-stone-700 mb-1">
                      Start Date
                    </label>
                    <input
                      type="date"
                      value={startDate}
                      onChange={(e) => setStartDate(e.target.value)}
                      className="w-full"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-stone-700 mb-1">
                      End Date
                    </label>
                    <input
                      type="date"
                      value={endDate}
                      onChange={(e) => setEndDate(e.target.value)}
                      className="w-full"
                    />
                  </div>
                </div>

                {/* Employee Filter */}
                <div>
                  <label className="block text-sm font-medium text-stone-700 mb-1">
                    Employee (Optional)
                  </label>
                  <select
                    value={selectedEmployee}
                    onChange={(e) => setSelectedEmployee(e.target.value)}
                    className="w-full"
                  >
                    <option value="">All Employees</option>
                    {employees.map((emp) => (
                      <option key={emp.employee_id} value={emp.employee_id}>
                        {emp.name} ({emp.employee_id})
                      </option>
                    ))}
                  </select>
                </div>

                {/* Generate Button */}
                <button
                  onClick={generateLegacyReport}
                  disabled={isLoading}
                  className="w-full py-3 px-4 bg-orange-600 text-white rounded-lg font-medium hover:bg-orange-700 transition-colors flex items-center justify-center gap-2 disabled:opacity-50"
                >
                  {isLoading ? (
                    <>
                      <Spinner size="sm" className="border-white/30 border-t-white" />
                      Generating...
                    </>
                  ) : (
                    <>
                      <FileText size={20} />
                      Generate Report
                    </>
                  )}
                </button>
              </div>
            </Card>

            {/* Attendance Report */}
            {attendanceReport && (
              <Card
                title="Attendance Report"
                description={`${formatDate(attendanceReport.start_date)} - ${formatDate(attendanceReport.end_date)}`}
              >
                {/* Summary */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                  <div className="p-4 bg-blue-50 rounded-lg text-center">
                    <p className="text-2xl font-bold text-blue-600">
                      {attendanceReport.summary.unique_employees}
                    </p>
                    <p className="text-sm text-blue-700">Employees</p>
                  </div>
                  <div className="p-4 bg-green-50 rounded-lg text-center">
                    <p className="text-2xl font-bold text-green-600">
                      {attendanceReport.summary.total_records}
                    </p>
                    <p className="text-sm text-green-700">Records</p>
                  </div>
                  <div className="p-4 bg-purple-50 rounded-lg text-center">
                    <p className="text-2xl font-bold text-purple-600">
                      {attendanceReport.summary.total_days}
                    </p>
                    <p className="text-sm text-purple-700">Days</p>
                  </div>
                  <div className="p-4 bg-yellow-50 rounded-lg text-center">
                    <p className="text-2xl font-bold text-yellow-600">
                      {attendanceReport.summary.total_hours_worked.toFixed(1)}h
                    </p>
                    <p className="text-sm text-yellow-700">Total Hours</p>
                  </div>
                </div>

                {/* Table */}
                <div className="table-container">
                  <table>
                    <thead>
                      <tr>
                        <th>Employee</th>
                        <th>Date</th>
                        <th>Check In</th>
                        <th>Check Out</th>
                        <th>Hours</th>
                      </tr>
                    </thead>
                    <tbody>
                      {attendanceReport.records.length === 0 ? (
                        <tr>
                          <td colSpan={5} className="text-center text-stone-500 py-8">
                            No records found
                          </td>
                        </tr>
                      ) : (
                        attendanceReport.records.map((record, idx) => (
                          <tr key={idx}>
                            <td>
                              <div>
                                <p className="font-medium">{record.employee_name}</p>
                                <p className="text-xs text-stone-500">{record.employee_id}</p>
                              </div>
                            </td>
                            <td>{formatDate(record.date)}</td>
                            <td>
                              {record.check_ins.map((t, i) => (
                                <span key={i} className="badge badge-success mr-1">
                                  {formatTime(t)}
                                </span>
                              ))}
                            </td>
                            <td>
                              {record.check_outs.map((t, i) => (
                                <span key={i} className="badge badge-info mr-1">
                                  {formatTime(t)}
                                </span>
                              ))}
                            </td>
                            <td>{record.total_hours?.toFixed(2) || '-'}</td>
                          </tr>
                        ))
                      )}
                    </tbody>
                  </table>
                </div>

                {/* Download Button */}
                <button
                  onClick={downloadCSV}
                  className="mt-4 py-2 px-4 bg-green-500 text-white rounded-lg font-medium hover:bg-green-600 transition-colors flex items-center gap-2"
                >
                  <Download size={18} />
                  Download CSV
                </button>
              </Card>
            )}

            {/* Violations Report */}
            {violationReport && (
              <Card
                title="Violations Report"
                description={`${formatDate(violationReport.start_date)} - ${formatDate(violationReport.end_date)}`}
              >
                {/* Summary */}
                <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mb-6">
                  <div className="p-4 bg-red-50 rounded-lg text-center">
                    <p className="text-2xl font-bold text-red-600">
                      {violationReport.summary.total_violations}
                    </p>
                    <p className="text-sm text-red-700">Total Violations</p>
                  </div>
                  <div className="p-4 bg-orange-50 rounded-lg text-center">
                    <p className="text-2xl font-bold text-orange-600">
                      {violationReport.summary.employees_with_violations}
                    </p>
                    <p className="text-sm text-orange-700">Employees</p>
                  </div>
                  <div className="p-4 bg-blue-50 rounded-lg text-center col-span-2 md:col-span-1">
                    <p className="text-2xl font-bold text-blue-600">
                      {Object.keys(violationReport.summary.violation_breakdown).length}
                    </p>
                    <p className="text-sm text-blue-700">Violation Types</p>
                  </div>
                </div>

                {/* Violation Breakdown */}
                {Object.keys(violationReport.summary.violation_breakdown).length > 0 && (
                  <div className="mb-6">
                    <h4 className="text-sm font-medium text-stone-700 mb-2">Breakdown by Type:</h4>
                    <div className="flex flex-wrap gap-2">
                      {Object.entries(violationReport.summary.violation_breakdown).map(([type, count]) => (
                        <span key={type} className="badge badge-danger">
                          {type}: {count}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {/* Table */}
                <div className="table-container">
                  <table>
                    <thead>
                      <tr>
                        <th>Person</th>
                        <th>Date/Time</th>
                        <th>Violations</th>
                        <th>Location</th>
                      </tr>
                    </thead>
                    <tbody>
                      {violationReport.violations.length === 0 ? (
                        <tr>
                          <td colSpan={4} className="text-center text-stone-500 py-8">
                            No violations found
                          </td>
                        </tr>
                      ) : (
                        violationReport.violations.map((record) => (
                          <tr key={record.id}>
                            <td>
                              <p className="font-medium">
                                {record.employee_name || 'Unknown'}
                              </p>
                              {record.employee_id && (
                                <p className="text-xs text-stone-500">{record.employee_id}</p>
                              )}
                            </td>
                            <td>
                              <div className="flex items-center gap-1 text-stone-600">
                                <Calendar size={14} />
                                {formatDate(record.timestamp)}
                              </div>
                              <div className="flex items-center gap-1 text-stone-500 text-sm">
                                <Clock size={12} />
                                {formatTime(record.timestamp)}
                              </div>
                            </td>
                            <td>
                              <div className="flex flex-wrap gap-1">
                                {record.violations.map((v, i) => (
                                  <span key={i} className="badge badge-danger text-xs">
                                    {v.label}
                                  </span>
                                ))}
                              </div>
                            </td>
                            <td className="text-stone-600">{record.location || '-'}</td>
                          </tr>
                        ))
                      )}
                    </tbody>
                  </table>
                </div>

                {/* Download Button */}
                <button
                  onClick={downloadCSV}
                  className="mt-4 py-2 px-4 bg-green-500 text-white rounded-lg font-medium hover:bg-green-600 transition-colors flex items-center gap-2"
                >
                  <Download size={18} />
                  Download CSV
                </button>
              </Card>
            )}
          </div>
        )}

        {/* Schedule Modal */}
        <ReportScheduleModal
          isOpen={showScheduleModal}
          onClose={() => {
            setShowScheduleModal(false);
            setEditingSchedule(undefined);
          }}
          onSave={handleCreateSchedule}
          schedule={editingSchedule}
          reportTypes={reportTypes}
          mineId={user?.mine_id}
        />
      </div>
    </AppLayout>
  );
}
