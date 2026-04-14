'use client';

import { ReportSchedule } from '@/types';

interface ScheduleListProps {
  schedules: ReportSchedule[];
  onEdit: (schedule: ReportSchedule) => void;
  onDelete: (scheduleId: string) => void;
  onToggle: (scheduleId: string, isActive: boolean) => void;
  onTest: (scheduleId: string) => void;
  loading?: boolean;
}

const FREQUENCY_LABELS: Record<string, string> = {
  daily: 'Daily',
  weekly: 'Weekly',
  monthly: 'Monthly',
};

const REPORT_TYPE_LABELS: Record<string, string> = {
  shift_summary: 'Shift Summary',
  shift_handover: 'Shift Handover',
  entry_exit_log: 'Entry/Exit Log',
  weekly_compliance: 'Weekly Compliance',
  high_risk_workers: 'High Risk Workers',
  zone_risk_analysis: 'Zone Risk Analysis',
  violation_trends: 'Violation Trends',
  daily_operations: 'Daily Operations',
  shift_performance: 'Shift Performance',
  worker_rankings: 'Worker Rankings',
  monthly_summary: 'Monthly Summary',
  mine_comparison: 'Mine Comparison',
  risk_heatmap: 'Risk Heatmap',
  critical_incidents: 'Critical Incidents',
  executive_summary: 'Executive Summary',
  kpi_dashboard: 'KPI Dashboard',
  regulatory_compliance: 'Regulatory Compliance',
  financial_impact: 'Financial Impact',
  worker_compliance_card: 'Compliance Card',
  worker_monthly_summary: 'Monthly Summary',
};

export default function ScheduleList({
  schedules,
  onEdit,
  onDelete,
  onToggle,
  onTest,
  loading,
}: ScheduleListProps) {
  const formatTime = (time: string) => {
    const [hours, minutes] = time.split(':');
    const h = parseInt(hours);
    const ampm = h >= 12 ? 'PM' : 'AM';
    const h12 = h % 12 || 12;
    return `${h12}:${minutes} ${ampm} UTC`;
  };

  const formatNextRun = (dateStr: string) => {
    const date = new Date(dateStr);
    return date.toLocaleDateString('en-US', {
      weekday: 'short',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  if (loading) {
    return (
      <div className="bg-white rounded-xl shadow-sm border border-stone-200 p-8 text-center">
        <svg className="animate-spin h-8 w-8 mx-auto text-stone-400" viewBox="0 0 24 24">
          <circle
            className="opacity-25"
            cx="12"
            cy="12"
            r="10"
            stroke="currentColor"
            strokeWidth="4"
            fill="none"
          />
          <path
            className="opacity-75"
            fill="currentColor"
            d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
          />
        </svg>
        <p className="mt-2 text-stone-500">Loading schedules...</p>
      </div>
    );
  }

  if (schedules.length === 0) {
    return (
      <div className="bg-white rounded-xl shadow-sm border border-stone-200 p-8 text-center">
        <svg
          className="mx-auto h-12 w-12 text-stone-400"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={1.5}
            d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"
          />
        </svg>
        <h3 className="mt-2 text-sm font-medium text-stone-800">No scheduled reports</h3>
        <p className="mt-1 text-sm text-stone-500">
          Create a schedule to automatically generate and email reports.
        </p>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-xl shadow-sm border border-stone-200 overflow-hidden">
      <div className="divide-y divide-stone-200">
        {schedules.map((schedule) => (
          <div
            key={schedule.id}
            className={`p-4 hover:bg-stone-50 transition-colors ${
              !schedule.is_active ? 'opacity-60' : ''
            }`}
          >
            <div className="flex items-start justify-between">
              <div className="flex-1">
                <div className="flex items-center gap-2">
                  <h4 className="font-medium text-stone-800">{schedule.name}</h4>
                  <span
                    className={`px-2 py-0.5 text-xs font-medium rounded-full ${
                      schedule.is_active
                        ? 'bg-green-100 text-green-700'
                        : 'bg-stone-100 text-stone-600'
                    }`}
                  >
                    {schedule.is_active ? 'Active' : 'Paused'}
                  </span>
                </div>

                <div className="mt-1 flex flex-wrap gap-x-4 gap-y-1 text-sm text-stone-500">
                  <span className="flex items-center gap-1">
                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                      />
                    </svg>
                    {REPORT_TYPE_LABELS[schedule.report_type] || schedule.report_type}
                  </span>

                  <span className="flex items-center gap-1">
                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
                      />
                    </svg>
                    {FREQUENCY_LABELS[schedule.schedule.frequency]} at {formatTime(schedule.schedule.time)}
                  </span>

                  <span className="flex items-center gap-1">
                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"
                      />
                    </svg>
                    {schedule.recipients.length} recipient(s)
                  </span>

                  <span className="flex items-center gap-1">
                    {schedule.config.format.map((f) => (
                      <span
                        key={f}
                        className={`px-1.5 py-0.5 text-xs rounded ${
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
                  </span>
                </div>

                {schedule.next_run && (
                  <p className="mt-2 text-xs text-stone-400">
                    Next run: {formatNextRun(schedule.next_run)}
                  </p>
                )}
              </div>

              {/* Actions */}
              <div className="flex items-center gap-1">
                <button
                  onClick={() => onTest(schedule.id)}
                  className="p-2 text-stone-400 hover:text-orange-600 hover:bg-orange-50 rounded transition-colors"
                  title="Test run"
                >
                  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"
                    />
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                    />
                  </svg>
                </button>

                <button
                  onClick={() => onToggle(schedule.id, !schedule.is_active)}
                  className={`p-2 rounded transition-colors ${
                    schedule.is_active
                      ? 'text-stone-400 hover:text-orange-600 hover:bg-orange-50'
                      : 'text-stone-400 hover:text-green-600 hover:bg-green-50'
                  }`}
                  title={schedule.is_active ? 'Pause schedule' : 'Resume schedule'}
                >
                  {schedule.is_active ? (
                    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M10 9v6m4-6v6m7-3a9 9 0 11-18 0 9 9 0 0118 0z"
                      />
                    </svg>
                  ) : (
                    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"
                      />
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                      />
                    </svg>
                  )}
                </button>

                <button
                  onClick={() => onEdit(schedule)}
                  className="p-2 text-stone-400 hover:text-stone-600 hover:bg-stone-100 rounded transition-colors"
                  title="Edit schedule"
                >
                  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"
                    />
                  </svg>
                </button>

                <button
                  onClick={() => {
                    if (confirm('Are you sure you want to delete this schedule?')) {
                      onDelete(schedule.id);
                    }
                  }}
                  className="p-2 text-stone-400 hover:text-red-600 hover:bg-red-50 rounded transition-colors"
                  title="Delete schedule"
                >
                  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                    />
                  </svg>
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
