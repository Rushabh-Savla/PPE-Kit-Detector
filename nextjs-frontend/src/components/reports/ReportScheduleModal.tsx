'use client';

import { useState, useEffect } from 'react';
import {
  ReportType,
  ReportFormat,
  ScheduleFrequency,
  EmailRecipient,
  ReportSchedule,
  CreateScheduleRequest,
  ReportTypeInfo,
} from '@/types';
import FormatSelector from './FormatSelector';
import EmailRecipientManager from './EmailRecipientManager';

interface ReportScheduleModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSave: (schedule: CreateScheduleRequest) => Promise<void>;
  schedule?: ReportSchedule;
  reportTypes: ReportTypeInfo[];
  mineId?: string;
}

const FREQUENCY_OPTIONS: { value: ScheduleFrequency; label: string }[] = [
  { value: 'daily', label: 'Daily' },
  { value: 'weekly', label: 'Weekly' },
  { value: 'monthly', label: 'Monthly' },
];

const DATE_RANGE_OPTIONS = [
  { value: 'previous_shift', label: 'Previous Shift' },
  { value: 'previous_day', label: 'Previous Day' },
  { value: 'previous_week', label: 'Previous Week' },
  { value: 'previous_month', label: 'Previous Month' },
];

const DAYS_OF_WEEK = [
  { value: 1, label: 'Monday' },
  { value: 2, label: 'Tuesday' },
  { value: 3, label: 'Wednesday' },
  { value: 4, label: 'Thursday' },
  { value: 5, label: 'Friday' },
  { value: 6, label: 'Saturday' },
  { value: 7, label: 'Sunday' },
];

export default function ReportScheduleModal({
  isOpen,
  onClose,
  onSave,
  schedule,
  reportTypes,
  mineId,
}: ReportScheduleModalProps) {
  const [name, setName] = useState('');
  const [reportType, setReportType] = useState<ReportType>('shift_summary');
  const [frequency, setFrequency] = useState<ScheduleFrequency>('daily');
  const [time, setTime] = useState('06:00');
  const [dayOfWeek, setDayOfWeek] = useState(1);
  const [dayOfMonth, setDayOfMonth] = useState(1);
  const [formats, setFormats] = useState<ReportFormat[]>(['pdf']);
  const [dateRange, setDateRange] = useState<string>('previous_day');
  const [recipients, setRecipients] = useState<EmailRecipient[]>([]);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');

  // Reset form when modal opens/closes or schedule changes
  useEffect(() => {
    if (isOpen) {
      if (schedule) {
        setName(schedule.name);
        setReportType(schedule.report_type);
        setFrequency(schedule.schedule.frequency);
        setTime(schedule.schedule.time);
        setDayOfWeek(schedule.schedule.day_of_week || 1);
        setDayOfMonth(schedule.schedule.day_of_month || 1);
        setFormats(schedule.config.format);
        setDateRange(schedule.config.date_range);
        setRecipients(schedule.recipients);
      } else {
        // Default values for new schedule
        setName('');
        setReportType('shift_summary');
        setFrequency('daily');
        setTime('06:00');
        setDayOfWeek(1);
        setDayOfMonth(1);
        setFormats(['pdf']);
        setDateRange('previous_day');
        setRecipients([]);
      }
      setError('');
    }
  }, [isOpen, schedule]);

  const selectedReportType = reportTypes.find((rt) => rt.id === reportType);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    if (!name.trim()) {
      setError('Schedule name is required');
      return;
    }
    if (formats.length === 0) {
      setError('Select at least one format');
      return;
    }
    if (recipients.length === 0) {
      setError('Add at least one recipient');
      return;
    }

    setSaving(true);
    try {
      const scheduleData: CreateScheduleRequest = {
        name: name.trim(),
        report_type: reportType,
        mine_id: mineId,
        schedule: {
          frequency,
          time,
          ...(frequency === 'weekly' && { day_of_week: dayOfWeek }),
          ...(frequency === 'monthly' && { day_of_month: dayOfMonth }),
        },
        recipients,
        config: {
          format: formats,
          date_range: dateRange as 'previous_shift' | 'previous_day' | 'previous_week' | 'previous_month',
        },
      };

      await onSave(scheduleData);
      onClose();
    } catch (err: any) {
      setError(err.message || 'Failed to save schedule');
    } finally {
      setSaving(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto">
      <div className="flex min-h-screen items-center justify-center p-4">
        {/* Backdrop */}
        <div
          className="fixed inset-0 bg-black bg-opacity-50 transition-opacity"
          onClick={onClose}
        />

        {/* Modal */}
        <div className="relative w-full max-w-2xl bg-white rounded-xl shadow-xl">
          {/* Header */}
          <div className="flex items-center justify-between px-6 py-4 border-b border-stone-200">
            <h2 className="text-xl font-semibold text-stone-800">
              {schedule ? 'Edit Schedule' : 'Create Report Schedule'}
            </h2>
            <button
              onClick={onClose}
              className="p-2 text-stone-400 hover:text-stone-600 transition-colors"
            >
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          {/* Form */}
          <form onSubmit={handleSubmit} className="p-6 space-y-6 max-h-[70vh] overflow-y-auto">
            {/* Schedule Name */}
            <div>
              <label className="block text-sm font-medium text-stone-700 mb-1">
                Schedule Name
              </label>
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="e.g., Daily Shift Summary"
                className="w-full px-3 py-2 border border-stone-300 rounded-lg focus:ring-2 focus:ring-orange-500 focus:border-orange-500"
              />
            </div>

            {/* Report Type */}
            <div>
              <label className="block text-sm font-medium text-stone-700 mb-1">
                Report Type
              </label>
              <select
                value={reportType}
                onChange={(e) => setReportType(e.target.value as ReportType)}
                className="w-full px-3 py-2 border border-stone-300 rounded-lg focus:ring-2 focus:ring-orange-500 focus:border-orange-500"
              >
                {reportTypes.map((rt) => (
                  <option key={rt.id} value={rt.id}>
                    {rt.name}
                  </option>
                ))}
              </select>
              {selectedReportType && (
                <p className="mt-1 text-sm text-stone-500">{selectedReportType.description}</p>
              )}
            </div>

            {/* Schedule Configuration */}
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-stone-700 mb-1">
                  Frequency
                </label>
                <select
                  value={frequency}
                  onChange={(e) => setFrequency(e.target.value as ScheduleFrequency)}
                  className="w-full px-3 py-2 border border-stone-300 rounded-lg focus:ring-2 focus:ring-orange-500 focus:border-orange-500"
                >
                  {FREQUENCY_OPTIONS.map((opt) => (
                    <option key={opt.value} value={opt.value}>
                      {opt.label}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-stone-700 mb-1">
                  Time (UTC)
                </label>
                <input
                  type="time"
                  value={time}
                  onChange={(e) => setTime(e.target.value)}
                  className="w-full px-3 py-2 border border-stone-300 rounded-lg focus:ring-2 focus:ring-orange-500 focus:border-orange-500"
                />
              </div>
            </div>

            {/* Day Selection */}
            {frequency === 'weekly' && (
              <div>
                <label className="block text-sm font-medium text-stone-700 mb-1">
                  Day of Week
                </label>
                <select
                  value={dayOfWeek}
                  onChange={(e) => setDayOfWeek(Number(e.target.value))}
                  className="w-full px-3 py-2 border border-stone-300 rounded-lg focus:ring-2 focus:ring-orange-500 focus:border-orange-500"
                >
                  {DAYS_OF_WEEK.map((day) => (
                    <option key={day.value} value={day.value}>
                      {day.label}
                    </option>
                  ))}
                </select>
              </div>
            )}

            {frequency === 'monthly' && (
              <div>
                <label className="block text-sm font-medium text-stone-700 mb-1">
                  Day of Month
                </label>
                <select
                  value={dayOfMonth}
                  onChange={(e) => setDayOfMonth(Number(e.target.value))}
                  className="w-full px-3 py-2 border border-stone-300 rounded-lg focus:ring-2 focus:ring-orange-500 focus:border-orange-500"
                >
                  {Array.from({ length: 28 }, (_, i) => i + 1).map((day) => (
                    <option key={day} value={day}>
                      {day}
                    </option>
                  ))}
                </select>
                <p className="mt-1 text-xs text-stone-500">
                  Days 29-31 will run on the last day of shorter months
                </p>
              </div>
            )}

            {/* Date Range */}
            <div>
              <label className="block text-sm font-medium text-stone-700 mb-1">
                Report Date Range
              </label>
              <select
                value={dateRange}
                onChange={(e) => setDateRange(e.target.value)}
                className="w-full px-3 py-2 border border-stone-300 rounded-lg focus:ring-2 focus:ring-orange-500 focus:border-orange-500"
              >
                {DATE_RANGE_OPTIONS.map((opt) => (
                  <option key={opt.value} value={opt.value}>
                    {opt.label}
                  </option>
                ))}
              </select>
            </div>

            {/* Format Selection */}
            <div>
              <label className="block text-sm font-medium text-stone-700 mb-2">
                Export Formats
              </label>
              <FormatSelector
                selectedFormats={formats}
                onChange={setFormats}
                multiSelect={true}
                availableFormats={selectedReportType?.available_formats || ['pdf', 'excel']}
              />
            </div>

            {/* Recipients */}
            <div>
              <label className="block text-sm font-medium text-stone-700 mb-2">
                Email Recipients
              </label>
              <EmailRecipientManager recipients={recipients} onChange={setRecipients} />
            </div>

            {/* Error Message */}
            {error && (
              <div className="p-3 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">
                {error}
              </div>
            )}
          </form>

          {/* Footer */}
          <div className="flex justify-end gap-3 px-6 py-4 border-t border-stone-200 bg-stone-50 rounded-b-xl">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 text-stone-700 bg-white border border-stone-300 rounded-lg hover:bg-stone-50 transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              onClick={handleSubmit}
              disabled={saving}
              className="px-4 py-2 bg-orange-600 text-white rounded-lg hover:bg-orange-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
            >
              {saving ? (
                <>
                  <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
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
                  Saving...
                </>
              ) : (
                <>
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M5 13l4 4L19 7"
                    />
                  </svg>
                  {schedule ? 'Update Schedule' : 'Create Schedule'}
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
