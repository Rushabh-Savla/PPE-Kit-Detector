'use client';

import { useState, useEffect } from 'react';
import { ReportType, ReportFormat, ReportTypeInfo, GenerateReportRequest } from '@/types';
import FormatSelector from './FormatSelector';
import { reportsApi } from '@/lib/api';

interface ReportGeneratorProps {
  reportTypes: ReportTypeInfo[];
  mineId?: string;
  onGenerated?: () => void;
}

export default function ReportGenerator({
  reportTypes,
  mineId,
  onGenerated,
}: ReportGeneratorProps) {
  const [selectedType, setSelectedType] = useState<ReportType | ''>('');
  const [format, setFormat] = useState<ReportFormat>('pdf');
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  const selectedReportType = reportTypes.find((rt) => rt.id === selectedType);

  // Set default dates on mount
  useEffect(() => {
    const today = new Date();
    const weekAgo = new Date(today);
    weekAgo.setDate(weekAgo.getDate() - 7);

    setEndDate(today.toISOString().split('T')[0]);
    setStartDate(weekAgo.toISOString().split('T')[0]);
  }, []);

  const handleGenerate = async () => {
    if (!selectedType) {
      setError('Please select a report type');
      return;
    }
    if (!startDate || !endDate) {
      setError('Please select date range');
      return;
    }

    setError('');
    setSuccess('');
    setGenerating(true);

    try {
      const request: GenerateReportRequest = {
        report_type: selectedType,
        format,
        start_date: startDate,
        end_date: endDate,
        mine_id: mineId,
      };

      const response = await reportsApi.generateReport(request);

      // Download the file
      const blob = await reportsApi.downloadReport(response.report_id);
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = response.file_name;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);

      setSuccess(`Report generated successfully: ${response.file_name}`);
      onGenerated?.();
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Failed to generate report');
    } finally {
      setGenerating(false);
    }
  };

  return (
    <div className="bg-white rounded-xl shadow-sm border border-stone-200 p-6">
      <h3 className="text-lg font-semibold text-stone-800 mb-4">Generate Report</h3>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
        {/* Report Type */}
        <div>
          <label className="block text-sm font-medium text-stone-700 mb-1">
            Report Type
          </label>
          <select
            value={selectedType}
            onChange={(e) => setSelectedType(e.target.value as ReportType)}
            className="w-full px-3 py-2 border border-stone-300 rounded-lg focus:ring-2 focus:ring-orange-500 focus:border-orange-500"
          >
            <option value="">Select report type...</option>
            {reportTypes.map((rt, index) => (
              <option key={`${rt.id}-${index}`} value={rt.id}>
                {rt.name}
              </option>
            ))}
          </select>
          {selectedReportType && (
            <p className="mt-1 text-xs text-stone-500">{selectedReportType.description}</p>
          )}
        </div>

        {/* Format */}
        <div>
          <label className="block text-sm font-medium text-stone-700 mb-1">
            Format
          </label>
          <FormatSelector
            selectedFormats={[format]}
            onChange={(formats) => setFormat(formats[0] || 'pdf')}
            availableFormats={selectedReportType?.available_formats || ['pdf', 'excel', 'csv']}
          />
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
        {/* Start Date */}
        <div>
          <label className="block text-sm font-medium text-stone-700 mb-1">
            Start Date
          </label>
          <input
            type="date"
            value={startDate}
            onChange={(e) => setStartDate(e.target.value)}
            className="w-full px-3 py-2 border border-stone-300 rounded-lg focus:ring-2 focus:ring-orange-500 focus:border-orange-500"
          />
        </div>

        {/* End Date */}
        <div>
          <label className="block text-sm font-medium text-stone-700 mb-1">
            End Date
          </label>
          <input
            type="date"
            value={endDate}
            onChange={(e) => setEndDate(e.target.value)}
            className="w-full px-3 py-2 border border-stone-300 rounded-lg focus:ring-2 focus:ring-orange-500 focus:border-orange-500"
          />
        </div>
      </div>

      {/* Quick Date Range Buttons */}
      <div className="flex flex-wrap gap-2 mb-4">
        <button
          type="button"
          onClick={() => {
            const today = new Date();
            setEndDate(today.toISOString().split('T')[0]);
            setStartDate(today.toISOString().split('T')[0]);
          }}
          className="px-3 py-1.5 text-sm bg-stone-100 text-stone-700 rounded-lg hover:bg-stone-200 transition-colors"
        >
          Today
        </button>
        <button
          type="button"
          onClick={() => {
            const today = new Date();
            const weekAgo = new Date(today);
            weekAgo.setDate(weekAgo.getDate() - 7);
            setEndDate(today.toISOString().split('T')[0]);
            setStartDate(weekAgo.toISOString().split('T')[0]);
          }}
          className="px-3 py-1.5 text-sm bg-stone-100 text-stone-700 rounded-lg hover:bg-stone-200 transition-colors"
        >
          Last 7 Days
        </button>
        <button
          type="button"
          onClick={() => {
            const today = new Date();
            const monthAgo = new Date(today);
            monthAgo.setDate(monthAgo.getDate() - 30);
            setEndDate(today.toISOString().split('T')[0]);
            setStartDate(monthAgo.toISOString().split('T')[0]);
          }}
          className="px-3 py-1.5 text-sm bg-stone-100 text-stone-700 rounded-lg hover:bg-stone-200 transition-colors"
        >
          Last 30 Days
        </button>
        <button
          type="button"
          onClick={() => {
            const today = new Date();
            const firstDay = new Date(today.getFullYear(), today.getMonth(), 1);
            setEndDate(today.toISOString().split('T')[0]);
            setStartDate(firstDay.toISOString().split('T')[0]);
          }}
          className="px-3 py-1.5 text-sm bg-stone-100 text-stone-700 rounded-lg hover:bg-stone-200 transition-colors"
        >
          This Month
        </button>
      </div>

      {/* Messages */}
      {error && (
        <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">
          {error}
        </div>
      )}
      {success && (
        <div className="mb-4 p-3 bg-green-50 border border-green-200 rounded-lg text-green-700 text-sm">
          {success}
        </div>
      )}

      {/* Generate Button */}
      <button
        onClick={handleGenerate}
        disabled={generating || !selectedType}
        className="w-full px-4 py-3 bg-orange-600 text-white rounded-lg hover:bg-orange-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
      >
        {generating ? (
          <>
            <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
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
            Generating Report...
          </>
        ) : (
          <>
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
              />
            </svg>
            Generate & Download
          </>
        )}
      </button>
    </div>
  );
}
