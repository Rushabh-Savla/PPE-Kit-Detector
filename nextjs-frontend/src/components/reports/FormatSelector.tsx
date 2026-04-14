'use client';

import { ReportFormat } from '@/types';

interface FormatSelectorProps {
  selectedFormats: ReportFormat[];
  onChange: (formats: ReportFormat[]) => void;
  multiSelect?: boolean;
  availableFormats?: ReportFormat[];
}

const FORMAT_CONFIG: Record<ReportFormat, { label: string; icon: string; color: string }> = {
  pdf: {
    label: 'PDF',
    icon: 'M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z',
    color: 'text-red-600 bg-red-50 border-red-200',
  },
  excel: {
    label: 'Excel',
    icon: 'M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 10V7m0 10a2 2 0 002 2h2a2 2 0 002-2V7a2 2 0 00-2-2h-2a2 2 0 00-2 2',
    color: 'text-green-600 bg-green-50 border-green-200',
  },
  csv: {
    label: 'CSV',
    icon: 'M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z',
    color: 'text-blue-600 bg-blue-50 border-blue-200',
  },
};

export default function FormatSelector({
  selectedFormats,
  onChange,
  multiSelect = false,
  availableFormats = ['pdf', 'excel', 'csv'],
}: FormatSelectorProps) {
  const handleSelect = (format: ReportFormat) => {
    if (multiSelect) {
      if (selectedFormats.includes(format)) {
        onChange(selectedFormats.filter((f) => f !== format));
      } else {
        onChange([...selectedFormats, format]);
      }
    } else {
      onChange([format]);
    }
  };

  return (
    <div className="flex gap-3">
      {availableFormats.map((format) => {
        const config = FORMAT_CONFIG[format];
        const isSelected = selectedFormats.includes(format);

        return (
          <button
            key={format}
            type="button"
            onClick={() => handleSelect(format)}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg border-2 transition-all ${
              isSelected
                ? `${config.color} border-current`
                : 'bg-white text-stone-600 border-stone-200 hover:border-stone-300'
            }`}
          >
            <svg
              className="w-5 h-5"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d={config.icon} />
            </svg>
            <span className="font-medium">{config.label}</span>
            {multiSelect && isSelected && (
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                <path
                  fillRule="evenodd"
                  d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                  clipRule="evenodd"
                />
              </svg>
            )}
          </button>
        );
      })}
    </div>
  );
}
