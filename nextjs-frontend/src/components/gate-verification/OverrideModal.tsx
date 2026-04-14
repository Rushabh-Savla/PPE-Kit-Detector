'use client';

import { useState } from 'react';
import { X, AlertTriangle, CheckCircle, XCircle, Shield } from 'lucide-react';
import type { VerificationItem, VerificationItemType } from '@/types';

interface OverrideModalProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: (reason: string) => void;
  items: Record<VerificationItemType, VerificationItem>;
  passedCount: number;
  totalChecks: number;
}

export function OverrideModal({
  isOpen,
  onClose,
  onConfirm,
  items,
  passedCount,
  totalChecks,
}: OverrideModalProps) {
  const [reason, setReason] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  if (!isOpen) return null;

  const handleConfirm = async () => {
    if (!reason.trim()) return;

    setIsSubmitting(true);
    try {
      await onConfirm(reason);
      setReason('');
    } finally {
      setIsSubmitting(false);
    }
  };

  const getItemStatus = (item: VerificationItem, showRfid: boolean) => {
    const checks = [];

    if (showRfid) {
      checks.push({
        label: 'RFID',
        passed: item.rfidStatus === 'passed',
      });
    }

    checks.push({
      label: 'ML Detection',
      passed: item.mlStatus === 'passed',
    });

    return checks;
  };

  const itemsArray: { type: VerificationItemType; label: string; showRfid: boolean }[] = [
    { type: 'helmet', label: 'Helmet', showRfid: true },
    { type: 'face', label: 'Face Recognition', showRfid: false },
    { type: 'vest', label: 'Safety Vest', showRfid: true },
    { type: 'shoes', label: 'Safety Shoes', showRfid: true },
  ];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/50 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Modal */}
      <div className="relative bg-white rounded-2xl shadow-2xl w-full max-w-lg mx-4 overflow-hidden">
        {/* Header */}
        <div className="bg-orange-500 text-white p-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Shield size={24} />
            <h2 className="text-xl font-bold">Manager Override</h2>
          </div>
          <button
            onClick={onClose}
            className="p-1 hover:bg-white/20 rounded-lg transition-colors"
          >
            <X size={24} />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-6">
          {/* Warning */}
          <div className="flex items-start gap-3 p-4 bg-amber-50 border border-amber-200 rounded-lg">
            <AlertTriangle className="text-amber-500 flex-shrink-0" size={24} />
            <div>
              <p className="font-semibold text-amber-800">Partial Verification</p>
              <p className="text-sm text-amber-700 mt-1">
                Only {passedCount} of {totalChecks} checks passed. Are you sure you want to
                override and allow entry?
              </p>
            </div>
          </div>

          {/* Verification Summary */}
          <div>
            <h3 className="font-semibold text-stone-700 mb-3">Verification Summary</h3>
            <div className="space-y-2">
              {itemsArray.map(({ type, label, showRfid }) => {
                const item = items[type];
                const checks = getItemStatus(item, showRfid);

                return (
                  <div
                    key={type}
                    className="flex items-center justify-between p-3 bg-stone-50 rounded-lg"
                  >
                    <span className="font-medium text-stone-700">{label}</span>
                    <div className="flex items-center gap-3">
                      {checks.map((check) => (
                        <div
                          key={check.label}
                          className="flex items-center gap-1 text-sm"
                        >
                          {check.passed ? (
                            <CheckCircle size={16} className="text-green-500" />
                          ) : (
                            <XCircle size={16} className="text-red-500" />
                          )}
                          <span
                            className={
                              check.passed ? 'text-green-700' : 'text-red-700'
                            }
                          >
                            {check.label}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Reason Input */}
          <div>
            <label className="block font-semibold text-stone-700 mb-2">
              Override Reason <span className="text-red-500">*</span>
            </label>
            <textarea
              value={reason}
              onChange={(e) => setReason(e.target.value)}
              placeholder="Enter the reason for override..."
              className="w-full h-24 px-4 py-3 border border-stone-300 rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-transparent"
              required
            />
            <p className="text-xs text-stone-500 mt-1">
              This will be logged for audit purposes.
            </p>
          </div>
        </div>

        {/* Footer */}
        <div className="border-t border-stone-200 p-4 flex gap-3 justify-end bg-stone-50">
          <button
            onClick={onClose}
            className="px-6 py-2.5 text-stone-700 font-medium hover:bg-stone-200 rounded-lg transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleConfirm}
            disabled={!reason.trim() || isSubmitting}
            className="px-6 py-2.5 bg-orange-500 text-white font-semibold rounded-lg hover:bg-orange-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
          >
            {isSubmitting ? (
              <>
                <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                Processing...
              </>
            ) : (
              <>
                <Shield size={18} />
                Confirm Override
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
