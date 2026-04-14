'use client';

import { useState } from 'react';
import { EmailRecipient } from '@/types';

interface EmailRecipientManagerProps {
  recipients: EmailRecipient[];
  onChange: (recipients: EmailRecipient[]) => void;
}

export default function EmailRecipientManager({
  recipients,
  onChange,
}: EmailRecipientManagerProps) {
  const [newEmail, setNewEmail] = useState('');
  const [newName, setNewName] = useState('');
  const [newType, setNewType] = useState<'to' | 'cc' | 'bcc'>('to');
  const [error, setError] = useState('');

  const validateEmail = (email: string) => {
    return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
  };

  const handleAdd = () => {
    if (!newEmail.trim()) {
      setError('Email is required');
      return;
    }
    if (!validateEmail(newEmail)) {
      setError('Invalid email format');
      return;
    }
    if (recipients.some((r) => r.email.toLowerCase() === newEmail.toLowerCase())) {
      setError('Email already added');
      return;
    }

    onChange([
      ...recipients,
      {
        email: newEmail.trim(),
        name: newName.trim() || newEmail.split('@')[0],
        type: newType,
      },
    ]);
    setNewEmail('');
    setNewName('');
    setError('');
  };

  const handleRemove = (email: string) => {
    onChange(recipients.filter((r) => r.email !== email));
  };

  const handleTypeChange = (email: string, type: 'to' | 'cc' | 'bcc') => {
    onChange(
      recipients.map((r) => (r.email === email ? { ...r, type } : r))
    );
  };

  const getTypeColor = (type: 'to' | 'cc' | 'bcc') => {
    switch (type) {
      case 'to':
        return 'bg-orange-100 text-orange-700';
      case 'cc':
        return 'bg-stone-100 text-stone-700';
      case 'bcc':
        return 'bg-yellow-100 text-yellow-700';
    }
  };

  return (
    <div className="space-y-4">
      {/* Add recipient form */}
      <div className="flex gap-2 flex-wrap">
        <input
          type="text"
          placeholder="Name (optional)"
          value={newName}
          onChange={(e) => setNewName(e.target.value)}
          className="flex-1 min-w-[150px] px-3 py-2 border border-stone-300 rounded-lg focus:ring-2 focus:ring-orange-500 focus:border-orange-500"
        />
        <input
          type="email"
          placeholder="Email address"
          value={newEmail}
          onChange={(e) => {
            setNewEmail(e.target.value);
            setError('');
          }}
          onKeyDown={(e) => e.key === 'Enter' && handleAdd()}
          className="flex-1 min-w-[200px] px-3 py-2 border border-stone-300 rounded-lg focus:ring-2 focus:ring-orange-500 focus:border-orange-500"
        />
        <select
          value={newType}
          onChange={(e) => setNewType(e.target.value as 'to' | 'cc' | 'bcc')}
          className="px-3 py-2 border border-stone-300 rounded-lg focus:ring-2 focus:ring-orange-500 focus:border-orange-500"
        >
          <option value="to">To</option>
          <option value="cc">CC</option>
          <option value="bcc">BCC</option>
        </select>
        <button
          type="button"
          onClick={handleAdd}
          className="px-4 py-2 bg-orange-600 text-white rounded-lg hover:bg-orange-700 transition-colors"
        >
          Add
        </button>
      </div>

      {error && <p className="text-red-500 text-sm">{error}</p>}

      {/* Recipients list */}
      {recipients.length > 0 ? (
        <div className="border border-stone-200 rounded-lg divide-y divide-stone-200">
          {recipients.map((recipient) => (
            <div
              key={recipient.email}
              className="flex items-center justify-between p-3 hover:bg-stone-50"
            >
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-full bg-stone-200 flex items-center justify-center text-stone-600 text-sm font-medium">
                  {recipient.name.charAt(0).toUpperCase()}
                </div>
                <div>
                  <p className="font-medium text-stone-800">{recipient.name}</p>
                  <p className="text-sm text-stone-500">{recipient.email}</p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <select
                  value={recipient.type}
                  onChange={(e) =>
                    handleTypeChange(recipient.email, e.target.value as 'to' | 'cc' | 'bcc')
                  }
                  className={`px-2 py-1 text-xs font-medium rounded ${getTypeColor(
                    recipient.type
                  )} border-0 focus:ring-2 focus:ring-orange-500`}
                >
                  <option value="to">To</option>
                  <option value="cc">CC</option>
                  <option value="bcc">BCC</option>
                </select>
                <button
                  type="button"
                  onClick={() => handleRemove(recipient.email)}
                  className="p-1 text-stone-400 hover:text-red-500 transition-colors"
                >
                  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M6 18L18 6M6 6l12 12"
                    />
                  </svg>
                </button>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="text-center py-6 text-stone-500 border border-dashed border-stone-300 rounded-lg">
          No recipients added yet. Add email addresses above.
        </div>
      )}

      {/* Summary */}
      {recipients.length > 0 && (
        <div className="flex gap-4 text-sm text-stone-500">
          <span>
            To: {recipients.filter((r) => r.type === 'to').length}
          </span>
          <span>
            CC: {recipients.filter((r) => r.type === 'cc').length}
          </span>
          <span>
            BCC: {recipients.filter((r) => r.type === 'bcc').length}
          </span>
        </div>
      )}
    </div>
  );
}
