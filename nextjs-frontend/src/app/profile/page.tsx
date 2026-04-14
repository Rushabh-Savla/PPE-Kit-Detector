'use client';

import { useEffect, useState } from 'react';
import {
  User,
  Mail,
  Phone,
  Building2,
  MapPin,
  Clock,
  Shield,
  Award,
  AlertTriangle,
  Calendar,
  Star,
  Flame,
  CheckCircle,
  Camera,
  Edit,
  Save,
  X,
} from 'lucide-react';
import AppLayout from '@/components/AppLayout';
import { Card } from '@/components/Card';
import { Spinner } from '@/components/Loading';
import { useAuthStore } from '@/lib/store';
import { authApi, workerApi } from '@/lib/api';
import type { Worker } from '@/types';

const badgeConfig: Record<string, { icon: React.ReactNode; color: string; label: string; description: string }> = {
  safety_star: {
    icon: <Star className="w-5 h-5" />,
    color: 'bg-yellow-100 text-yellow-700 border-yellow-200',
    label: 'Safety Star',
    description: 'Recognized for outstanding safety compliance'
  },
  perfect_record: {
    icon: <CheckCircle className="w-5 h-5" />,
    color: 'bg-green-100 text-green-700 border-green-200',
    label: 'Perfect Record',
    description: 'No violations in the past 90 days'
  },
  streak_7: {
    icon: <Flame className="w-5 h-5" />,
    color: 'bg-orange-100 text-orange-700 border-orange-200',
    label: '7 Day Streak',
    description: '7 consecutive days of full compliance'
  },
  streak_30: {
    icon: <Flame className="w-5 h-5" />,
    color: 'bg-red-100 text-red-700 border-red-200',
    label: '30 Day Streak',
    description: '30 consecutive days of full compliance'
  },
  team_player: {
    icon: <Award className="w-5 h-5" />,
    color: 'bg-blue-100 text-blue-700 border-blue-200',
    label: 'Team Player',
    description: 'Helped fellow workers maintain compliance'
  },
};

const shiftLabels: Record<string, string> = {
  day: 'Day Shift (6:00 AM - 2:00 PM)',
  afternoon: 'Afternoon Shift (2:00 PM - 10:00 PM)',
  night: 'Night Shift (10:00 PM - 6:00 AM)',
};

export default function ProfilePage() {
  const { worker, user, userType } = useAuthStore();
  const [workerData, setWorkerData] = useState<Worker | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isEditing, setIsEditing] = useState(false);
  const [editForm, setEditForm] = useState({
    phone: '',
    emergency_contact: '',
  });
  const [saving, setSaving] = useState(false);
  const [changePassword, setChangePassword] = useState(false);
  const [passwordForm, setPasswordForm] = useState({
    currentPassword: '',
    newPassword: '',
    confirmPassword: '',
  });
  const [passwordError, setPasswordError] = useState('');

  useEffect(() => {
    loadWorkerData();
  }, []);

  const loadWorkerData = async () => {
    try {
      setLoading(true);
      if (userType === 'worker' && worker) {
        const data = await workerApi.get(worker.id);
        setWorkerData(data);
        setEditForm({
          phone: data.phone || '',
          emergency_contact: data.emergency_contact || '',
        });
      }
    } catch (err) {
      setError('Failed to load profile data');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async () => {
    if (!workerData) return;
    try {
      setSaving(true);
      await workerApi.update(workerData.id, editForm);
      await loadWorkerData();
      setIsEditing(false);
    } catch (err) {
      console.error('Failed to update profile:', err);
    } finally {
      setSaving(false);
    }
  };

  const handlePasswordChange = async () => {
    setPasswordError('');

    if (passwordForm.newPassword !== passwordForm.confirmPassword) {
      setPasswordError('New passwords do not match');
      return;
    }

    if (passwordForm.newPassword.length < 6) {
      setPasswordError('Password must be at least 6 characters');
      return;
    }

    try {
      setSaving(true);
      await authApi.changePassword(passwordForm.currentPassword, passwordForm.newPassword);
      setChangePassword(false);
      setPasswordForm({ currentPassword: '', newPassword: '', confirmPassword: '' });
      alert('Password changed successfully!');
    } catch (err: any) {
      setPasswordError(err.response?.data?.detail || 'Failed to change password');
    } finally {
      setSaving(false);
    }
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

  // For staff users
  if (userType === 'staff' && user) {
    return (
      <AppLayout>
        <div className="max-w-2xl mx-auto space-y-6">
          <h1 className="text-2xl font-bold text-stone-800">My Profile</h1>

          <Card>
            <div className="flex items-center gap-4 mb-6">
              <div className="w-20 h-20 bg-blue-100 rounded-full flex items-center justify-center">
                <User className="w-10 h-10 text-blue-600" />
              </div>
              <div>
                <h2 className="text-xl font-bold text-stone-800">{user.full_name}</h2>
                <p className="text-stone-500">@{user.username}</p>
                <span className="inline-block mt-2 px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-sm font-medium">
                  {user.role.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                </span>
              </div>
            </div>

            <div className="space-y-4 border-t pt-4">
              <div className="flex items-center gap-3">
                <Mail className="w-5 h-5 text-stone-400" />
                <span className="text-stone-700">{user.email || 'No email set'}</span>
              </div>
              <div className="flex items-center gap-3">
                <Phone className="w-5 h-5 text-stone-400" />
                <span className="text-stone-700">{user.phone || 'No phone set'}</span>
              </div>
            </div>

            <div className="mt-6 pt-6 border-t">
              <button
                onClick={() => setChangePassword(true)}
                className="btn btn-secondary"
              >
                Change Password
              </button>
            </div>
          </Card>

          {/* Password Change Modal */}
          {changePassword && (
            <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
              <div className="bg-white rounded-xl p-6 max-w-md w-full mx-4">
                <h3 className="text-lg font-bold mb-4">Change Password</h3>
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-stone-700 mb-1">Current Password</label>
                    <input
                      type="password"
                      value={passwordForm.currentPassword}
                      onChange={(e) => setPasswordForm({ ...passwordForm, currentPassword: e.target.value })}
                      className="w-full px-3 py-2 border border-stone-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-stone-700 mb-1">New Password</label>
                    <input
                      type="password"
                      value={passwordForm.newPassword}
                      onChange={(e) => setPasswordForm({ ...passwordForm, newPassword: e.target.value })}
                      className="w-full px-3 py-2 border border-stone-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-stone-700 mb-1">Confirm New Password</label>
                    <input
                      type="password"
                      value={passwordForm.confirmPassword}
                      onChange={(e) => setPasswordForm({ ...passwordForm, confirmPassword: e.target.value })}
                      className="w-full px-3 py-2 border border-stone-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                  {passwordError && (
                    <p className="text-red-500 text-sm">{passwordError}</p>
                  )}
                </div>
                <div className="flex justify-end gap-3 mt-6">
                  <button
                    onClick={() => setChangePassword(false)}
                    className="btn btn-secondary"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={handlePasswordChange}
                    disabled={saving}
                    className="btn btn-primary"
                  >
                    {saving ? 'Saving...' : 'Change Password'}
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
      </AppLayout>
    );
  }

  // For workers
  if (!workerData) {
    return (
      <AppLayout>
        <div className="text-center py-12">
          <p className="text-red-500">{error || 'Failed to load profile'}</p>
        </div>
      </AppLayout>
    );
  }

  return (
    <AppLayout>
      <div className="max-w-2xl mx-auto space-y-6">
        {/* Profile Header */}
        <div className="bg-gradient-to-r from-orange-500 to-orange-600 rounded-2xl p-6 text-white">
          <div className="flex items-center gap-4">
            <div className="w-20 h-20 bg-white/20 rounded-full flex items-center justify-center">
              <User className="w-10 h-10 text-white" />
            </div>
            <div className="flex-1">
              <h1 className="text-2xl font-bold">{workerData.name}</h1>
              <p className="text-orange-100">ID: {workerData.employee_id}</p>
              <div className="flex items-center gap-2 mt-2">
                <span className="px-3 py-1 bg-white/20 rounded-full text-sm">
                  {workerData.department || 'General'}
                </span>
              </div>
            </div>
            <button
              onClick={() => setIsEditing(!isEditing)}
              className="p-2 bg-white/20 rounded-lg hover:bg-white/30 transition-colors"
            >
              {isEditing ? <X size={20} /> : <Edit size={20} />}
            </button>
          </div>
        </div>

        {/* Compliance Score */}
        <Card>
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-stone-500">Compliance Score</p>
              <p className={`text-4xl font-bold ${
                workerData.compliance_score >= 90 ? 'text-green-600' :
                workerData.compliance_score >= 70 ? 'text-yellow-600' : 'text-red-600'
              }`}>
                {workerData.compliance_score}%
              </p>
            </div>
            <div className={`w-16 h-16 rounded-full flex items-center justify-center ${
              workerData.compliance_score >= 90 ? 'bg-green-100' :
              workerData.compliance_score >= 70 ? 'bg-yellow-100' : 'bg-red-100'
            }`}>
              <Shield className={`w-8 h-8 ${
                workerData.compliance_score >= 90 ? 'text-green-600' :
                workerData.compliance_score >= 70 ? 'text-yellow-600' : 'text-red-600'
              }`} />
            </div>
          </div>
          <div className="mt-4">
            <div className="w-full bg-stone-200 rounded-full h-3">
              <div
                className={`h-3 rounded-full ${
                  workerData.compliance_score >= 90 ? 'bg-green-500' :
                  workerData.compliance_score >= 70 ? 'bg-yellow-500' : 'bg-red-500'
                }`}
                style={{ width: `${workerData.compliance_score}%` }}
              />
            </div>
          </div>
          <div className="flex items-center justify-between mt-4 text-sm">
            <span className="text-stone-500">Total Violations: {workerData.total_violations}</span>
            <span className="text-stone-500">Member since: {new Date(workerData.created_at).toLocaleDateString()}</span>
          </div>
        </Card>

        {/* Work Information */}
        <Card title="Work Information">
          <div className="space-y-4">
            <div className="flex items-center gap-3">
              <Building2 className="w-5 h-5 text-stone-400" />
              <div>
                <p className="text-sm text-stone-500">Mine</p>
                <p className="font-medium">{workerData.mine_name || 'Not Assigned'}</p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <MapPin className="w-5 h-5 text-stone-400" />
              <div>
                <p className="text-sm text-stone-500">Zone</p>
                <p className="font-medium">{workerData.zone_name || 'Not Assigned'}</p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <Clock className="w-5 h-5 text-stone-400" />
              <div>
                <p className="text-sm text-stone-500">Assigned Shift</p>
                <p className="font-medium">{shiftLabels[workerData.assigned_shift] || workerData.assigned_shift}</p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <Camera className="w-5 h-5 text-stone-400" />
              <div>
                <p className="text-sm text-stone-500">Face Registration</p>
                <p className={`font-medium ${workerData.face_registered ? 'text-green-600' : 'text-yellow-600'}`}>
                  {workerData.face_registered ? 'Registered' : 'Not Registered'}
                </p>
              </div>
            </div>
          </div>
        </Card>

        {/* Contact Information */}
        <Card title="Contact Information">
          {isEditing ? (
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-stone-700 mb-1">Phone Number</label>
                <input
                  type="tel"
                  value={editForm.phone}
                  onChange={(e) => setEditForm({ ...editForm, phone: e.target.value })}
                  className="w-full px-3 py-2 border border-stone-300 rounded-lg focus:ring-2 focus:ring-orange-500"
                  placeholder="Enter phone number"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-stone-700 mb-1">Emergency Contact</label>
                <input
                  type="tel"
                  value={editForm.emergency_contact}
                  onChange={(e) => setEditForm({ ...editForm, emergency_contact: e.target.value })}
                  className="w-full px-3 py-2 border border-stone-300 rounded-lg focus:ring-2 focus:ring-orange-500"
                  placeholder="Enter emergency contact"
                />
              </div>
              <div className="flex gap-3">
                <button
                  onClick={handleSave}
                  disabled={saving}
                  className="flex items-center gap-2 px-4 py-2 bg-orange-600 text-white rounded-lg hover:bg-orange-700"
                >
                  <Save size={16} />
                  {saving ? 'Saving...' : 'Save Changes'}
                </button>
                <button
                  onClick={() => setIsEditing(false)}
                  className="px-4 py-2 bg-stone-200 text-stone-700 rounded-lg hover:bg-stone-300"
                >
                  Cancel
                </button>
              </div>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="flex items-center gap-3">
                <Phone className="w-5 h-5 text-stone-400" />
                <div>
                  <p className="text-sm text-stone-500">Phone</p>
                  <p className="font-medium">{workerData.phone || 'Not set'}</p>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <AlertTriangle className="w-5 h-5 text-stone-400" />
                <div>
                  <p className="text-sm text-stone-500">Emergency Contact</p>
                  <p className="font-medium">{workerData.emergency_contact || 'Not set'}</p>
                </div>
              </div>
            </div>
          )}
        </Card>

        {/* Badges */}
        {workerData.badges.length > 0 && (
          <Card title="Earned Badges">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {workerData.badges.map((badge) => {
                const config = badgeConfig[badge] || {
                  icon: <Award className="w-5 h-5" />,
                  color: 'bg-stone-100 text-stone-700 border-stone-200',
                  label: badge,
                  description: 'Achievement unlocked'
                };
                return (
                  <div
                    key={badge}
                    className={`flex items-center gap-3 p-4 rounded-xl border ${config.color}`}
                  >
                    <div className="flex-shrink-0">{config.icon}</div>
                    <div>
                      <p className="font-medium">{config.label}</p>
                      <p className="text-xs opacity-70">{config.description}</p>
                    </div>
                  </div>
                );
              })}
            </div>
          </Card>
        )}

        {/* Change Password */}
        <Card title="Security">
          <button
            onClick={() => setChangePassword(true)}
            className="btn btn-secondary"
          >
            Change Password
          </button>
        </Card>

        {/* Password Change Modal */}
        {changePassword && (
          <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
            <div className="bg-white rounded-xl p-6 max-w-md w-full mx-4">
              <h3 className="text-lg font-bold mb-4">Change Password</h3>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-stone-700 mb-1">Current Password</label>
                  <input
                    type="password"
                    value={passwordForm.currentPassword}
                    onChange={(e) => setPasswordForm({ ...passwordForm, currentPassword: e.target.value })}
                    className="w-full px-3 py-2 border border-stone-300 rounded-lg focus:ring-2 focus:ring-orange-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-stone-700 mb-1">New Password</label>
                  <input
                    type="password"
                    value={passwordForm.newPassword}
                    onChange={(e) => setPasswordForm({ ...passwordForm, newPassword: e.target.value })}
                    className="w-full px-3 py-2 border border-stone-300 rounded-lg focus:ring-2 focus:ring-orange-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-stone-700 mb-1">Confirm New Password</label>
                  <input
                    type="password"
                    value={passwordForm.confirmPassword}
                    onChange={(e) => setPasswordForm({ ...passwordForm, confirmPassword: e.target.value })}
                    className="w-full px-3 py-2 border border-stone-300 rounded-lg focus:ring-2 focus:ring-orange-500"
                  />
                </div>
                {passwordError && (
                  <p className="text-red-500 text-sm">{passwordError}</p>
                )}
              </div>
              <div className="flex justify-end gap-3 mt-6">
                <button
                  onClick={() => setChangePassword(false)}
                  className="btn btn-secondary"
                >
                  Cancel
                </button>
                <button
                  onClick={handlePasswordChange}
                  disabled={saving}
                  className="px-4 py-2 bg-orange-600 text-white rounded-lg hover:bg-orange-700"
                >
                  {saving ? 'Saving...' : 'Change Password'}
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </AppLayout>
  );
}
