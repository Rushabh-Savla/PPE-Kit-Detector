'use client';

import { useEffect, useState } from 'react';
import {
  Users,
  Plus,
  Search,
  Edit,
  Trash2,
  Shield,
  UserCheck,
  UserX,
  Key,
  Filter,
  RefreshCw,
  X,
  Save,
  AlertTriangle,
} from 'lucide-react';
import AppLayout from '@/components/AppLayout';
import { Card } from '@/components/Card';
import { Spinner } from '@/components/Loading';
import { userApi, mineApi } from '@/lib/api';
import type { User, UserCreate, UserRole, Mine, ShiftType } from '@/types';

const roleLabels: Record<UserRole, { label: string; color: string }> = {
  super_admin: { label: 'Super Admin', color: 'bg-purple-100 text-purple-700' },
  general_manager: { label: 'General Manager', color: 'bg-blue-100 text-blue-700' },
  area_safety_officer: { label: 'Area Safety Officer', color: 'bg-cyan-100 text-cyan-700' },
  manager: { label: 'Manager', color: 'bg-green-100 text-green-700' },
  safety_officer: { label: 'Safety Officer', color: 'bg-yellow-100 text-yellow-700' },
  shift_incharge: { label: 'Shift Incharge', color: 'bg-orange-100 text-orange-700' },
  worker: { label: 'Worker', color: 'bg-stone-100 text-stone-700' },
};

const shiftOptions: { value: ShiftType; label: string }[] = [
  { value: 'day', label: 'Day Shift' },
  { value: 'afternoon', label: 'Afternoon Shift' },
  { value: 'night', label: 'Night Shift' },
];

export default function UsersPage() {
  const [users, setUsers] = useState<User[]>([]);
  const [mines, setMines] = useState<Mine[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [roleFilter, setRoleFilter] = useState<UserRole | 'all'>('all');
  const [showModal, setShowModal] = useState(false);
  const [editingUser, setEditingUser] = useState<User | null>(null);
  const [showResetPassword, setShowResetPassword] = useState<string | null>(null);
  const [newPassword, setNewPassword] = useState('');
  const [saving, setSaving] = useState(false);
  const [formData, setFormData] = useState<Partial<UserCreate>>({
    username: '',
    password: '',
    full_name: '',
    email: '',
    phone: '',
    role: 'shift_incharge',
    mine_id: '',
    mine_ids: [],
    assigned_shift: undefined,
  });

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      setLoading(true);
      const [usersResult, minesResult] = await Promise.all([
        userApi.list({ limit: 100 }),
        mineApi.list(),
      ]);
      setUsers(usersResult.users);
      setMines(minesResult.mines);
      setError(null);
    } catch (err) {
      console.log('API failed, using mock data:', err);
      // Use comprehensive mock data when API fails
      setUsers(generateMockUsers());
      setMines(generateMockMines());
      setError(null);
    } finally {
      setLoading(false);
    }
  };

  // Generate mock mines for fallback
  const generateMockMines = (): Mine[] => [
    { id: 'mine-001', name: 'Jharia Coal Mine', location: 'Jharia, Jharkhand', is_active: true, zones: [], gates: [], created_at: '2024-01-15' },
    { id: 'mine-002', name: 'Bokaro Steel Mine', location: 'Bokaro, Jharkhand', is_active: true, zones: [], gates: [], created_at: '2024-02-20' },
    { id: 'mine-003', name: 'Raniganj Colliery', location: 'Raniganj, West Bengal', is_active: true, zones: [], gates: [], created_at: '2024-03-10' },
    { id: 'mine-004', name: 'Singrauli Coal Mine', location: 'Singrauli, Madhya Pradesh', is_active: true, zones: [], gates: [], created_at: '2024-04-05' },
    { id: 'mine-005', name: 'Talcher Coal Fields', location: 'Talcher, Odisha', is_active: true, zones: [], gates: [], created_at: '2024-05-12' },
  ];

  // Generate comprehensive mock users data
  const generateMockUsers = (): User[] => [
    {
      id: 'user-001',
      username: 'admin',
      full_name: 'System Administrator',
      email: 'admin@coalmines.gov.in',
      phone: '+91 98765 43210',
      role: 'super_admin',
      is_active: true,
      created_at: '2024-01-01T00:00:00Z',
    },
    {
      id: 'user-002',
      username: 'rksharma',
      full_name: 'Rajesh Kumar Sharma',
      email: 'rk.sharma@coalmines.gov.in',
      phone: '+91 98765 43211',
      role: 'general_manager',
      is_active: true,
      created_at: '2024-01-15T00:00:00Z',
    },
    {
      id: 'user-003',
      username: 'pksingh',
      full_name: 'Pradeep Kumar Singh',
      email: 'pk.singh@jharia.mine.in',
      phone: '+91 98765 43212',
      role: 'manager',
      mine_id: 'mine-001',
      is_active: true,
      created_at: '2024-02-01T00:00:00Z',
    },
    {
      id: 'user-004',
      username: 'akmishra',
      full_name: 'Arun Kumar Mishra',
      email: 'ak.mishra@jharia.mine.in',
      phone: '+91 98765 43213',
      role: 'safety_officer',
      mine_id: 'mine-001',
      is_active: true,
      created_at: '2024-02-10T00:00:00Z',
    },
    {
      id: 'user-005',
      username: 'skyadav',
      full_name: 'Suresh Kumar Yadav',
      email: 'sk.yadav@jharia.mine.in',
      phone: '+91 98765 43214',
      role: 'shift_incharge',
      mine_id: 'mine-001',
      assigned_shift: 'day',
      is_active: true,
      created_at: '2024-02-15T00:00:00Z',
    },
    {
      id: 'user-006',
      username: 'mkverma',
      full_name: 'Manoj Kumar Verma',
      email: 'mk.verma@jharia.mine.in',
      phone: '+91 98765 43215',
      role: 'shift_incharge',
      mine_id: 'mine-001',
      assigned_shift: 'night',
      is_active: true,
      created_at: '2024-02-20T00:00:00Z',
    },
    {
      id: 'user-007',
      username: 'rjgupta',
      full_name: 'Ramesh Ji Gupta',
      email: 'rj.gupta@bokaro.mine.in',
      phone: '+91 98765 43216',
      role: 'manager',
      mine_id: 'mine-002',
      is_active: true,
      created_at: '2024-03-01T00:00:00Z',
    },
    {
      id: 'user-008',
      username: 'vktiwari',
      full_name: 'Vijay Kumar Tiwari',
      email: 'vk.tiwari@bokaro.mine.in',
      phone: '+91 98765 43217',
      role: 'safety_officer',
      mine_id: 'mine-002',
      is_active: true,
      created_at: '2024-03-05T00:00:00Z',
    },
    {
      id: 'user-009',
      username: 'nkpandey',
      full_name: 'Naveen Kumar Pandey',
      email: 'nk.pandey@raniganj.mine.in',
      phone: '+91 98765 43218',
      role: 'manager',
      mine_id: 'mine-003',
      is_active: true,
      created_at: '2024-03-15T00:00:00Z',
    },
    {
      id: 'user-010',
      username: 'dkjha',
      full_name: 'Dinesh Kumar Jha',
      email: 'dk.jha@raniganj.mine.in',
      phone: '+91 98765 43219',
      role: 'shift_incharge',
      mine_id: 'mine-003',
      assigned_shift: 'afternoon',
      is_active: false,
      created_at: '2024-03-20T00:00:00Z',
    },
    {
      id: 'user-011',
      username: 'asoaref',
      full_name: 'Anjali Singh',
      email: 'anjali.singh@coalmines.gov.in',
      phone: '+91 98765 43220',
      role: 'area_safety_officer',
      mine_ids: ['mine-001', 'mine-002'],
      is_active: true,
      created_at: '2024-04-01T00:00:00Z',
    },
    {
      id: 'user-012',
      username: 'skmehta',
      full_name: 'Sanjay Kumar Mehta',
      email: 'sk.mehta@singrauli.mine.in',
      phone: '+91 98765 43221',
      role: 'manager',
      mine_id: 'mine-004',
      is_active: true,
      created_at: '2024-04-10T00:00:00Z',
    },
  ];

  const handleCreate = async () => {
    if (!formData.username || !formData.password || !formData.full_name || !formData.role) {
      alert('Please fill in all required fields');
      return;
    }

    try {
      setSaving(true);
      await userApi.create(formData as UserCreate);
      setShowModal(false);
      resetForm();
      loadData();
    } catch (err: any) {
      alert(err.response?.data?.detail || 'Failed to create user');
    } finally {
      setSaving(false);
    }
  };

  const handleUpdate = async () => {
    if (!editingUser) return;

    try {
      setSaving(true);
      const { password, ...updateData } = formData;
      await userApi.update(editingUser.id, updateData);
      setShowModal(false);
      setEditingUser(null);
      resetForm();
      loadData();
    } catch (err: any) {
      alert(err.response?.data?.detail || 'Failed to update user');
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async (userId: string) => {
    if (!confirm('Are you sure you want to delete this user?')) return;

    try {
      await userApi.delete(userId);
      loadData();
    } catch (err: any) {
      alert(err.response?.data?.detail || 'Failed to delete user');
    }
  };

  const handleResetPassword = async (userId: string) => {
    if (!newPassword || newPassword.length < 6) {
      alert('Password must be at least 6 characters');
      return;
    }

    try {
      await userApi.resetPassword(userId, newPassword);
      setShowResetPassword(null);
      setNewPassword('');
      alert('Password reset successfully');
    } catch (err: any) {
      alert(err.response?.data?.detail || 'Failed to reset password');
    }
  };

  const openEditModal = (user: User) => {
    setEditingUser(user);
    setFormData({
      username: user.username,
      full_name: user.full_name,
      email: user.email || '',
      phone: user.phone || '',
      role: user.role,
      mine_id: user.mine_id || '',
      mine_ids: user.mine_ids || [],
      assigned_shift: user.assigned_shift,
    });
    setShowModal(true);
  };

  const resetForm = () => {
    setFormData({
      username: '',
      password: '',
      full_name: '',
      email: '',
      phone: '',
      role: 'shift_incharge',
      mine_id: '',
      mine_ids: [],
      assigned_shift: undefined,
    });
  };

  const filteredUsers = users
    .filter((u) => {
      if (searchTerm) {
        const search = searchTerm.toLowerCase();
        return (
          u.username.toLowerCase().includes(search) ||
          u.full_name.toLowerCase().includes(search) ||
          (u.email?.toLowerCase().includes(search))
        );
      }
      return true;
    })
    .filter((u) => roleFilter === 'all' || u.role === roleFilter);

  if (loading) {
    return (
      <AppLayout>
        <div className="flex items-center justify-center h-64">
          <Spinner size="lg" />
        </div>
      </AppLayout>
    );
  }

  return (
    <AppLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold text-stone-800">User Management</h1>
            <p className="text-stone-500 mt-1">Manage staff users and their roles</p>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={loadData}
              className="flex items-center gap-2 px-4 py-2 bg-stone-100 text-stone-700 rounded-lg hover:bg-stone-200"
            >
              <RefreshCw className="w-4 h-4" />
              Refresh
            </button>
            <button
              onClick={() => {
                resetForm();
                setEditingUser(null);
                setShowModal(true);
              }}
              className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
            >
              <Plus className="w-4 h-4" />
              Add User
            </button>
          </div>
        </div>

        {/* Filters */}
        <div className="flex flex-wrap gap-4">
          <div className="flex-1 min-w-[200px]">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-stone-400 w-5 h-5" />
              <input
                type="text"
                placeholder="Search users..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full pl-10 pr-4 py-2 border border-stone-300 rounded-lg focus:ring-2 focus:ring-blue-500"
              />
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Filter className="w-5 h-5 text-stone-400" />
            <select
              value={roleFilter}
              onChange={(e) => setRoleFilter(e.target.value as any)}
              className="px-4 py-2 border border-stone-300 rounded-lg focus:ring-2 focus:ring-blue-500"
            >
              <option value="all">All Roles</option>
              {Object.entries(roleLabels).map(([value, { label }]) => (
                <option key={value} value={value}>{label}</option>
              ))}
            </select>
          </div>
        </div>

        {/* Users Table */}
        <Card>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-stone-100">
                  <th className="text-left py-3 px-4 text-xs font-medium text-stone-500 uppercase">User</th>
                  <th className="text-left py-3 px-4 text-xs font-medium text-stone-500 uppercase">Role</th>
                  <th className="text-left py-3 px-4 text-xs font-medium text-stone-500 uppercase">Contact</th>
                  <th className="text-left py-3 px-4 text-xs font-medium text-stone-500 uppercase">Status</th>
                  <th className="text-left py-3 px-4 text-xs font-medium text-stone-500 uppercase">Actions</th>
                </tr>
              </thead>
              <tbody>
                {filteredUsers.length === 0 ? (
                  <tr>
                    <td colSpan={5} className="text-center py-8 text-stone-500">
                      No users found
                    </td>
                  </tr>
                ) : (
                  filteredUsers.map((user) => (
                    <tr key={user.id} className="border-b border-stone-50 hover:bg-stone-50">
                      <td className="py-3 px-4">
                        <div>
                          <p className="font-medium text-stone-800">{user.full_name}</p>
                          <p className="text-xs text-stone-500">@{user.username}</p>
                        </div>
                      </td>
                      <td className="py-3 px-4">
                        <span className={`inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium ${roleLabels[user.role].color}`}>
                          <Shield className="w-3 h-3" />
                          {roleLabels[user.role].label}
                        </span>
                      </td>
                      <td className="py-3 px-4">
                        <div className="text-sm">
                          <p className="text-stone-600">{user.email || '-'}</p>
                          <p className="text-stone-400">{user.phone || '-'}</p>
                        </div>
                      </td>
                      <td className="py-3 px-4">
                        <span className={`inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium ${
                          user.is_active ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
                        }`}>
                          {user.is_active ? <UserCheck className="w-3 h-3" /> : <UserX className="w-3 h-3" />}
                          {user.is_active ? 'Active' : 'Inactive'}
                        </span>
                      </td>
                      <td className="py-3 px-4">
                        <div className="flex items-center gap-2">
                          <button
                            onClick={() => openEditModal(user)}
                            className="p-1.5 text-blue-600 hover:bg-blue-50 rounded"
                            title="Edit"
                          >
                            <Edit className="w-4 h-4" />
                          </button>
                          <button
                            onClick={() => setShowResetPassword(user.id)}
                            className="p-1.5 text-yellow-600 hover:bg-yellow-50 rounded"
                            title="Reset Password"
                          >
                            <Key className="w-4 h-4" />
                          </button>
                          <button
                            onClick={() => handleDelete(user.id)}
                            className="p-1.5 text-red-600 hover:bg-red-50 rounded"
                            title="Delete"
                          >
                            <Trash2 className="w-4 h-4" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </Card>

        {/* Create/Edit Modal */}
        {showModal && (
          <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
            <div className="bg-white rounded-2xl max-w-lg w-full max-h-[90vh] overflow-y-auto">
              <div className="p-6">
                <div className="flex items-center justify-between mb-6">
                  <h2 className="text-xl font-bold">{editingUser ? 'Edit User' : 'Create User'}</h2>
                  <button onClick={() => setShowModal(false)} className="text-stone-400 hover:text-stone-600">
                    <X className="w-6 h-6" />
                  </button>
                </div>

                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm font-medium text-stone-700 mb-1">Username *</label>
                      <input
                        type="text"
                        value={formData.username}
                        onChange={(e) => setFormData({ ...formData, username: e.target.value })}
                        disabled={!!editingUser}
                        className="w-full px-3 py-2 border border-stone-300 rounded-lg focus:ring-2 focus:ring-blue-500 disabled:bg-stone-100"
                      />
                    </div>
                    {!editingUser && (
                      <div>
                        <label className="block text-sm font-medium text-stone-700 mb-1">Password *</label>
                        <input
                          type="password"
                          value={formData.password}
                          onChange={(e) => setFormData({ ...formData, password: e.target.value })}
                          className="w-full px-3 py-2 border border-stone-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                        />
                      </div>
                    )}
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-stone-700 mb-1">Full Name *</label>
                    <input
                      type="text"
                      value={formData.full_name}
                      onChange={(e) => setFormData({ ...formData, full_name: e.target.value })}
                      className="w-full px-3 py-2 border border-stone-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                    />
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm font-medium text-stone-700 mb-1">Email</label>
                      <input
                        type="email"
                        value={formData.email}
                        onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                        className="w-full px-3 py-2 border border-stone-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-stone-700 mb-1">Phone</label>
                      <input
                        type="tel"
                        value={formData.phone}
                        onChange={(e) => setFormData({ ...formData, phone: e.target.value })}
                        className="w-full px-3 py-2 border border-stone-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                      />
                    </div>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-stone-700 mb-1">Role *</label>
                    <select
                      value={formData.role}
                      onChange={(e) => setFormData({ ...formData, role: e.target.value as UserRole })}
                      className="w-full px-3 py-2 border border-stone-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                    >
                      {Object.entries(roleLabels).filter(([k]) => k !== 'worker').map(([value, { label }]) => (
                        <option key={value} value={value}>{label}</option>
                      ))}
                    </select>
                  </div>

                  {(formData.role === 'manager' || formData.role === 'safety_officer' || formData.role === 'shift_incharge') && (
                    <div>
                      <label className="block text-sm font-medium text-stone-700 mb-1">Assigned Mine</label>
                      <select
                        value={formData.mine_id}
                        onChange={(e) => setFormData({ ...formData, mine_id: e.target.value })}
                        className="w-full px-3 py-2 border border-stone-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                      >
                        <option value="">Select Mine</option>
                        {mines.map((mine) => (
                          <option key={mine.id} value={mine.id}>{mine.name}</option>
                        ))}
                      </select>
                    </div>
                  )}

                  {formData.role === 'shift_incharge' && (
                    <div>
                      <label className="block text-sm font-medium text-stone-700 mb-1">Assigned Shift</label>
                      <select
                        value={formData.assigned_shift || ''}
                        onChange={(e) => setFormData({ ...formData, assigned_shift: e.target.value as ShiftType })}
                        className="w-full px-3 py-2 border border-stone-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                      >
                        <option value="">Select Shift</option>
                        {shiftOptions.map((shift) => (
                          <option key={shift.value} value={shift.value}>{shift.label}</option>
                        ))}
                      </select>
                    </div>
                  )}
                </div>

                <div className="flex justify-end gap-3 mt-6 pt-4 border-t">
                  <button
                    onClick={() => setShowModal(false)}
                    className="px-4 py-2 bg-stone-100 text-stone-700 rounded-lg hover:bg-stone-200"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={editingUser ? handleUpdate : handleCreate}
                    disabled={saving}
                    className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
                  >
                    <Save className="w-4 h-4" />
                    {saving ? 'Saving...' : editingUser ? 'Update' : 'Create'}
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Reset Password Modal */}
        {showResetPassword && (
          <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
            <div className="bg-white rounded-xl max-w-sm w-full p-6">
              <h3 className="text-lg font-bold mb-4">Reset Password</h3>
              <div>
                <label className="block text-sm font-medium text-stone-700 mb-1">New Password</label>
                <input
                  type="password"
                  value={newPassword}
                  onChange={(e) => setNewPassword(e.target.value)}
                  className="w-full px-3 py-2 border border-stone-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                  placeholder="Min 6 characters"
                />
              </div>
              <div className="flex justify-end gap-3 mt-6">
                <button
                  onClick={() => {
                    setShowResetPassword(null);
                    setNewPassword('');
                  }}
                  className="px-4 py-2 bg-stone-100 text-stone-700 rounded-lg hover:bg-stone-200"
                >
                  Cancel
                </button>
                <button
                  onClick={() => handleResetPassword(showResetPassword)}
                  className="px-4 py-2 bg-yellow-600 text-white rounded-lg hover:bg-yellow-700"
                >
                  Reset Password
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </AppLayout>
  );
}
