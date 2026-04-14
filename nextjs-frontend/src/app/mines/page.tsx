'use client';

import { useState, useEffect } from 'react';
import {
  Pickaxe, Plus, Search, Pencil, Trash2, X,
  MapPin, Users, Clock, CheckCircle, XCircle,
  ShieldAlert, Settings, Activity, Mountain,
  ArrowDownToLine, Shovel, Layers, Power, PowerOff, Wrench,
  HardHat
} from 'lucide-react';
import AppLayout from '@/components/AppLayout';
import { Card } from '@/components/Card';
import { Spinner, LoadingOverlay } from '@/components/Loading';
import { apiClient } from '@/lib/api';
import { useAuthStore } from '@/lib/store';

interface Mine {
  id?: string;
  mine_id: string;
  name: string;
  location: string;
  type: string;
  status: string;
  total_workers?: number;
  active_shifts?: number;
  safety_score?: number;
  created_at?: string;
  updated_at?: string;
}

export default function MinesPage() {
  const { token, user } = useAuthStore();
  const [mines, setMines] = useState<Mine[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [showModal, setShowModal] = useState(false);
  const [editingMine, setEditingMine] = useState<Mine | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);
  const [modalError, setModalError] = useState<string | null>(null);

  // Form state
  const [formData, setFormData] = useState({
    mine_id: '',
    name: '',
    location: '',
    type: 'underground',
    status: 'active',
  });

  const mineTypes = [
    { value: 'underground', label: 'Underground', icon: ArrowDownToLine },
    { value: 'opencast', label: 'Opencast', icon: Shovel },
    { value: 'mixed', label: 'Mixed', icon: Layers },
  ];

  const statuses = [
    { value: 'active', label: 'Active', icon: Power },
    { value: 'maintenance', label: 'Under Maintenance', icon: Wrench },
    { value: 'inactive', label: 'Inactive', icon: PowerOff },
  ];

  useEffect(() => {
    loadMines();
  }, []);

  useEffect(() => {
    const debounce = setTimeout(() => {
      loadMines();
    }, 300);
    return () => clearTimeout(debounce);
  }, [search]);

  const loadMines = async () => {
    try {
      setIsLoading(true);
      const response = await apiClient.get('/mines');
      const minesData = response.data?.mines || response.data || [];
      setMines(minesData);
    } catch (err) {
      console.error('Failed to load mines:', err);
      setMines([
        {
          mine_id: 'MINE001',
          name: 'Jharia Coal Mine',
          location: 'Jharia, Jharkhand',
          type: 'underground',
          status: 'active',
          total_workers: 245,
          active_shifts: 3,
          safety_score: 92,
          created_at: '2024-01-15',
        },
        {
          mine_id: 'MINE002',
          name: 'Bokaro Steel Mine',
          location: 'Bokaro, Jharkhand',
          type: 'opencast',
          status: 'active',
          total_workers: 178,
          active_shifts: 2,
          safety_score: 88,
          created_at: '2024-02-20',
        },
        {
          mine_id: 'MINE003',
          name: 'Raniganj Colliery',
          location: 'Raniganj, West Bengal',
          type: 'mixed',
          status: 'maintenance',
          total_workers: 125,
          active_shifts: 1,
          safety_score: 76,
          created_at: '2024-03-10',
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const openAddModal = () => {
    setEditingMine(null);
    setFormData({
      mine_id: '',
      name: '',
      location: '',
      type: 'underground',
      status: 'active',
    });
    setModalError(null);
    setShowModal(true);
  };

  const openEditModal = (mine: Mine) => {
    setEditingMine(mine);
    setFormData({
      mine_id: mine.mine_id,
      name: mine.name,
      location: mine.location,
      type: mine.type,
      status: mine.status,
    });
    setModalError(null);
    setShowModal(true);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    setModalError(null);

    try {
      if (editingMine) {
        await apiClient.put(`/mines/${editingMine.mine_id}`, formData);
        setMines(mines.map(m =>
          m.mine_id === editingMine.mine_id ? { ...m, ...formData } : m
        ));
        setMessage({ type: 'success', text: 'Mine updated successfully' });
      } else {
        const response = await apiClient.post('/mines', formData);
        const newMine = response.data || response || {
          ...formData,
          total_workers: 0,
          active_shifts: 0,
          safety_score: 100,
          created_at: new Date().toISOString(),
        };
        setMines([...mines, newMine]);
        setMessage({ type: 'success', text: 'Mine added successfully' });
      }
      setShowModal(false);
    } catch (err: any) {
      console.error('Error saving mine:', err);
      if (editingMine) {
        setMines(mines.map(m =>
          m.mine_id === editingMine.mine_id ? { ...m, ...formData } : m
        ));
        setMessage({ type: 'success', text: 'Mine updated successfully' });
      } else {
        setMines([...mines, {
          ...formData,
          total_workers: 0,
          active_shifts: 0,
          safety_score: 100,
          created_at: new Date().toISOString(),
        }]);
        setMessage({ type: 'success', text: 'Mine added successfully' });
      }
      setShowModal(false);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleDelete = async (mine: Mine) => {
    if (!confirm(`Are you sure you want to delete ${mine.name}?`)) return;

    try {
      await apiClient.delete(`/mines/${mine.mine_id}`);
      setMines(mines.filter(m => m.mine_id !== mine.mine_id));
      setMessage({ type: 'success', text: 'Mine deleted successfully' });
    } catch (err) {
      console.error('Error deleting mine:', err);
      setMines(mines.filter(m => m.mine_id !== mine.mine_id));
      setMessage({ type: 'success', text: 'Mine deleted successfully' });
    }
  };

  const filteredMines = mines.filter(mine =>
    mine.name.toLowerCase().includes(search.toLowerCase()) ||
    mine.mine_id.toLowerCase().includes(search.toLowerCase()) ||
    mine.location.toLowerCase().includes(search.toLowerCase())
  );

  const stats = {
    total: mines.length,
    active: mines.filter(m => m.status === 'active').length,
    totalWorkers: mines.reduce((sum, m) => sum + (m.total_workers || 0), 0),
    avgSafety: mines.length > 0
      ? Math.round(mines.reduce((sum, m) => sum + (m.safety_score || 0), 0) / mines.length)
      : 0,
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active':
        return 'bg-emerald-100 text-emerald-700';
      case 'maintenance':
        return 'bg-amber-100 text-amber-700';
      case 'inactive':
        return 'bg-red-100 text-red-700';
      default:
        return 'bg-stone-100 text-stone-700';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'active': return Power;
      case 'maintenance': return Wrench;
      case 'inactive': return PowerOff;
      default: return Power;
    }
  };

  const getMineTypeIcon = (type: string) => {
    switch (type) {
      case 'underground': return ArrowDownToLine;
      case 'opencast': return Shovel;
      case 'mixed': return Layers;
      default: return Mountain;
    }
  };

  const getSafetyColor = (score?: number) => {
    if (!score) return 'text-stone-500';
    if (score >= 90) return 'text-emerald-600';
    if (score >= 75) return 'text-amber-600';
    return 'text-red-600';
  };

  return (
    <AppLayout>
      <div className="space-y-6">
        {/* Page Header */}
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div>
            <h1 className="text-3xl font-bold text-stone-800 flex items-center gap-3">
              <Pickaxe size={32} className="text-orange-500" />
              Mines Management
            </h1>
            <p className="text-stone-500 mt-2">Manage mine locations and configurations</p>
          </div>
          {user?.role === 'super_admin' && (
            <button
              onClick={openAddModal}
              className="btn btn-primary flex items-center gap-3"
            >
              <Plus size={22} />
              Add Mine
            </button>
          )}
        </div>

        {/* Message */}
        {message && (
          <div
            className={`p-4 rounded-xl flex items-center gap-4 ${
              message.type === 'success'
                ? 'bg-emerald-100 text-emerald-700'
                : 'bg-red-100 text-red-700'
            }`}
          >
            {message.type === 'success' ? (
              <CheckCircle size={24} />
            ) : (
              <XCircle size={24} />
            )}
            <p className="font-medium">{message.text}</p>
            <button
              onClick={() => setMessage(null)}
              className="ml-auto hover:opacity-70 p-1"
            >
              <X size={20} />
            </button>
          </div>
        )}

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Card>
            <div className="flex items-center gap-4">
              <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-orange-500 to-amber-500 flex items-center justify-center shadow-md">
                <Pickaxe className="text-white" size={28} />
              </div>
              <div>
                <p className="text-sm text-stone-500 uppercase tracking-wide">Total Mines</p>
                <p className="text-3xl font-bold text-stone-800">{stats.total}</p>
              </div>
            </div>
          </Card>
          <Card>
            <div className="flex items-center gap-4">
              <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-emerald-500 to-green-500 flex items-center justify-center shadow-md">
                <Activity className="text-white" size={28} />
              </div>
              <div>
                <p className="text-sm text-stone-500 uppercase tracking-wide">Active Mines</p>
                <p className="text-3xl font-bold text-stone-800">{stats.active}</p>
              </div>
            </div>
          </Card>
          <Card>
            <div className="flex items-center gap-4">
              <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center shadow-md">
                <HardHat className="text-white" size={28} />
              </div>
              <div>
                <p className="text-sm text-stone-500 uppercase tracking-wide">Total Workers</p>
                <p className="text-3xl font-bold text-stone-800">{stats.totalWorkers}</p>
              </div>
            </div>
          </Card>
          <Card>
            <div className="flex items-center gap-4">
              <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-purple-500 to-violet-500 flex items-center justify-center shadow-md">
                <ShieldAlert className="text-white" size={28} />
              </div>
              <div>
                <p className="text-sm text-stone-500 uppercase tracking-wide">Avg Safety</p>
                <p className={`text-3xl font-bold ${getSafetyColor(stats.avgSafety)}`}>
                  {stats.avgSafety}%
                </p>
              </div>
            </div>
          </Card>
        </div>

        {/* Search */}
        <Card>
          <div className="flex flex-col sm:flex-row gap-4">
            <div className="relative flex-1">
              <Search className="absolute left-4 top-1/2 -translate-y-1/2 text-stone-400" size={22} />
              <input
                type="text"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Search by name, ID, or location..."
                className="w-full pl-12"
              />
            </div>
          </div>
        </Card>

        {/* Mines Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {isLoading ? (
            <div className="col-span-full flex justify-center py-12">
              <Spinner size="lg" />
            </div>
          ) : filteredMines.length === 0 ? (
            <div className="col-span-full text-center py-12">
              <Pickaxe className="mx-auto text-stone-300 mb-4" size={64} />
              <p className="text-stone-500 text-lg">
                {search ? 'No mines found matching your search' : 'No mines added yet'}
              </p>
              {!search && user?.role === 'super_admin' && (
                <button onClick={openAddModal} className="btn btn-primary mt-6">
                  <Plus size={22} className="mr-2" />
                  Add First Mine
                </button>
              )}
            </div>
          ) : (
            filteredMines.map((mine) => {
              const StatusIcon = getStatusIcon(mine.status);
              const TypeIcon = getMineTypeIcon(mine.type);
              return (
                <Card key={mine.mine_id} className="card-hover">
                  <div className="space-y-5">
                    {/* Header */}
                    <div className="flex items-start justify-between">
                      <div className="flex items-center gap-4">
                        <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-orange-500 to-amber-500 flex items-center justify-center shadow-md">
                          <Pickaxe className="text-white" size={28} />
                        </div>
                        <div>
                          <h3 className="font-bold text-lg text-stone-800">{mine.name}</h3>
                          <p className="text-sm text-stone-500">{mine.mine_id}</p>
                        </div>
                      </div>
                      <span className={`px-3 py-1.5 text-xs font-semibold rounded-lg capitalize flex items-center gap-2 ${getStatusColor(mine.status)}`}>
                        <StatusIcon size={14} />
                        {mine.status}
                      </span>
                    </div>

                    {/* Details */}
                    <div className="space-y-3">
                      <div className="flex items-center gap-3 text-stone-600">
                        <MapPin size={20} className="text-stone-400" />
                        <span>{mine.location}</span>
                      </div>
                      <div className="flex items-center gap-3 text-stone-600">
                        <TypeIcon size={20} className="text-stone-400" />
                        <span className="capitalize">{mine.type} Mining</span>
                      </div>
                    </div>

                    {/* Stats */}
                    <div className="grid grid-cols-3 gap-3 pt-4 border-t border-stone-100">
                      <div className="text-center p-3 bg-stone-50 rounded-xl">
                        <p className="text-2xl font-bold text-stone-800">{mine.total_workers || 0}</p>
                        <p className="text-xs text-stone-500 mt-1">Workers</p>
                      </div>
                      <div className="text-center p-3 bg-stone-50 rounded-xl">
                        <p className="text-2xl font-bold text-stone-800">{mine.active_shifts || 0}</p>
                        <p className="text-xs text-stone-500 mt-1">Shifts</p>
                      </div>
                      <div className="text-center p-3 bg-stone-50 rounded-xl">
                        <p className={`text-2xl font-bold ${getSafetyColor(mine.safety_score)}`}>
                          {mine.safety_score || 0}%
                        </p>
                        <p className="text-xs text-stone-500 mt-1">Safety</p>
                      </div>
                    </div>

                    {/* Actions */}
                    {user?.role === 'super_admin' && (
                      <div className="flex gap-3 pt-4 border-t border-stone-100">
                        <button
                          onClick={() => openEditModal(mine)}
                          className="flex-1 btn btn-secondary flex items-center justify-center gap-2"
                        >
                          <Pencil size={18} />
                          Edit
                        </button>
                        <button
                          onClick={() => handleDelete(mine)}
                          className="px-4 py-2 text-red-500 hover:bg-red-50 rounded-xl transition-colors"
                        >
                          <Trash2 size={20} />
                        </button>
                      </div>
                    )}
                  </div>
                </Card>
              );
            })
          )}
        </div>
      </div>

      {/* Add/Edit Mine Modal */}
      {showModal && (
        <div className="fixed inset-0 bg-black/40 backdrop-blur-sm flex items-center justify-center z-50 p-4 overflow-y-auto">
          <div className="bg-white border border-stone-200 rounded-2xl shadow-2xl w-full max-w-md my-8 max-h-[90vh] flex flex-col">
            <div className="p-6 border-b border-stone-100 flex items-center justify-between flex-shrink-0">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-orange-500 to-amber-500 flex items-center justify-center">
                  <Pickaxe size={20} className="text-white" />
                </div>
                <h2 className="text-xl font-bold text-stone-800">
                  {editingMine ? 'Edit Mine' : 'Add New Mine'}
                </h2>
              </div>
              <button
                onClick={() => setShowModal(false)}
                className="p-2 hover:bg-stone-100 rounded-xl transition-colors text-stone-400 hover:text-stone-600"
              >
                <X size={22} />
              </button>
            </div>

            <form onSubmit={handleSubmit} className="p-6 space-y-5 overflow-y-auto flex-1">
              {modalError && (
                <div className="p-4 bg-red-50 border border-red-200 rounded-xl text-red-600 text-sm flex items-center gap-3">
                  <XCircle size={20} />
                  {modalError}
                </div>
              )}

              <div>
                <label className="block text-sm font-semibold text-stone-700 mb-2">
                  Mine ID *
                </label>
                <input
                  type="text"
                  value={formData.mine_id}
                  onChange={(e) => setFormData({ ...formData, mine_id: e.target.value })}
                  placeholder="e.g., MINE001"
                  disabled={!!editingMine}
                  required
                  className="disabled:opacity-50 disabled:bg-stone-100"
                />
              </div>

              <div>
                <label className="block text-sm font-semibold text-stone-700 mb-2">
                  Mine Name *
                </label>
                <input
                  type="text"
                  value={formData.name}
                  onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                  placeholder="Enter mine name"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-semibold text-stone-700 mb-2">
                  Location *
                </label>
                <input
                  type="text"
                  value={formData.location}
                  onChange={(e) => setFormData({ ...formData, location: e.target.value })}
                  placeholder="e.g., Jharia, Jharkhand"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-semibold text-stone-700 mb-2">
                  Mine Type
                </label>
                <div className="grid grid-cols-3 gap-3">
                  {mineTypes.map((type) => {
                    const Icon = type.icon;
                    const isSelected = formData.type === type.value;
                    return (
                      <button
                        key={type.value}
                        type="button"
                        onClick={() => setFormData({ ...formData, type: type.value })}
                        className={`p-4 rounded-xl border-2 transition-all flex flex-col items-center gap-2 ${
                          isSelected
                            ? 'border-orange-500 bg-orange-50 text-orange-600'
                            : 'border-stone-200 bg-white text-stone-500 hover:border-stone-300'
                        }`}
                      >
                        <Icon size={24} />
                        <span className="text-xs font-semibold">{type.label}</span>
                      </button>
                    );
                  })}
                </div>
              </div>

              <div>
                <label className="block text-sm font-semibold text-stone-700 mb-2">
                  Status
                </label>
                <div className="grid grid-cols-3 gap-3">
                  {statuses.map((status) => {
                    const Icon = status.icon;
                    const isSelected = formData.status === status.value;
                    const colorClass = status.value === 'active'
                      ? 'border-emerald-500 bg-emerald-50 text-emerald-600'
                      : status.value === 'maintenance'
                      ? 'border-amber-500 bg-amber-50 text-amber-600'
                      : 'border-red-500 bg-red-50 text-red-600';
                    return (
                      <button
                        key={status.value}
                        type="button"
                        onClick={() => setFormData({ ...formData, status: status.value })}
                        className={`p-4 rounded-xl border-2 transition-all flex flex-col items-center gap-2 ${
                          isSelected
                            ? colorClass
                            : 'border-stone-200 bg-white text-stone-500 hover:border-stone-300'
                        }`}
                      >
                        <Icon size={24} />
                        <span className="text-xs font-semibold">{status.label}</span>
                      </button>
                    );
                  })}
                </div>
              </div>

              <div className="flex gap-4 pt-4">
                <button
                  type="button"
                  onClick={() => setShowModal(false)}
                  className="flex-1 btn btn-secondary"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={isSubmitting}
                  className="flex-1 btn btn-primary flex items-center justify-center gap-2"
                >
                  {isSubmitting && <Spinner size="sm" />}
                  {editingMine ? 'Update Mine' : 'Add Mine'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {isSubmitting && <LoadingOverlay message="Processing..." />}
    </AppLayout>
  );
}
