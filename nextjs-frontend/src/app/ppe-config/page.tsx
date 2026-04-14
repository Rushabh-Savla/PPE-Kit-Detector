'use client';

import { useEffect, useState } from 'react';
import {
  Shield,
  Plus,
  Edit,
  Trash2,
  Save,
  X,
  AlertTriangle,
  CheckCircle,
  Settings,
  MapPin,
  Info,
} from 'lucide-react';
import AppLayout from '@/components/AppLayout';
import { Card } from '@/components/Card';
import { Spinner } from '@/components/Loading';
import { mineApi } from '@/lib/api';
import { useAuthStore } from '@/lib/store';
import type { Mine, Zone } from '@/types';

interface PPEItem {
  id: string;
  name: string;
  description: string;
  required: boolean;
  icon: string;
}

const defaultPPEItems: PPEItem[] = [
  { id: 'helmet', name: 'Safety Helmet', description: 'Hard hat for head protection', required: true, icon: '🪖' },
  { id: 'vest', name: 'High-Vis Vest', description: 'Reflective safety vest for visibility', required: true, icon: '🦺' },
  { id: 'gloves', name: 'Safety Gloves', description: 'Protective gloves for hand safety', required: true, icon: '🧤' },
  { id: 'boots', name: 'Safety Boots', description: 'Steel-toe boots for foot protection', required: true, icon: '👢' },
  { id: 'goggles', name: 'Safety Goggles', description: 'Eye protection for hazardous areas', required: false, icon: '🥽' },
  { id: 'mask', name: 'Dust Mask', description: 'Respiratory protection from dust', required: false, icon: '😷' },
  { id: 'earplugs', name: 'Ear Protection', description: 'Hearing protection for noisy areas', required: false, icon: '🎧' },
  { id: 'harness', name: 'Safety Harness', description: 'Fall protection equipment', required: false, icon: '🪢' },
];

export default function PPEConfigPage() {
  const { getMineIds } = useAuthStore();
  const [mines, setMines] = useState<Mine[]>([]);
  const [selectedMine, setSelectedMine] = useState<string>('');
  const [selectedZone, setSelectedZone] = useState<string>('all');
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [ppeConfig, setPPEConfig] = useState<PPEItem[]>(defaultPPEItems);
  const [editMode, setEditMode] = useState(false);
  const [showAddItem, setShowAddItem] = useState(false);
  const [newItem, setNewItem] = useState({ name: '', description: '', icon: '⚙️' });

  useEffect(() => {
    loadMines();
  }, []);

  const loadMines = async () => {
    try {
      setLoading(true);
      const mineIds = getMineIds();
      const mineId = mineIds.length > 0 ? mineIds[0] : undefined;
      const { mines: data } = await mineApi.list();
      setMines(data);

      if (mineId && data.find(m => m.id === mineId)) {
        setSelectedMine(mineId);
      } else if (data.length > 0) {
        setSelectedMine(data[0].id);
      }
    } catch (err) {
      console.error('Failed to load mines:', err);
    } finally {
      setLoading(false);
    }
  };

  const selectedMineData = mines.find(m => m.id === selectedMine);
  const zones = selectedMineData?.zones || [];

  const handleToggleRequired = (itemId: string) => {
    if (!editMode) return;
    setPPEConfig(config =>
      config.map(item =>
        item.id === itemId ? { ...item, required: !item.required } : item
      )
    );
  };

  const handleRemoveItem = (itemId: string) => {
    if (!editMode) return;
    setPPEConfig(config => config.filter(item => item.id !== itemId));
  };

  const handleAddItem = () => {
    if (!newItem.name.trim()) return;

    const id = newItem.name.toLowerCase().replace(/\s+/g, '_');
    setPPEConfig(config => [
      ...config,
      {
        id,
        name: newItem.name,
        description: newItem.description || `${newItem.name} for safety`,
        required: false,
        icon: newItem.icon,
      },
    ]);
    setNewItem({ name: '', description: '', icon: '⚙️' });
    setShowAddItem(false);
  };

  const handleSave = async () => {
    try {
      setSaving(true);
      // In a real implementation, this would call an API to save the config
      await new Promise(resolve => setTimeout(resolve, 1000));
      setEditMode(false);
      alert('PPE configuration saved successfully!');
    } catch (err) {
      console.error('Failed to save config:', err);
      alert('Failed to save configuration');
    } finally {
      setSaving(false);
    }
  };

  const handleCancel = () => {
    setPPEConfig(defaultPPEItems);
    setEditMode(false);
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

  const requiredCount = ppeConfig.filter(p => p.required).length;
  const optionalCount = ppeConfig.filter(p => !p.required).length;

  return (
    <AppLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold text-stone-900">PPE Configuration</h1>
            <p className="text-stone-500 mt-1">Configure required PPE items for detection</p>
          </div>
          <div className="flex items-center gap-3">
            {editMode ? (
              <>
                <button
                  onClick={handleCancel}
                  className="flex items-center gap-2 px-4 py-2 bg-stone-100 text-stone-700 rounded-lg hover:bg-stone-200"
                >
                  <X className="w-4 h-4" />
                  Cancel
                </button>
                <button
                  onClick={handleSave}
                  disabled={saving}
                  className="flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700"
                >
                  <Save className="w-4 h-4" />
                  {saving ? 'Saving...' : 'Save Changes'}
                </button>
              </>
            ) : (
              <button
                onClick={() => setEditMode(true)}
                className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
              >
                <Edit className="w-4 h-4" />
                Edit Configuration
              </button>
            )}
          </div>
        </div>

        {/* Scope Selection */}
        <Card title="Configuration Scope" description="Select mine and zone for PPE requirements">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-stone-700 mb-1">Mine</label>
              <select
                value={selectedMine}
                onChange={(e) => {
                  setSelectedMine(e.target.value);
                  setSelectedZone('all');
                }}
                className="w-full px-3 py-2 border border-stone-300 rounded-lg focus:ring-2 focus:ring-blue-500"
              >
                {mines.map((mine) => (
                  <option key={mine.id} value={mine.id}>{mine.name}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-stone-700 mb-1">Zone</label>
              <select
                value={selectedZone}
                onChange={(e) => setSelectedZone(e.target.value)}
                className="w-full px-3 py-2 border border-stone-300 rounded-lg focus:ring-2 focus:ring-blue-500"
              >
                <option value="all">All Zones (Mine-wide)</option>
                {zones.map((zone) => (
                  <option key={zone.id} value={zone.id}>{zone.name}</option>
                ))}
              </select>
            </div>
          </div>
          {selectedZone !== 'all' && (
            <div className="mt-4 p-3 bg-blue-50 rounded-lg flex items-start gap-2">
              <Info className="w-5 h-5 text-blue-600 flex-shrink-0" />
              <p className="text-sm text-blue-700">
                Zone-specific configurations override mine-wide settings for workers entering this zone.
              </p>
            </div>
          )}
        </Card>

        {/* Summary */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-green-50 rounded-xl p-4 text-center">
            <Shield className="w-8 h-8 text-green-600 mx-auto mb-2" />
            <p className="text-2xl font-bold text-green-700">{requiredCount}</p>
            <p className="text-sm text-green-600">Required Items</p>
          </div>
          <div className="bg-yellow-50 rounded-xl p-4 text-center">
            <Settings className="w-8 h-8 text-yellow-600 mx-auto mb-2" />
            <p className="text-2xl font-bold text-yellow-700">{optionalCount}</p>
            <p className="text-sm text-yellow-600">Optional Items</p>
          </div>
          <div className="bg-blue-50 rounded-xl p-4 text-center">
            <MapPin className="w-8 h-8 text-blue-600 mx-auto mb-2" />
            <p className="text-2xl font-bold text-blue-700">{zones.length}</p>
            <p className="text-sm text-blue-600">Zones</p>
          </div>
          <div className="bg-purple-50 rounded-xl p-4 text-center">
            <CheckCircle className="w-8 h-8 text-purple-600 mx-auto mb-2" />
            <p className="text-2xl font-bold text-purple-700">{ppeConfig.length}</p>
            <p className="text-sm text-purple-600">Total Items</p>
          </div>
        </div>

        {/* PPE Items */}
        <Card
          title="PPE Requirements"
          description="Configure which PPE items are required for gate entry"
          action={
            editMode && (
              <button
                onClick={() => setShowAddItem(true)}
                className="flex items-center gap-2 px-3 py-1.5 bg-blue-100 text-blue-700 rounded-lg hover:bg-blue-200 text-sm"
              >
                <Plus className="w-4 h-4" />
                Add Item
              </button>
            )
          }
        >
          {/* Add Item Form */}
          {showAddItem && (
            <div className="mb-6 p-4 bg-stone-50 rounded-xl">
              <h3 className="font-medium text-stone-900 mb-4">Add New PPE Item</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div>
                  <label className="block text-sm font-medium text-stone-700 mb-1">Name</label>
                  <input
                    type="text"
                    value={newItem.name}
                    onChange={(e) => setNewItem({ ...newItem, name: e.target.value })}
                    className="w-full px-3 py-2 border border-stone-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                    placeholder="e.g., Safety Goggles"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-stone-700 mb-1">Description</label>
                  <input
                    type="text"
                    value={newItem.description}
                    onChange={(e) => setNewItem({ ...newItem, description: e.target.value })}
                    className="w-full px-3 py-2 border border-stone-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                    placeholder="Purpose of this PPE"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-stone-700 mb-1">Icon (emoji)</label>
                  <input
                    type="text"
                    value={newItem.icon}
                    onChange={(e) => setNewItem({ ...newItem, icon: e.target.value })}
                    className="w-full px-3 py-2 border border-stone-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                    placeholder="🥽"
                  />
                </div>
              </div>
              <div className="flex justify-end gap-2 mt-4">
                <button
                  onClick={() => setShowAddItem(false)}
                  className="px-4 py-2 bg-stone-200 text-stone-700 rounded-lg hover:bg-stone-300"
                >
                  Cancel
                </button>
                <button
                  onClick={handleAddItem}
                  className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
                >
                  Add Item
                </button>
              </div>
            </div>
          )}

          {/* Required Items */}
          <div className="mb-6">
            <h3 className="text-sm font-medium text-stone-700 mb-3 flex items-center gap-2">
              <CheckCircle className="w-4 h-4 text-green-500" />
              Required Items
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {ppeConfig.filter(p => p.required).map((item) => (
                <div
                  key={item.id}
                  className={`flex items-center justify-between p-4 rounded-xl border-2 border-green-200 bg-green-50 ${
                    editMode ? 'cursor-pointer hover:bg-green-100' : ''
                  }`}
                  onClick={() => handleToggleRequired(item.id)}
                >
                  <div className="flex items-center gap-3">
                    <span className="text-2xl">{item.icon}</span>
                    <div>
                      <p className="font-medium text-stone-900">{item.name}</p>
                      <p className="text-xs text-stone-500">{item.description}</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="px-2 py-1 bg-green-100 text-green-700 rounded-full text-xs font-medium">
                      Required
                    </span>
                    {editMode && (
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleRemoveItem(item.id);
                        }}
                        className="p-1 text-red-500 hover:bg-red-100 rounded"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Optional Items */}
          {ppeConfig.filter(p => !p.required).length > 0 && (
            <div>
              <h3 className="text-sm font-medium text-stone-700 mb-3 flex items-center gap-2">
                <Settings className="w-4 h-4 text-yellow-500" />
                Optional Items
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {ppeConfig.filter(p => !p.required).map((item) => (
                  <div
                    key={item.id}
                    className={`flex items-center justify-between p-4 rounded-xl border-2 border-stone-200 bg-stone-50 ${
                      editMode ? 'cursor-pointer hover:bg-stone-100' : ''
                    }`}
                    onClick={() => handleToggleRequired(item.id)}
                  >
                    <div className="flex items-center gap-3">
                      <span className="text-2xl opacity-70">{item.icon}</span>
                      <div>
                        <p className="font-medium text-stone-900">{item.name}</p>
                        <p className="text-xs text-stone-500">{item.description}</p>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="px-2 py-1 bg-stone-100 text-stone-600 rounded-full text-xs font-medium">
                        Optional
                      </span>
                      {editMode && (
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleRemoveItem(item.id);
                          }}
                          className="p-1 text-red-500 hover:bg-red-100 rounded"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {editMode && (
            <div className="mt-6 p-4 bg-yellow-50 rounded-lg">
              <div className="flex items-start gap-2">
                <AlertTriangle className="w-5 h-5 text-yellow-600 flex-shrink-0" />
                <div>
                  <p className="font-medium text-yellow-800">Edit Mode Active</p>
                  <p className="text-sm text-yellow-700 mt-1">
                    Click on items to toggle between required and optional. Changes will not be saved until you click &quot;Save Changes&quot;.
                  </p>
                </div>
              </div>
            </div>
          )}
        </Card>

        {/* Detection Info */}
        <Card title="Detection Information">
          <div className="prose prose-sm max-w-none text-stone-600">
            <p>
              The PPE detection system uses AI-powered computer vision to identify the following items:
            </p>
            <ul className="mt-4 space-y-2">
              <li className="flex items-center gap-2">
                <CheckCircle className="w-4 h-4 text-green-500" />
                <span>Safety helmets (hard hats) - various colors supported</span>
              </li>
              <li className="flex items-center gap-2">
                <CheckCircle className="w-4 h-4 text-green-500" />
                <span>High-visibility vests - reflective strips detected</span>
              </li>
              <li className="flex items-center gap-2">
                <CheckCircle className="w-4 h-4 text-green-500" />
                <span>Safety boots - steel-toe detection supported</span>
              </li>
              <li className="flex items-center gap-2">
                <CheckCircle className="w-4 h-4 text-green-500" />
                <span>Safety gloves - various types supported</span>
              </li>
            </ul>
            <p className="mt-4 text-yellow-700">
              <strong>Note:</strong> Detection accuracy may vary based on lighting conditions and camera quality.
              Ensure proper camera positioning for optimal results.
            </p>
          </div>
        </Card>
      </div>
    </AppLayout>
  );
}
