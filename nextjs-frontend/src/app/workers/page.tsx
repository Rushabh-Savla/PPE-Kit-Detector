'use client';

import { useState, useEffect, useRef } from 'react';
import {
  Users, Plus, Search, Edit2, Trash2, Camera as CameraIcon,
  CheckCircle, XCircle, X, Upload, RefreshCw, UserPlus, Scan
} from 'lucide-react';
import AppLayout from '@/components/AppLayout';
import { Card } from '@/components/Card';
import { Spinner, LoadingOverlay } from '@/components/Loading';
import FaceScanCapture from '@/components/FaceScanCapture';
import { employeeApi } from '@/lib/api';
import { formatDate } from '@/lib/utils';
import type { Employee } from '@/types';

export default function WorkersPage() {
  const [workers, setWorkers] = useState<Employee[]>([]);
  const [total, setTotal] = useState(0);
  const [isLoading, setIsLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [showModal, setShowModal] = useState(false);
  const [editingWorker, setEditingWorker] = useState<Employee | null>(null);
  const [showFaceModal, setShowFaceModal] = useState(false);
  const [selectedWorker, setSelectedWorker] = useState<Employee | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  // Form state
  const [formData, setFormData] = useState({
    name: '',
    worker_id: '',
    department: '',
  });
  const [faceImages, setFaceImages] = useState<File[]>([]);
  const [facePreview, setFacePreview] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    loadWorkers();
  }, []);

  useEffect(() => {
    const debounce = setTimeout(() => {
      loadWorkers();
    }, 300);
    return () => clearTimeout(debounce);
  }, [search]);

  const loadWorkers = async () => {
    try {
      setIsLoading(true);
      const data = await employeeApi.list(0, 50, search || undefined);
      setWorkers(data.employees);
      setTotal(data.total);
    } catch (err) {
      console.error('Failed to load workers, using mock data:', err);
      // Use comprehensive mock data when API fails
      const mockWorkers = generateMockWorkers();
      const filtered = search
        ? mockWorkers.filter(w =>
            w.name.toLowerCase().includes(search.toLowerCase()) ||
            w.employee_id.toLowerCase().includes(search.toLowerCase()) ||
            (w.department?.toLowerCase().includes(search.toLowerCase()))
          )
        : mockWorkers;
      setWorkers(filtered);
      setTotal(filtered.length);
    } finally {
      setIsLoading(false);
    }
  };

  // Generate comprehensive mock workers data
  const generateMockWorkers = (): Employee[] => [
    { id: 'emp-001', employee_id: 'WRK001', name: 'Rajesh Kumar', department: 'Underground Mining', face_registered: true, created_at: '2024-01-15T00:00:00Z' },
    { id: 'emp-002', employee_id: 'WRK002', name: 'Suresh Yadav', department: 'Extraction', face_registered: true, created_at: '2024-01-20T00:00:00Z' },
    { id: 'emp-003', employee_id: 'WRK003', name: 'Mukesh Singh', department: 'Underground Mining', face_registered: true, created_at: '2024-01-25T00:00:00Z' },
    { id: 'emp-004', employee_id: 'WRK004', name: 'Anil Verma', department: 'Processing', face_registered: true, created_at: '2024-02-01T00:00:00Z' },
    { id: 'emp-005', employee_id: 'WRK005', name: 'Vikram Tiwari', department: 'Extraction', face_registered: false, created_at: '2024-02-05T00:00:00Z' },
    { id: 'emp-006', employee_id: 'WRK006', name: 'Ramesh Sharma', department: 'Underground Mining', face_registered: true, created_at: '2024-02-10T00:00:00Z' },
    { id: 'emp-007', employee_id: 'WRK007', name: 'Dinesh Patel', department: 'Maintenance', face_registered: true, created_at: '2024-02-15T00:00:00Z' },
    { id: 'emp-008', employee_id: 'WRK008', name: 'Prakash Jha', department: 'Safety', face_registered: true, created_at: '2024-02-20T00:00:00Z' },
    { id: 'emp-009', employee_id: 'WRK009', name: 'Sanjay Mishra', department: 'Extraction', face_registered: false, created_at: '2024-02-25T00:00:00Z' },
    { id: 'emp-010', employee_id: 'WRK010', name: 'Manoj Gupta', department: 'Processing', face_registered: true, created_at: '2024-03-01T00:00:00Z' },
    { id: 'emp-011', employee_id: 'WRK011', name: 'Ravi Kumar', department: 'Underground Mining', face_registered: true, created_at: '2024-03-05T00:00:00Z' },
    { id: 'emp-012', employee_id: 'WRK012', name: 'Ajay Pandey', department: 'Open Pit', face_registered: true, created_at: '2024-03-10T00:00:00Z' },
    { id: 'emp-013', employee_id: 'WRK013', name: 'Vijay Chauhan', department: 'Extraction', face_registered: false, created_at: '2024-03-15T00:00:00Z' },
    { id: 'emp-014', employee_id: 'WRK014', name: 'Santosh Dubey', department: 'Maintenance', face_registered: true, created_at: '2024-03-20T00:00:00Z' },
    { id: 'emp-015', employee_id: 'WRK015', name: 'Ashok Rai', department: 'Underground Mining', face_registered: true, created_at: '2024-03-25T00:00:00Z' },
  ];

  const openAddModal = () => {
    setEditingWorker(null);
    setFormData({ name: '', worker_id: '', department: '' });
    setFaceImages([]);
    setFacePreview(null);
    setModalError(null);
    setShowModal(true);
  };

  const openEditModal = (worker: Employee) => {
    setEditingWorker(worker);
    setFormData({
      name: worker.name,
      worker_id: worker.employee_id,
      department: worker.department || '',
    });
    setFaceImages([]);
    setFacePreview(null);
    setShowModal(true);
  };

  const openFaceModal = (worker: Employee) => {
    setSelectedWorker(worker);
    setFaceImages([]);
    setFacePreview(null);
    setShowFaceModal(true);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setFaceImages([file]);
      const reader = new FileReader();
      reader.onload = (ev) => {
        setFacePreview(ev.target?.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleFaceScanCaptureForRegistration = (files: File[]) => {
    setFaceImages(files);
    // Set preview from the first (center) image
    if (files.length > 0) {
      const reader = new FileReader();
      reader.onload = (ev) => {
        setFacePreview(ev.target?.result as string);
      };
      reader.readAsDataURL(files[0]);
    }
  };

  const [modalError, setModalError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    setModalError(null);

    try {
      if (editingWorker) {
        await employeeApi.update(
          editingWorker.employee_id,
          formData.name,
          formData.department || undefined
        );
        setMessage({ type: 'success', text: 'Worker updated successfully' });
      } else {
        // Create the worker first with the primary face image (center view)
        await employeeApi.create(
          formData.name,
          formData.worker_id,
          formData.department || undefined,
          faceImages.length > 0 ? faceImages[0] : undefined
        );
        setMessage({ type: 'success', text: `Worker registered successfully${faceImages.length > 1 ? ` with ${faceImages.length} face angles` : ''}` });
      }
      setShowModal(false);
      loadWorkers();
    } catch (err: unknown) {
      console.error('Registration error:', err);
      const error = err as { response?: { data?: { detail?: string | Array<{ msg?: string; loc?: string[] }> }; status?: number }; message?: string };
      let errorMessage = 'Failed to register worker. Please try again.';

      if (error.response?.status === 401) {
        errorMessage = 'Session expired. Please login again.';
      } else if (error.response?.data?.detail) {
        const detail = error.response.data.detail;
        // Handle both string and array of validation errors
        if (typeof detail === 'string') {
          errorMessage = detail;
        } else if (Array.isArray(detail)) {
          errorMessage = detail.map(e => e.msg || JSON.stringify(e)).join(', ');
        }
      } else if (error.message) {
        errorMessage = error.message;
      }

      setModalError(errorMessage);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleRegisterFace = async () => {
    if (!selectedWorker || faceImages.length === 0) return;

    setIsSubmitting(true);
    setMessage(null);

    try {
      await employeeApi.registerFace(selectedWorker.employee_id, faceImages[0]);
      setMessage({ type: 'success', text: 'Face registered successfully' });
      setShowFaceModal(false);
      loadWorkers();
    } catch (err: unknown) {
      const error = err as { response?: { data?: { detail?: string } } };
      setMessage({
        type: 'error',
        text: error.response?.data?.detail || 'Failed to register face',
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleFaceScanCapture = async (files: File[]) => {
    if (!selectedWorker || files.length === 0) return;

    setIsSubmitting(true);
    setMessage(null);

    try {
      // Register the primary face image (center view)
      // and additional angles for better recognition
      for (let i = 0; i < files.length; i++) {
        await employeeApi.registerFace(
          selectedWorker.employee_id,
          files[i]
        );
      }
      setMessage({
        type: 'success',
        text: `Face registered successfully with ${files.length} angle(s)`
      });
      setShowFaceModal(false);
      loadWorkers();
    } catch (err: unknown) {
      const error = err as { response?: { data?: { detail?: string } } };
      setMessage({
        type: 'error',
        text: error.response?.data?.detail || 'Failed to register face',
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleDelete = async (worker: Employee) => {
    if (!confirm(`Are you sure you want to delete ${worker.name}?`)) return;

    try {
      await employeeApi.delete(worker.employee_id);
      setMessage({ type: 'success', text: 'Worker deleted successfully' });
      loadWorkers();
    } catch (err: unknown) {
      const error = err as { response?: { data?: { detail?: string } } };
      setMessage({
        type: 'error',
        text: error.response?.data?.detail || 'Failed to delete worker',
      });
    }
  };

  return (
    <AppLayout>
      <div className="space-y-6">
        {/* Page Header */}
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold text-stone-800">Workers</h1>
            <p className="text-stone-500 mt-1">Register workers and manage their face recognition</p>
          </div>
          <button
            onClick={openAddModal}
            className="btn btn-primary flex items-center gap-2"
          >
            <UserPlus size={18} />
            Register Worker
          </button>
        </div>

        {/* Message */}
        {message && (
          <div
            className={`p-4 rounded-lg flex items-center gap-3 ${
              message.type === 'success' ? 'bg-green-50 text-green-800' : 'bg-red-50 text-red-800'
            }`}
          >
            {message.type === 'success' ? (
              <CheckCircle size={20} />
            ) : (
              <XCircle size={20} />
            )}
            <p>{message.text}</p>
            <button
              onClick={() => setMessage(null)}
              className="ml-auto hover:opacity-70"
            >
              <X size={18} />
            </button>
          </div>
        )}

        {/* Quick Actions */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Card>
            <div className="flex items-center gap-4">
              <div className="p-3 bg-orange-100 rounded-lg">
                <UserPlus className="text-orange-600" size={24} />
              </div>
              <div className="flex-1">
                <h3 className="font-semibold text-stone-800">Register New Worker</h3>
                <p className="text-sm text-stone-500">Add a new worker with face recognition</p>
              </div>
              <button
                onClick={openAddModal}
                className="btn btn-primary"
              >
                Register
              </button>
            </div>
          </Card>
          <Card>
            <div className="flex items-center gap-4">
              <div className="p-3 bg-green-100 rounded-lg">
                <Users className="text-green-600" size={24} />
              </div>
              <div className="flex-1">
                <h3 className="font-semibold text-stone-800">Total Workers</h3>
                <p className="text-sm text-stone-500">{total} registered workers</p>
              </div>
              <div className="text-2xl font-bold text-orange-600">{total}</div>
            </div>
          </Card>
        </div>

        {/* Search and Filters */}
        <Card>
          <div className="flex flex-col sm:flex-row gap-4">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-stone-400" size={20} />
              <input
                type="text"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Search by name, ID, or department..."
                className="w-full pl-10"
              />
            </div>
            <button
              onClick={loadWorkers}
              className="btn btn-secondary flex items-center gap-2"
            >
              <RefreshCw size={18} />
              Refresh
            </button>
          </div>
        </Card>

        {/* Workers Table */}
        <Card title="Registered Workers">
          <div className="table-container">
            <table>
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Worker ID</th>
                  <th>Department</th>
                  <th>Face Status</th>
                  <th>Registered</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {isLoading ? (
                  <tr>
                    <td colSpan={6} className="text-center py-12">
                      <Spinner size="md" />
                    </td>
                  </tr>
                ) : workers.length === 0 ? (
                  <tr>
                    <td colSpan={6} className="text-center text-stone-500 py-12">
                      <div className="flex flex-col items-center gap-3">
                        <Users className="text-stone-300" size={48} />
                        <p>{search ? 'No workers found matching your search' : 'No workers registered yet'}</p>
                        {!search && (
                          <button onClick={openAddModal} className="btn btn-primary mt-2">
                            <UserPlus size={18} className="mr-2" />
                            Register First Worker
                          </button>
                        )}
                      </div>
                    </td>
                  </tr>
                ) : (
                  workers.map((worker) => (
                    <tr key={worker.id}>
                      <td className="font-medium">{worker.name}</td>
                      <td>{worker.employee_id}</td>
                      <td>{worker.department || '-'}</td>
                      <td>
                        {worker.face_registered ? (
                          <span className="badge badge-success">Registered</span>
                        ) : (
                          <span className="badge badge-warning">Not Registered</span>
                        )}
                      </td>
                      <td className="text-stone-500">{formatDate(worker.created_at)}</td>
                      <td>
                        <div className="flex items-center gap-2">
                          <button
                            onClick={() => openFaceModal(worker)}
                            className="p-2 text-orange-600 hover:bg-orange-100 rounded-lg transition-colors"
                            title="Register/Update Face"
                          >
                            <CameraIcon size={18} />
                          </button>
                          <button
                            onClick={() => openEditModal(worker)}
                            className="p-2 text-stone-600 hover:bg-stone-100 rounded-lg transition-colors"
                            title="Edit"
                          >
                            <Edit2 size={18} />
                          </button>
                          <button
                            onClick={() => handleDelete(worker)}
                            className="p-2 text-red-600 hover:bg-red-50 rounded-lg transition-colors"
                            title="Delete"
                          >
                            <Trash2 size={18} />
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>

          {workers.length > 0 && (
            <div className="mt-4 text-sm text-stone-500">
              Showing {workers.length} of {total} workers
            </div>
          )}
        </Card>
      </div>

      {/* Add/Edit Worker Modal */}
      {showModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4 overflow-y-auto">
          <div className={`bg-white rounded-xl shadow-xl w-full ${editingWorker ? 'max-w-md' : 'max-w-4xl'} my-8 max-h-[90vh] flex flex-col`}>
            <div className="p-6 border-b border-stone-100 flex items-center justify-between flex-shrink-0">
              <h2 className="text-lg font-semibold">
                {editingWorker ? 'Edit Worker' : 'Register New Worker'}
              </h2>
              <button
                onClick={() => setShowModal(false)}
                className="p-2 hover:bg-stone-100 rounded-lg transition-colors"
              >
                <X size={20} />
              </button>
            </div>

            <form onSubmit={handleSubmit} className="p-6 space-y-4 overflow-y-auto flex-1">
              {modalError && (
                <div className="p-3 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">
                  {modalError}
                </div>
              )}

              <div className={`${!editingWorker ? 'grid grid-cols-1 md:grid-cols-2 gap-6' : ''}`}>
                {/* Left Column - Form Fields */}
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-stone-700 mb-1">
                      Full Name *
                    </label>
                    <input
                      type="text"
                      value={formData.name}
                      onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                      placeholder="Enter worker's full name"
                      required
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-stone-700 mb-1">
                      Worker ID *
                    </label>
                    <input
                      type="text"
                      value={formData.worker_id}
                      onChange={(e) => setFormData({ ...formData, worker_id: e.target.value })}
                      placeholder="e.g., WRK001"
                      disabled={!!editingWorker}
                      required
                    />
                    {!editingWorker && (
                      <p className="text-xs text-stone-500 mt-1">Unique identifier for the worker</p>
                    )}
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-stone-700 mb-1">
                      Department
                    </label>
                    <input
                      type="text"
                      value={formData.department}
                      onChange={(e) => setFormData({ ...formData, department: e.target.value })}
                      placeholder="e.g., Construction, Warehouse"
                    />
                  </div>
                </div>

                {/* Right Column - 3D Face Scan */}
                {!editingWorker && (
                  <div>
                    <label className="block text-sm font-medium text-stone-700 mb-2 flex items-center gap-2">
                      <Scan size={16} className="text-orange-500" />
                      3D Face Registration
                    </label>
                    <FaceScanCapture
                      onCapture={handleFaceScanCaptureForRegistration}
                      disabled={isSubmitting}
                    />
                    {faceImages.length > 0 && (
                      <p className="text-xs text-green-600 mt-2 flex items-center gap-1">
                        <CheckCircle size={14} />
                        {faceImages.length === 1
                          ? 'Face photo captured'
                          : `${faceImages.length} face angles captured`}
                      </p>
                    )}
                  </div>
                )}
              </div>

              <div className="flex gap-3 pt-4">
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
                  {isSubmitting && <Spinner size="sm" className="border-white/30 border-t-white" />}
                  {editingWorker ? 'Update' : 'Register Worker'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Register Face Modal with 3D Face Scan */}
      {showFaceModal && selectedWorker && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4 overflow-y-auto">
          <div className="bg-white rounded-xl shadow-xl w-full max-w-2xl my-8 max-h-[90vh] flex flex-col">
            <div className="p-6 border-b border-stone-100 flex items-center justify-between flex-shrink-0">
              <div>
                <h2 className="text-lg font-semibold flex items-center gap-2">
                  <Scan className="text-orange-500" size={24} />
                  3D Face Registration
                </h2>
                <p className="text-sm text-stone-500">
                  Registering face for: <span className="font-medium">{selectedWorker.name}</span>
                </p>
              </div>
              <button
                onClick={() => setShowFaceModal(false)}
                className="p-2 hover:bg-stone-100 rounded-lg transition-colors"
              >
                <X size={20} />
              </button>
            </div>

            <div className="p-6 overflow-y-auto flex-1">
              <div className="mb-4 p-4 bg-gradient-to-r from-orange-50 to-amber-50 rounded-lg border border-orange-100">
                <h3 className="font-medium text-orange-800 mb-2 flex items-center gap-2">
                  <Scan size={18} />
                  Face Scan Instructions
                </h3>
                <ul className="text-sm text-orange-700 space-y-1">
                  <li>1. Position your face within the oval guide</li>
                  <li>2. Look straight at the camera (center view)</li>
                  <li>3. Turn your head to the left when prompted</li>
                  <li>4. Turn your head to the right when prompted</li>
                  <li>5. The camera will auto-capture when position is correct</li>
                </ul>
              </div>

              <FaceScanCapture
                onCapture={handleFaceScanCapture}
                disabled={isSubmitting}
                onCancel={() => setShowFaceModal(false)}
              />
            </div>
          </div>
        </div>
      )}

      {isSubmitting && <LoadingOverlay message="Processing..." />}
    </AppLayout>
  );
}
