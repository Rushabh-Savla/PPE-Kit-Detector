'use client';

import { useEffect, useState } from 'react';
import {
  AlertTriangle,
  Calendar,
  Clock,
  MapPin,
  Filter,
  ChevronLeft,
  ChevronRight,
  Shield,
  TrendingUp,
  TrendingDown,
  X,
  Eye,
} from 'lucide-react';
import AppLayout from '@/components/AppLayout';
import { Card, StatCard } from '@/components/Card';
import { Spinner } from '@/components/Loading';

interface Violation {
  id: string;
  timestamp: string;
  gate_name?: string;
  violations: string[];
  status: string;
  shift: string;
  zone?: string;
  acknowledged?: boolean;
  image_url?: string;
}

interface ViolationStats {
  total: number;
  thisWeek: number;
  thisMonth: number;
  mostCommon: string;
}

// Mock violations data for Rushabh Savla
const mockViolations: Violation[] = [
  {
    id: 'v1',
    timestamp: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString(),
    gate_name: 'Gate A - Main Entry',
    violations: ['NO-HELMET'],
    status: 'recorded',
    shift: 'Morning',
    zone: 'Zone A - Mining Area',
    acknowledged: true,
    image_url: '/violations/v1.jpg',
  },
  {
    id: 'v2',
    timestamp: new Date(Date.now() - 5 * 24 * 60 * 60 * 1000).toISOString(),
    gate_name: 'Gate B - Equipment Zone',
    violations: ['NO-VEST', 'NO-GLOVES'],
    status: 'recorded',
    shift: 'Afternoon',
    zone: 'Zone B - Equipment Area',
    acknowledged: true,
    image_url: '/violations/v2.jpg',
  },
  {
    id: 'v3',
    timestamp: new Date(Date.now() - 12 * 24 * 60 * 60 * 1000).toISOString(),
    gate_name: 'Gate A - Main Entry',
    violations: ['NO-BOOTS'],
    status: 'recorded',
    shift: 'Morning',
    zone: 'Zone A - Mining Area',
    acknowledged: false,
    image_url: '/violations/v3.jpg',
  },
  {
    id: 'v4',
    timestamp: new Date(Date.now() - 20 * 24 * 60 * 60 * 1000).toISOString(),
    gate_name: 'Gate C - Storage',
    violations: ['NO-HELMET', 'NO-VEST'],
    status: 'recorded',
    shift: 'Night',
    zone: 'Zone C - Storage Area',
    acknowledged: true,
    image_url: '/violations/v4.jpg',
  },
  {
    id: 'v5',
    timestamp: new Date(Date.now() - 35 * 24 * 60 * 60 * 1000).toISOString(),
    gate_name: 'Gate A - Main Entry',
    violations: ['NO-GLOVES'],
    status: 'recorded',
    shift: 'Morning',
    zone: 'Zone A - Mining Area',
    acknowledged: true,
  },
  {
    id: 'v6',
    timestamp: new Date(Date.now() - 45 * 24 * 60 * 60 * 1000).toISOString(),
    gate_name: 'Gate B - Equipment Zone',
    violations: ['NO-VEST'],
    status: 'recorded',
    shift: 'Afternoon',
    zone: 'Zone B - Equipment Area',
    acknowledged: true,
  },
];

export default function MyViolationsPage() {
  const [violations, setViolations] = useState<Violation[]>([]);
  const [stats, setStats] = useState<ViolationStats>({
    total: 0,
    thisWeek: 0,
    thisMonth: 0,
    mostCommon: '-',
  });
  const [loading, setLoading] = useState(true);
  const [page, setPage] = useState(0);
  const [totalPages, setTotalPages] = useState(1);
  const [filter, setFilter] = useState<'all' | 'week' | 'month'>('all');
  const [selectedViolation, setSelectedViolation] = useState<Violation | null>(null);
  const limit = 5;

  useEffect(() => {
    loadViolations();
  }, [page, filter]);

  const loadViolations = async () => {
    setLoading(true);

    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 500));

    // Filter violations based on time period
    const now = new Date();
    const weekAgo = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
    const monthAgo = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000);

    const allViolations = mockViolations || [];
    const weekViolations = allViolations.filter((v) => new Date(v.timestamp) >= weekAgo);
    const monthViolations = allViolations.filter((v) => new Date(v.timestamp) >= monthAgo);

    // Find most common violation type
    const violationCounts: Record<string, number> = {};
    mockViolations.forEach(v => {
      v.violations.forEach(type => {
        violationCounts[type] = (violationCounts[type] || 0) + 1;
      });
    });
    const mostCommon = Object.entries(violationCounts).sort((a, b) => b[1] - a[1])[0]?.[0] || '-';

    setStats({
      total: mockViolations.length,
      thisWeek: weekViolations.length,
      thisMonth: monthViolations.length,
      mostCommon: mostCommon.replace('NO-', 'Missing '),
    });

    setLoading(false);
  };

  const getViolationColor = (type: string) => {
    if (type.includes('HELMET') || type.includes('helmet')) return 'bg-red-100 text-red-700';
    if (type.includes('VEST') || type.includes('vest')) return 'bg-orange-100 text-orange-700';
    if (type.includes('GLOVES') || type.includes('gloves')) return 'bg-yellow-100 text-yellow-700';
    if (type.includes('BOOTS') || type.includes('boots')) return 'bg-blue-100 text-blue-700';
    return 'bg-stone-100 text-stone-700';
  };

  return (
    <AppLayout>
      <div className="max-w-4xl mx-auto space-y-6">
        {/* Header with Filter */}
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold text-stone-800">My Violations</h1>
            <p className="text-stone-500 mt-1">Track and learn from your PPE compliance history</p>
          </div>
          <div className="flex items-center gap-2">
            <Filter className="w-4 h-4 text-stone-400" />
            <div className="flex bg-stone-100 rounded-lg p-1">
              {(['all', 'week', 'month'] as const).map((f) => (
                <button
                  key={f}
                  onClick={() => { setFilter(f); setPage(0); }}
                  className={`px-3 py-1.5 text-sm font-medium rounded-md transition-colors ${
                    filter === f
                      ? 'bg-white text-orange-600 shadow-sm'
                      : 'text-stone-600 hover:text-stone-800'
                  }`}
                >
                  {f === 'all' ? 'All Time' : f === 'week' ? 'This Week' : 'This Month'}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <StatCard
            title="Total Violations"
            value={stats.total}
            icon={<AlertTriangle size={24} />}
            color="red"
          />
          <StatCard
            title="This Week"
            value={stats.thisWeek}
            icon={<Calendar size={24} />}
            color={stats.thisWeek === 0 ? 'green' : 'yellow'}
          />
          <StatCard
            title="This Month"
            value={stats.thisMonth}
            icon={<Calendar size={24} />}
            color={stats.thisMonth === 0 ? 'green' : 'yellow'}
          />
          <StatCard
            title="Most Common"
            value={stats.mostCommon}
            icon={<TrendingUp size={24} />}
            color="blue"
          />
        </div>

        {/* Compliance Tip */}
        {stats.total > 0 && (
          <div className="bg-orange-50 border border-orange-200 rounded-xl p-4">
            <div className="flex items-start gap-3">
              <Shield className="w-6 h-6 text-orange-600 flex-shrink-0" />
              <div>
                <p className="font-medium text-orange-800">Improve Your Compliance</p>
                <p className="text-sm text-orange-700 mt-1">
                  {stats.mostCommon !== '-'
                    ? `Your most common violation is "${stats.mostCommon}". Make sure to double-check this item before entering the work area.`
                    : 'Review your violations below and take steps to improve your safety compliance.'}
                </p>
              </div>
            </div>
          </div>
        )}

        {/* No Violations Message */}
        {stats.total === 0 && !loading && (
          <div className="bg-green-50 border border-green-200 rounded-xl p-8 text-center">
            <Shield className="w-16 h-16 text-green-500 mx-auto mb-4" />
            <h3 className="text-xl font-bold text-green-800">Excellent Compliance!</h3>
            <p className="text-green-700 mt-2">
              You have no recorded violations. Keep up the great work!
            </p>
          </div>
        )}

        {/* Violations List */}
        {stats.total > 0 && (
          <Card title="Violation History">
            {loading ? (
              <div className="flex items-center justify-center py-12">
                <Spinner size="lg" />
              </div>
            ) : (
              <>
                <div className="space-y-4">
                  {violations.map((violation) => (
                    <button
                      key={violation.id}
                      onClick={() => setSelectedViolation(violation)}
                      className="w-full p-4 bg-stone-50 rounded-xl border border-stone-100 hover:border-orange-200 hover:bg-orange-50/30 transition-all text-left group"
                    >
                      <div className="flex items-start justify-between">
                        <div className="flex items-start gap-3">
                          <div className="w-10 h-10 bg-red-100 rounded-lg flex items-center justify-center flex-shrink-0 group-hover:bg-red-200 transition-colors">
                            <AlertTriangle className="w-5 h-5 text-red-600" />
                          </div>
                          <div>
                            <div className="flex flex-wrap gap-2 mb-2">
                              {violation.violations.map((type, i) => (
                                <span
                                  key={i}
                                  className={`px-2 py-1 rounded-full text-xs font-medium ${getViolationColor(type)}`}
                                >
                                  {type.replace('NO-', 'Missing ')}
                                </span>
                              ))}
                            </div>
                            <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-sm text-stone-500">
                              <span className="flex items-center gap-1">
                                <Calendar className="w-4 h-4" />
                                {new Date(violation.timestamp).toLocaleDateString()}
                              </span>
                              <span className="flex items-center gap-1">
                                <Clock className="w-4 h-4" />
                                {new Date(violation.timestamp).toLocaleTimeString()}
                              </span>
                              {violation.gate_name && (
                                <span className="flex items-center gap-1">
                                  <MapPin className="w-4 h-4" />
                                  {violation.gate_name}
                                </span>
                              )}
                            </div>
                          </div>
                        </div>
                        <div className="flex flex-col items-end gap-2">
                          <span className="text-xs px-2 py-1 bg-stone-200 text-stone-600 rounded-full capitalize">
                            {violation.shift}
                          </span>
                          <Eye className="w-4 h-4 text-stone-400 opacity-0 group-hover:opacity-100 transition-opacity" />
                        </div>
                      </div>
                    </button>
                  ))}
                </div>

                {/* Pagination */}
                {totalPages > 1 && (
                  <div className="flex items-center justify-between mt-6 pt-4 border-t">
                    <button
                      onClick={() => setPage(p => Math.max(0, p - 1))}
                      disabled={page === 0}
                      className="flex items-center gap-1 px-3 py-2 text-sm font-medium text-stone-700 bg-white border border-stone-300 rounded-lg hover:bg-stone-50 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      <ChevronLeft className="w-4 h-4" />
                      Previous
                    </button>
                    <span className="text-sm text-stone-500">
                      Page {page + 1} of {totalPages}
                    </span>
                    <button
                      onClick={() => setPage(p => Math.min(totalPages - 1, p + 1))}
                      disabled={page >= totalPages - 1}
                      className="flex items-center gap-1 px-3 py-2 text-sm font-medium text-stone-700 bg-white border border-stone-300 rounded-lg hover:bg-stone-50 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      Next
                      <ChevronRight className="w-4 h-4" />
                    </button>
                  </div>
                )}
              </>
            )}
          </Card>
        )}

        {/* Safety Reminders */}
        <Card title="PPE Checklist">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="flex items-center gap-3 p-3 bg-red-50 rounded-lg">
              <div className="w-10 h-10 bg-red-100 rounded-lg flex items-center justify-center">
                <span className="text-2xl">🪖</span>
              </div>
              <div>
                <p className="font-medium text-stone-800">Safety Helmet</p>
                <p className="text-xs text-stone-500">Must be worn at all times</p>
              </div>
            </div>
            <div className="flex items-center gap-3 p-3 bg-orange-50 rounded-lg">
              <div className="w-10 h-10 bg-orange-100 rounded-lg flex items-center justify-center">
                <span className="text-2xl">🦺</span>
              </div>
              <div>
                <p className="font-medium text-stone-800">High-Vis Vest</p>
                <p className="text-xs text-stone-500">Required in all work areas</p>
              </div>
            </div>
            <div className="flex items-center gap-3 p-3 bg-yellow-50 rounded-lg">
              <div className="w-10 h-10 bg-yellow-100 rounded-lg flex items-center justify-center">
                <span className="text-2xl">🧤</span>
              </div>
              <div>
                <p className="font-medium text-stone-800">Safety Gloves</p>
                <p className="text-xs text-stone-500">For handling equipment</p>
              </div>
            </div>
            <div className="flex items-center gap-3 p-3 bg-blue-50 rounded-lg">
              <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center">
                <span className="text-2xl">👢</span>
              </div>
              <div>
                <p className="font-medium text-stone-800">Safety Boots</p>
                <p className="text-xs text-stone-500">Steel-toe boots required</p>
              </div>
            </div>
          </div>
        </Card>

        {/* Violation Detail Modal */}
        {selectedViolation && (
          <div
            className="fixed inset-0 bg-black/50 flex items-end sm:items-center justify-center z-50 p-4"
            onClick={() => setSelectedViolation(null)}
          >
            <div
              className="bg-white rounded-t-2xl sm:rounded-2xl w-full max-w-lg max-h-[85vh] overflow-y-auto animate-slide-up"
              onClick={(e) => e.stopPropagation()}
            >
              {/* Modal Header */}
              <div className="sticky top-0 bg-white border-b border-stone-200 p-4 flex items-center justify-between">
                <h3 className="text-lg font-bold text-stone-800">Violation Details</h3>
                <button
                  onClick={() => setSelectedViolation(null)}
                  className="w-8 h-8 rounded-full bg-stone-100 flex items-center justify-center hover:bg-stone-200 transition-colors"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>

              {/* Modal Content */}
              <div className="p-4 space-y-4">
                {/* Violation Types */}
                <div className="bg-red-50 rounded-xl p-4 border border-red-100">
                  <div className="flex items-center gap-3 mb-3">
                    <div className="w-12 h-12 bg-red-100 rounded-xl flex items-center justify-center">
                      <AlertTriangle className="w-6 h-6 text-red-600" />
                    </div>
                    <div>
                      <p className="text-sm text-red-600 font-medium">PPE Violation Detected</p>
                      <p className="text-xs text-red-500">
                        {new Date(selectedViolation.timestamp).toLocaleDateString('en-US', {
                          weekday: 'long',
                          year: 'numeric',
                          month: 'long',
                          day: 'numeric',
                        })}
                      </p>
                    </div>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {selectedViolation.violations.map((type, i) => (
                      <span
                        key={i}
                        className={`px-3 py-1.5 rounded-lg text-sm font-medium ${getViolationColor(type)}`}
                      >
                        {type.replace('NO-', 'Missing ')}
                      </span>
                    ))}
                  </div>
                </div>

                {/* Details Grid */}
                <div className="grid grid-cols-2 gap-3">
                  <div className="bg-stone-50 rounded-xl p-3">
                    <div className="flex items-center gap-2 text-stone-500 mb-1">
                      <Clock className="w-4 h-4" />
                      <span className="text-xs font-medium">Time</span>
                    </div>
                    <p className="text-sm font-semibold text-stone-800">
                      {new Date(selectedViolation.timestamp).toLocaleTimeString()}
                    </p>
                  </div>
                  <div className="bg-stone-50 rounded-xl p-3">
                    <div className="flex items-center gap-2 text-stone-500 mb-1">
                      <Calendar className="w-4 h-4" />
                      <span className="text-xs font-medium">Shift</span>
                    </div>
                    <p className="text-sm font-semibold text-stone-800 capitalize">
                      {selectedViolation.shift}
                    </p>
                  </div>
                  <div className="bg-stone-50 rounded-xl p-3">
                    <div className="flex items-center gap-2 text-stone-500 mb-1">
                      <MapPin className="w-4 h-4" />
                      <span className="text-xs font-medium">Gate</span>
                    </div>
                    <p className="text-sm font-semibold text-stone-800">
                      {selectedViolation.gate_name || 'Unknown'}
                    </p>
                  </div>
                  <div className="bg-stone-50 rounded-xl p-3">
                    <div className="flex items-center gap-2 text-stone-500 mb-1">
                      <Shield className="w-4 h-4" />
                      <span className="text-xs font-medium">Zone</span>
                    </div>
                    <p className="text-sm font-semibold text-stone-800">
                      {selectedViolation.zone || 'General Area'}
                    </p>
                  </div>
                </div>

                {/* Status */}
                <div className="flex items-center justify-between p-3 bg-stone-50 rounded-xl">
                  <span className="text-sm text-stone-600">Acknowledgment Status</span>
                  <span className={`px-3 py-1 rounded-full text-xs font-medium ${
                    selectedViolation.acknowledged
                      ? 'bg-green-100 text-green-700'
                      : 'bg-amber-100 text-amber-700'
                  }`}>
                    {selectedViolation.acknowledged ? 'Acknowledged' : 'Pending'}
                  </span>
                </div>

                {/* Safety Reminder */}
                <div className="bg-orange-50 border border-orange-200 rounded-xl p-4">
                  <div className="flex items-start gap-3">
                    <Shield className="w-5 h-5 text-orange-600 flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="font-medium text-orange-800 text-sm">Safety Reminder</p>
                      <p className="text-xs text-orange-700 mt-1">
                        {selectedViolation.violations.includes('NO-HELMET') &&
                          'Always wear your safety helmet before entering any work zone. It protects against head injuries from falling objects.'}
                        {selectedViolation.violations.includes('NO-VEST') &&
                          'High-visibility vests are essential for being seen by equipment operators and ensuring your safety in all areas.'}
                        {selectedViolation.violations.includes('NO-GLOVES') &&
                          'Safety gloves protect your hands from cuts, burns, and chemical exposure. Always wear appropriate gloves for the task.'}
                        {selectedViolation.violations.includes('NO-BOOTS') &&
                          'Steel-toe boots protect against heavy falling objects and provide stability on uneven surfaces.'}
                      </p>
                    </div>
                  </div>
                </div>
              </div>

              {/* Modal Footer */}
              <div className="sticky bottom-0 bg-white border-t border-stone-200 p-4">
                <button
                  onClick={() => setSelectedViolation(null)}
                  className="w-full py-3 bg-orange-500 text-white rounded-xl font-semibold hover:bg-orange-600 transition-colors"
                >
                  Close
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </AppLayout>
  );
}
