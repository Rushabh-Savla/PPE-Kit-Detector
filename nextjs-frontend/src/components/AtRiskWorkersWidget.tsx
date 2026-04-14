'use client';

import { useEffect, useState } from 'react';
import { AlertTriangle, TrendingUp, ShieldAlert, Brain, ExternalLink } from 'lucide-react';
import { useRouter } from 'next/navigation';
import { Card } from '@/components/Card';
import { Spinner } from '@/components/Loading';
import { predictionApi } from '@/lib/api';
import type { AtRiskWorkersResponse } from '@/types';

const RISK_COLORS = {
  low: '#10b981',
  medium: '#f59e0b',
  high: '#f97316',
  critical: '#ef4444',
};

interface AtRiskWorkersWidgetProps {
  limit?: number;
  showTitle?: boolean;
  compact?: boolean;
}

export default function AtRiskWorkersWidget({
  limit = 5,
  showTitle = true,
  compact = false,
}: AtRiskWorkersWidgetProps) {
  const router = useRouter();
  const [data, setData] = useState<AtRiskWorkersResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadData();
  }, [limit]);

  const loadData = async () => {
    try {
      setLoading(true);
      const result = await predictionApi.getAtRiskWorkers({ limit });
      setData(result);
    } catch (err) {
      console.error('Failed to load at-risk workers:', err);
      setError('Failed to load data');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <Card>
        {showTitle && (
          <h3 className="text-lg font-semibold text-stone-800 mb-4 flex items-center gap-2">
            <Brain className="w-5 h-5 text-purple-600" />
            At-Risk Workers (ML Prediction)
          </h3>
        )}
        <div className="flex items-center justify-center py-8">
          <Spinner />
        </div>
      </Card>
    );
  }

  if (error || !data) {
    return (
      <Card>
        {showTitle && (
          <h3 className="text-lg font-semibold text-stone-800 mb-4 flex items-center gap-2">
            <Brain className="w-5 h-5 text-purple-600" />
            At-Risk Workers (ML Prediction)
          </h3>
        )}
        <div className="text-center py-8">
          <p className="text-red-500 text-sm">{error || 'Failed to load data'}</p>
          <button onClick={loadData} className="mt-2 text-blue-600 text-sm hover:underline">
            Retry
          </button>
        </div>
      </Card>
    );
  }

  const criticalCount = data.by_category.critical || 0;
  const highCount = data.by_category.high || 0;
  const mediumCount = data.by_category.medium || 0;

  return (
    <Card>
      {showTitle && (
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-stone-800 flex items-center gap-2">
            <Brain className="w-5 h-5 text-purple-600" />
            At-Risk Workers
            <span className="text-sm font-normal text-stone-500">(AI Prediction)</span>
          </h3>
          <button
            onClick={() => router.push('/predictive-analysis')}
            className="text-purple-600 hover:text-purple-700 flex items-center gap-1 text-sm font-medium transition-colors"
          >
            View All
            <ExternalLink className="w-4 h-4" />
          </button>
        </div>
      )}

      {/* Summary Stats */}
      {!compact && (
        <div className="grid grid-cols-3 gap-3 mb-4">
          <div className="bg-red-50 border border-red-200 rounded-lg p-3">
            <div className="flex items-center gap-2 mb-1">
              <ShieldAlert className="w-4 h-4 text-red-600" />
              <span className="text-xs font-medium text-red-600">Critical</span>
            </div>
            <div className="text-2xl font-bold text-red-700">{criticalCount}</div>
          </div>
          <div className="bg-orange-50 border border-orange-200 rounded-lg p-3">
            <div className="flex items-center gap-2 mb-1">
              <AlertTriangle className="w-4 h-4 text-orange-600" />
              <span className="text-xs font-medium text-orange-600">High</span>
            </div>
            <div className="text-2xl font-bold text-orange-700">{highCount}</div>
          </div>
          <div className="bg-amber-50 border border-amber-200 rounded-lg p-3">
            <div className="flex items-center gap-2 mb-1">
              <TrendingUp className="w-4 h-4 text-amber-600" />
              <span className="text-xs font-medium text-amber-600">Medium</span>
            </div>
            <div className="text-2xl font-bold text-amber-700">{mediumCount}</div>
          </div>
        </div>
      )}

      {/* Workers List */}
      <div className="space-y-3">
        {data.workers.slice(0, limit).map((worker) => {
          const riskColor = RISK_COLORS[worker.risk_category];

          return (
            <div
              key={worker.worker_id}
              className="border border-stone-200 rounded-lg p-3 hover:bg-stone-50 transition-colors cursor-pointer"
              onClick={() => router.push(`/workers?id=${worker.worker_id}`)}
            >
              <div className="flex items-start justify-between mb-2">
                <div className="flex-1">
                  <h4 className="font-medium text-stone-900 text-sm">{worker.worker_name}</h4>
                  <p className="text-xs text-stone-500">{worker.employee_id}</p>
                </div>
                <div className="flex items-center gap-2">
                  <span
                    className="px-2 py-1 text-xs font-semibold rounded-full text-white"
                    style={{ backgroundColor: riskColor }}
                  >
                    {worker.risk_score.toFixed(0)}
                  </span>
                  {worker.requires_intervention && (
                    <AlertTriangle className="w-4 h-4 text-red-600" />
                  )}
                </div>
              </div>

              {/* Risk Progress Bar */}
              <div className="mb-2">
                <div className="h-2 bg-stone-200 rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all"
                    style={{
                      width: `${worker.risk_score}%`,
                      backgroundColor: riskColor,
                    }}
                  />
                </div>
              </div>

              {/* Main Issue */}
              <div className="flex items-start gap-2">
                <span className="text-xs text-stone-600 line-clamp-2">{worker.main_issue}</span>
              </div>

              {compact && worker.requires_intervention && (
                <div className="mt-2 text-xs font-medium text-red-600 flex items-center gap-1">
                  <ShieldAlert className="w-3 h-3" />
                  Intervention Required
                </div>
              )}
            </div>
          );
        })}

        {data.workers.length === 0 && (
          <div className="text-center py-8 text-stone-500">
            <ShieldAlert className="w-12 h-12 mx-auto mb-2 text-stone-300" />
            <p className="text-sm">No at-risk workers detected</p>
            <p className="text-xs text-stone-400 mt-1">All workers are performing well!</p>
          </div>
        )}
      </div>

      {/* View All Link */}
      {data.workers.length > 0 && (
        <button
          onClick={() => router.push('/predictive-analysis')}
          className="w-full mt-4 py-2 text-center text-sm font-medium text-purple-600 hover:text-purple-700 hover:bg-purple-50 rounded-lg transition-colors"
        >
          View All {data.total_at_risk} At-Risk Workers →
        </button>
      )}
    </Card>
  );
}
