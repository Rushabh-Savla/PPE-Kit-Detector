import { cn } from '@/lib/utils';
import { ReactNode } from 'react';

interface CardProps {
  children: ReactNode;
  className?: string;
  title?: string;
  description?: string;
  icon?: ReactNode;
  action?: ReactNode;
}

export function Card({ children, className, title, description, icon, action }: CardProps) {
  return (
    <div className={cn(
      'bg-white rounded-2xl border border-slate-200 shadow-sm',
      className
    )}>
      {(title || description || action) && (
        <div className="px-6 py-5 border-b border-slate-100 flex items-center justify-between gap-4">
          <div className="flex items-center gap-4">
            {icon && (
              <div className="w-11 h-11 rounded-xl bg-gradient-to-br from-orange-200 to-amber-200 flex items-center justify-center shadow-sm border border-orange-300">
                {icon}
              </div>
            )}
            <div>
              {title && <h3 className="text-base font-semibold text-slate-700">{title}</h3>}
              {description && <p className="text-sm text-slate-500 mt-0.5">{description}</p>}
            </div>
          </div>
          {action && <div>{action}</div>}
        </div>
      )}
      <div className="p-6">{children}</div>
    </div>
  );
}

interface StatCardProps {
  title: string;
  value: string | number;
  icon: ReactNode;
  trend?: {
    value: number;
    positive: boolean;
  };
  color?: 'orange' | 'green' | 'red' | 'amber' | 'blue' | 'purple' | 'cyan' | 'yellow';
  subtitle?: string;
}

export function StatCard({ title, value, icon, trend, color = 'orange', subtitle }: StatCardProps) {
  const colorClasses = {
    orange: 'from-orange-200 to-amber-200 border-orange-300 text-orange-600',
    green: 'from-emerald-100 to-teal-200 border-emerald-300 text-emerald-600',
    red: 'from-red-100 to-rose-200 border-red-300 text-red-600',
    amber: 'from-amber-100 to-yellow-200 border-amber-300 text-amber-600',
    blue: 'from-blue-100 to-sky-200 border-blue-300 text-blue-600',
    purple: 'from-purple-100 to-violet-200 border-purple-300 text-purple-600',
    cyan: 'from-cyan-100 to-teal-200 border-cyan-300 text-cyan-600',
    yellow: 'from-yellow-100 to-amber-200 border-yellow-300 text-yellow-600',
  };

  const selectedColors = colorClasses[color as keyof typeof colorClasses] || colorClasses.orange;
  const [bgGradient, borderColor, iconColor] = selectedColors.split(' ');

  return (
    <div className="bg-white rounded-2xl border border-slate-200 p-5 shadow-sm card-hover transition-all duration-300 min-h-[180px]">
      <div className="flex flex-col items-center justify-center text-center h-full gap-2">
        {/* Icon at top */}
        <div className={cn(
          'w-12 h-12 rounded-xl bg-gradient-to-br flex items-center justify-center shadow-sm border flex-shrink-0',
          bgGradient,
          borderColor,
          iconColor
        )}>
          {icon}
        </div>

        {/* Value */}
        <p className="text-2xl font-bold text-slate-700 leading-tight">{value}</p>

        {/* Title - smaller text below */}
        <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider leading-tight">{title}</p>

        {/* Subtitle */}
        {subtitle && (
          <p className="text-[11px] text-slate-400 leading-tight">{subtitle}</p>
        )}

        {/* Trend indicator */}
        {trend && (
          <div className={cn(
            'inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold',
            trend.positive
              ? 'bg-emerald-100 text-emerald-700 border border-emerald-300'
              : 'bg-red-100 text-red-700 border border-red-300'
          )}>
            <span className="text-[10px]">{trend.positive ? '▲' : '▼'}</span>
            {Math.abs(trend.value)}%
          </div>
        )}
      </div>
    </div>
  );
}
