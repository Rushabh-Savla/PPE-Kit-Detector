'use client';

interface CountdownTimerProps {
  timeRemaining: number; // in seconds
  totalTime?: number; // total duration in seconds
  size?: 'sm' | 'md' | 'lg';
}

const SIZE_CONFIG = {
  sm: { width: 80, strokeWidth: 4, fontSize: 'text-xl' },
  md: { width: 120, strokeWidth: 6, fontSize: 'text-3xl' },
  lg: { width: 160, strokeWidth: 8, fontSize: 'text-5xl' },
};

export function CountdownTimer({
  timeRemaining,
  totalTime = 10,
  size = 'md',
}: CountdownTimerProps) {
  const { width, strokeWidth, fontSize } = SIZE_CONFIG[size];
  const radius = (width - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const progress = timeRemaining / totalTime;
  const strokeDashoffset = circumference * (1 - progress);

  // Color based on time remaining
  let strokeColor = 'stroke-green-500';
  let textColor = 'text-green-600';

  if (timeRemaining <= 3) {
    strokeColor = 'stroke-red-500';
    textColor = 'text-red-600';
  } else if (timeRemaining <= 5) {
    strokeColor = 'stroke-amber-500';
    textColor = 'text-amber-600';
  }

  return (
    <div className="relative inline-flex items-center justify-center">
      <svg
        width={width}
        height={width}
        className="transform -rotate-90"
      >
        {/* Background circle */}
        <circle
          cx={width / 2}
          cy={width / 2}
          r={radius}
          fill="none"
          stroke="currentColor"
          strokeWidth={strokeWidth}
          className="text-stone-200"
        />
        {/* Progress circle */}
        <circle
          cx={width / 2}
          cy={width / 2}
          r={radius}
          fill="none"
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          className={`${strokeColor} transition-all duration-100`}
          style={{
            strokeDasharray: circumference,
            strokeDashoffset,
          }}
        />
      </svg>
      {/* Time display */}
      <div className="absolute inset-0 flex items-center justify-center">
        <span className={`${fontSize} font-bold ${textColor} tabular-nums`}>
          {Math.ceil(timeRemaining)}
        </span>
      </div>
    </div>
  );
}

// Simple text countdown for compact spaces
export function CountdownText({
  timeRemaining,
}: {
  timeRemaining: number;
}) {
  let textColor = 'text-green-600';

  if (timeRemaining <= 3) {
    textColor = 'text-red-600';
  } else if (timeRemaining <= 5) {
    textColor = 'text-amber-600';
  }

  return (
    <span className={`font-mono font-bold ${textColor}`}>
      {Math.ceil(timeRemaining)}s
    </span>
  );
}
