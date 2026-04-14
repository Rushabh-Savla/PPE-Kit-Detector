'use client';

import { ReactNode } from 'react';
import Sidebar from './Sidebar';
import AuthGuard from './AuthGuard';
import { useSidebarStore } from '@/lib/store';
import { cn } from '@/lib/utils';

interface AppLayoutProps {
  children: ReactNode;
}

export default function AppLayout({ children }: AppLayoutProps) {
  const { isCollapsed } = useSidebarStore();

  return (
    <AuthGuard>
      <div className="min-h-screen bg-white">
        <Sidebar />
        <main className={cn(
          'min-h-screen transition-all duration-300 ease-in-out',
          isCollapsed ? 'lg:ml-[72px]' : 'lg:ml-64'
        )}>
          <div className="p-4 lg:p-8 pt-16 lg:pt-8">{children}</div>
        </main>
      </div>
    </AuthGuard>
  );
}
