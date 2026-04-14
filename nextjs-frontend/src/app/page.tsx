'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { useAuthStore } from '@/lib/store';
import AppLayout from '@/components/AppLayout';
import { PageLoading } from '@/components/Loading';

// Role-specific dashboard imports
import ShiftInchargeDashboard from '@/components/dashboards/ShiftInchargeDashboard';
import SafetyOfficerDashboard from '@/components/dashboards/SafetyOfficerDashboard';
import ManagerDashboard from '@/components/dashboards/ManagerDashboard';
import AreaSafetyOfficerDashboard from '@/components/dashboards/AreaSafetyOfficerDashboard';
import GeneralManagerDashboard from '@/components/dashboards/GeneralManagerDashboard';
import WorkerDashboard from '@/components/dashboards/WorkerDashboard';
import SuperAdminDashboard from '@/components/dashboards/SuperAdminDashboard';

export default function DashboardPage() {
  const router = useRouter();
  const { getRole, token } = useAuthStore();
  const [mounted, setMounted] = useState(false);
  const [isCheckingAuth, setIsCheckingAuth] = useState(true);

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    if (mounted) {
      // Check if user is authenticated
      const storedToken = localStorage.getItem('token');
      if (!storedToken && !token) {
        // Not authenticated, redirect to landing page
        router.push('/landing');
      } else {
        setIsCheckingAuth(false);
      }
    }
  }, [mounted, token, router]);

  // Wait for component mount and auth check
  if (!mounted || isCheckingAuth) {
    return <PageLoading />;
  }

  const role = getRole();

  // If role is null after hydration, show loading (auth might still be processing)
  if (!role) {
    return (
      <AppLayout>
        <PageLoading />
      </AppLayout>
    );
  }

  // Render role-specific dashboard
  const renderDashboard = () => {
    switch (role) {
      case 'super_admin':
        return <SuperAdminDashboard />;
      case 'general_manager':
        return <GeneralManagerDashboard />;
      case 'area_safety_officer':
        return <AreaSafetyOfficerDashboard />;
      case 'manager':
        return <ManagerDashboard />;
      case 'safety_officer':
        return <SafetyOfficerDashboard />;
      case 'shift_incharge':
        return <ShiftInchargeDashboard />;
      case 'worker':
        return <WorkerDashboard />;
      default:
        // Fallback - super admin dashboard as it has most access
        return <SuperAdminDashboard />;
    }
  };

  return (
    <AppLayout>
      {renderDashboard()}
    </AppLayout>
  );
}
