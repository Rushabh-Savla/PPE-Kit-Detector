'use client';

import { useEffect, useState } from 'react';
import { useRouter, usePathname } from 'next/navigation';
import { useAuthStore } from '@/lib/store';
import { authApi } from '@/lib/api';
import { PageLoading } from './Loading';

const publicPaths = ['/login', '/landing'];

export default function AuthGuard({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const pathname = usePathname();
  const { token, logout } = useAuthStore();
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const checkAuth = async () => {
      const isPublicPath = publicPaths.includes(pathname);

      if (!token && !isPublicPath) {
        router.push('/login');
        setIsLoading(false);
        return;
      }

      if (token) {
        // Skip verification for mock tokens (used in development)
        if (token.startsWith('mock-worker-token-')) {
          if (pathname === '/login') {
            router.push('/');
          }
          setIsLoading(false);
          return;
        }

        try {
          const response = await authApi.verify();
          // authApi.verify() returns { valid: boolean, user_type, username, role, mine_id }
          if (response.valid) {
            // Token is valid - redirect away from login if on login page
            if (pathname === '/login') {
              router.push('/');
            }
          } else {
            // Token is invalid
            logout();
            if (!isPublicPath) {
              router.push('/login');
            }
          }
        } catch {
          logout();
          if (!isPublicPath) {
            router.push('/login');
          }
        }
      }

      setIsLoading(false);
    };

    checkAuth();
  }, [token, pathname, router, logout]);

  if (isLoading) {
    return <PageLoading />;
  }

  return <>{children}</>;
}
