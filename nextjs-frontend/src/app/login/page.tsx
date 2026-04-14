'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import Image from 'next/image';
import { authApi } from '@/lib/api';
import { useAuthStore } from '@/lib/store';
import { HardHat, UserCog, Loader2, AlertCircle, KeyRound, User } from 'lucide-react';

type LoginMode = 'staff' | 'worker';

export default function LoginPage() {
  const router = useRouter();
  const { setStaffAuth, setWorkerAuth } = useAuthStore();

  const [mode, setMode] = useState<LoginMode>('staff');
  const [username, setUsername] = useState('');
  const [employeeId, setEmployeeId] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      if (mode === 'staff') {
        const response = await authApi.login(username, password);
        console.log('Login response:', response);

        if (!response.access_token) {
          throw new Error('No access token in response');
        }

        const token = response.access_token;
        const user = {
          ...response.user,
          mine_ids: response.user.mine_ids || [],
        };

        // Store in localStorage
        localStorage.setItem('token', token);
        localStorage.setItem('user', JSON.stringify(user));
        localStorage.setItem('userType', 'staff');

        console.log('Token stored:', localStorage.getItem('token'));

        setStaffAuth(token, user);
      } else {
        // Use mock data for worker login
        const mockWorker = {
          id: 'w1',
          employee_id: employeeId || 'EMP-2024-001',
          name: 'Rushabh Savla',
          department: 'Operations',
          mine_id: 'mine1',
          mine_name: 'Central Mining Complex',
          zone_id: 'zone1',
          zone_name: 'Zone A - Mining Area',
          assigned_shift: 'day' as const,
          phone: '+91 98765 43210',
          emergency_contact: '+91 98765 43211',
          face_registered: true,
          is_active: true,
          created_at: new Date().toISOString(),
          compliance_score: 87,
          total_violations: 6,
          badges: ['safety_star', 'streak_7'],
        };
        const mockToken = 'mock-worker-token-' + Date.now();

        // Store in localStorage
        localStorage.setItem('token', mockToken);
        localStorage.setItem('worker', JSON.stringify(mockWorker));
        localStorage.setItem('userType', 'worker');

        setWorkerAuth(mockToken, mockWorker);
      }
      // Small delay to ensure state is persisted before navigation
      await new Promise(resolve => setTimeout(resolve, 100));
      router.push('/');
    } catch (err: unknown) {
      const errorObj = err as { response?: { data?: { detail?: string | Array<{ msg: string }> } } };
      const detail = errorObj?.response?.data?.detail;

      // Handle Pydantic validation errors (array of objects with msg field)
      if (Array.isArray(detail)) {
        setError(detail.map(e => e.msg).join(', ') || 'Validation error');
      } else if (typeof detail === 'string') {
        setError(detail);
      } else if (detail && typeof detail === 'object') {
        setError(JSON.stringify(detail));
      } else {
        setError('Login failed. Please check your credentials.');
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-stone-50 via-slate-50 to-stone-100">
      {/* Subtle Background Pattern */}
      <div className="absolute inset-0 opacity-[0.02]">
        <div className="absolute inset-0" style={{
          backgroundImage: `url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23000000' fill-opacity='1'%3E%3Cpath d='m36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")`,
        }}></div>
      </div>

      <div className="relative z-10 w-full max-w-md px-6">
        {/* Logo and Title */}
        <div className="text-center mb-4">
          <div className="inline-flex items-center justify-center w-40 h-40 mb-4">
            <Image
              src="/raksham-logo.png"
              alt="Raksham Logo"
              width={1000}
              height={1000}
              className="object-contain drop-shadow-lg"
              priority
            />
          </div>
          {/* <h1 className="text-4xl font-bold text-slate-800 mb-2.5 tracking-tight">Raksham</h1> */}
          <p className="text-slate-500 text-base font-medium">Mine Safety & PPE Compliance System</p>
        </div>

        {/* Login Card */}
        <div className="bg-white border border-slate-200 rounded-3xl shadow-2xl shadow-slate-200/50 p-8">
          {/* Mode Toggle */}
          <div className="flex bg-slate-50 rounded-2xl p-1.5 mb-8 border border-slate-100">
            <button
              type="button"
              onClick={() => {
                setMode('staff');
                setError('');
              }}
              className={`flex-1 flex items-center justify-center gap-2.5 py-3.5 rounded-xl text-sm font-semibold transition-all ${
                mode === 'staff'
                  ? 'bg-gradient-to-r from-orange-200 to-amber-200 text-orange-700 shadow-md'
                  : 'text-slate-500 hover:text-slate-700'
              }`}
            >
              <UserCog size={19} strokeWidth={2.5} />
              Staff Login
            </button>
            <button
              type="button"
              onClick={() => {
                setMode('worker');
                setError('');
              }}
              className={`flex-1 flex items-center justify-center gap-2.5 py-3.5 rounded-xl text-sm font-semibold transition-all ${
                mode === 'worker'
                  ? 'bg-gradient-to-r from-emerald-100 to-teal-100 text-emerald-700 shadow-md'
                  : 'text-slate-500 hover:text-slate-700'
              }`}
            >
              <HardHat size={19} strokeWidth={2.5} />
              Worker Login
            </button>
          </div>

          {/* Login Form */}
          <form onSubmit={handleSubmit} className="space-y-6">
            {mode === 'staff' ? (
              <div>
                <label htmlFor="username" className="block text-sm font-semibold text-slate-700 mb-2.5">
                  Username
                </label>
                <div className="relative">
                  <User className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400" size={20} strokeWidth={2.5} />
                  <input
                    type="text"
                    id="username"
                    value={username}
                    onChange={(e) => setUsername(e.target.value)}
                    placeholder="Enter your username"
                    className="w-full with-icon"
                    style={{ paddingLeft: '48px' }}
                    required
                  />
                </div>
              </div>
            ) : (
              <div>
                <label htmlFor="employeeId" className="block text-sm font-semibold text-slate-700 mb-2.5">
                  Employee ID
                </label>
                <div className="relative">
                  <HardHat className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400" size={20} strokeWidth={2.5} />
                  <input
                    type="text"
                    id="employeeId"
                    value={employeeId}
                    onChange={(e) => setEmployeeId(e.target.value)}
                    placeholder="Enter your Employee ID (e.g., W001)"
                    className="w-full with-icon"
                    style={{ paddingLeft: '48px' }}
                    required
                  />
                </div>
              </div>
            )}

            <div>
              <label htmlFor="password" className="block text-sm font-semibold text-slate-700 mb-2.5">
                Password
              </label>
              <div className="relative">
                <KeyRound className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400" size={20} strokeWidth={2.5} />
                <input
                  type="password"
                  id="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="Enter your password"
                  className="w-full with-icon"
                  style={{ paddingLeft: '48px' }}
                  required
                />
              </div>
            </div>

            {error && (
              <div className="bg-red-50 border-1.5 border-red-300 text-red-700 px-4 py-3.5 rounded-2xl text-sm flex items-center gap-3">
                <AlertCircle size={20} strokeWidth={2.5} />
                <span className="font-medium">{error}</span>
              </div>
            )}

            <button
              type="submit"
              disabled={loading}
              className="w-full btn btn-primary py-4 text-base font-semibold"
            >
              {loading ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" strokeWidth={2.5} />
                  Signing in...
                </>
              ) : (
                <>Sign In</>
              )}
            </button>
          </form>

          {/* Help Text */}
          <div className="mt-8 pt-7 border-t border-slate-100">
            <p className="text-center text-sm text-slate-500 leading-relaxed">
              {mode === 'staff' ? (
                <>
                  Staff members use their assigned username and password.
                  <br />
                  Contact your administrator if you need access.
                </>
              ) : (
                <>
                  Workers use their Employee ID and password provided by their supervisor.
                  <br />
                  Need help? Contact your Shift Incharge.
                </>
              )}
            </p>
          </div>
        </div>

        {/* Demo Credentials */}
        <div className="mt-6 bg-white/80 backdrop-blur-sm border border-slate-200 rounded-2xl p-6 shadow-lg">
          <h3 className="text-slate-700 font-semibold mb-4 text-center text-sm">Demo Credentials</h3>
          <div className="grid grid-cols-2 gap-3 text-xs">
            {mode === 'staff' ? (
              <>
                <div className="bg-gradient-to-br from-orange-100 to-orange-50 rounded-xl p-3 border border-orange-200">
                  <p className="text-orange-700 font-semibold mb-1">Super Admin</p>
                  <p className="text-slate-600">superadmin / admin123</p>
                </div>
                <div className="bg-gradient-to-br from-amber-100 to-amber-50 rounded-xl p-3 border border-amber-200">
                  <p className="text-amber-700 font-semibold mb-1">General Manager</p>
                  <p className="text-slate-600">gm / gm123</p>
                </div>
                <div className="bg-gradient-to-br from-yellow-100 to-yellow-50 rounded-xl p-3 border border-yellow-200">
                  <p className="text-yellow-700 font-semibold mb-1">Area Safety Officer</p>
                  <p className="text-slate-600">aso / aso123</p>
                </div>
                <div className="bg-gradient-to-br from-lime-100 to-lime-50 rounded-xl p-3 border border-lime-200">
                  <p className="text-lime-700 font-semibold mb-1">Manager</p>
                  <p className="text-slate-600">manager1 / manager123</p>
                </div>
                <div className="bg-gradient-to-br from-emerald-100 to-emerald-50 rounded-xl p-3 border border-emerald-200">
                  <p className="text-emerald-700 font-semibold mb-1">Safety Officer</p>
                  <p className="text-slate-600">safety1 / safety123</p>
                </div>
                <div className="bg-gradient-to-br from-cyan-100 to-cyan-50 rounded-xl p-3 border border-cyan-200">
                  <p className="text-cyan-700 font-semibold mb-1">Shift Incharge</p>
                  <p className="text-slate-600">shift_day1 / shift123</p>
                </div>
              </>
            ) : (
              <>
                <div className="bg-gradient-to-br from-emerald-100 to-emerald-50 rounded-xl p-3 border border-emerald-200">
                  <p className="text-emerald-700 font-semibold mb-1">Worker (Mine 1)</p>
                  <p className="text-slate-600">W001 / worker123</p>
                </div>
                <div className="bg-gradient-to-br from-teal-100 to-teal-50 rounded-xl p-3 border border-teal-200">
                  <p className="text-teal-700 font-semibold mb-1">Worker (Mine 2)</p>
                  <p className="text-slate-600">W101 / worker123</p>
                </div>
              </>
            )}
          </div>
        </div>

        {/* Footer */}
        <p className="text-center text-slate-400 text-sm mt-8">
          Raksham - Mine Safety System v2.0
        </p>
      </div>
    </div>
  );
}
