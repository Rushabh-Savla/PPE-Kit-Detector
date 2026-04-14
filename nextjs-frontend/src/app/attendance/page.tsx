'use client';

import { useState, useEffect } from 'react';
import { LogIn, LogOut, Clock, CheckCircle, XCircle, RefreshCw } from 'lucide-react';
import AppLayout from '@/components/AppLayout';
import Camera from '@/components/Camera';
import { Card } from '@/components/Card';
import { Spinner } from '@/components/Loading';
import { attendanceApi } from '@/lib/api';
import { formatTime, formatDate } from '@/lib/utils';
import type { TodayAttendance, DetectionResult } from '@/types';

type AttendanceMode = 'check_in' | 'check_out';

export default function AttendancePage() {
  const [mode, setMode] = useState<AttendanceMode>('check_in');
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState<{
    success: boolean;
    message: string;
    image?: string;
    detections?: DetectionResult['detections'];
  } | null>(null);
  const [todayAttendance, setTodayAttendance] = useState<TodayAttendance | null>(null);
  const [capturedFile, setCapturedFile] = useState<File | null>(null);

  useEffect(() => {
    loadTodayAttendance();
  }, []);

  const loadTodayAttendance = async () => {
    try {
      const data = await attendanceApi.getToday();
      setTodayAttendance(data);
    } catch (err) {
      console.error('Failed to load attendance:', err);
    }
  };

  const handleCapture = (file: File) => {
    setCapturedFile(file);
    setResult(null);
  };

  const handleSubmit = async () => {
    if (!capturedFile) return;

    setIsProcessing(true);
    setResult(null);

    try {
      const response = mode === 'check_in'
        ? await attendanceApi.checkIn(capturedFile)
        : await attendanceApi.checkOut(capturedFile);

      // Response is the attendance record - attendance was successful
      setResult({
        success: true,
        message: mode === 'check_in' ? 'Check-in successful!' : 'Check-out successful!',
        image: response.image,
      });

      loadTodayAttendance();
    } catch (err: unknown) {
      const error = err as { response?: { data?: { error?: string } } };
      setResult({
        success: false,
        message: error.response?.data?.error || 'Failed to process attendance',
      });
    } finally {
      setIsProcessing(false);
    }
  };

  const resetForm = () => {
    setCapturedFile(null);
    setResult(null);
  };

  return (
    <AppLayout>
      <div className="space-y-6">
        {/* Page Header */}
        <div>
          <h1 className="text-2xl font-bold text-stone-800">Attendance</h1>
          <p className="text-stone-500 mt-1">Mark attendance using face recognition</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left Column - Camera */}
          <div className="space-y-4">
            {/* Mode Toggle */}
            <Card>
              <div className="flex gap-2">
                <button
                  onClick={() => { setMode('check_in'); resetForm(); }}
                  className={`flex-1 py-3 px-4 rounded-lg font-medium transition-all flex items-center justify-center gap-2 ${
                    mode === 'check_in'
                      ? 'bg-green-500 text-white'
                      : 'bg-stone-100 text-stone-600 hover:bg-stone-200'
                  }`}
                >
                  <LogIn size={20} />
                  Check In
                </button>
                <button
                  onClick={() => { setMode('check_out'); resetForm(); }}
                  className={`flex-1 py-3 px-4 rounded-lg font-medium transition-all flex items-center justify-center gap-2 ${
                    mode === 'check_out'
                      ? 'bg-blue-500 text-white'
                      : 'bg-stone-100 text-stone-600 hover:bg-stone-200'
                  }`}
                >
                  <LogOut size={20} />
                  Check Out
                </button>
              </div>
            </Card>

            {/* Camera */}
            <Card title={`${mode === 'check_in' ? 'Check In' : 'Check Out'} Camera`}>
              <Camera onCapture={handleCapture} disabled={isProcessing} />

              {capturedFile && !result && (
                <button
                  onClick={handleSubmit}
                  disabled={isProcessing}
                  className={`w-full mt-4 py-3 px-4 rounded-lg font-medium transition-all flex items-center justify-center gap-2 ${
                    mode === 'check_in'
                      ? 'bg-green-500 hover:bg-green-600 text-white'
                      : 'bg-blue-500 hover:bg-blue-600 text-white'
                  } disabled:opacity-50`}
                >
                  {isProcessing ? (
                    <>
                      <Spinner size="sm" className="border-white/30 border-t-white" />
                      Processing...
                    </>
                  ) : (
                    <>
                      {mode === 'check_in' ? <LogIn size={20} /> : <LogOut size={20} />}
                      Submit {mode === 'check_in' ? 'Check In' : 'Check Out'}
                    </>
                  )}
                </button>
              )}
            </Card>

            {/* Result */}
            {result && (
              <Card>
                <div className={`p-4 rounded-lg ${result.success ? 'bg-green-50' : 'bg-red-50'}`}>
                  <div className="flex items-start gap-3">
                    {result.success ? (
                      <CheckCircle className="text-green-500 flex-shrink-0" size={24} />
                    ) : (
                      <XCircle className="text-red-500 flex-shrink-0" size={24} />
                    )}
                    <div>
                      <p className={`font-medium ${result.success ? 'text-green-800' : 'text-red-800'}`}>
                        {result.success ? 'Success!' : 'Failed'}
                      </p>
                      <p className={`text-sm ${result.success ? 'text-green-600' : 'text-red-600'}`}>
                        {result.message}
                      </p>
                    </div>
                  </div>
                </div>

                {result.image && (
                  <div className="mt-4">
                    <p className="text-sm font-medium text-stone-700 mb-2">Detection Result:</p>
                    <img
                      src={result.image}
                      alt="Detection result"
                      className="w-full rounded-lg"
                    />
                  </div>
                )}

                <button
                  onClick={resetForm}
                  className="w-full mt-4 py-2 px-4 bg-stone-100 text-stone-700 rounded-lg font-medium hover:bg-stone-200 transition-colors flex items-center justify-center gap-2"
                >
                  <RefreshCw size={18} />
                  Take Another Photo
                </button>
              </Card>
            )}
          </div>

          {/* Right Column - Today's Attendance */}
          <Card title="Today's Attendance" description={todayAttendance?.date || 'Loading...'}>
            {/* Summary */}
            {todayAttendance && (
              <div className="grid grid-cols-2 gap-4 mb-6">
                <div className="p-4 bg-green-50 rounded-lg text-center">
                  <p className="text-2xl font-bold text-green-600">{todayAttendance.present}</p>
                  <p className="text-sm text-green-700">Present</p>
                </div>
                <div className="p-4 bg-stone-100 rounded-lg text-center">
                  <p className="text-2xl font-bold text-stone-600">{todayAttendance.absent}</p>
                  <p className="text-sm text-stone-700">Absent</p>
                </div>
              </div>
            )}

            {/* Attendance List */}
            <div className="space-y-3 max-h-[500px] overflow-y-auto">
              {!todayAttendance ? (
                <div className="text-center py-8">
                  <Spinner size="md" />
                </div>
              ) : todayAttendance.attendance.length === 0 ? (
                <p className="text-center text-stone-500 py-8">No attendance recorded today</p>
              ) : (
                todayAttendance.attendance.map((record) => (
                  <div
                    key={record.employee_id}
                    className="flex items-center justify-between p-3 bg-stone-50 rounded-lg"
                  >
                    <div className="flex items-center gap-3">
                      <div className={`w-2 h-2 rounded-full ${record.is_present ? 'bg-green-500' : 'bg-stone-400'}`} />
                      <div>
                        <p className="font-medium text-stone-800">{record.employee_name}</p>
                        <p className="text-xs text-stone-500">ID: {record.employee_id}</p>
                      </div>
                    </div>
                    <div className="text-right">
                      {record.check_ins.length > 0 && (
                        <div className="flex items-center gap-1 text-green-600 text-sm">
                          <LogIn size={14} />
                          {formatTime(record.check_ins[0])}
                        </div>
                      )}
                      {record.check_outs.length > 0 && (
                        <div className="flex items-center gap-1 text-blue-600 text-sm">
                          <LogOut size={14} />
                          {formatTime(record.check_outs[record.check_outs.length - 1])}
                        </div>
                      )}
                    </div>
                  </div>
                ))
              )}
            </div>

            <button
              onClick={loadTodayAttendance}
              className="w-full mt-4 py-2 px-4 bg-stone-100 text-stone-700 rounded-lg font-medium hover:bg-stone-200 transition-colors flex items-center justify-center gap-2"
            >
              <RefreshCw size={18} />
              Refresh
            </button>
          </Card>
        </div>
      </div>
    </AppLayout>
  );
}
