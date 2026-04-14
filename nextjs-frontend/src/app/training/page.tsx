'use client';

import { useState } from 'react';
import {
  GraduationCap,
  PlayCircle,
  CheckCircle,
  Clock,
  Award,
  BookOpen,
  Shield,
  AlertTriangle,
  ChevronRight,
  Lock,
  Star,
} from 'lucide-react';
import AppLayout from '@/components/AppLayout';
import { Card } from '@/components/Card';

interface TrainingModule {
  id: string;
  title: string;
  description: string;
  duration: string;
  category: 'safety' | 'ppe' | 'emergency' | 'general';
  status: 'completed' | 'in_progress' | 'locked' | 'available';
  progress?: number;
  completedDate?: string;
}

const trainingModules: TrainingModule[] = [
  {
    id: '1',
    title: 'PPE Fundamentals',
    description: 'Learn about the essential personal protective equipment required in mining operations.',
    duration: '15 min',
    category: 'ppe',
    status: 'completed',
    completedDate: '2024-01-15',
  },
  {
    id: '2',
    title: 'Helmet Safety Standards',
    description: 'Understanding proper helmet usage, maintenance, and inspection procedures.',
    duration: '10 min',
    category: 'ppe',
    status: 'completed',
    completedDate: '2024-01-16',
  },
  {
    id: '3',
    title: 'High-Visibility Vest Requirements',
    description: 'Why high-visibility clothing is essential and when it must be worn.',
    duration: '8 min',
    category: 'ppe',
    status: 'in_progress',
    progress: 60,
  },
  {
    id: '4',
    title: 'Emergency Evacuation Procedures',
    description: 'Step-by-step guide to emergency evacuation routes and assembly points.',
    duration: '20 min',
    category: 'emergency',
    status: 'available',
  },
  {
    id: '5',
    title: 'Fire Safety in Mines',
    description: 'Identifying fire hazards and proper use of fire extinguishers.',
    duration: '25 min',
    category: 'emergency',
    status: 'available',
  },
  {
    id: '6',
    title: 'First Aid Basics',
    description: 'Essential first aid knowledge for common workplace injuries.',
    duration: '30 min',
    category: 'safety',
    status: 'locked',
  },
  {
    id: '7',
    title: 'Hazard Recognition',
    description: 'Identifying and reporting potential hazards in the workplace.',
    duration: '20 min',
    category: 'safety',
    status: 'locked',
  },
  {
    id: '8',
    title: 'Workplace Safety Culture',
    description: 'Building a culture of safety awareness and responsibility.',
    duration: '15 min',
    category: 'general',
    status: 'locked',
  },
];

const categoryConfig: Record<string, { color: string; icon: React.ReactNode; label: string }> = {
  safety: { color: 'bg-green-100 text-green-700', icon: <Shield className="w-4 h-4" />, label: 'Safety' },
  ppe: { color: 'bg-blue-100 text-blue-700', icon: <BookOpen className="w-4 h-4" />, label: 'PPE' },
  emergency: { color: 'bg-red-100 text-red-700', icon: <AlertTriangle className="w-4 h-4" />, label: 'Emergency' },
  general: { color: 'bg-purple-100 text-purple-700', icon: <GraduationCap className="w-4 h-4" />, label: 'General' },
};

export default function TrainingPage() {
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [selectedModule, setSelectedModule] = useState<TrainingModule | null>(null);

  const filteredModules = selectedCategory === 'all'
    ? trainingModules
    : trainingModules.filter(m => m.category === selectedCategory);

  const completedCount = trainingModules.filter(m => m.status === 'completed').length;
  const totalCount = trainingModules.length;
  const progressPercentage = Math.round((completedCount / totalCount) * 100);

  const getStatusBadge = (status: TrainingModule['status'], progress?: number) => {
    switch (status) {
      case 'completed':
        return (
          <span className="flex items-center gap-1 px-2 py-1 bg-green-100 text-green-700 rounded-full text-xs font-medium">
            <CheckCircle className="w-3 h-3" />
            Completed
          </span>
        );
      case 'in_progress':
        return (
          <span className="flex items-center gap-1 px-2 py-1 bg-blue-100 text-blue-700 rounded-full text-xs font-medium">
            <PlayCircle className="w-3 h-3" />
            {progress}% Complete
          </span>
        );
      case 'available':
        return (
          <span className="flex items-center gap-1 px-2 py-1 bg-orange-100 text-orange-700 rounded-full text-xs font-medium">
            <PlayCircle className="w-3 h-3" />
            Start
          </span>
        );
      case 'locked':
        return (
          <span className="flex items-center gap-1 px-2 py-1 bg-stone-100 text-stone-500 rounded-full text-xs font-medium">
            <Lock className="w-3 h-3" />
            Locked
          </span>
        );
    }
  };

  return (
    <AppLayout>
      <div className="max-w-4xl mx-auto space-y-6">
        {/* Header */}
        <div>
          <h1 className="text-2xl font-bold text-stone-800">Training Center</h1>
          <p className="text-stone-500 mt-1">Complete training modules to improve your safety knowledge</p>
        </div>

        {/* Progress Overview */}
        <div className="bg-gradient-to-r from-orange-500 to-orange-600 rounded-2xl p-6 text-white">
          <div className="flex items-center justify-between mb-4">
            <div>
              <p className="text-orange-100 text-sm">Overall Progress</p>
              <p className="text-3xl font-bold mt-1">{completedCount} / {totalCount} Modules</p>
            </div>
            <div className="w-20 h-20 relative">
              <svg className="w-full h-full transform -rotate-90">
                <circle
                  cx="40"
                  cy="40"
                  r="35"
                  fill="none"
                  stroke="rgba(255,255,255,0.2)"
                  strokeWidth="6"
                />
                <circle
                  cx="40"
                  cy="40"
                  r="35"
                  fill="none"
                  stroke="white"
                  strokeWidth="6"
                  strokeDasharray={`${progressPercentage * 2.2} 220`}
                  strokeLinecap="round"
                />
              </svg>
              <div className="absolute inset-0 flex items-center justify-center">
                <span className="text-lg font-bold">{progressPercentage}%</span>
              </div>
            </div>
          </div>
          <div className="w-full bg-white/20 rounded-full h-2">
            <div
              className="bg-white h-2 rounded-full transition-all"
              style={{ width: `${progressPercentage}%` }}
            />
          </div>
        </div>

        {/* Category Filter */}
        <div className="flex flex-wrap gap-2">
          <button
            onClick={() => setSelectedCategory('all')}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              selectedCategory === 'all'
                ? 'bg-orange-600 text-white'
                : 'bg-stone-100 text-stone-700 hover:bg-stone-200'
            }`}
          >
            All Modules
          </button>
          {Object.entries(categoryConfig).map(([key, config]) => (
            <button
              key={key}
              onClick={() => setSelectedCategory(key)}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                selectedCategory === key
                  ? 'bg-orange-600 text-white'
                  : 'bg-stone-100 text-stone-700 hover:bg-stone-200'
              }`}
            >
              {config.icon}
              {config.label}
            </button>
          ))}
        </div>

        {/* Training Modules */}
        <div className="space-y-4">
          {filteredModules.map((module) => {
            const category = categoryConfig[module.category];
            return (
              <div
                key={module.id}
                onClick={() => module.status !== 'locked' && setSelectedModule(module)}
                className={`p-4 bg-white rounded-xl border border-stone-200 transition-all ${
                  module.status === 'locked'
                    ? 'opacity-60 cursor-not-allowed'
                    : 'hover:shadow-md cursor-pointer hover:border-orange-300'
                }`}
              >
                <div className="flex items-start gap-4">
                  <div className={`w-12 h-12 rounded-xl flex items-center justify-center flex-shrink-0 ${category.color}`}>
                    <GraduationCap className="w-6 h-6" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-start justify-between gap-2">
                      <div>
                        <h3 className="font-semibold text-stone-800">{module.title}</h3>
                        <p className="text-sm text-stone-500 mt-1">{module.description}</p>
                      </div>
                      {getStatusBadge(module.status, module.progress)}
                    </div>
                    <div className="flex items-center gap-4 mt-3">
                      <span className={`inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs ${category.color}`}>
                        {category.icon}
                        {category.label}
                      </span>
                      <span className="flex items-center gap-1 text-xs text-stone-500">
                        <Clock className="w-3 h-3" />
                        {module.duration}
                      </span>
                      {module.completedDate && (
                        <span className="text-xs text-stone-400">
                          Completed: {new Date(module.completedDate).toLocaleDateString()}
                        </span>
                      )}
                    </div>
                    {module.status === 'in_progress' && module.progress && (
                      <div className="mt-3">
                        <div className="w-full bg-stone-200 rounded-full h-1.5">
                          <div
                            className="bg-blue-500 h-1.5 rounded-full"
                            style={{ width: `${module.progress}%` }}
                          />
                        </div>
                      </div>
                    )}
                  </div>
                  {module.status !== 'locked' && (
                    <ChevronRight className="w-5 h-5 text-stone-400" />
                  )}
                </div>
              </div>
            );
          })}
        </div>

        {/* Achievements */}
        <Card title="Achievements">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center p-4 bg-yellow-50 rounded-xl border border-yellow-200">
              <div className="w-12 h-12 bg-yellow-100 rounded-full flex items-center justify-center mx-auto mb-2">
                <Star className="w-6 h-6 text-yellow-600" />
              </div>
              <p className="text-sm font-medium text-stone-800">Quick Learner</p>
              <p className="text-xs text-stone-500">Complete 3 modules</p>
              <p className="text-xs text-yellow-600 mt-1">{completedCount >= 3 ? 'Unlocked!' : `${completedCount}/3`}</p>
            </div>
            <div className={`text-center p-4 rounded-xl border ${completedCount >= 5 ? 'bg-green-50 border-green-200' : 'bg-stone-50 border-stone-200'}`}>
              <div className={`w-12 h-12 rounded-full flex items-center justify-center mx-auto mb-2 ${completedCount >= 5 ? 'bg-green-100' : 'bg-stone-200'}`}>
                <Shield className={`w-6 h-6 ${completedCount >= 5 ? 'text-green-600' : 'text-stone-400'}`} />
              </div>
              <p className="text-sm font-medium text-stone-800">Safety Expert</p>
              <p className="text-xs text-stone-500">Complete 5 modules</p>
              <p className={`text-xs mt-1 ${completedCount >= 5 ? 'text-green-600' : 'text-stone-500'}`}>
                {completedCount >= 5 ? 'Unlocked!' : `${completedCount}/5`}
              </p>
            </div>
            <div className={`text-center p-4 rounded-xl border ${completedCount >= totalCount ? 'bg-purple-50 border-purple-200' : 'bg-stone-50 border-stone-200'}`}>
              <div className={`w-12 h-12 rounded-full flex items-center justify-center mx-auto mb-2 ${completedCount >= totalCount ? 'bg-purple-100' : 'bg-stone-200'}`}>
                <Award className={`w-6 h-6 ${completedCount >= totalCount ? 'text-purple-600' : 'text-stone-400'}`} />
              </div>
              <p className="text-sm font-medium text-stone-800">Training Champion</p>
              <p className="text-xs text-stone-500">Complete all modules</p>
              <p className={`text-xs mt-1 ${completedCount >= totalCount ? 'text-purple-600' : 'text-stone-500'}`}>
                {completedCount >= totalCount ? 'Unlocked!' : `${completedCount}/${totalCount}`}
              </p>
            </div>
            <div className="text-center p-4 bg-stone-50 rounded-xl border border-stone-200">
              <div className="w-12 h-12 bg-stone-200 rounded-full flex items-center justify-center mx-auto mb-2">
                <GraduationCap className="w-6 h-6 text-stone-400" />
              </div>
              <p className="text-sm font-medium text-stone-800">Perfect Score</p>
              <p className="text-xs text-stone-500">Pass all quizzes</p>
              <p className="text-xs text-stone-500 mt-1">Coming Soon</p>
            </div>
          </div>
        </Card>

        {/* Module Detail Modal */}
        {selectedModule && (
          <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
            <div className="bg-white rounded-2xl max-w-lg w-full max-h-[90vh] overflow-y-auto">
              <div className="p-6">
                <div className="flex items-start justify-between mb-4">
                  <div className={`w-12 h-12 rounded-xl flex items-center justify-center ${categoryConfig[selectedModule.category].color}`}>
                    <GraduationCap className="w-6 h-6" />
                  </div>
                  <button
                    onClick={() => setSelectedModule(null)}
                    className="text-stone-400 hover:text-stone-600"
                  >
                    ✕
                  </button>
                </div>
                <h2 className="text-xl font-bold text-stone-800">{selectedModule.title}</h2>
                <p className="text-stone-500 mt-2">{selectedModule.description}</p>

                <div className="flex items-center gap-4 mt-4">
                  <span className={`inline-flex items-center gap-1 px-3 py-1 rounded-full text-sm ${categoryConfig[selectedModule.category].color}`}>
                    {categoryConfig[selectedModule.category].icon}
                    {categoryConfig[selectedModule.category].label}
                  </span>
                  <span className="flex items-center gap-1 text-sm text-stone-500">
                    <Clock className="w-4 h-4" />
                    {selectedModule.duration}
                  </span>
                </div>

                {selectedModule.status === 'completed' ? (
                  <div className="mt-6 p-4 bg-green-50 rounded-xl">
                    <div className="flex items-center gap-2 text-green-700">
                      <CheckCircle className="w-5 h-5" />
                      <span className="font-medium">Module Completed</span>
                    </div>
                    <p className="text-sm text-green-600 mt-1">
                      Completed on {selectedModule.completedDate ? new Date(selectedModule.completedDate).toLocaleDateString() : 'N/A'}
                    </p>
                  </div>
                ) : selectedModule.status === 'in_progress' ? (
                  <div className="mt-6">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm text-stone-600">Progress</span>
                      <span className="text-sm font-medium">{selectedModule.progress}%</span>
                    </div>
                    <div className="w-full bg-stone-200 rounded-full h-2">
                      <div
                        className="bg-blue-500 h-2 rounded-full"
                        style={{ width: `${selectedModule.progress}%` }}
                      />
                    </div>
                  </div>
                ) : null}

                <div className="mt-6 flex gap-3">
                  {selectedModule.status === 'completed' ? (
                    <button className="flex-1 py-3 bg-stone-100 text-stone-700 rounded-xl font-medium hover:bg-stone-200">
                      Review Module
                    </button>
                  ) : (
                    <button className="flex-1 py-3 bg-orange-600 text-white rounded-xl font-medium hover:bg-orange-700">
                      {selectedModule.status === 'in_progress' ? 'Continue' : 'Start'} Module
                    </button>
                  )}
                  <button
                    onClick={() => setSelectedModule(null)}
                    className="px-6 py-3 bg-stone-100 text-stone-700 rounded-xl font-medium hover:bg-stone-200"
                  >
                    Close
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </AppLayout>
  );
}
