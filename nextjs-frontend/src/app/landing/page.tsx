'use client';

import { useEffect, useState, useRef } from 'react';
import { useRouter } from 'next/navigation';
import Image from 'next/image';
import { useAuthStore } from '@/lib/store';
import {
  Shield, Eye, Wind, Brain, UserCheck, AlertTriangle,
  BarChart3, ChevronRight, Play, CheckCircle2, ArrowRight,
  HardHat, Shirt, Glasses, Hand, CircleDot, Footprints,
  Activity, Users, Clock, Zap, MapPin, Bell, FileText,
  ChevronDown, Menu, X, Sparkles, TrendingUp, Lock
} from 'lucide-react';

// Animated counter hook
function useCounter(end: number, duration: number = 2000, start: number = 0) {
  const [count, setCount] = useState(start);
  const [isVisible, setIsVisible] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting && !isVisible) {
          setIsVisible(true);
        }
      },
      { threshold: 0.1 }
    );

    if (ref.current) {
      observer.observe(ref.current);
    }

    return () => observer.disconnect();
  }, [isVisible]);

  useEffect(() => {
    if (!isVisible) return;

    let startTime: number;
    const animate = (timestamp: number) => {
      if (!startTime) startTime = timestamp;
      const progress = Math.min((timestamp - startTime) / duration, 1);
      setCount(Math.floor(progress * (end - start) + start));
      if (progress < 1) {
        requestAnimationFrame(animate);
      }
    };
    requestAnimationFrame(animate);
  }, [isVisible, end, start, duration]);

  return { count, ref };
}

// Feature card component
function FeatureCard({
  icon: Icon,
  title,
  description,
  color,
  delay
}: {
  icon: React.ElementType;
  title: string;
  description: string;
  color: string;
  delay: number;
}) {
  return (
    <div
      className={`group bg-white rounded-3xl p-8 border border-slate-200 shadow-sm hover:shadow-xl transition-all duration-500 hover:-translate-y-2 animate-fade-up`}
      style={{ animationDelay: `${delay}ms` }}
    >
      <div className={`w-16 h-16 rounded-2xl bg-gradient-to-br ${color} flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300 shadow-lg`}>
        <Icon className="w-8 h-8 text-white" strokeWidth={2} />
      </div>
      <h3 className="text-xl font-bold text-slate-800 mb-3">{title}</h3>
      <p className="text-slate-500 leading-relaxed">{description}</p>
    </div>
  );
}

// PPE Item component for animation
function PPEItem({
  icon: Icon,
  label,
  detected,
  delay
}: {
  icon: React.ElementType;
  label: string;
  detected: boolean;
  delay: number;
}) {
  return (
    <div
      className={`flex items-center gap-3 p-3 rounded-xl transition-all duration-300 animate-slide-in-right ${
        detected
          ? 'bg-emerald-50 border border-emerald-200'
          : 'bg-red-50 border border-red-200'
      }`}
      style={{ animationDelay: `${delay}ms` }}
    >
      <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${
        detected ? 'bg-emerald-100' : 'bg-red-100'
      }`}>
        <Icon className={`w-5 h-5 ${detected ? 'text-emerald-600' : 'text-red-600'}`} />
      </div>
      <span className="font-medium text-slate-700">{label}</span>
      <div className="ml-auto">
        {detected ? (
          <CheckCircle2 className="w-5 h-5 text-emerald-500" />
        ) : (
          <X className="w-5 h-5 text-red-500" />
        )}
      </div>
    </div>
  );
}

// Stat component
function StatItem({ value, label, suffix = '' }: { value: number; label: string; suffix?: string }) {
  const { count, ref } = useCounter(value, 2000);
  return (
    <div ref={ref} className="text-center">
      <div className="text-4xl md:text-5xl font-bold text-white mb-2">
        {count}{suffix}
      </div>
      <div className="text-orange-100 font-medium">{label}</div>
    </div>
  );
}

export default function LandingPage() {
  const router = useRouter();
  const { token } = useAuthStore();
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const [activeFeature, setActiveFeature] = useState(0);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  // Redirect authenticated users to dashboard
  useEffect(() => {
    if (mounted) {
      const storedToken = localStorage.getItem('token');
      if (storedToken || token) {
        router.push('/');
      }
    }
  }, [mounted, token, router]);

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 50);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  // Auto-rotate features
  useEffect(() => {
    const interval = setInterval(() => {
      setActiveFeature((prev) => (prev + 1) % 4);
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  const features = [
    {
      icon: Eye,
      title: 'AI Vision Detection',
      description: 'Real-time PPE detection using YOLOv8. Identifies helmets, vests, goggles, gloves, masks & safety shoes in milliseconds.',
      color: 'from-orange-400 to-amber-500',
    },
    {
      icon: Wind,
      title: 'Gas Monitoring',
      description: 'Continuous monitoring of Methane (CH₄) and Carbon Monoxide (CO) levels with instant evacuation alerts.',
      color: 'from-emerald-400 to-teal-500',
    },
    {
      icon: Brain,
      title: 'Predictive Analytics',
      description: 'ML-powered risk prediction identifies at-risk workers before incidents occur. Prevent, don\'t just react.',
      color: 'from-violet-400 to-purple-500',
    },
    {
      icon: UserCheck,
      title: 'Face Recognition',
      description: 'Biometric attendance and gate access using FaceNet. Know exactly who is in the mine at all times.',
      color: 'from-blue-400 to-cyan-500',
    },
    {
      icon: AlertTriangle,
      title: 'SOS Emergency',
      description: 'One-tap emergency alerts with location tracking. Audio broadcasts and automatic evacuation triggers.',
      color: 'from-red-400 to-rose-500',
    },
    {
      icon: BarChart3,
      title: 'Smart Dashboards',
      description: '7 role-specific dashboards from Worker to Super Admin. Everyone sees exactly what they need.',
      color: 'from-amber-400 to-yellow-500',
    },
  ];

  const dashboardRoles = [
    { role: 'Super Admin', desc: 'Complete system oversight', color: 'bg-orange-100 text-orange-700 border-orange-300' },
    { role: 'General Manager', desc: 'Organization-wide KPIs', color: 'bg-amber-100 text-amber-700 border-amber-300' },
    { role: 'Area Safety Officer', desc: 'Multi-mine analytics', color: 'bg-yellow-100 text-yellow-700 border-yellow-300' },
    { role: 'Manager', desc: 'Mine-level management', color: 'bg-lime-100 text-lime-700 border-lime-300' },
    { role: 'Safety Officer', desc: 'Compliance enforcement', color: 'bg-emerald-100 text-emerald-700 border-emerald-300' },
    { role: 'Shift Incharge', desc: 'Real-time gate control', color: 'bg-teal-100 text-teal-700 border-teal-300' },
    { role: 'Worker', desc: 'Personal safety tracking', color: 'bg-cyan-100 text-cyan-700 border-cyan-300' },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-b from-stone-50 via-white to-orange-50/30">
      {/* Navigation */}
      <nav className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
        scrolled ? 'bg-white/95 backdrop-blur-md shadow-lg' : 'bg-transparent'
      }`}>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-20">
            {/* Logo */}
            <div className="flex items-center gap-3">
              <Image
                src="/raksham-logo.png"
                alt="Raksham"
                width={220}
                height={80}
                className="h-28 w-auto"
              />
            </div>

            {/* Desktop Navigation */}
            <div className="hidden md:flex items-center gap-8">
              <a href="#features" className="text-slate-600 hover:text-orange-500 font-medium transition-colors">Features</a>
              <a href="#how-it-works" className="text-slate-600 hover:text-orange-500 font-medium transition-colors">How It Works</a>
              <a href="#demo" className="text-slate-600 hover:text-orange-500 font-medium transition-colors">Demo</a>
              <button
                onClick={() => router.push('/login')}
                className="btn btn-primary"
              >
                Sign In
                <ArrowRight className="w-4 h-4" />
              </button>
            </div>

            {/* Mobile Menu Button */}
            <button
              onClick={() => setIsMenuOpen(!isMenuOpen)}
              className="md:hidden p-2 rounded-xl bg-slate-100"
            >
              {isMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
            </button>
          </div>
        </div>

        {/* Mobile Menu */}
        {isMenuOpen && (
          <div className="md:hidden bg-white border-t border-slate-200 shadow-lg animate-slide-down">
            <div className="px-4 py-6 space-y-4">
              <a href="#features" className="block text-slate-600 hover:text-orange-500 font-medium py-2">Features</a>
              <a href="#how-it-works" className="block text-slate-600 hover:text-orange-500 font-medium py-2">How It Works</a>
              <a href="#demo" className="block text-slate-600 hover:text-orange-500 font-medium py-2">Demo</a>
              <button
                onClick={() => router.push('/login')}
                className="w-full btn btn-primary"
              >
                Sign In
              </button>
            </div>
          </div>
        )}
      </nav>

      {/* Hero Section */}
      <section className="relative pt-32 pb-20 lg:pt-40 lg:pb-32 overflow-hidden">
        {/* Background Elements */}
        <div className="absolute inset-0 overflow-hidden">
          <div className="absolute -top-40 -right-40 w-96 h-96 bg-gradient-to-br from-orange-200 to-amber-200 rounded-full blur-3xl opacity-40 animate-float" />
          <div className="absolute -bottom-40 -left-40 w-96 h-96 bg-gradient-to-br from-emerald-200 to-teal-200 rounded-full blur-3xl opacity-40 animate-float-delayed" />
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-gradient-to-br from-orange-100 to-transparent rounded-full blur-3xl opacity-30" />
        </div>

        {/* Grid Pattern */}
        <div className="absolute inset-0 opacity-[0.03]">
          <div className="absolute inset-0" style={{
            backgroundImage: `url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23000000' fill-opacity='1'%3E%3Cpath d='m36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")`,
          }} />
        </div>

        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid lg:grid-cols-2 gap-12 lg:gap-16 items-center">
            {/* Left Content */}
            <div className="text-center lg:text-left flex flex-col justify-center">
              {/* <div className="inline-flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-orange-100 to-amber-100 rounded-full border border-orange-200 mb-8 animate-fade-up">
                <Sparkles className="w-4 h-4 text-orange-500" />
                <span className="text-sm font-semibold text-orange-700">AI-Powered Mine Safety</span>
              </div> */}

              <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold text-slate-800 mb-6 leading-tight animate-fade-up animation-delay-100">
                Every Worker
                <span className="bg-gradient-to-r from-orange-500 to-amber-500 bg-clip-text text-transparent"> Home Safe</span>.
                <br />Every Shift.
              </h1>

              <p className="text-lg sm:text-xl text-slate-600 mb-10 max-w-xl mx-auto lg:mx-0 leading-relaxed animate-fade-up animation-delay-200">
                Raksham combines AI vision, predictive analytics, and real-time monitoring to protect your mining workforce like never before.
              </p>

              <div className="flex flex-col sm:flex-row gap-4 justify-center lg:justify-start animate-fade-up animation-delay-300">
                <button
                  onClick={() => router.push('/login')}
                  className="btn btn-primary text-lg px-8 py-4 shadow-xl shadow-orange-200/50 hover:shadow-2xl hover:shadow-orange-300/50"
                >
                  Get Started
                  <ArrowRight className="w-5 h-5" />
                </button>
                <a
                  href="#demo"
                  className="btn btn-secondary text-lg px-8 py-4 group"
                >
                  <Play className="w-5 h-5 text-orange-500 group-hover:scale-110 transition-transform" />
                  Watch Demo
                </a>
              </div>

              {/* Trust Badges */}
              <div className="mt-12 flex flex-wrap items-center justify-center lg:justify-start gap-6 animate-fade-up animation-delay-400">
                <div className="flex items-center gap-2 text-slate-500">
                  <Shield className="w-5 h-5 text-emerald-500" />
                  <span className="text-sm font-medium">DGMS Compliant</span>
                </div>
                <div className="flex items-center gap-2 text-slate-500">
                  <Lock className="w-5 h-5 text-blue-500" />
                  <span className="text-sm font-medium">Enterprise Security</span>
                </div>
                <div className="flex items-center gap-2 text-slate-500">
                  <Zap className="w-5 h-5 text-amber-500" />
                  <span className="text-sm font-medium">Real-time Processing</span>
                </div>
              </div>
            </div>

            {/* Right Content - Video Demo */}
            <div className="relative animate-fade-up animation-delay-200">
              <div className="relative bg-white rounded-3xl shadow-2xl border border-slate-200 overflow-hidden">
                {/* Demo Header */}
                <div className="bg-gradient-to-r from-slate-800 to-slate-900 px-6 py-4 flex items-center gap-3">
                  <div className="flex gap-2">
                    <div className="w-3 h-3 rounded-full bg-red-400" />
                    <div className="w-3 h-3 rounded-full bg-amber-400" />
                    <div className="w-3 h-3 rounded-full bg-emerald-400" />
                  </div>
                  <div className="flex items-center gap-2 ml-4">
                    <div className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
                    <span className="text-white/80 text-sm font-medium">Live PPE Detection</span>
                  </div>
                </div>

                {/* Video */}
                <div className="relative">
                  <video
                    autoPlay
                    loop
                    muted
                    playsInline
                    className="w-full h-auto"
                  >
                    <source src="/landing-video.mp4" type="video/mp4" />
                    Your browser does not support the video tag.
                  </video>

                  {/* Scanning Effect Overlay */}
                  <div className="absolute inset-0 bg-gradient-to-b from-orange-500/10 to-transparent animate-scan pointer-events-none" />
                </div>
              </div>

              {/* Floating Elements */}
              <div className="absolute -top-6 -right-6 w-20 h-20 bg-gradient-to-br from-emerald-400 to-teal-500 rounded-2xl flex items-center justify-center shadow-xl animate-float">
                <Shield className="w-10 h-10 text-white" />
              </div>
              <div className="absolute -bottom-4 -left-4 w-16 h-16 bg-gradient-to-br from-orange-400 to-amber-500 rounded-2xl flex items-center justify-center shadow-xl animate-float-delayed">
                <Eye className="w-8 h-8 text-white" />
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-16 bg-gradient-to-r from-orange-500 via-amber-500 to-orange-600 relative overflow-hidden">
        <div className="absolute inset-0 opacity-20" style={{
          backgroundImage: 'radial-gradient(circle, rgba(255,255,255,0.3) 1px, transparent 1px)',
          backgroundSize: '20px 20px'
        }} />

        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            <StatItem value={99} label="Detection Accuracy" suffix="%" />
            <StatItem value={500} label="Response Time" suffix="ms" />
            <StatItem value={24} label="Monitoring" suffix="/7" />
            <StatItem value={6} label="PPE Items Tracked" suffix="+" />
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-24 relative">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <div className="inline-flex items-center gap-2 px-4 py-2 bg-orange-100 rounded-full border border-orange-200 mb-6">
              <Sparkles className="w-4 h-4 text-orange-500" />
              <span className="text-sm font-semibold text-orange-700">Comprehensive Protection</span>
            </div>
            <h2 className="text-3xl sm:text-4xl lg:text-5xl font-bold text-slate-800 mb-6">
              One Platform. <span className="text-orange-500">Complete Safety.</span>
            </h2>
            <p className="text-lg text-slate-600 max-w-2xl mx-auto">
              Everything you need to protect your workforce, from AI-powered detection to predictive analytics.
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {features.map((feature, index) => (
              <FeatureCard
                key={feature.title}
                icon={feature.icon}
                title={feature.title}
                description={feature.description}
                color={feature.color}
                delay={index * 100}
              />
            ))}
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section id="how-it-works" className="py-24 bg-gradient-to-b from-slate-50 to-white relative">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <div className="inline-flex items-center gap-2 px-4 py-2 bg-emerald-100 rounded-full border border-emerald-200 mb-6">
              <Zap className="w-4 h-4 text-emerald-500" />
              <span className="text-sm font-semibold text-emerald-700">Quick Setup</span>
            </div>
            <h2 className="text-3xl sm:text-4xl lg:text-5xl font-bold text-slate-800 mb-6">
              Up and Running in <span className="text-emerald-500">3 Steps</span>
            </h2>
          </div>

          <div className="grid md:grid-cols-3 gap-8 relative">
            {/* Connection Line */}
            <div className="hidden md:block absolute top-24 left-1/4 right-1/4 h-1 bg-gradient-to-r from-orange-300 via-amber-300 to-emerald-300 rounded-full" />

            {[
              { step: 1, icon: Activity, title: 'Connect Cameras', desc: 'Hook up your existing CCTV infrastructure. No special hardware needed.', color: 'from-orange-400 to-amber-500' },
              { step: 2, icon: Users, title: 'Register Workers', desc: 'Upload worker photos and IDs. Face recognition ready in minutes.', color: 'from-amber-400 to-yellow-500' },
              { step: 3, icon: Shield, title: 'Start Protecting', desc: 'AI begins monitoring immediately. Real-time alerts and compliance tracking.', color: 'from-emerald-400 to-teal-500' },
            ].map((item, index) => (
              <div key={item.step} className="relative animate-fade-up" style={{ animationDelay: `${index * 150}ms` }}>
                <div className="bg-white rounded-3xl p-8 border border-slate-200 shadow-lg hover:shadow-xl transition-all duration-300 text-center relative z-10">
                  <div className={`w-20 h-20 mx-auto rounded-2xl bg-gradient-to-br ${item.color} flex items-center justify-center mb-6 shadow-lg`}>
                    <item.icon className="w-10 h-10 text-white" />
                  </div>
                  <div className="absolute -top-3 -right-3 w-10 h-10 rounded-full bg-slate-800 text-white font-bold flex items-center justify-center text-lg shadow-lg">
                    {item.step}
                  </div>
                  <h3 className="text-xl font-bold text-slate-800 mb-3">{item.title}</h3>
                  <p className="text-slate-500">{item.desc}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Role-Based Dashboards */}
      <section className="py-24 relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-orange-50 via-white to-amber-50" />

        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative">
          <div className="grid lg:grid-cols-2 gap-16 items-center">
            <div>
              <div className="inline-flex items-center gap-2 px-4 py-2 bg-violet-100 rounded-full border border-violet-200 mb-6">
                <BarChart3 className="w-4 h-4 text-violet-500" />
                <span className="text-sm font-semibold text-violet-700">Role-Based Access</span>
              </div>
              <h2 className="text-3xl sm:text-4xl font-bold text-slate-800 mb-6">
                7 Specialized Dashboards for <span className="text-violet-500">Every Role</span>
              </h2>
              <p className="text-lg text-slate-600 mb-8">
                From workers checking their compliance to super admins overseeing the entire operation, everyone gets exactly what they need.
              </p>

              <div className="space-y-3">
                {dashboardRoles.map((role, index) => (
                  <div
                    key={role.role}
                    className={`flex items-center justify-between p-4 rounded-2xl border ${role.color} animate-fade-up`}
                    style={{ animationDelay: `${index * 50}ms` }}
                  >
                    <span className="font-semibold">{role.role}</span>
                    <span className="text-sm opacity-80">{role.desc}</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="relative">
              <div className="bg-white rounded-3xl shadow-2xl border border-slate-200 overflow-hidden">
                <div className="bg-gradient-to-r from-slate-100 to-slate-50 px-6 py-4 border-b border-slate-200">
                  <div className="flex items-center gap-4">
                    <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-orange-400 to-amber-500 flex items-center justify-center">
                      <BarChart3 className="w-5 h-5 text-white" />
                    </div>
                    <div>
                      <div className="font-semibold text-slate-800">Safety Dashboard</div>
                      <div className="text-sm text-slate-500">Real-time overview</div>
                    </div>
                  </div>
                </div>

                <div className="p-6 space-y-4">
                  {/* Mock Dashboard Cards */}
                  <div className="grid grid-cols-3 gap-4">
                    {[
                      { label: 'Active Workers', value: '247', color: 'bg-emerald-100 text-emerald-700' },
                      { label: 'Violations Today', value: '3', color: 'bg-red-100 text-red-700' },
                      { label: 'Compliance', value: '96%', color: 'bg-orange-100 text-orange-700' },
                    ].map((stat) => (
                      <div key={stat.label} className={`p-4 rounded-2xl ${stat.color}`}>
                        <div className="text-2xl font-bold">{stat.value}</div>
                        <div className="text-xs font-medium opacity-80">{stat.label}</div>
                      </div>
                    ))}
                  </div>

                  {/* Mock Chart */}
                  <div className="h-40 bg-gradient-to-t from-orange-100 to-white rounded-2xl flex items-end justify-around p-4">
                    {[40, 65, 45, 80, 55, 90, 70].map((height, i) => (
                      <div
                        key={i}
                        className="w-8 bg-gradient-to-t from-orange-400 to-amber-300 rounded-t-lg animate-grow-height"
                        style={{ height: `${height}%`, animationDelay: `${i * 100}ms` }}
                      />
                    ))}
                  </div>

                  {/* Recent Alerts */}
                  <div className="space-y-2">
                    {[
                      { type: 'warning', msg: 'Worker W042 - Missing goggles at Gate A' },
                      { type: 'success', msg: 'All workers compliant in Zone B' },
                    ].map((alert, i) => (
                      <div
                        key={i}
                        className={`flex items-center gap-3 p-3 rounded-xl ${
                          alert.type === 'warning' ? 'bg-amber-50 border border-amber-200' : 'bg-emerald-50 border border-emerald-200'
                        }`}
                      >
                        {alert.type === 'warning' ? (
                          <AlertTriangle className="w-5 h-5 text-amber-500" />
                        ) : (
                          <CheckCircle2 className="w-5 h-5 text-emerald-500" />
                        )}
                        <span className="text-sm text-slate-700">{alert.msg}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Floating Badge */}
              <div className="absolute -top-4 -right-4 bg-gradient-to-br from-violet-500 to-purple-600 text-white px-4 py-2 rounded-full text-sm font-bold shadow-xl animate-bounce-subtle">
                Live Data
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Predictive Analytics Section */}
      <section className="py-24 bg-gradient-to-b from-violet-50 to-white relative overflow-hidden">
        <div className="absolute inset-0">
          <div className="absolute top-0 left-1/4 w-96 h-96 bg-violet-200/50 rounded-full blur-3xl" />
          <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-purple-200/50 rounded-full blur-3xl" />
        </div>

        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative">
          <div className="text-center mb-16">
            <div className="inline-flex items-center gap-2 px-4 py-2 bg-violet-100 rounded-full border border-violet-200 mb-6">
              <Brain className="w-4 h-4 text-violet-600" />
              <span className="text-sm font-semibold text-violet-700">Predictive AI</span>
            </div>
            <h2 className="text-3xl sm:text-4xl lg:text-5xl font-bold text-slate-800 mb-6">
              Predict. Prevent. <span className="text-violet-600">Protect.</span>
            </h2>
            <p className="text-lg text-slate-600 max-w-2xl mx-auto">
              Our ML models don't just detect problems—they predict them before they happen.
            </p>
          </div>

          <div className="grid lg:grid-cols-2 gap-12 items-center">
            <div className="bg-white rounded-3xl p-8 border border-slate-200 shadow-xl">
              <div className="flex items-center gap-4 mb-6">
                <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center">
                  <Users className="w-8 h-8 text-white" />
                </div>
                <div>
                  <div className="text-slate-800 font-bold text-xl">Rajesh Kumar</div>
                  <div className="text-slate-500">Employee ID: W-2847</div>
                </div>
              </div>

              {/* Risk Meter */}
              <div className="mb-6">
                <div className="flex justify-between items-center mb-2">
                  <span className="text-slate-700 font-medium">Risk Level</span>
                  <span className="text-amber-600 font-bold">HIGH</span>
                </div>
                <div className="h-4 bg-slate-200 rounded-full overflow-hidden">
                  <div className="h-full w-3/4 bg-gradient-to-r from-emerald-400 via-amber-400 to-red-400 rounded-full animate-grow-width" />
                </div>
              </div>

              {/* Risk Factors */}
              <div className="space-y-3 mb-6">
                <div className="text-slate-700 font-semibold mb-2">Risk Factors:</div>
                {[
                  { icon: TrendingUp, text: 'Compliance dropping (92% → 78%)', color: 'text-amber-600' },
                  { icon: Clock, text: '3 late arrivals this week', color: 'text-orange-600' },
                  { icon: Glasses, text: 'Goggles violations: 4x above average', color: 'text-red-600' },
                ].map((factor, i) => (
                  <div key={i} className="flex items-center gap-3 text-slate-600">
                    <factor.icon className={`w-5 h-5 ${factor.color}`} />
                    <span>{factor.text}</span>
                  </div>
                ))}
              </div>

              {/* AI Recommendation */}
              <div className="bg-gradient-to-r from-violet-100 to-purple-100 rounded-2xl p-4 border border-violet-200">
                <div className="flex items-center gap-2 mb-2">
                  <Brain className="w-5 h-5 text-violet-600" />
                  <span className="text-violet-700 font-semibold">AI Recommendation</span>
                </div>
                <p className="text-slate-700">
                  "Schedule refresher training by Friday. <span className="text-emerald-600 font-semibold">87% confidence</span> this prevents incident."
                </p>
              </div>
            </div>

            <div className="space-y-6">
              {[
                { title: 'Violation Prediction', desc: 'Predicts future violation counts based on historical patterns', accuracy: '94%' },
                { title: 'Attendance Analysis', desc: 'Identifies workers likely to have attendance issues', accuracy: '91%' },
                { title: 'Risk Scoring', desc: 'Multi-factor risk assessment for proactive intervention', accuracy: '89%' },
              ].map((model, index) => (
                <div
                  key={model.title}
                  className="bg-white rounded-2xl p-6 border border-slate-200 shadow-lg hover:shadow-xl transition-all duration-300 animate-fade-up"
                  style={{ animationDelay: `${index * 100}ms` }}
                >
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="text-slate-800 font-semibold text-lg">{model.title}</h4>
                    <span className="text-emerald-600 font-bold">{model.accuracy}</span>
                  </div>
                  <p className="text-slate-500">{model.desc}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section id="demo" className="py-24 relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-orange-500 via-amber-500 to-orange-600" />
        <div className="absolute inset-0 opacity-20" style={{
          backgroundImage: 'radial-gradient(circle, rgba(255,255,255,0.3) 1px, transparent 1px)',
          backgroundSize: '20px 20px'
        }} />

        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 relative text-center">
          <div className="bg-white/10 backdrop-blur-sm rounded-3xl p-12 border border-white/20">
            <h2 className="text-3xl sm:text-4xl lg:text-5xl font-bold text-white mb-6">
              Every Second Counts in Safety
            </h2>
            <p className="text-xl text-orange-100 mb-10 max-w-2xl mx-auto">
              Your workers deserve the best protection. Start safeguarding your team with AI-powered safety today.
            </p>

            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <button
                onClick={() => router.push('/login')}
                className="bg-white text-orange-600 hover:bg-orange-50 font-semibold text-lg px-10 py-4 rounded-2xl shadow-xl hover:shadow-2xl transition-all duration-300 flex items-center justify-center gap-2"
              >
                Start Protecting Today
                <ArrowRight className="w-5 h-5" />
              </button>
            </div>

            <div className="mt-10 flex flex-wrap items-center justify-center gap-8 text-orange-100">
              <div className="flex items-center gap-2">
                <CheckCircle2 className="w-5 h-5" />
                <span>Free Demo</span>
              </div>
              <div className="flex items-center gap-2">
                <CheckCircle2 className="w-5 h-5" />
                <span>No Credit Card</span>
              </div>
              <div className="flex items-center gap-2">
                <CheckCircle2 className="w-5 h-5" />
                <span>24/7 Support</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-gradient-to-b from-slate-100 to-slate-50 text-slate-600 py-16 border-t border-slate-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid md:grid-cols-4 gap-12 mb-12">
            <div className="md:col-span-2">
              <Image
                src="/raksham-logo.png"
                alt="Raksham"
                width={240}
                height={100}
                className="h-28 w-auto mb-4"
              />
              <p className="text-slate-500 max-w-md">
                AI-powered mine safety system protecting workers with real-time PPE detection, gas monitoring, and predictive analytics.
              </p>
            </div>

            <div>
              <h4 className="text-slate-800 font-semibold mb-4">Features</h4>
              <ul className="space-y-2">
                <li><a href="#features" className="hover:text-orange-500 transition-colors">PPE Detection</a></li>
                <li><a href="#features" className="hover:text-orange-500 transition-colors">Gas Monitoring</a></li>
                <li><a href="#features" className="hover:text-orange-500 transition-colors">Face Recognition</a></li>
                <li><a href="#features" className="hover:text-orange-500 transition-colors">Predictive Analytics</a></li>
              </ul>
            </div>

            <div>
              <h4 className="text-slate-800 font-semibold mb-4">Contact</h4>
              <ul className="space-y-2">
                <li className="flex items-center gap-2">
                  <MapPin className="w-4 h-4 text-orange-500" />
                  <span>India</span>
                </li>
                <li className="flex items-center gap-2">
                  <Bell className="w-4 h-4 text-orange-500" />
                  <span>24/7 Support</span>
                </li>
              </ul>
            </div>
          </div>

          <div className="border-t border-slate-200 pt-8 text-center">
            <p className="text-slate-500">
              © 2024 Raksham - Mine Safety System. Built with care for worker safety.
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}
