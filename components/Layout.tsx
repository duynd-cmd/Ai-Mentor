import React from 'react';
import { useMemory } from '../contexts/MemoryContext';
import { ViewState } from '../types';
import { 
  Home, 
  BookOpen, 
  Brain, 
  FileText, 
  Clock, 
  LogOut, 
  GraduationCap, 
  Menu,
  X,
  Search
} from 'lucide-react';

interface LayoutProps {
  children: React.ReactNode;
}

export const Layout: React.FC<LayoutProps> = ({ children }) => {
  const { view, setView, memory, logout } = useMemory();
  const [isMobileMenuOpen, setIsMobileMenuOpen] = React.useState(false);

  // Hide sidebar on Landing, Auth, and Onboarding pages
  const isPublicPage = view === ViewState.LANDING || view === ViewState.AUTH || view === ViewState.ONBOARDING;

  const navItems = [
    { id: ViewState.DASHBOARD, label: 'Tổng quan', icon: Home },
    { id: ViewState.PLANNER, label: 'Lộ trình', icon: BookOpen },
    { id: ViewState.RESOURCES, label: 'Tài liệu', icon: Search },
    { id: ViewState.SCRIBA, label: 'Scriba AI', icon: Brain },
    { id: ViewState.TIMER, label: 'Đồng hồ', icon: Clock },
    { id: ViewState.NOTES, label: 'Ghi chú', icon: FileText },
  ];

  const handleNav = (v: ViewState) => {
    setView(v);
    setIsMobileMenuOpen(false);
  };

  if (isPublicPage) {
    return <div className="min-h-screen bg-cream-50 font-sans text-slate-800">{children}</div>;
  }

  return (
    <div className="flex min-h-screen bg-cream-50 font-sans text-slate-800">
      {/* Mobile Header */}
      <div className="md:hidden fixed top-0 w-full bg-cream-100 border-b border-cream-200 z-50 flex items-center justify-between p-4">
        <div className="flex items-center gap-2">
           <div className="bg-sage-200 p-1.5 rounded-lg">
              <GraduationCap className="w-5 h-5 text-slate-800" />
           </div>
           <span className="font-serif font-bold text-lg">AI Mentor</span>
        </div>
        <button onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}>
          {isMobileMenuOpen ? <X /> : <Menu />}
        </button>
      </div>

      {/* Sidebar */}
      <aside className={`
        fixed inset-y-0 left-0 z-40 w-64 bg-cream-100 border-r border-cream-200 transform transition-transform duration-200 ease-in-out
        ${isMobileMenuOpen ? 'translate-x-0' : '-translate-x-full'}
        md:translate-x-0 md:static md:block
      `}>
        <div className="flex flex-col h-full p-6">
          <div className="hidden md:flex items-center gap-3 mb-10">
            <div className="bg-sage-200 p-2 rounded-xl border border-sage-300">
              <GraduationCap className="w-6 h-6 text-slate-800" />
            </div>
            <div>
              <h1 className="font-serif font-bold text-xl leading-tight">AI Mentor</h1>
            </div>
          </div>

          {/* User Profile Snippet */}
          <div className="mb-8 p-4 bg-white/50 rounded-2xl border border-cream-200">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-full bg-slate-800 text-cream-50 flex items-center justify-center font-serif text-lg">
                {memory.name.charAt(0)}
              </div>
              <div>
                <p className="font-medium text-xs text-slate-500">Xin chào,</p>
                <p className="font-bold text-slate-900 truncate max-w-[120px]">{memory.name}</p>
                <p className="text-xs text-sage-600 font-medium">{memory.grade}</p>
              </div>
            </div>
          </div>

          <nav className="flex-1 space-y-1">
            <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-4 px-2">Công cụ học tập</p>
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = view === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => handleNav(item.id)}
                  className={`
                    w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-colors
                    ${isActive 
                      ? 'bg-slate-800 text-cream-50 shadow-sm' 
                      : 'text-slate-600 hover:bg-cream-200 hover:text-slate-900'}
                  `}
                >
                  <Icon className={`w-5 h-5 ${isActive ? 'text-sage-300' : 'text-slate-400'}`} />
                  {item.label}
                </button>
              );
            })}
          </nav>

          <div className="mt-auto pt-6 border-t border-cream-200">
            <button 
              onClick={logout}
              className="flex items-center gap-3 px-3 py-2.5 text-slate-500 hover:text-slate-900 text-sm font-medium w-full hover:bg-cream-200 rounded-xl transition-colors"
            >
              <LogOut className="w-5 h-5" />
              Đăng xuất
            </button>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 pt-20 md:pt-0 overflow-y-auto h-screen relative">
        <div className="max-w-7xl mx-auto p-6 md:p-10">
          {children}
        </div>
      </main>
    </div>
  );
};