'use client';
import Link from 'next/link';
import { usePathname } from 'next/navigation';

const navItems = [
  { name: '홈', href: '/' },
  { name: '주가 예측', href: '/predict' },
  { name: '예측 해석', href: '/xai' },
  { name: '창업 커뮤니티', href: '/community' },
  { name: '초보자 가이드', href: '/guide' },
  { name: '기회 찾기', href: '/chances' },
];


export default function Header() {
  const pathname = usePathname();

  return (
    <header className="w-full bg-white shadow-sm fixed top-0 left-0 z-50">
      <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
        {/* Logo */}
        <div className="flex items-center gap-3">
          <img src="/logo.png" alt="Logo" className="w-10 h-10" />
          <div className="text-xl font-bold text-gray-700 leading-tight">
            <div>RISE</div>
            <div className="text-xs text-gray-500">PLATFORM</div>
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex gap-8 text-sm font-semibold">
          {navItems.map(({ name, href }) => (
            <Link
              key={name}
              href={href}
              className={`relative transition-all duration-200 ${
                pathname === href ? 'text-blue-600' : 'text-gray-600 hover:text-blue-600'
              }`}
            >
              {name}
              {pathname === href && (
                <span className="absolute -bottom-1 left-0 w-full h-[1px] bg-blue-600" />
              )}
            </Link>
          ))}
        </nav>
      </div>
    </header>
  );
}
