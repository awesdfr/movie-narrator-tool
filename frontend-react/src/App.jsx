import React from "react";

export default function App() {
  const sidebarItems = [
    { id: "new", icon: "◻", label: "新线程", active: true },
    { id: "auto", icon: "◷", label: "自动化", active: false },
    { id: "skill", icon: "✦", label: "技能", active: false },
  ];

  const threadItems = [
    { name: "my_video_project", sub: "无线程", active: false },
    { name: "movie-narrator-tool-main", sub: "", active: true },
    { name: "Improve video match and nar...", sub: "28 分", active: false },
  ];

  const suggestions = [
    { icon: "🎮", title: "Build a classic Snake game in this repo." },
    { icon: "📄", title: "Create a one-page PDF that summarizes this app." },
    { icon: "✏️", title: "Create a plan to optimize onboarding flow." },
  ];

  return (
    <div className="min-h-screen overflow-hidden bg-[radial-gradient(circle_at_top_left,_rgba(129,140,248,0.22),_transparent_28%),radial-gradient(circle_at_top_right,_rgba(56,189,248,0.18),_transparent_24%),radial-gradient(circle_at_bottom_right,_rgba(192,132,252,0.14),_transparent_22%),linear-gradient(180deg,_#f5f2f8_0%,_#efebf3_50%,_#ebe7f0_100%)] text-zinc-800">
      <div className="flex min-h-screen gap-4 p-4">
        <Sidebar sidebarItems={sidebarItems} threadItems={threadItems} />
        <MainPanel suggestions={suggestions} />
      </div>
    </div>
  );
}

function Sidebar({ sidebarItems, threadItems }) {
  return (
    <aside className="flex w-[290px] shrink-0 flex-col justify-between rounded-[28px] border border-white/40 bg-white/35 shadow-[0_12px_40px_rgba(160,160,180,0.18)] backdrop-blur-2xl">
      <div>
        <div className="flex items-center gap-3 border-b border-white/30 px-5 py-5">
          <div className="h-7 w-7 rounded-full bg-gradient-to-br from-indigo-500 via-sky-400 to-violet-400 shadow-md" />
          <div className="text-[18px] font-semibold tracking-tight">
            Codex
          </div>
        </div>

        <div className="space-y-2 px-4 py-4">
          {sidebarItems.map((item) => (
            <button
              key={item.id}
              className={`flex w-full items-center gap-3 rounded-2xl px-4 py-3 text-left text-sm transition-all duration-200 ${
                item.active
                  ? "bg-white/65 shadow-sm"
                  : "hover:bg-white/45 active:scale-[0.99]"
              }`}
            >
              <span className="text-base opacity-70">{item.icon}</span>
              <span>{item.label}</span>
            </button>
          ))}
        </div>

        <div className="px-4 pt-2">
          <div className="mb-3 flex items-center justify-between px-2 text-xs text-zinc-500">
            <span>线程</span>
            <div className="flex items-center gap-3">
              <button className="opacity-70 hover:opacity-100">⊞</button>
              <button className="opacity-70 hover:opacity-100">≡</button>
            </div>
          </div>

          <div className="space-y-2">
            {threadItems.map((item) => (
              <button
                key={item.name}
                className={`w-full rounded-2xl px-4 py-3 text-left text-sm transition-all ${
                  item.active ? "bg-white/55 shadow-sm" : "hover:bg-white/35"
                }`}
              >
                <div className="truncate text-zinc-700">{item.name}</div>
                {item.sub ? (
                  <div className="mt-2 text-xs text-zinc-400">{item.sub}</div>
                ) : null}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="p-4">
        <button className="w-full rounded-2xl border border-white/30 bg-white/50 px-4 py-3 text-sm shadow-sm transition hover:bg-white/70">
          设置
        </button>
      </div>
    </aside>
  );
}

function MainPanel({ suggestions }) {
  return (
    <main className="relative flex-1 overflow-hidden rounded-[32px] border border-white/35 bg-white/40 shadow-[0_16px_48px_rgba(160,160,180,0.18)] backdrop-blur-2xl">
      <Header />
      <div className="relative flex h-[calc(100vh-32px-82px)] flex-col items-center justify-between px-10 py-10">
        <div className="pointer-events-none absolute left-1/2 top-20 h-44 w-44 -translate-x-1/2 rounded-full bg-violet-300/30 blur-3xl" />
        <div className="pointer-events-none absolute bottom-20 right-16 h-36 w-36 rounded-full bg-sky-300/25 blur-3xl" />
        <div className="pointer-events-none absolute left-16 top-40 h-28 w-28 rounded-full bg-white/30 blur-2xl" />

        <section className="flex flex-1 flex-col items-center justify-center text-center">
          <div className="mb-6 flex h-20 w-20 items-center justify-center rounded-full border border-white/55 bg-white/45 text-3xl shadow-[0_12px_30px_rgba(180,180,200,0.22)] backdrop-blur-xl">
            ☁
          </div>
          <h1 className="text-5xl font-semibold tracking-tight text-zinc-800">
            开始构建
          </h1>
          <button className="mt-3 text-2xl text-zinc-500 transition hover:text-zinc-700">
            movie-narrator-tool-main ▾
          </button>
          <div className="mt-16 grid w-full max-w-4xl grid-cols-1 gap-4 md:grid-cols-3">
            {suggestions.map((item) => (
              <SuggestionCard
                key={item.title}
                icon={item.icon}
                title={item.title}
              />
            ))}
          </div>
        </section>

        <PromptPanel />
      </div>
    </main>
  );
}

function Header() {
  return (
    <div className="flex items-center justify-between border-b border-white/30 px-6 py-5">
      <div className="text-lg font-semibold">新线程</div>
      <div className="flex items-center gap-3">
        <button className="rounded-xl border border-white/40 bg-white/60 px-4 py-2 text-sm shadow-sm transition hover:bg-white/80">
          📁 打开
        </button>
        <TopActionButton label="－" />
        <TopActionButton label="□" />
        <TopActionButton label="×" />
      </div>
    </div>
  );
}

function TopActionButton({ label }) {
  return (
    <button className="flex h-10 w-10 items-center justify-center rounded-xl bg-white/50 text-sm shadow-sm transition hover:bg-white/70">
      {label}
    </button>
  );
}

function SuggestionCard({ icon, title }) {
  return (
    <button className="rounded-[24px] border border-white/40 bg-white/45 p-5 text-left shadow-[0_10px_30px_rgba(180,180,200,0.14)] backdrop-blur-xl transition duration-200 hover:-translate-y-0.5 hover:bg-white/60">
      <div className="mb-4 text-lg">{icon}</div>
      <div className="text-sm leading-7 text-zinc-700">{title}</div>
    </button>
  );
}

function PromptPanel() {
  return (
    <div className="w-full max-w-4xl">
      <div className="rounded-[28px] border border-white/40 bg-white/55 p-4 shadow-[0_14px_40px_rgba(170,170,190,0.16)] backdrop-blur-2xl">
        <div className="min-h-[92px] rounded-[20px] border border-white/20 bg-white/35 px-4 py-4 text-sm text-zinc-400">
          向 Codex 任意提问，@ 添加文件，/ 调出命令
        </div>
        <div className="mt-4 flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3 text-sm text-zinc-500">
            <SmallButton>＋</SmallButton>
            <TagButton>GPT-5.4 ▾</TagButton>
            <TagButton>超高 ▾</TagButton>
          </div>
          <div className="flex items-center gap-3">
            <SmallButton>🎙</SmallButton>
            <button className="flex h-11 w-11 items-center justify-center rounded-full bg-zinc-700 text-white shadow-lg transition hover:scale-[1.03]">
              ↑
            </button>
          </div>
        </div>
      </div>

      <div className="mt-4 flex flex-wrap items-center justify-between gap-3 px-2 text-sm text-zinc-500">
        <div className="flex items-center gap-3">
          <TagButton>💻 本地 ▾</TagButton>
          <TagButton>默认权限 ▾</TagButton>
        </div>
        <div className="text-zinc-400">✣ 创建 Git 存储库</div>
      </div>
    </div>
  );
}

function SmallButton({ children }) {
  return (
    <button className="flex h-10 w-10 items-center justify-center rounded-xl bg-white/55 shadow-sm transition hover:bg-white/75">
      {children}
    </button>
  );
}

function TagButton({ children }) {
  return (
    <button className="rounded-xl bg-white/55 px-4 py-2 shadow-sm transition hover:bg-white/75">
      {children}
    </button>
  );
}

