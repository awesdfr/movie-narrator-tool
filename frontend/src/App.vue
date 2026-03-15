<template>
  <el-config-provider :locale="locale">
    <div class="app-layout">

      <!-- Sidebar -->
      <aside class="sidebar" :class="{ collapsed: sidebarCollapsed }">
        <div class="sidebar-logo" @click="$router.push('/')">
          <div class="logo-icon">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
              <path d="M15 10l4.553-2.277A1 1 0 0121 8.723v6.554a1 1 0 01-1.447.894L15 14M3 8a2 2 0 012-2h10a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2V8z"
                stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
          </div>
          <span class="logo-text">解说重制</span>
        </div>

        <nav class="sidebar-nav">
          <div class="nav-section">
            <router-link to="/" class="nav-item" :class="{ active: $route.name === 'Home' }">
              <svg class="nav-icon" viewBox="0 0 24 24" fill="none">
                <path d="M3 9.5L12 3l9 6.5V20a1 1 0 01-1 1H4a1 1 0 01-1-1V9.5z"
                  stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/>
                <path d="M9 21V12h6v9" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/>
              </svg>
              <span class="nav-label">项目列表</span>
            </router-link>

            <router-link to="/settings" class="nav-item" :class="{ active: $route.name === 'Settings' }">
              <svg class="nav-icon" viewBox="0 0 24 24" fill="none">
                <circle cx="12" cy="12" r="3" stroke="currentColor" stroke-width="1.8"/>
                <path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 010 2.83 2 2 0 01-2.83 0l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-4 0v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83-2.83l.06-.06A1.65 1.65 0 004.68 15a1.65 1.65 0 00-1.51-1H3a2 2 0 010-4h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 012.83-2.83l.06.06A1.65 1.65 0 009 4.68a1.65 1.65 0 001-1.51V3a2 2 0 014 0v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 2.83l-.06.06A1.65 1.65 0 0019.4 9a1.65 1.65 0 001.51 1H21a2 2 0 010 4h-.09a1.65 1.65 0 00-1.51 1z"
                  stroke="currentColor" stroke-width="1.8"/>
              </svg>
              <span class="nav-label">设置</span>
            </router-link>
          </div>

          <!-- Project sub-nav -->
          <div v-if="isInProject" class="nav-section project-nav">
            <div class="nav-section-label">当前项目</div>

            <router-link :to="`/project/${projectId}`" class="nav-item"
              :class="{ active: $route.name === 'Project' }">
              <svg class="nav-icon" viewBox="0 0 24 24" fill="none">
                <rect x="3" y="3" width="7" height="7" rx="1" stroke="currentColor" stroke-width="1.8"/>
                <rect x="14" y="3" width="7" height="7" rx="1" stroke="currentColor" stroke-width="1.8"/>
                <rect x="3" y="14" width="7" height="7" rx="1" stroke="currentColor" stroke-width="1.8"/>
                <rect x="14" y="14" width="7" height="7" rx="1" stroke="currentColor" stroke-width="1.8"/>
              </svg>
              <span class="nav-label">概览</span>
            </router-link>

            <router-link :to="`/project/${projectId}/editor`" class="nav-item"
              :class="{ active: $route.name === 'Editor' }">
              <svg class="nav-icon" viewBox="0 0 24 24" fill="none">
                <path d="M11 4H4a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2v-7"
                  stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>
                <path d="M18.5 2.5a2.121 2.121 0 013 3L12 15l-4 1 1-4 9.5-9.5z"
                  stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>
              </svg>
              <span class="nav-label">片段编辑</span>
            </router-link>

            <router-link :to="`/project/${projectId}/timeline`" class="nav-item"
              :class="{ active: $route.name === 'Timeline' }">
              <svg class="nav-icon" viewBox="0 0 24 24" fill="none">
                <line x1="3" y1="6" x2="21" y2="6" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>
                <line x1="3" y1="12" x2="21" y2="12" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>
                <line x1="3" y1="18" x2="21" y2="18" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>
                <rect x="6" y="4" width="5" height="4" rx="1" fill="currentColor" opacity=".4"/>
                <rect x="13" y="10" width="4" height="4" rx="1" fill="currentColor" opacity=".4"/>
                <rect x="8" y="16" width="6" height="4" rx="1" fill="currentColor" opacity=".4"/>
              </svg>
              <span class="nav-label">时间轴</span>
            </router-link>

            <router-link :to="`/project/${projectId}/preview`" class="nav-item"
              :class="{ active: $route.name === 'Preview' }">
              <svg class="nav-icon" viewBox="0 0 24 24" fill="none">
                <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"
                  stroke="currentColor" stroke-width="1.8"/>
                <circle cx="12" cy="12" r="3" stroke="currentColor" stroke-width="1.8"/>
              </svg>
              <span class="nav-label">预览对应</span>
            </router-link>
          </div>
        </nav>

        <button class="sidebar-collapse" @click="sidebarCollapsed = !sidebarCollapsed" title="收起侧边栏">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
            <path d="M15 18l-6-6 6-6" stroke="currentColor" stroke-width="2"
              stroke-linecap="round" stroke-linejoin="round"
              :style="{ transform: sidebarCollapsed ? 'rotate(180deg)' : '', transformOrigin: 'center' }"/>
          </svg>
        </button>
      </aside>

      <!-- Main area -->
      <div class="main-area">
        <!-- Topbar -->
        <header class="topbar">
          <div class="topbar-left">
            <div v-if="isInProject" class="breadcrumb">
              <router-link to="/" class="bc-item">项目列表</router-link>
              <span class="bc-sep">›</span>
              <span class="bc-item current">{{ pageTitle }}</span>
            </div>
            <div v-else class="topbar-title">{{ pageTitle }}</div>
          </div>
          <div class="topbar-right">
            <div class="topbar-hint" v-if="appVersion">
              <span class="version-badge">v{{ appVersion }}</span>
            </div>
          </div>
        </header>

        <!-- Page content -->
        <main class="page-content">
          <router-view />
        </main>
      </div>
    </div>
  </el-config-provider>
</template>

<script setup>
import { computed, ref } from 'vue'
import { useRoute } from 'vue-router'
import { useSettingsStore } from '@/stores/settings'
import zhCn from 'element-plus/dist/locale/zh-cn.mjs'
import en from 'element-plus/dist/locale/en.mjs'

const settingsStore = useSettingsStore()
const route = useRoute()
const sidebarCollapsed = ref(false)
const appVersion = ref('2.0')

const locale = computed(() => settingsStore.language === 'zh-CN' ? zhCn : en)

const isInProject = computed(() =>
  ['Project', 'Editor', 'Timeline', 'Preview', 'Process'].includes(route.name)
)

const projectId = computed(() => route.params.id)

const pageTitle = computed(() => {
  const map = {
    Home: '项目列表',
    Project: '概览',
    Editor: '片段编辑',
    Timeline: '时间轴',
    Preview: '预览对应',
    Process: '处理中',
    Settings: '设置',
  }
  return map[route.name] || '页面'
})
</script>

<style lang="scss">
.app-layout {
  display: flex;
  min-height: 100vh;
  background: var(--bg-base);
}

// ── Sidebar ──────────────────────────────────────────────────────────────────
.sidebar {
  width: var(--sidebar-width);
  min-height: 100vh;
  background: var(--sidebar-bg);
  border-right: 1px solid var(--sidebar-border);
  display: flex;
  flex-direction: column;
  flex-shrink: 0;
  transition: width .2s ease;
  position: sticky;
  top: 0;
  height: 100vh;
  overflow: hidden;

  &.collapsed {
    width: 60px;
    .logo-text, .nav-label, .nav-section-label { opacity: 0; pointer-events: none; }
    .sidebar-logo { justify-content: center; padding: 0; }
    .nav-item { justify-content: center; padding: 0 0 0 0; }
    .project-nav { border-top: none; }
  }
}

.sidebar-logo {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 0 16px;
  height: 56px;
  cursor: pointer;
  border-bottom: 1px solid var(--sidebar-border);
  transition: background .15s;
  &:hover { background: var(--bg-hover); }

  .logo-icon {
    width: 34px;
    height: 34px;
    background: var(--accent-dim);
    border: 1px solid rgba(91,108,248,.3);
    border-radius: 9px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--accent);
    flex-shrink: 0;
  }

  .logo-text {
    font-size: 15px;
    font-weight: 700;
    color: var(--text-primary);
    white-space: nowrap;
    transition: opacity .2s;
  }
}

.sidebar-nav {
  flex: 1;
  overflow-y: auto;
  padding: 10px 8px;
}

.nav-section {
  margin-bottom: 6px;
  &.project-nav {
    margin-top: 12px;
    padding-top: 12px;
    border-top: 1px solid var(--border-faint);
  }
}

.nav-section-label {
  font-size: 10px;
  font-weight: 700;
  color: var(--text-muted);
  letter-spacing: .8px;
  text-transform: uppercase;
  padding: 4px 10px 8px;
  transition: opacity .2s;
}

.nav-item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 0 10px;
  height: 38px;
  border-radius: 8px;
  color: var(--text-secondary);
  text-decoration: none;
  font-size: 13.5px;
  font-weight: 500;
  transition: all .15s;
  margin-bottom: 2px;
  white-space: nowrap;

  &:hover { background: var(--bg-hover); color: var(--text-primary); }
  &.active { background: var(--accent-dim); color: var(--accent); }

  .nav-icon {
    width: 17px;
    height: 17px;
    flex-shrink: 0;
  }

  .nav-label { transition: opacity .2s; }
}

.sidebar-collapse {
  margin: 8px;
  height: 32px;
  background: transparent;
  border: 1px solid var(--border-faint);
  border-radius: 7px;
  color: var(--text-muted);
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all .15s;
  &:hover { background: var(--bg-hover); color: var(--text-secondary); }
}

// ── Main area ────────────────────────────────────────────────────────────────
.main-area {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  min-height: 100vh;
}

// ── Topbar ───────────────────────────────────────────────────────────────────
.topbar {
  height: 56px;
  background: var(--bg-surface);
  border-bottom: 1px solid var(--border-faint);
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 24px;
  flex-shrink: 0;
  position: sticky;
  top: 0;
  z-index: 10;
}

.topbar-left { display: flex; align-items: center; gap: 8px; }
.topbar-right { display: flex; align-items: center; gap: 10px; }
.topbar-title { font-size: 15px; font-weight: 600; color: var(--text-primary); }

.breadcrumb {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 13px;
  .bc-item {
    color: var(--text-secondary);
    text-decoration: none;
    transition: color .15s;
    &:hover { color: var(--text-primary); }
    &.current { color: var(--text-primary); font-weight: 500; }
  }
  .bc-sep { color: var(--text-muted); }
}

.version-badge {
  font-size: 11px;
  padding: 2px 8px;
  background: var(--bg-elevated);
  border: 1px solid var(--border-faint);
  border-radius: 20px;
  color: var(--text-muted);
  font-weight: 500;
}

// ── Page content ─────────────────────────────────────────────────────────────
.page-content {
  flex: 1;
  padding: 24px;
  overflow-y: auto;
}
</style>
