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
          <span class="logo-text">视频匹配</span>
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
import zhCn from 'element-plus/dist/locale/zh-cn.mjs'

const route = useRoute()
const sidebarCollapsed = ref(false)
const appVersion = ref('2.0')

const locale = zhCn

const isInProject = computed(() =>
  route.name === 'Project'
)

const projectId = computed(() => route.params.id)

const pageTitle = computed(() => {
  const map = {
    Home: '项目列表',
    Project: '概览',
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
