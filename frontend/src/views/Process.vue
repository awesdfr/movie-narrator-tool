<template>
  <div class="process-page">

    <!-- Main status card -->
    <div class="status-card">
      <!-- Stage icon -->
      <div class="stage-icon" :class="stageIconClass">
        <svg v-if="progress.stage === 'completed'" width="40" height="40" viewBox="0 0 24 24" fill="none">
          <path d="M22 11.08V12a10 10 0 11-5.93-9.14" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
          <path d="M22 4L12 14.01l-3-3" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
        <svg v-else-if="progress.stage === 'error'" width="40" height="40" viewBox="0 0 24 24" fill="none">
          <circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="2"/>
          <path d="M15 9l-6 6M9 9l6 6" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
        </svg>
        <svg v-else width="40" height="40" viewBox="0 0 24 24" fill="none" class="spin">
          <path d="M21 12a9 9 0 11-6.219-8.56" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"/>
        </svg>
      </div>

      <!-- Stage text -->
      <div class="stage-info">
        <div class="stage-name">{{ getStageText(progress.stage) }}</div>
        <div class="stage-msg">{{ progress.message || '等待中…' }}</div>
      </div>

      <!-- Progress bar -->
      <div v-if="progress.stage !== 'completed' && progress.stage !== 'error'" class="prog-track">
        <div class="prog-fill" :style="{ width: (progress.progress || 0) + '%' }"></div>
        <span class="prog-pct">{{ progress.progress || 0 }}%</span>
      </div>

      <!-- Actions -->
      <div class="stage-actions">
        <el-button
          v-if="progress.stage === 'completed'"
          type="primary"
          @click="$router.push(`/project/${projectId}/editor`)"
        >
          查看片段
        </el-button>
        <el-button
          v-if="projectStore.processing"
          type="danger"
          @click="handleStop"
        >
          停止处理
        </el-button>
        <el-button
          v-if="progress.stage === 'error'"
          type="primary"
          @click="handleRetry"
        >
          重新处理
        </el-button>
      </div>
    </div>

    <!-- Parallel tasks -->
    <div v-if="parallelTasks.length > 0" class="parallel-card">
      <div class="pc-head">
        <span class="pc-title">并行任务</span>
        <span class="pc-badge" :class="getParallelBadgeClass()">{{ completedTasks }}/{{ totalTasks }} 完成</span>
      </div>
      <div class="task-list">
        <div v-for="task in parallelTasks" :key="task.id" class="task-item">
          <div class="task-meta">
            <span class="task-name">{{ task.name }}</span>
            <span class="task-status" :class="task.status">{{ getTaskStatusText(task.status) }}</span>
          </div>
          <div class="task-prog-track">
            <div
              class="task-prog-fill"
              :class="task.status"
              :style="{ width: task.progress + '%' }"
            ></div>
          </div>
        </div>
      </div>
    </div>

    <!-- Log panel -->
    <div class="log-card">
      <div class="log-head">
        <span class="log-title">处理日志</span>
        <span class="log-count">{{ logs.length }} 条</span>
      </div>
      <div class="log-body" ref="logBody">
        <div v-if="logs.length === 0" class="log-empty">等待日志输出…</div>
        <div v-for="(log, index) in logs" :key="index" class="log-line">
          <span class="log-time">{{ log.time }}</span>
          <span class="log-msg" :class="log.type">{{ log.message }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted, nextTick } from 'vue'
import { useRoute } from 'vue-router'
import { useProjectStore } from '@/stores/project'
import wsService from '@/api/websocket'

const route = useRoute()
const projectStore = useProjectStore()

const projectId = computed(() => route.params.id)
const progress = computed(() => projectStore.progress)

const logs = ref([])
const parallelTasks = ref([])
const totalTasks = ref(0)
const completedTasks = ref(0)
const logBody = ref(null)

const stageIconClass = computed(() => {
  if (progress.value?.stage === 'completed') return 'done'
  if (progress.value?.stage === 'error') return 'err'
  return 'running'
})

onMounted(async () => {
  await wsService.connect(projectId.value)

  wsService.on('progress', (data) => {
    addLog(data.message, data.stage === 'error' ? 'error' : 'info')

    if (data.parallel_tasks) {
      parallelTasks.value = data.parallel_tasks
      totalTasks.value = data.total_tasks || data.parallel_tasks.length
      completedTasks.value = data.completed_tasks || data.parallel_tasks.filter(t => t.status === 'completed').length
    }

    if (data.stage !== progress.value?.stage) {
      parallelTasks.value = []
    }
  })

  addLog('开始监听处理进度...', 'info')
})

onUnmounted(() => {
  wsService.disconnect()
})

function getStageText(stage) {
  const texts = {
    '': '准备中',
    starting: '启动中',
    analyzing: '视频分析',
    recognizing: '语音识别',
    matching: '帧匹配（并行）',
    polishing: '文案润色（并行）',
    generating_tts: 'TTS生成（并行）',
    completed: '处理完成',
    error: '处理出错'
  }
  return texts[stage] || stage
}

function getTaskStatusText(status) {
  return { pending: '等待', running: '处理中', completed: '完成', error: '失败' }[status] || status
}

function getParallelBadgeClass() {
  if (completedTasks.value === totalTasks.value) return 'success'
  if (completedTasks.value > 0) return 'partial'
  return 'pending'
}

function addLog(message, type = 'info') {
  const now = new Date()
  const time = now.toLocaleTimeString('zh-CN')
  logs.value.push({ time, message, type })
  if (logs.value.length > 100) logs.value.shift()
  nextTick(() => {
    if (logBody.value) logBody.value.scrollTop = logBody.value.scrollHeight
  })
}

async function handleStop() {
  await projectStore.stopProcessing(projectId.value)
  addLog('处理已停止', 'warning')
}

async function handleRetry() {
  logs.value = []
  parallelTasks.value = []
  await projectStore.startProcessing(projectId.value)
}
</script>

<style lang="scss" scoped>
.process-page {
  max-width: 860px;
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

// ── Status card ───────────────────────────────────────────────────────────────
.status-card {
  background: var(--card-bg);
  border: 1px solid var(--border-faint);
  border-radius: 12px;
  padding: 32px 28px;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 20px;
  text-align: center;
}

.stage-icon {
  width: 80px;
  height: 80px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;

  &.running {
    background: rgba(91,108,248,.12);
    color: var(--accent);
    border: 2px solid rgba(91,108,248,.3);
  }
  &.done {
    background: rgba(103,194,58,.12);
    color: #67c23a;
    border: 2px solid rgba(103,194,58,.3);
  }
  &.err {
    background: rgba(245,108,108,.12);
    color: #f56c6c;
    border: 2px solid rgba(245,108,108,.3);
  }
}

.spin {
  animation: spin 1.2s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.stage-info { display: flex; flex-direction: column; gap: 6px; }

.stage-name {
  font-size: 22px;
  font-weight: 700;
  color: var(--text-primary);
}

.stage-msg {
  font-size: 13px;
  color: var(--text-muted);
  max-width: 500px;
}

// ── Progress bar ──────────────────────────────────────────────────────────────
.prog-track {
  width: 100%;
  max-width: 480px;
  height: 22px;
  background: var(--bg-elevated);
  border-radius: 11px;
  position: relative;
  overflow: hidden;
  border: 1px solid var(--border-faint);
}

.prog-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--accent) 0%, #7b8cf8 100%);
  border-radius: 11px;
  transition: width .4s ease;
  min-width: 4px;
}

.prog-pct {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  font-weight: 700;
  color: #fff;
  text-shadow: 0 1px 2px rgba(0,0,0,.4);
}

.stage-actions { display: flex; gap: 10px; }

// ── Parallel card ─────────────────────────────────────────────────────────────
.parallel-card {
  background: var(--card-bg);
  border: 1px solid var(--border-faint);
  border-radius: 10px;
  overflow: hidden;
}

.pc-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 16px;
  border-bottom: 1px solid var(--border-faint);
  background: var(--bg-elevated);
}

.pc-title { font-size: 13px; font-weight: 600; color: var(--text-primary); }

.pc-badge {
  font-size: 11px;
  font-weight: 700;
  padding: 2px 8px;
  border-radius: 4px;
  &.success { background: rgba(103,194,58,.15); color: #67c23a; }
  &.partial { background: rgba(230,162,60,.15); color: #e6a23c; }
  &.pending { background: var(--bg-elevated); color: var(--text-muted); border: 1px solid var(--border-faint); }
}

.task-list {
  max-height: 360px;
  overflow-y: auto;
  padding: 8px 16px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.task-item { display: flex; flex-direction: column; gap: 6px; }

.task-meta {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.task-name { font-size: 13px; font-weight: 500; color: var(--text-primary); }

.task-status {
  font-size: 11px;
  font-weight: 600;
  padding: 2px 7px;
  border-radius: 4px;
  &.pending { background: var(--bg-elevated); color: var(--text-muted); }
  &.running { background: rgba(91,108,248,.15); color: var(--accent); }
  &.completed { background: rgba(103,194,58,.15); color: #67c23a; }
  &.error { background: rgba(245,108,108,.15); color: #f56c6c; }
}

.task-prog-track {
  height: 6px;
  background: var(--bg-elevated);
  border-radius: 3px;
  overflow: hidden;
}

.task-prog-fill {
  height: 100%;
  border-radius: 3px;
  transition: width .3s ease;
  &.running { background: var(--accent); }
  &.completed { background: #67c23a; }
  &.error { background: #f56c6c; }
  &.pending { background: var(--text-muted); }
}

// ── Log panel ─────────────────────────────────────────────────────────────────
.log-card {
  background: var(--card-bg);
  border: 1px solid var(--border-faint);
  border-radius: 10px;
  overflow: hidden;
}

.log-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 16px;
  border-bottom: 1px solid var(--border-faint);
  background: var(--bg-elevated);
}

.log-title { font-size: 13px; font-weight: 600; color: var(--text-primary); }
.log-count { font-size: 11px; color: var(--text-muted); }

.log-body {
  max-height: 320px;
  overflow-y: auto;
  padding: 10px 16px;
  font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
  font-size: 12px;
  background: #090b12;
}

.log-empty {
  color: var(--text-muted);
  font-style: italic;
  padding: 8px 0;
}

.log-line {
  display: flex;
  gap: 12px;
  padding: 3px 0;
  border-bottom: 1px solid rgba(255,255,255,.03);
  &:last-child { border-bottom: none; }
}

.log-time {
  color: #4a5066;
  flex-shrink: 0;
  user-select: none;
}

.log-msg {
  color: #9ba3bf;
  line-height: 1.5;
  word-break: break-word;
  &.error { color: #f56c6c; }
  &.warning { color: #e6a23c; }
  &.info { color: #9ba3bf; }
}
</style>
