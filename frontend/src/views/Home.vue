<template>
  <div class="home-page">

    <!-- Header -->
    <div class="page-header">
      <div>
        <h1 class="page-title">我的项目</h1>
        <p class="page-subtitle">管理电影解说重制任务</p>
      </div>
      <el-button type="primary" size="large" @click="openCreate">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" style="margin-right:6px">
          <path d="M12 5v14M5 12h14" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"/>
        </svg>
        新建项目
      </el-button>
    </div>

    <!-- Workflow guide -->
    <div class="workflow-card">
      <div class="workflow-title">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" style="flex-shrink:0">
          <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/>
        </svg>
        工作流程
      </div>
      <div class="workflow-steps">
        <div class="step">
          <div class="step-num">1</div>
          <div class="step-text">准备解说视频 + 原电影文件</div>
        </div>
        <div class="step">
          <div class="step-num">2</div>
          <div class="step-text">自动匹配解说画面到原电影对应片段</div>
        </div>
        <div class="step">
          <div class="step-num">3</div>
          <div class="step-text">AI 提取解说词并润色，生成新 TTS 语音</div>
        </div>
        <div class="step">
          <div class="step-num">4</div>
          <div class="step-text">原电影高清画面 + 新解说语音，导出成片</div>
        </div>
      </div>
    </div>

    <!-- Loading skeleton -->
    <div v-if="projectStore.loading" class="skeleton-grid">
      <el-skeleton v-for="i in 6" :key="i" animated class="skeleton-card">
        <template #template>
          <el-skeleton-item variant="rect" style="height:100px;border-radius:10px"/>
        </template>
      </el-skeleton>
    </div>

    <!-- Empty state -->
    <div v-else-if="projectStore.projects.length === 0" class="empty-state">
      <svg width="56" height="56" viewBox="0 0 24 24" fill="none" style="opacity:.3;margin-bottom:16px">
        <path d="M22 19a2 2 0 01-2 2H4a2 2 0 01-2-2V5a2 2 0 012-2h5l2 3h9a2 2 0 012 2z"
          stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"/>
      </svg>
      <p class="empty-text">暂无项目，点击右上角按钮新建</p>
    </div>

    <!-- Project grid -->
    <div v-else class="project-grid">
      <div
        v-for="project in projectStore.projects"
        :key="project.id"
        class="project-card"
        @click="goToProject(project)"
      >
        <!-- Status stripe -->
        <div class="card-stripe" :class="getStatusClass(project.status)"></div>

        <div class="card-body">
          <div class="card-top">
            <div class="project-name">{{ project.name }}</div>
            <el-tag :type="getStatusType(project.status)" size="small">
              {{ getStatusText(project.status) }}
            </el-tag>
          </div>

          <div class="card-meta">
            <span class="meta-item">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" style="opacity:.5">
                <rect x="3" y="4" width="18" height="18" rx="2" stroke="currentColor" stroke-width="2"/>
                <path d="M16 2v4M8 2v4M3 10h18" stroke="currentColor" stroke-width="2"/>
              </svg>
              {{ formatDate(project.updated_at) }}
            </span>
            <span class="meta-item" v-if="project.segment_count">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" style="opacity:.5">
                <path d="M3 6h18M3 12h18M3 18h18" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
              </svg>
              {{ project.segment_count }} 个片段
            </span>
          </div>

          <!-- Mini progress bar if processing -->
          <div v-if="isProcessing(project.status)" class="mini-progress">
            <div class="mini-bar pulsing"></div>
          </div>
        </div>

        <!-- Actions -->
        <div class="card-actions" @click.stop>
          <button type="button" class="action-btn" title="复制" @click.stop="duplicateProject(project)">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none">
              <rect x="9" y="9" width="13" height="13" rx="2" stroke="currentColor" stroke-width="2"/>
              <path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"
                stroke="currentColor" stroke-width="2"/>
            </svg>
          </button>
          <button type="button" class="action-btn danger" title="删除" @click.stop="confirmDelete(project)">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none">
              <polyline points="3 6 5 6 21 6" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
              <path d="M19 6l-1 14a2 2 0 01-2 2H8a2 2 0 01-2-2L5 6"
                stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
              <path d="M10 11v6M14 11v6M9 6V4a1 1 0 011-1h4a1 1 0 011 1v2"
                stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
            </svg>
          </button>
        </div>
      </div>
    </div>

    <!-- Create dialog -->
    <el-dialog
      v-model="showCreateDialog"
      title="新建项目"
      width="580px"
      :close-on-click-modal="false"
      destroy-on-close
    >
      <el-form
        ref="createFormRef"
        :model="createForm"
        :rules="createRules"
        label-width="110px"
        label-position="left"
      >
        <el-form-item label="项目名称" prop="name">
          <el-input v-model="createForm.name" placeholder="例如：变形金刚3 解说重制" />
        </el-form-item>

        <el-form-item label="原电影" prop="movie_path">
          <div class="file-select-row">
            <el-select v-model="createForm.movie_path" placeholder="选择电影文件"
              filterable :loading="loadingMovies" class="file-select">
              <el-option v-for="file in movieFiles" :key="file.path" :label="file.name" :value="file.path">
                <div class="file-option">
                  <span class="file-name">{{ file.name }}</span>
                  <span class="file-meta">{{ file.size_display }} · {{ file.modified_time }}</span>
                </div>
              </el-option>
            </el-select>
            <div class="file-actions">
              <button type="button" class="icon-btn" @click="refreshFiles('movies')" title="刷新">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
                  <polyline points="23 4 23 10 17 10" stroke="currentColor" stroke-width="2"/>
                  <path d="M20.49 15a9 9 0 11-2.12-9.36L23 10" stroke="currentColor" stroke-width="2"/>
                </svg>
              </button>
              <button type="button" class="icon-btn" @click="openFolder('movies')" title="打开文件夹">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
                  <path d="M22 19a2 2 0 01-2 2H4a2 2 0 01-2-2V5a2 2 0 012-2h5l2 3h9a2 2 0 012 2z"
                    stroke="currentColor" stroke-width="2"/>
                </svg>
              </button>
            </div>
          </div>
          <div class="form-tip">高清原始电影，请放入 videos/movies 目录</div>
        </el-form-item>

        <el-form-item label="解说视频" prop="narration_path">
          <div class="file-select-row">
            <el-select v-model="createForm.narration_path" placeholder="选择解说视频"
              filterable :loading="loadingNarrations" class="file-select">
              <el-option v-for="file in narrationFiles" :key="file.path" :label="file.name" :value="file.path">
                <div class="file-option">
                  <span class="file-name">{{ file.name }}</span>
                  <span class="file-meta">{{ file.size_display }} · {{ file.modified_time }}</span>
                </div>
              </el-option>
            </el-select>
            <div class="file-actions">
              <button type="button" class="icon-btn" @click="refreshFiles('narrations')" title="刷新">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
                  <polyline points="23 4 23 10 17 10" stroke="currentColor" stroke-width="2"/>
                  <path d="M20.49 15a9 9 0 11-2.12-9.36L23 10" stroke="currentColor" stroke-width="2"/>
                </svg>
              </button>
              <button type="button" class="icon-btn" @click="openFolder('narrations')" title="打开文件夹">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
                  <path d="M22 19a2 2 0 01-2 2H4a2 2 0 01-2-2V5a2 2 0 012-2h5l2 3h9a2 2 0 012 2z"
                    stroke="currentColor" stroke-width="2"/>
                </svg>
              </button>
            </div>
          </div>
          <div class="form-tip">B站电影解说视频，用作参考。请放入 videos/narrations 目录</div>
        </el-form-item>

        <el-form-item label="参考音频">
          <div class="file-select-row">
            <el-select v-model="createForm.reference_audio_path" placeholder="可选，用于声纹识别"
              filterable clearable :loading="loadingAudio" class="file-select">
              <el-option v-for="file in audioFiles" :key="file.path" :label="file.name" :value="file.path">
                <div class="file-option">
                  <span class="file-name">{{ file.name }}</span>
                  <span class="file-meta">{{ file.size_display }}</span>
                </div>
              </el-option>
            </el-select>
            <div class="file-actions">
              <button type="button" class="icon-btn" @click="refreshFiles('reference_audio')" title="刷新">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
                  <polyline points="23 4 23 10 17 10" stroke="currentColor" stroke-width="2"/>
                  <path d="M20.49 15a9 9 0 11-2.12-9.36L23 10" stroke="currentColor" stroke-width="2"/>
                </svg>
              </button>
              <button type="button" class="icon-btn" @click="openFolder('reference_audio')" title="打开文件夹">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
                  <path d="M22 19a2 2 0 01-2 2H4a2 2 0 01-2-2V5a2 2 0 012-2h5l2 3h9a2 2 0 012 2z"
                    stroke="currentColor" stroke-width="2"/>
                </svg>
              </button>
            </div>
          </div>
          <div class="form-tip">提供解说者音频样本，用于区分解说和电影对白</div>
        </el-form-item>

        <el-form-item label="TTS 参考音">
          <div class="file-select-row">
            <el-select v-model="createForm.tts_reference_audio_path" placeholder="可选，TTS 音色克隆"
              filterable clearable :loading="loadingAudio" class="file-select">
              <el-option v-for="file in audioFiles" :key="file.path" :label="file.name" :value="file.path">
                <div class="file-option">
                  <span class="file-name">{{ file.name }}</span>
                </div>
              </el-option>
            </el-select>
            <div class="file-actions">
              <button type="button" class="icon-btn" @click="refreshFiles('reference_audio')" title="刷新">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
                  <polyline points="23 4 23 10 17 10" stroke="currentColor" stroke-width="2"/>
                  <path d="M20.49 15a9 9 0 11-2.12-9.36L23 10" stroke="currentColor" stroke-width="2"/>
                </svg>
              </button>
            </div>
          </div>
          <div class="form-tip">用于生成新解说的音色参考（默认使用上方参考音频）</div>
        </el-form-item>

        <el-form-item label="SRT 字幕">
          <div class="file-select-row">
            <el-select v-model="createForm.subtitle_path" placeholder="可选，跳过语音识别"
              filterable clearable :loading="loadingSubtitles" class="file-select">
              <el-option v-for="file in subtitleFiles" :key="file.path" :label="file.name" :value="file.path">
                <div class="file-option">
                  <span class="file-name">{{ file.name }}</span>
                  <span class="file-meta">{{ file.size_display }}</span>
                </div>
              </el-option>
            </el-select>
            <div class="file-actions">
              <button type="button" class="icon-btn" @click="refreshFiles('subtitles')" title="刷新">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
                  <polyline points="23 4 23 10 17 10" stroke="currentColor" stroke-width="2"/>
                  <path d="M20.49 15a9 9 0 11-2.12-9.36L23 10" stroke="currentColor" stroke-width="2"/>
                </svg>
              </button>
              <button type="button" class="icon-btn" @click="openFolder('subtitles')" title="打开文件夹">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
                  <path d="M22 19a2 2 0 01-2 2H4a2 2 0 01-2-2V5a2 2 0 012-2h5l2 3h9a2 2 0 012 2z"
                    stroke="currentColor" stroke-width="2"/>
                </svg>
              </button>
            </div>
          </div>
          <div class="form-tip">导入字幕可跳过 Whisper 识别步骤，大幅加速处理</div>
        </el-form-item>
      </el-form>

      <template #footer>
        <el-button @click="showCreateDialog = false">取消</el-button>
        <el-button type="primary" :loading="creating" @click="handleCreate">创建项目</el-button>
      </template>
    </el-dialog>

  </div>
</template>

<script setup>
import { nextTick, onMounted, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { useProjectStore } from '@/stores/project'
import { projectApi, filesApi } from '@/api'

const router = useRouter()
const projectStore = useProjectStore()

const showCreateDialog = ref(false)
const creating = ref(false)
const createFormRef = ref(null)

const createForm = ref({
  name: '', movie_path: '', narration_path: '',
  reference_audio_path: '', tts_reference_audio_path: '', subtitle_path: ''
})

const EMPTY_CREATE_FORM = {
  name: '',
  movie_path: '',
  narration_path: '',
  reference_audio_path: '',
  tts_reference_audio_path: '',
  subtitle_path: ''
}

const movieFiles      = ref([])
const narrationFiles  = ref([])
const audioFiles      = ref([])
const subtitleFiles   = ref([])
const loadingMovies      = ref(false)
const loadingNarrations  = ref(false)
const loadingAudio       = ref(false)
const loadingSubtitles   = ref(false)

const createRules = {
  name:           [{ required: true, message: '请输入项目名称', trigger: 'blur' }],
  movie_path:     [{ required: true, message: '请选择原电影文件', trigger: 'change' }],
  narration_path: [{ required: true, message: '请选择解说视频文件', trigger: 'change' }],
}

onMounted(() => projectStore.fetchProjects())

function openCreate() {
  createForm.value = { ...EMPTY_CREATE_FORM }
  showCreateDialog.value = true
}

function firstPath(files) {
  return files.find(file => file?.path)?.path || ''
}

function normalizeOptionalPath(path) {
  const value = typeof path === 'string' ? path.trim() : ''
  return value || null
}

function buildCreatePayload() {
  return {
    name: createForm.value.name.trim(),
    movie_path: createForm.value.movie_path.trim(),
    narration_path: createForm.value.narration_path.trim(),
    reference_audio_path: normalizeOptionalPath(createForm.value.reference_audio_path),
    tts_reference_audio_path: normalizeOptionalPath(createForm.value.tts_reference_audio_path),
    subtitle_path: normalizeOptionalPath(createForm.value.subtitle_path)
  }
}

async function preloadCreateDialog() {
  await Promise.all(['movies', 'narrations', 'reference_audio', 'subtitles'].map(type => refreshFiles(type)))

  if (!createForm.value.movie_path) {
    createForm.value.movie_path = firstPath(movieFiles.value)
  }
  if (!createForm.value.narration_path) {
    createForm.value.narration_path = firstPath(narrationFiles.value)
  }

  await nextTick()
  createFormRef.value?.clearValidate?.()
}

function getStatusClass(status) {
  return {
    created: 'stripe-info',
    analyzing: 'stripe-warning',
    matching: 'stripe-warning',
    recognizing: 'stripe-warning',
    ready_for_polish: 'stripe-success',
    polishing: 'stripe-warning',
    ready_for_tts: 'stripe-success',
    generating_tts: 'stripe-warning',
    completed: 'stripe-accent',
    error: 'stripe-danger',
  }[status] || 'stripe-info'
}

function getStatusType(status) {
  return { created:'info', analyzing:'warning', matching:'warning', recognizing:'warning',
    ready_for_polish:'success', polishing:'warning', ready_for_tts:'success',
    generating_tts:'warning', completed:'success', error:'danger' }[status] || 'info'
}

function getStatusText(status) {
  return { created:'已创建', analyzing:'分析中', matching:'匹配中', recognizing:'识别中',
    ready_for_polish:'待润色', polishing:'润色中', ready_for_tts:'待生成 TTS',
    generating_tts:'生成 TTS 中', completed:'已完成', error:'错误' }[status] || status
}

function isProcessing(status) {
  return ['analyzing', 'matching', 'recognizing', 'polishing', 'generating_tts'].includes(status)
}

function formatDate(dateStr) {
  if (!dateStr) return ''
  return new Date(dateStr).toLocaleString('zh-CN', {
    month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit'
  })
}

function goToProject(project) {
  router.push(`/project/${project.id}`)
}

async function handleCreate() {
  const payload = buildCreatePayload()

  if (!payload.name) {
    ElMessage.error('请输入项目名称')
    return
  }
  if (!payload.movie_path) {
    ElMessage.error('请选择原电影文件')
    return
  }
  if (!payload.narration_path) {
    ElMessage.error('请选择解说视频文件')
    return
  }

  createForm.value = {
    ...createForm.value,
    ...payload,
    reference_audio_path: payload.reference_audio_path || '',
    tts_reference_audio_path: payload.tts_reference_audio_path || '',
    subtitle_path: payload.subtitle_path || ''
  }

  const valid = await createFormRef.value.validate().catch(() => false)
  if (!valid) return
  creating.value = true
  try {
    const project = await projectStore.createProject(payload)
    ElMessage.success('项目创建成功')
    showCreateDialog.value = false
    createForm.value = { ...EMPTY_CREATE_FORM }
    router.push(`/project/${project.id}`)
  } catch {} finally { creating.value = false }
}

async function duplicateProject(project) {
  try {
    await projectApi.duplicate(project.id)
    ElMessage.success('项目已复制')
    projectStore.fetchProjects()
  } catch {}
}

function confirmDelete(project) {
  ElMessageBox.confirm(`确定要删除"${project.name}"吗？此操作不可恢复。`, '删除确认', {
    confirmButtonText: '删除', cancelButtonText: '取消', type: 'warning'
  }).then(async () => {
    await projectStore.deleteProject(project.id)
    ElMessage.success('已删除')
  }).catch(() => {})
}

async function refreshFiles(type) {
  try {
    if (type === 'movies')          { loadingMovies.value     = true; movieFiles.value     = await filesApi.listMovies() }
    if (type === 'narrations')      { loadingNarrations.value = true; narrationFiles.value = await filesApi.listNarrations() }
    if (type === 'reference_audio') { loadingAudio.value      = true; audioFiles.value     = await filesApi.listReferenceAudio() }
    if (type === 'subtitles')       { loadingSubtitles.value  = true; subtitleFiles.value  = await filesApi.listSubtitles() }
  } catch {}
  finally { loadingMovies.value = loadingNarrations.value = loadingAudio.value = loadingSubtitles.value = false }
}

async function openFolder(type) {
  try { await filesApi.openFolder(type); ElMessage.success('已打开文件夹，放入文件后点击刷新') } catch {}
}

watch(showCreateDialog, async value => {
  if (value) {
    await preloadCreateDialog()
  }
})
</script>

<style lang="scss" scoped>
.home-page { max-width: 1100px; }

// Workflow card
.workflow-card {
  background: var(--bg-surface);
  border: 1px solid var(--border-faint);
  border-radius: 10px;
  margin-bottom: 24px;
  overflow: hidden;
}

.workflow-title {
  display: flex;
  align-items: center;
  gap: 7px;
  padding: 12px 16px;
  font-size: 11px;
  font-weight: 700;
  color: var(--text-muted);
  letter-spacing: .6px;
  text-transform: uppercase;
  border-bottom: 1px solid var(--border-faint);
  background: var(--bg-elevated);
}

.workflow-steps {
  display: flex;
  .step {
    flex: 1;
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 14px 16px;
    border-right: 1px solid var(--border-faint);
    &:last-child { border-right: none; }
    .step-num {
      width: 20px; height: 20px; border-radius: 50%;
      background: var(--accent-dim); color: var(--accent);
      font-size: 10px; font-weight: 700;
      display: flex; align-items: center; justify-content: center;
      flex-shrink: 0; margin-top: 1px;
    }
    .step-text { font-size: 12px; color: var(--text-secondary); line-height: 1.5; }
  }
}

// Skeleton
.skeleton-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 14px;
}
.skeleton-card { border-radius: 10px; overflow: hidden; }

// Project grid
.project-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 14px;
}

.project-card {
  background: var(--bg-surface);
  border: 1px solid var(--border-faint);
  border-radius: 10px;
  cursor: pointer;
  transition: border-color .15s, transform .15s, box-shadow .15s;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  position: relative;

  &:hover {
    border-color: var(--border-default);
    transform: translateY(-2px);
    box-shadow: var(--shadow-md);
    .card-actions { opacity: 1; }
  }
}

.card-stripe {
  height: 3px;
  flex-shrink: 0;
  &.stripe-info    { background: var(--text-muted); }
  &.stripe-warning { background: var(--warning); }
  &.stripe-success { background: var(--success); }
  &.stripe-accent  { background: linear-gradient(90deg, var(--accent), var(--purple)); }
  &.stripe-danger  { background: var(--danger); }
}

.card-body { padding: 14px 16px 12px; flex: 1; }

.card-top {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 10px;
  margin-bottom: 10px;
}

.project-name {
  font-size: 14px;
  font-weight: 600;
  color: var(--text-primary);
  line-height: 1.4;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  flex: 1;
  min-width: 0;
}

.card-meta {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
}

.meta-item {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 11px;
  color: var(--text-muted);
}

.mini-progress {
  margin-top: 10px;
  height: 2px;
  background: var(--bg-overlay);
  border-radius: 2px;
  overflow: hidden;
}

.mini-bar {
  height: 100%;
  width: 40%;
  background: var(--warning);
  border-radius: 2px;
  &.pulsing {
    animation: slide 1.5s ease-in-out infinite;
  }
}

@keyframes slide {
  0%   { transform: translateX(-100%); }
  100% { transform: translateX(300%); }
}

.card-actions {
  display: flex;
  gap: 6px;
  padding: 8px 12px;
  border-top: 1px solid var(--border-faint);
  opacity: 0;
  transition: opacity .15s;
  background: var(--bg-elevated);
}

.action-btn {
  width: 28px; height: 28px;
  border-radius: 6px;
  background: transparent;
  border: 1px solid var(--border-subtle);
  color: var(--text-secondary);
  cursor: pointer;
  display: flex; align-items: center; justify-content: center;
  transition: all .15s;
  &:hover { background: var(--bg-overlay); color: var(--text-primary); border-color: var(--border-default); }
  &.danger:hover { background: var(--danger-dim); color: var(--danger); border-color: rgba(239,68,68,.3); }
}

// File icon button
.file-actions {
  display: flex;
  gap: 4px;
  flex-shrink: 0;
}

.icon-btn {
  width: 32px; height: 32px;
  border-radius: 6px;
  background: var(--bg-elevated);
  border: 1px solid var(--border-subtle);
  color: var(--text-secondary);
  cursor: pointer;
  display: flex; align-items: center; justify-content: center;
  transition: all .15s;
  &:hover { background: var(--bg-overlay); color: var(--text-primary); }
}
</style>
