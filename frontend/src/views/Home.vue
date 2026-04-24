<template>
  <div class="home-page">
    <div class="page-header">
      <div>
        <h1 class="page-title">视频匹配项目</h1>
        <p class="page-subtitle">导入原电影和解说视频，自动匹配画面并导出剪映草稿。</p>
      </div>
      <el-button type="primary" size="large" @click="openCreate">
        新建匹配项目
      </el-button>
    </div>

    <div class="workflow-card">
      <div class="workflow-title">工作流程</div>
      <div class="workflow-steps">
        <div class="step"><span>1</span>选择原电影</div>
        <div class="step"><span>2</span>选择解说视频</div>
        <div class="step"><span>3</span>自动画面匹配</div>
        <div class="step"><span>4</span>导出剪映草稿或走 API</div>
      </div>
    </div>

    <div v-if="projectStore.loading" class="empty-state">加载项目中...</div>
    <div v-else-if="projectStore.projects.length === 0" class="empty-state">
      暂无项目，点击右上角新建匹配项目。
    </div>

    <div v-else class="project-grid">
      <div
        v-for="project in projectStore.projects"
        :key="project.id"
        class="project-card"
        @click="goToProject(project)"
      >
        <div class="card-stripe" :class="getStatusClass(project.status)"></div>
        <div class="card-body">
          <div class="card-top">
            <div class="project-name">{{ project.name }}</div>
            <el-tag :type="getStatusType(project.status)" size="small">
              {{ getStatusText(project.status) }}
            </el-tag>
          </div>
          <div class="card-meta">
            <span>{{ formatDate(project.updated_at) }}</span>
            <span v-if="project.segment_count">{{ project.segment_count }} 个片段</span>
          </div>
          <div v-if="isProcessing(project.status)" class="mini-progress">
            <div class="mini-bar"></div>
          </div>
        </div>
        <div class="card-actions" @click.stop>
          <button type="button" class="action-btn" title="复制项目" @click.stop="duplicateProject(project)">复制</button>
          <button type="button" class="action-btn danger" title="删除项目" @click.stop="confirmDelete(project)">删除</button>
        </div>
      </div>
    </div>

    <el-dialog
      v-model="showCreateDialog"
      title="新建视频匹配项目"
      width="620px"
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
          <el-input v-model="createForm.name" placeholder="例如：肖申克画面匹配" />
        </el-form-item>

        <el-form-item label="原电影" prop="movie_path">
          <div class="file-select-row">
            <el-select v-model="createForm.movie_path" placeholder="选择原电影文件" filterable :loading="loadingMovies" class="file-select">
              <el-option v-for="file in movieFiles" :key="file.path" :label="file.name" :value="file.path">
                <div class="file-option">
                  <span class="file-name">{{ file.name }}</span>
                  <span class="file-meta">{{ file.size_display }} · {{ file.modified_time }}</span>
                </div>
              </el-option>
            </el-select>
            <el-button @click="refreshFiles('movies')">刷新</el-button>
            <el-button @click="openFolder('movies')">目录</el-button>
          </div>
          <div class="form-tip">放入 videos/movies 目录。</div>
        </el-form-item>

        <el-form-item label="解说视频" prop="narration_path">
          <div class="file-select-row">
            <el-select v-model="createForm.narration_path" placeholder="选择解说视频" filterable :loading="loadingNarrations" class="file-select">
              <el-option v-for="file in narrationFiles" :key="file.path" :label="file.name" :value="file.path">
                <div class="file-option">
                  <span class="file-name">{{ file.name }}</span>
                  <span class="file-meta">{{ file.size_display }} · {{ file.modified_time }}</span>
                </div>
              </el-option>
            </el-select>
            <el-button @click="refreshFiles('narrations')">刷新</el-button>
            <el-button @click="openFolder('narrations')">目录</el-button>
          </div>
          <div class="form-tip">放入 videos/narrations 目录。</div>
        </el-form-item>

        <el-form-item label="SRT 字幕">
          <div class="file-select-row">
            <el-select v-model="createForm.subtitle_path" placeholder="可选，跳过语音识别提速" filterable clearable :loading="loadingSubtitles" class="file-select">
              <el-option v-for="file in subtitleFiles" :key="file.path" :label="file.name" :value="file.path">
                <div class="file-option">
                  <span class="file-name">{{ file.name }}</span>
                  <span class="file-meta">{{ file.size_display }}</span>
                </div>
              </el-option>
            </el-select>
            <el-button @click="refreshFiles('subtitles')">刷新</el-button>
            <el-button @click="openFolder('subtitles')">目录</el-button>
          </div>
          <div class="form-tip">有字幕就导入，没有字幕会自动识别解说音频。</div>
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
import { nextTick, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { filesApi, projectApi } from '@/api'
import { useProjectStore } from '@/stores/project'

const router = useRouter()
const projectStore = useProjectStore()

const showCreateDialog = ref(false)
const creating = ref(false)
const createFormRef = ref(null)
const createForm = ref({ name: '', movie_path: '', narration_path: '', subtitle_path: '' })

const movieFiles = ref([])
const narrationFiles = ref([])
const subtitleFiles = ref([])
const loadingMovies = ref(false)
const loadingNarrations = ref(false)
const loadingSubtitles = ref(false)

const createRules = {
  name: [{ required: true, message: '请输入项目名称', trigger: 'blur' }],
  movie_path: [{ required: true, message: '请选择原电影文件', trigger: 'change' }],
  narration_path: [{ required: true, message: '请选择解说视频文件', trigger: 'change' }],
}

onMounted(() => projectStore.fetchProjects())

async function openCreate() {
  createForm.value = { name: '', movie_path: '', narration_path: '', subtitle_path: '' }
  showCreateDialog.value = true
  await preloadCreateDialog()
}

async function preloadCreateDialog() {
  await Promise.all(['movies', 'narrations', 'subtitles'].map(type => refreshFiles(type)))
  createForm.value.movie_path ||= firstPath(movieFiles.value)
  createForm.value.narration_path ||= firstPath(narrationFiles.value)
  await nextTick()
  createFormRef.value?.clearValidate?.()
}

function firstPath(files) {
  return files.find(file => file?.path)?.path || ''
}

function optionalPath(path) {
  const value = typeof path === 'string' ? path.trim() : ''
  return value || null
}

async function handleCreate() {
  await createFormRef.value?.validate()
  creating.value = true
  try {
    const payload = {
      name: createForm.value.name.trim(),
      movie_path: createForm.value.movie_path.trim(),
      narration_path: createForm.value.narration_path.trim(),
      subtitle_path: optionalPath(createForm.value.subtitle_path),
    }
    const project = await projectStore.createProject(payload)
    showCreateDialog.value = false
    ElMessage.success('项目已创建')
    router.push(`/project/${project.id}`)
  } finally {
    creating.value = false
  }
}

async function refreshFiles(type) {
  if (type === 'movies') loadingMovies.value = true
  if (type === 'narrations') loadingNarrations.value = true
  if (type === 'subtitles') loadingSubtitles.value = true
  try {
    if (type === 'movies') movieFiles.value = await filesApi.listMovies()
    if (type === 'narrations') narrationFiles.value = await filesApi.listNarrations()
    if (type === 'subtitles') subtitleFiles.value = await filesApi.listSubtitles()
  } finally {
    if (type === 'movies') loadingMovies.value = false
    if (type === 'narrations') loadingNarrations.value = false
    if (type === 'subtitles') loadingSubtitles.value = false
  }
}

async function openFolder(type) {
  await filesApi.openFolder(type)
  ElMessage.success('已打开目录，放入文件后点击刷新')
}

function goToProject(project) {
  router.push(`/project/${project.id}`)
}

async function duplicateProject(project) {
  const copy = await projectApi.duplicate(project.id, `${project.name} 副本`)
  await projectStore.fetchProjects()
  router.push(`/project/${copy.id}`)
}

async function confirmDelete(project) {
  await ElMessageBox.confirm(`确定删除项目「${project.name}」？`, '删除项目', { type: 'warning' })
  await projectStore.deleteProject(project.id)
  ElMessage.success('项目已删除')
}

function getStatusClass(status) {
  return {
    created: 'stripe-info',
    analyzing: 'stripe-warning',
    matching: 'stripe-warning',
    recognizing: 'stripe-warning',
    ready_for_polish: 'stripe-success',
    ready_for_tts: 'stripe-success',
    completed: 'stripe-accent',
    error: 'stripe-danger',
  }[status] || 'stripe-info'
}

function getStatusType(status) {
  return {
    created: 'info',
    analyzing: 'warning',
    matching: 'warning',
    recognizing: 'warning',
    ready_for_polish: 'success',
    ready_for_tts: 'success',
    completed: 'success',
    error: 'danger',
  }[status] || 'info'
}

function getStatusText(status) {
  return {
    created: '已创建',
    analyzing: '分析中',
    matching: '匹配中',
    recognizing: '识别中',
    ready_for_polish: '已匹配',
    ready_for_tts: '已匹配',
    completed: '已完成',
    error: '错误',
  }[status] || status
}

function isProcessing(status) {
  return ['analyzing', 'matching', 'recognizing'].includes(status)
}

function formatDate(dateStr) {
  if (!dateStr) return ''
  return new Date(dateStr).toLocaleString('zh-CN', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  })
}
</script>

<style scoped lang="scss">
.home-page { max-width: 1160px; margin: 0 auto; }
.page-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 18px;
}
.page-title { margin: 0; font-size: 26px; color: var(--text-primary); }
.page-subtitle { margin: 8px 0 0; color: var(--text-secondary); }
.workflow-card {
  margin-bottom: 20px;
  padding: 18px;
  border: 1px solid var(--border);
  border-radius: 14px;
  background: linear-gradient(135deg, rgba(64, 224, 208, .08), rgba(76, 110, 245, .08));
}
.workflow-title { font-weight: 700; margin-bottom: 14px; color: var(--text-primary); }
.workflow-steps { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }
.step {
  padding: 12px;
  border-radius: 12px;
  background: rgba(255,255,255,.04);
  color: var(--text-secondary);
  font-size: 13px;
}
.step span {
  display: inline-flex;
  width: 22px;
  height: 22px;
  align-items: center;
  justify-content: center;
  margin-right: 8px;
  border-radius: 50%;
  background: var(--accent);
  color: white;
  font-weight: 700;
}
.empty-state {
  padding: 60px 20px;
  text-align: center;
  border: 1px dashed var(--border);
  border-radius: 14px;
  color: var(--text-muted);
}
.project-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 14px; }
.project-card {
  position: relative;
  overflow: hidden;
  border: 1px solid var(--border);
  border-radius: 14px;
  background: var(--bg-card);
  cursor: pointer;
}
.card-stripe { height: 3px; background: var(--accent); }
.stripe-warning { background: var(--warning); }
.stripe-success { background: var(--success); }
.stripe-danger { background: var(--danger); }
.stripe-accent { background: var(--accent); }
.card-body { padding: 16px; }
.card-top { display: flex; justify-content: space-between; gap: 10px; align-items: center; }
.project-name { font-weight: 700; color: var(--text-primary); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.card-meta { display: flex; gap: 12px; margin-top: 12px; color: var(--text-muted); font-size: 12px; }
.mini-progress { height: 3px; margin-top: 14px; background: var(--bg-hover); overflow: hidden; border-radius: 999px; }
.mini-bar { width: 45%; height: 100%; background: var(--accent); animation: pulse 1.2s infinite; }
.card-actions { display: flex; gap: 8px; padding: 0 16px 16px; }
.action-btn {
  border: 1px solid var(--border);
  border-radius: 8px;
  background: transparent;
  color: var(--text-secondary);
  padding: 5px 9px;
  cursor: pointer;
}
.action-btn.danger { color: var(--danger); }
.file-select-row { display: flex; width: 100%; gap: 8px; align-items: center; }
.file-select { flex: 1; }
.file-option { display: flex; justify-content: space-between; gap: 12px; }
.file-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.file-meta, .form-tip { color: var(--text-muted); font-size: 12px; }
.form-tip { margin-top: 6px; }
@keyframes pulse { 0% { transform: translateX(-100%); } 100% { transform: translateX(240%); } }
@media (max-width: 780px) {
  .page-header { align-items: flex-start; gap: 12px; flex-direction: column; }
  .workflow-steps { grid-template-columns: 1fr; }
  .file-select-row { align-items: stretch; flex-direction: column; }
}
</style>
