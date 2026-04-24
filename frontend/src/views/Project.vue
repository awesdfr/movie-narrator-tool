<template>
  <div class="project-page" v-loading="projectStore.loading">
    <div v-if="project">
      <div class="page-header">
        <div>
          <div class="project-title">{{ project.name }}</div>
          <div class="project-subtitle">
            <el-tag :type="getStatusType(project.status)" size="small">{{ getStatusText(project.status) }}</el-tag>
            <span>匹配版本 {{ project.match_version || 'v2_alignment_pipeline' }}</span>
          </div>
        </div>
      </div>

      <div v-if="hasSegments" class="stats-grid">
        <div class="stat-card"><div class="stat-value">{{ segmentStats.total }}</div><div class="stat-label">总片段</div></div>
        <div class="stat-card success"><div class="stat-value">{{ segmentStats.matched }}</div><div class="stat-label">已匹配</div></div>
        <div class="stat-card accent"><div class="stat-value">{{ segmentStats.autoAccepted }}</div><div class="stat-label">自动通过</div></div>
        <div class="stat-card warning"><div class="stat-value">{{ segmentStats.reviewRequired }}</div><div class="stat-label">待复核</div></div>
        <div class="stat-card"><div class="stat-value">{{ percent(segmentStats.coverageRate) }}</div><div class="stat-label">匹配覆盖率</div></div>
        <div class="stat-card"><div class="stat-value">{{ percent(segmentStats.timelineCoverage) }}</div><div class="stat-label">时间线覆盖率</div></div>
        <div class="stat-card"><div class="stat-value">{{ percent(segmentStats.matchedAvgConfidence) }}</div><div class="stat-label">平均置信度</div></div>
        <div class="stat-card" :class="hasVisualAudit ? 'success' : ''"><div class="stat-value">{{ visualAuditText }}</div><div class="stat-label">导出审计</div></div>
      </div>

      <div class="project-grid">
        <el-card class="main-card">
          <template #header>
            <div class="card-header">
              <span>视频匹配控制</span>
              <span v-if="projectStore.processing" class="processing-badge"><i></i> 处理中</span>
            </div>
          </template>

          <div v-if="projectStore.processing" class="processing-view">
            <div class="processing-stage">{{ getStageText(projectStore.progress.stage) }}</div>
            <el-progress :percentage="projectStore.progress.progress" :stroke-width="12" :text-inside="true" />
            <div class="processing-message">{{ projectStore.progress.message }}</div>
            <el-button type="danger" size="small" @click="handleStop">停止处理</el-button>
          </div>

          <template v-else>
            <div class="action-row">
              <el-button type="primary" :disabled="!canProcess" @click="handleStartProcess">开始匹配</el-button>
              <el-button :disabled="!hasSegments" @click="handleResegment">重切段</el-button>
              <el-button type="warning" :disabled="!hasSegments || weakSegmentCount === 0" @click="handleRematchWeak">
                弱片段重查 {{ weakSegmentCount }}
              </el-button>
              <el-button :disabled="!hasSegments" @click="subtitleDialogVisible = true">框选字幕区</el-button>
              <el-button type="success" :disabled="segmentStats.matched === 0" :loading="exporting" @click="handleExportJianying">
                导出剪映草稿
              </el-button>
            </div>

            <div v-if="hasSegments" class="status-banner">
              已匹配 {{ segmentStats.matched }}/{{ segmentStats.total }} 段，自动通过 {{ segmentStats.autoAccepted }} 段，
              待复核 {{ segmentStats.reviewRequired }} 段，平均置信度 {{ percent(segmentStats.matchedAvgConfidence) }}。
            </div>
            <div v-else class="status-banner muted">
              当前项目还没有片段。点击“开始匹配”后会自动识别/切段并匹配到原电影画面。
            </div>
          </template>
        </el-card>

        <el-card class="side-card">
          <template #header><span>项目与导出</span></template>
          <div class="info-row"><span>原电影</span><b>{{ shortPath(project.movie_path) }}</b></div>
          <div class="info-row"><span>解说视频</span><b>{{ shortPath(project.narration_path) }}</b></div>
          <div class="info-row"><span>电影时长</span><b>{{ formatDuration(project.movie_duration) }}</b></div>
          <div class="info-row"><span>解说时长</span><b>{{ formatDuration(project.narration_duration) }}</b></div>
          <div class="info-row"><span>字幕遮罩</span><b>{{ subtitleMaskModeText }}</b></div>
          <div class="info-row"><span>剪映草稿</span><b>{{ shortPath(project.last_jianying_draft_path) }}</b></div>

          <div class="side-actions">
            <el-button :disabled="!project.last_jianying_draft_path" :loading="auditRunning" @click="handleRunVisualAudit">
              运行导出审计
            </el-button>
            <el-button :disabled="!project.visual_audit_report_path" @click="handleOpenVisualAuditReport">
              审计报告
            </el-button>
            <el-button :disabled="!hasSegments" @click="handleOpenMatchReport">
              匹配报告
            </el-button>
          </div>
        </el-card>
      </div>

      <el-card v-if="hasSegments" class="segments-card">
        <template #header>
          <div class="card-header">
            <span>匹配片段</span>
            <span class="hint">只保留视频匹配需要看的字段</span>
          </div>
        </template>
        <el-table :data="segmentRows" height="440" size="small">
          <el-table-column type="index" width="56" />
          <el-table-column label="解说时间" min-width="150">
            <template #default="{ row }">{{ formatTime(row.narration_start) }} - {{ formatTime(row.narration_end) }}</template>
          </el-table-column>
          <el-table-column label="电影时间" min-width="150">
            <template #default="{ row }">{{ row.movie_start == null ? '--' : `${formatTime(row.movie_start)} - ${formatTime(row.movie_end)}` }}</template>
          </el-table-column>
          <el-table-column label="置信度" width="100">
            <template #default="{ row }">{{ percent(row.match_confidence || 0) }}</template>
          </el-table-column>
          <el-table-column label="状态" width="120">
            <template #default="{ row }">
              <el-tag size="small" :type="row.review_required ? 'warning' : 'success'">
                {{ row.review_required ? '待复核' : '通过' }}
              </el-tag>
            </template>
          </el-table-column>
          <el-table-column label="类型" width="110">
            <template #default="{ row }">{{ row.match_type || row.alignment_status || '--' }}</template>
          </el-table-column>
          <el-table-column label="操作" width="120" fixed="right">
            <template #default="{ row }">
              <el-button link type="primary" :loading="rematchingId === row.id" @click="handleRematchSegment(row)">
                重查
              </el-button>
            </template>
          </el-table-column>
        </el-table>
      </el-card>
    </div>

    <SubtitleRegionDialog
      v-if="project"
      v-model="subtitleDialogVisible"
      :project="project"
    />
  </div>
</template>

<script setup>
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { useRoute } from 'vue-router'
import { ElMessage } from 'element-plus'
import { previewApi } from '@/api'
import SubtitleRegionDialog from '@/components/SubtitleRegionDialog.vue'
import { useProjectStore } from '@/stores/project'

const route = useRoute()
const projectStore = useProjectStore()
const project = computed(() => projectStore.currentProject)

const subtitleDialogVisible = ref(false)
const exporting = ref(false)
const auditRunning = ref(false)
const rematchingId = ref('')

const hasSegments = computed(() => (project.value?.segments?.length || 0) > 0)
const hasVisualAudit = computed(() => project.value?.visual_audit_score != null)
const canProcess = computed(() => project.value && ['created', 'error', 'completed', 'ready_for_polish', 'ready_for_tts'].includes(project.value.status))
const segmentRows = computed(() => project.value?.segments || [])

const weakSegmentCount = computed(() => {
  return segmentRows.value.filter(segment => {
    if (!segment.use_segment || segment.skip_matching || segment.segment_type === 'non_movie' || segment.is_manual_match) return false
    if (segment.movie_start == null || segment.movie_end == null) return true
    if (segment.review_required || ['needs_review', 'unmatched'].includes(segment.alignment_status)) return true
    if (['inferred', 'fallback'].includes(segment.match_type)) return true
    return Number(segment.match_confidence || 0) < 0.78
  }).length
})

const segmentStats = computed(() => {
  const segs = segmentRows.value
  const matched = segs.filter(segment => segment.movie_start != null && segment.movie_end != null)
  const timelineEnd = segs.reduce((maxEnd, segment) => Math.max(maxEnd, Number(segment.narration_end || 0)), 0)
  const narrationDuration = Number(project.value?.narration_duration || 0)
  return {
    total: segs.length,
    matched: matched.length,
    autoAccepted: segs.filter(segment => segment.alignment_status === 'auto_accepted').length,
    reviewRequired: segs.filter(segment => segment.review_required).length,
    coverageRate: segs.length ? matched.length / segs.length : 0,
    timelineCoverage: narrationDuration > 0 ? Math.min(1, timelineEnd / narrationDuration) : 0,
    matchedAvgConfidence: matched.length ? matched.reduce((sum, segment) => sum + Number(segment.match_confidence || 0), 0) / matched.length : 0,
  }
})

const visualAuditText = computed(() => project.value?.visual_audit_score == null ? '未审计' : percent(project.value.visual_audit_score))
const subtitleMaskModeText = computed(() => ({
  hybrid: '自动+手动',
  manual_only: '仅手动',
  auto_only: '仅自动',
}[project.value?.subtitle_mask_mode] || '自动+手动'))

onMounted(() => projectStore.fetchProject(route.params.id))
onUnmounted(() => projectStore.clearCurrentProject())

function percent(value) {
  return `${((Number(value) || 0) * 100).toFixed(0)}%`
}

function shortPath(path) {
  if (!path) return '--'
  return path.replace(/\\/g, '/').split('/').slice(-3).join('/')
}

function formatDuration(seconds) {
  if (!seconds) return '--'
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  const s = Math.floor(seconds % 60)
  return h > 0 ? `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}` : `${m}:${String(s).padStart(2, '0')}`
}

function formatTime(seconds) {
  return formatDuration(Number(seconds || 0))
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

function getStageText(stage) {
  return {
    starting: '启动中',
    analyzing: '视频分析',
    recognizing: '语音与分段',
    matching: '视频匹配',
    completed: '匹配完成',
    error: '处理失败',
  }[stage] || stage
}

async function handleStartProcess() {
  await projectStore.startProcessing(route.params.id)
}

async function handleStop() {
  await projectStore.stopProcessing(route.params.id)
  ElMessage.info('处理已停止')
}

async function handleResegment() {
  await projectStore.resegment(route.params.id, { preserve_manual_matches: true })
  ElMessage.success('已重新切段')
}

async function handleRematchWeak() {
  await projectStore.rematchWeakProject(route.params.id, { preserve_manual_matches: true })
}

async function handleRematchSegment(segment) {
  rematchingId.value = segment.id
  try {
    await projectStore.rematchSegment(route.params.id, segment.id, {})
    ElMessage.success('片段已重查')
  } finally {
    rematchingId.value = ''
  }
}

async function handleExportJianying() {
  exporting.value = true
  try {
    const result = await previewApi.exportJianying(route.params.id, 'restore_draft')
    await projectStore.fetchProject(route.params.id)
    ElMessage.success(`剪映草稿已导出: ${result.draft_path}`)
  } finally {
    exporting.value = false
  }
}

async function handleRunVisualAudit() {
  auditRunning.value = true
  try {
    const result = await previewApi.runVisualAudit(route.params.id, {})
    await projectStore.fetchProject(route.params.id)
    ElMessage.success(`审计完成: ${percent(result.summary?.score_average || 0)}`)
  } finally {
    auditRunning.value = false
  }
}

function handleOpenMatchReport() {
  window.open(previewApi.getMatchReportUrl(route.params.id), '_blank', 'noopener')
}

function handleOpenVisualAuditReport() {
  window.open(previewApi.getVisualAuditReportUrl(route.params.id), '_blank', 'noopener')
}
</script>

<style scoped lang="scss">
.project-page { max-width: 1220px; margin: 0 auto; }
.page-header { margin-bottom: 18px; }
.project-title { font-size: 28px; font-weight: 800; color: var(--text-primary); }
.project-subtitle { display: flex; align-items: center; gap: 10px; margin-top: 8px; color: var(--text-muted); }
.stats-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
  margin-bottom: 16px;
}
.stat-card {
  padding: 16px;
  border: 1px solid var(--border);
  border-radius: 13px;
  background: var(--bg-card);
}
.stat-card.success { border-color: rgba(36, 186, 99, .45); }
.stat-card.accent { border-color: rgba(91, 108, 248, .45); }
.stat-card.warning { border-color: rgba(245, 158, 11, .45); }
.stat-value { font-size: 25px; font-weight: 800; color: var(--text-primary); }
.stat-label { margin-top: 6px; color: var(--text-muted); font-size: 12px; }
.project-grid { display: grid; grid-template-columns: minmax(0, 1fr) 360px; gap: 16px; margin-bottom: 16px; }
.card-header { display: flex; justify-content: space-between; align-items: center; gap: 12px; }
.processing-badge { color: var(--warning); font-size: 13px; }
.processing-badge i {
  display: inline-block;
  width: 8px;
  height: 8px;
  margin-right: 6px;
  border-radius: 50%;
  background: currentColor;
}
.processing-view { display: flex; flex-direction: column; gap: 14px; }
.processing-stage { font-weight: 700; color: var(--text-primary); }
.processing-message { color: var(--text-secondary); font-size: 13px; }
.action-row { display: flex; flex-wrap: wrap; gap: 10px; }
.status-banner {
  margin-top: 14px;
  padding: 12px 14px;
  border-radius: 10px;
  background: rgba(64, 224, 208, .08);
  color: var(--text-secondary);
  line-height: 1.6;
}
.status-banner.muted { background: var(--bg-hover); color: var(--text-muted); }
.info-row {
  display: grid;
  grid-template-columns: 86px minmax(0, 1fr);
  gap: 10px;
  padding: 10px 0;
  border-bottom: 1px solid var(--border-faint);
  font-size: 13px;
}
.info-row span { color: var(--text-muted); }
.info-row b { color: var(--text-primary); font-weight: 500; word-break: break-all; }
.side-actions { display: grid; grid-template-columns: 1fr; gap: 10px; margin-top: 14px; }
.segments-card { margin-bottom: 24px; }
.hint { color: var(--text-muted); font-size: 12px; font-weight: 400; }
@media (max-width: 980px) {
  .stats-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .project-grid { grid-template-columns: 1fr; }
}
@media (max-width: 620px) {
  .stats-grid { grid-template-columns: 1fr; }
}
</style>
