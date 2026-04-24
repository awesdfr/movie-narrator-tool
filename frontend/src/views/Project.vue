<template>
  <div class="project-page" v-loading="projectStore.loading">

    <div v-if="project">
      <!-- Header -->
      <div class="page-header">
        <div>
          <div class="project-header-name">{{ project.name }}</div>
          <div class="project-header-sub">
            <el-tag :type="getStatusType(project.status)" size="small">
              {{ getStatusText(project.status) }}
            </el-tag>
            <span class="header-version">匹配版本 {{ project.match_version || 'v2' }}</span>
          </div>
        </div>
      </div>

      <!-- Stats grid -->
      <div v-if="hasSegments" class="stats-grid">
        <div class="stat-card">
          <div class="stat-value">{{ segmentStats.total }}</div>
          <div class="stat-label">总片段</div>
        </div>
        <div class="stat-card success">
          <div class="stat-value">{{ segmentStats.matched }}</div>
          <div class="stat-label">已匹配</div>
        </div>
        <div class="stat-card accent">
          <div class="stat-value">{{ segmentStats.autoAccepted }}</div>
          <div class="stat-label">自动通过</div>
        </div>
        <div class="stat-card warning">
          <div class="stat-value">{{ segmentStats.reviewRequired }}</div>
          <div class="stat-label">待复核</div>
        </div>
        <div class="stat-card purple">
          <div class="stat-value">{{ percent(segmentStats.coverageRate) }}</div>
          <div class="stat-label">匹配覆盖率</div>
        </div>
        <div class="stat-card">
          <div class="stat-value">{{ percent(segmentStats.timelineCoverage) }}</div>
          <div class="stat-label">时间线覆盖率</div>
        </div>
        <div class="stat-card">
          <div class="stat-value">{{ percent(segmentStats.matchedAvgConfidence) }}</div>
          <div class="stat-label">已匹配段平均置信度</div>
        </div>
        <div class="stat-card">
          <div class="stat-value">{{ boundaryText }}</div>
          <div class="stat-label">平均边界误差</div>
        </div>
        <div class="stat-card" :class="hasBenchmark ? 'success' : ''">
          <div class="stat-value">{{ benchmarkText }}</div>
          <div class="stat-label">Benchmark</div>
        </div>
        <div class="stat-card" :class="hasVisualAudit ? 'success' : ''">
          <div class="stat-value">{{ visualAuditText }}</div>
          <div class="stat-label">画面审计</div>
        </div>
      </div>

      <!-- Main panels -->
      <div class="project-panels">

        <!-- Left: processing -->
        <div class="panel-main">
          <el-card>
            <template #header>
              <div class="card-header">
                <span>处理控制</span>
                <div v-if="projectStore.processing" class="processing-badge">
                  <span class="pulse-dot"></span> 处理中
                </div>
              </div>
            </template>

            <!-- Active processing view -->
            <div v-if="projectStore.processing" class="processing-view">
              <div class="processing-stage">{{ getStageText(projectStore.progress.stage) }}</div>
              <el-progress
                :percentage="projectStore.progress.progress"
                :stroke-width="12"
                :text-inside="true"
                style="margin-bottom:10px"
              />
              <div class="processing-message">{{ projectStore.progress.message }}</div>
              <el-button type="danger" size="small" style="margin-top:16px" @click="handleStop">
                停止处理
              </el-button>
            </div>

            <!-- Action buttons -->
            <div v-else>
              <div class="action-row">
                <el-button type="primary" :disabled="!canProcess" @click="handleStartProcess">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" style="margin-right:5px">
                    <polygon points="5 3 19 12 5 21 5 3" fill="currentColor"/>
                  </svg>
                  开始处理
                </el-button>
                <el-button :disabled="!hasSegments" @click="handleResegment">重切段</el-button>
                <el-button type="warning" :disabled="!hasSegments || weakSegmentCount === 0" @click="handleRematchWeak">
                  弱片段重查 {{ weakSegmentCount }}
                </el-button>
                <el-button type="warning" :disabled="!canPolish" @click="handleStartPolish">
                  ✦ 开始润色
                </el-button>
                <el-button type="success" :disabled="!canGenerateTTS" @click="handleBatchTTS">
                  ▶ 一键 TTS
                </el-button>
              </div>

              <div v-if="project.status === 'ready_for_polish'" class="status-banner info">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" style="flex-shrink:0">
                  <circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="2"/>
                  <path d="M12 16v-4M12 8h.01" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
                </svg>
                匹配完成 — 已匹配 {{ segmentStats.matched }}/{{ segmentStats.total }} 段，
                自动通过 {{ segmentStats.autoAccepted }} 段，待复核 {{ segmentStats.reviewRequired }} 段。
                匹配覆盖 {{ percent(segmentStats.coverageRate) }}，时间线覆盖 {{ percent(segmentStats.timelineCoverage) }}，
                已匹配段平均置信度 {{ percent(segmentStats.matchedAvgConfidence) }}，不等于整体准确率。
              </div>

              <div v-if="project.status === 'ready_for_tts'" class="status-banner success">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" style="flex-shrink:0">
                  <polyline points="20 6 9 17 4 12" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
                </svg>
                润色完成 — 匹配覆盖 {{ percent(segmentStats.coverageRate) }}，已匹配段平均置信度 {{ percent(segmentStats.matchedAvgConfidence) }}，平均边界误差 {{ boundaryText }}。
              </div>
            </div>
          </el-card>

          <!-- Benchmark notice -->
          <el-card class="benchmark-card" v-if="hasSegments">
            <template #header><span>Benchmark 评测</span></template>
            <div class="benchmark-actions">
              <el-button size="small" :loading="benchmarkRunning" @click="handleRunBenchmark">
                运行 Benchmark
              </el-button>
              <el-button
                size="small"
                text
                :disabled="!project?.benchmark_report_path"
                @click="handleOpenBenchmarkReport"
              >
                打开报告
              </el-button>
            </div>
            <div v-if="!hasBenchmark" class="benchmark-empty">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" style="color:var(--warning);flex-shrink:0">
                <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"
                  stroke="currentColor" stroke-width="2"/>
                <path d="M12 9v4M12 17h.01" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
              </svg>
              <div>
                <div class="benchmark-note-title">尚未绑定 benchmark</div>
                <div class="benchmark-note-desc">没有标注集时，高匹配率仅作为目标策略，不代表已达到对应准确率。</div>
              </div>
            </div>
            <div v-else class="benchmark-ok">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" style="color:var(--success);flex-shrink:0">
                <polyline points="20 6 9 17 4 12" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"/>
              </svg>
              <div>
                <div class="benchmark-note-title" style="color:var(--success)">Benchmark 评测：{{ benchmarkText }}</div>
                <div class="benchmark-note-desc">{{ benchmarkDescription }}</div>
              </div>
            </div>
            <div v-if="hasBenchmark" class="benchmark-metrics">
              <div>误匹配率 {{ percent(project?.benchmark_false_match_rate) }}</div>
              <div>低置信召回 {{ percent(project?.benchmark_low_confidence_recall) }}</div>
            </div>
          </el-card>

          <el-card class="readiness-card" v-if="hasSegments">
            <template #header>
              <div class="card-header">
                <span>商用就绪</span>
                <el-tag :type="readinessStatusType" size="small">{{ readinessStatusText }}</el-tag>
              </div>
            </template>
            <div class="benchmark-actions">
              <el-button size="small" :loading="auditRunning" @click="handleRunVisualAudit">
                运行画面审计
              </el-button>
              <el-button
                size="small"
                text
                :disabled="!project?.visual_audit_report_path"
                @click="handleOpenVisualAuditReport"
              >
                打开审计报告
              </el-button>
            </div>
            <div class="benchmark-note-desc readiness-summary">
              {{ visualAuditDescription }}
            </div>
            <div class="readiness-flags">
              <el-checkbox v-model="rightsConfirmed">已确认商业版权</el-checkbox>
              <el-checkbox v-model="platformRiskAcknowledged">已接受平台查重风险</el-checkbox>
              <el-button size="small" :loading="savingReadinessFlags" @click="handleSaveReadinessFlags">
                保存确认
              </el-button>
            </div>
            <div v-if="readiness?.blockers?.length" class="readiness-blockers">
              <div v-for="blocker in readiness.blockers" :key="blocker.code" class="readiness-blocker">
                <span class="readiness-blocker-code">{{ blocker.code }}</span>
                <span>{{ blocker.message }}</span>
              </div>
            </div>
            <div v-else class="benchmark-ok">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" style="color:var(--success);flex-shrink:0">
                <polyline points="20 6 9 17 4 12" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"/>
              </svg>
              <div>
                <div class="benchmark-note-title" style="color:var(--success)">当前项目没有商用阻断项</div>
                <div class="benchmark-note-desc">这只表示流程条件齐备，不代表平台一定放行，也不替代人工法务判断。</div>
              </div>
            </div>
          </el-card>
        </div>

        <!-- Right: info + quick actions -->
        <div class="panel-side">
          <el-card class="info-card">
            <template #header><span>项目信息</span></template>
            <div class="info-list">
              <div class="info-row">
                <span class="info-label">原电影</span>
                <span class="info-value path-value">{{ shortPath(project.movie_path) }}</span>
              </div>
              <div class="info-row">
                <span class="info-label">解说视频</span>
                <span class="info-value path-value">{{ shortPath(project.narration_path) }}</span>
              </div>
              <div class="info-row">
                <span class="info-label">电影时长</span>
                <span class="info-value">{{ formatDuration(project.movie_duration) }}</span>
              </div>
              <div class="info-row">
                <span class="info-label">解说时长</span>
                <span class="info-value">{{ formatDuration(project.narration_duration) }}</span>
              </div>
              <div class="info-row">
                <span class="info-label">字幕遮罩</span>
                <span class="info-value">{{ subtitleMaskModeText }}</span>
              </div>
              <div class="info-row">
                <span class="info-label">字幕框</span>
                <span class="info-value">{{ subtitleRegionSummary }}</span>
              </div>
            </div>
          </el-card>

          <!-- Quick navigation -->
          <el-card class="quicknav-card">
            <template #header><span>快速导航</span></template>
            <div class="quicknav-grid">
              <router-link :to="`/project/${project.id}/editor`"
                class="quicknav-btn" :class="{ disabled: !hasSegments }">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                  <path d="M11 4H4a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2v-7"
                    stroke="currentColor" stroke-width="2"/>
                  <path d="M18.5 2.5a2.121 2.121 0 013 3L12 15l-4 1 1-4 9.5-9.5z"
                    stroke="currentColor" stroke-width="2"/>
                </svg>
                片段编辑
              </router-link>

              <router-link :to="`/project/${project.id}/timeline`"
                class="quicknav-btn" :class="{ disabled: !hasSegments }">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                  <line x1="3" y1="6" x2="21" y2="6" stroke="currentColor" stroke-width="2"/>
                  <line x1="3" y1="12" x2="21" y2="12" stroke="currentColor" stroke-width="2"/>
                  <line x1="3" y1="18" x2="21" y2="18" stroke="currentColor" stroke-width="2"/>
                </svg>
                时间轴
              </router-link>

              <router-link :to="`/project/${project.id}/preview`"
                class="quicknav-btn" :class="{ disabled: !hasSegments }">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                  <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"
                    stroke="currentColor" stroke-width="2"/>
                  <circle cx="12" cy="12" r="3" stroke="currentColor" stroke-width="2"/>
                </svg>
                预览对应
              </router-link>

              <button class="quicknav-btn" @click="subtitleDialogVisible = true">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                  <rect x="3" y="3" width="18" height="18" rx="2" stroke="currentColor" stroke-width="2"/>
                  <line x1="8" y1="12" x2="16" y2="12" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
                  <line x1="8" y1="8" x2="12" y2="8" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
                </svg>
                框选字幕区
              </button>
            </div>

            <div v-if="hasSegments" class="creative-section">
              <div class="export-label">创作配置</div>
              <div class="creative-config-grid">
                <el-select v-model="exportMode" size="small">
                  <el-option label="创作草稿" value="creative_draft" />
                  <el-option label="复原草稿" value="restore_draft" />
                </el-select>
                <el-select v-model="creativeTemplate" size="small">
                  <el-option label="剧情拆解" value="story_mix" />
                  <el-option label="高能切点" value="highlight_punch" />
                  <el-option label="说明型" value="explain_focus" />
                </el-select>
              </div>
              <div class="creative-actions">
                <el-button size="small" :loading="savingCreativeConfig" @click="handleSaveCreativeConfig">
                  保存创作配置
                </el-button>
                <span v-if="creativePlan" class="creative-summary-text">
                  {{ creativePlan.segment_count }} 段 / {{ creativePlan.unit_count }} 个镜头单元
                </span>
              </div>
              <div v-loading="creativePlanLoading" class="creative-preview">
                <template v-if="creativePlan">
                  <div class="creative-badges">
                    <span class="plan-badge exact">Exact {{ creativeCounts.exact }}</span>
                    <span class="plan-badge inferred">Inferred {{ creativeCounts.inferred }}</span>
                    <span class="plan-badge fallback">Fallback {{ creativeCounts.fallback }}</span>
                  </div>
                  <div class="creative-list">
                    <div v-for="item in creativeSummaryRows" :key="item.segment_id" class="creative-row">
                      <div class="creative-row-head">
                        <span class="creative-row-index">#{{ (item.index ?? 0) + 1 }}</span>
                        <span class="plan-tag" :class="item.match_type">{{ item.match_type }}</span>
                      </div>
                      <div class="creative-row-text">{{ item.summary_text || '未生成摘要' }}</div>
                      <div class="creative-row-meta">
                        {{ item.units.length }} 单元
                        <span v-if="item.notes?.length"> · {{ item.notes.join(' / ') }}</span>
                      </div>
                    </div>
                  </div>
                </template>
                <div v-else class="creative-empty">暂无创作规划，先完成匹配或重切片段。</div>
              </div>
            </div>

            <div class="export-section">
              <div class="export-label">导出</div>
              <div class="export-row">
                <button class="export-btn" :disabled="!hasSegments" @click="handleExportJianying">
                  剪映草稿
                </button>
                <button class="export-btn" :disabled="!hasSegments" @click="handleExportSubtitle">
                  SRT 字幕
                </button>
                <button class="export-btn" :disabled="segmentStats.matched === 0" @click="handleExportDaVinci">
                  DaVinci XML
                </button>
                <button class="export-btn" :disabled="!hasSegments" @click="handleExportMaterialBasket">
                  素材篮
                </button>
                <button class="export-btn" :disabled="!hasSegments" @click="handleOpenMatchReport">
                  匹配报告
                </button>
              </div>
            </div>
          </el-card>
        </div>
      </div>
    </div>

    <SubtitleRegionDialog
      v-if="project"
      v-model="subtitleDialogVisible"
      :project="project"
    />
  </div>
</template>

<script setup>
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import { useRoute } from 'vue-router'
import { ElMessage } from 'element-plus'
import { previewApi } from '@/api'
import SubtitleRegionDialog from '@/components/SubtitleRegionDialog.vue'
import { useProjectStore } from '@/stores/project'

const route = useRoute()
const projectStore = useProjectStore()
const project = computed(() => projectStore.currentProject)
const subtitleDialogVisible = ref(false)
const creativePlan = ref(null)
const creativePlanLoading = ref(false)
const savingCreativeConfig = ref(false)
const benchmarkRunning = ref(false)
const auditRunning = ref(false)
const savingReadinessFlags = ref(false)
const readiness = ref(null)
const rightsConfirmed = ref(false)
const platformRiskAcknowledged = ref(false)
const exportMode = ref('restore_draft')
const creativeTemplate = ref('story_mix')

const hasSegments    = computed(() => (project.value?.segments?.length || 0) > 0)
const hasBenchmark   = computed(() => project.value?.benchmark_accuracy != null)
const hasVisualAudit = computed(() => project.value?.visual_audit_score != null)
const canProcess     = computed(() => project.value && ['created','error','ready_for_polish','ready_for_tts','completed'].includes(project.value.status))
const canPolish      = computed(() => project.value?.status === 'ready_for_polish' && hasSegments.value)
const canGenerateTTS = computed(() => ['ready_for_tts','completed'].includes(project.value?.status) && hasSegments.value)
const weakSegmentCount = computed(() => {
  const segs = project.value?.segments || []
  return segs.filter(segment => {
    if (!segment.use_segment || segment.skip_matching || segment.segment_type === 'non_movie' || segment.is_manual_match) return false
    if (segment.movie_start == null || segment.movie_end == null) return true
    if (segment.review_required || ['needs_review','unmatched'].includes(segment.alignment_status)) return true
    if (['inferred','fallback'].includes(segment.match_type)) return true
    if (Number(segment.match_confidence || 0) < 0.78) return true
    const visual = Number(segment.visual_confidence || 0)
    return visual > 0 && visual < 0.70
  }).length
})

const segmentStats = computed(() => {
  const segs = project.value?.segments || []
  const matched   = segs.filter(s => s.movie_start != null)
  const autoAcc   = segs.filter(s => s.alignment_status === 'auto_accepted').length
  const needReview = segs.filter(s => s.review_required).length
  const skipped   = segs.filter(s => s.skip_matching).length
  const errors    = matched.filter(s => s.estimated_boundary_error != null).map(s => s.estimated_boundary_error)
  const narrationDuration = Number(project.value?.narration_duration || 0)
  const lastSegmentEnd = segs.reduce((maxEnd, seg) => Math.max(maxEnd, Number(seg.narration_end || 0)), 0)
  return {
    total: segs.length, matched: matched.length, autoAccepted: autoAcc,
    reviewRequired: needReview, skipped,
    coverageRate: segs.length ? matched.length / segs.length : 0,
    timelineCoverage: narrationDuration > 0 ? Math.min(1, lastSegmentEnd / narrationDuration) : 0,
    matchedAvgConfidence: matched.length ? matched.reduce((a,s) => a+(s.match_confidence||0),0)/matched.length : 0,
    avgBoundaryError: errors.length ? errors.reduce((a,v)=>a+v,0)/errors.length : null,
  }
})

const boundaryText      = computed(() => segmentStats.value.avgBoundaryError == null ? '--' : `${segmentStats.value.avgBoundaryError.toFixed(1)}s`)
const benchmarkText     = computed(() => project.value?.benchmark_accuracy == null ? '未评测' : percent(project.value.benchmark_accuracy))
const visualAuditText   = computed(() => project.value?.visual_audit_score == null ? '未审计' : percent(project.value.visual_audit_score))
const benchmarkDescription = computed(() => {
  if (!project.value?.benchmark_manifest) return '已写入 benchmark 分数，建议固定当前清单，后续调参基于同一样本集。'
  return `评测基于 ${project.value.benchmark_manifest}。`
})
const visualAuditDescription = computed(() => {
  if (!project.value?.visual_audit_report_path) return '尚未运行导出画面审计，当前只知道匹配过程分数，不知道导出后真实观感。'
  const below = project.value?.visual_audit_below_threshold ?? 0
  const threshold = project.value?.visual_audit_threshold ?? 0.66
  const metric = project.value?.visual_audit_metric || 'identity'
  return `导出审计 ${visualAuditText.value}，低于阈值 ${threshold.toFixed(2)} 的采样点 ${below} 个，指标 ${metric}。`
})
const readinessStatusText = computed(() => ({
  ready: '可商用',
  conditional: '有条件',
  blocked: '阻断'
}[readiness.value?.status] || '未评估'))
const readinessStatusType = computed(() => ({
  ready: 'success',
  conditional: 'warning',
  blocked: 'danger'
}[readiness.value?.status] || 'info'))
const subtitleMaskModeText = computed(() => ({
  hybrid:'自动+手动', manual_only:'仅手动', auto_only:'仅自动'
}[project.value?.subtitle_mask_mode] || '自动+手动'))
const subtitleRegionSummary = computed(() => {
  const n = project.value?.narration_subtitle_regions?.length || 0
  const m = project.value?.movie_subtitle_regions?.length || 0
  return `解说 ${n} 个 / 电影 ${m} 个`
})

const creativeCounts = computed(() => creativePlan.value?.counts || { exact: 0, inferred: 0, fallback: 0 })
const creativeSummaryRows = computed(() => (creativePlan.value?.segments || []).slice(0, 6))

watch(project, value => {
  if (!value) return
  exportMode.value = 'restore_draft'
  creativeTemplate.value = value.creative_template || 'story_mix'
  rightsConfirmed.value = !!value.rights_confirmed
  platformRiskAcknowledged.value = !!value.platform_risk_acknowledged
})

onMounted(async () => {
  await projectStore.fetchProject(route.params.id)
  await loadReadiness()
  if (projectStore.currentProject?.segments?.length) {
    await loadCreativePlan()
  }
})
onUnmounted(() => projectStore.clearCurrentProject())

function percent(v) { return `${((v||0)*100).toFixed(0)}%` }
function shortPath(p) { if (!p) return '--'; const parts = p.replace(/\\/g,'/').split('/'); return parts.slice(-2).join('/') }
function formatDuration(s) {
  if (!s) return '--'
  const h = Math.floor(s/3600), m = Math.floor((s%3600)/60), sec = Math.floor(s%60)
  return h > 0 ? `${h}:${String(m).padStart(2,'0')}:${String(sec).padStart(2,'0')}` : `${m}:${String(sec).padStart(2,'0')}`
}

function getStatusType(status) {
  return {created:'info',analyzing:'warning',matching:'warning',recognizing:'warning',
    ready_for_polish:'success',polishing:'warning',ready_for_tts:'success',
    generating_tts:'warning',completed:'success',error:'danger'}[status]||'info'
}
function getStatusText(status) {
  return {created:'已创建',analyzing:'分析中',matching:'匹配中',recognizing:'识别中',
    ready_for_polish:'待润色',polishing:'润色中',ready_for_tts:'待生成 TTS',
    generating_tts:'生成 TTS 中',completed:'已完成',error:'错误'}[status]||status
}
function getStageText(stage) {
  return {starting:'启动中',analyzing:'视频分析',recognizing:'语音与分段',matching:'视频匹配',
    ready_for_polish:'匹配完成',polishing:'文案润色',ready_for_tts:'等待生成 TTS',
    generating_tts:'TTS 生成',completed:'处理完成',error:'处理失败'}[stage]||stage
}

async function handleStartProcess() { await projectStore.startProcessing(route.params.id) }
async function handleStop() { await projectStore.stopProcessing(route.params.id); ElMessage.info('处理已停止') }
async function handleResegment() {
  const result = await projectStore.resegment(route.params.id)
  ElMessage.success(`已重切段，共 ${result.segments?.length||0} 段`)
}
async function handleRematchWeak() {
  await projectStore.rematchWeakProject(route.params.id, {
    preserve_manual_matches: true,
    confidence_threshold: 0.78,
    visual_threshold: 0.70,
    include_inferred: true,
    include_review_required: true
  })
}
async function handleStartPolish() { await projectStore.startPolishing(route.params.id, 'movie_pro') }
async function handleBatchTTS()    { await projectStore.batchGenerateTTS(route.params.id) }
async function handleExportJianying() {
  const result = await previewApi.exportJianying(route.params.id, exportMode.value)
  await projectStore.fetchProject(route.params.id)
  await loadReadiness()
  ElMessage.success(`导出成功: ${result.draft_path}`)
}
async function handleExportSubtitle() {
  await previewApi.exportSubtitle(route.params.id, 'srt')
  ElMessage.success('字幕导出成功')
}
async function handleExportMaterialBasket() {
  const result = await previewApi.exportMaterialBasket(route.params.id)
  ElMessage.success(`素材篮导出成功: ${result.basket_path}`)
}
async function handleSaveCreativeConfig() {
  savingCreativeConfig.value = true
  try {
    await projectStore.updateProject(route.params.id, {
      default_export_mode: exportMode.value,
      creative_template: creativeTemplate.value
    })
    await loadCreativePlan()
    ElMessage.success('创作配置已保存')
  } finally {
    savingCreativeConfig.value = false
  }
}
async function loadCreativePlan() {
  creativePlanLoading.value = true
  try {
    creativePlan.value = await previewApi.getCreativePlan(route.params.id)
  } finally {
    creativePlanLoading.value = false
  }
}
async function loadReadiness() {
  readiness.value = await previewApi.getCommercialReadiness(route.params.id)
}
async function handleRunBenchmark() {
  const current = project.value?.benchmark_manifest || ''
  const input = window.prompt('输入 benchmark manifest JSON 路径，留空则使用上次路径。', current)
  if (input === null) return
  benchmarkRunning.value = true
  try {
    const result = await previewApi.evaluateBenchmark(route.params.id, input.trim() || null)
    await projectStore.fetchProject(route.params.id)
    await loadReadiness()
    ElMessage.success(`Benchmark 完成: ${percent(result.metrics?.accuracy || 0)}`)
  } finally {
    benchmarkRunning.value = false
  }
}
async function handleRunVisualAudit() {
  const current = project.value?.last_jianying_draft_path || ''
  const input = window.prompt('输入剪映草稿目录或 draft_content.json，留空则使用最近一次导出。', current)
  if (input === null) return
  auditRunning.value = true
  try {
    const result = await previewApi.runVisualAudit(route.params.id, { draft_path: input.trim() || null })
    await projectStore.fetchProject(route.params.id)
    await loadReadiness()
    ElMessage.success(`画面审计完成: ${percent(result.summary?.score_average || 0)}`)
  } finally {
    auditRunning.value = false
  }
}
async function handleSaveReadinessFlags() {
  savingReadinessFlags.value = true
  try {
    await projectStore.updateProject(route.params.id, {
      rights_confirmed: rightsConfirmed.value,
      platform_risk_acknowledged: platformRiskAcknowledged.value
    })
    await projectStore.fetchProject(route.params.id)
    await loadReadiness()
    ElMessage.success('商用确认已保存')
  } finally {
    savingReadinessFlags.value = false
  }
}
function handleOpenMatchReport() { window.open(previewApi.getMatchReportUrl(route.params.id),'_blank','noopener') }
function handleExportDaVinci()   { window.open(previewApi.getDaVinciXmlUrl(route.params.id),'_blank','noopener') }
function handleOpenBenchmarkReport() { window.open(previewApi.getBenchmarkReportUrl(route.params.id),'_blank','noopener') }
function handleOpenVisualAuditReport() { window.open(previewApi.getVisualAuditReportUrl(route.params.id),'_blank','noopener') }
</script>

<style lang="scss" scoped>
.project-page { max-width: 1200px; }

.project-header-name {
  font-size: 22px;
  font-weight: 700;
  color: var(--text-primary);
  margin-bottom: 8px;
}
.project-header-sub {
  display: flex;
  align-items: center;
  gap: 10px;
}
.header-version {
  font-size: 11px;
  color: var(--text-muted);
  background: var(--bg-elevated);
  padding: 2px 8px;
  border-radius: 10px;
  border: 1px solid var(--border-faint);
}

// Stats
.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(130px, 1fr));
  gap: 12px;
  margin-bottom: 24px;
}

// Panels
.project-panels {
  display: grid;
  grid-template-columns: 1fr 360px;
  gap: 20px;
  align-items: start;
}
.panel-main { display: flex; flex-direction: column; gap: 16px; }
.panel-side  { display: flex; flex-direction: column; gap: 16px; }

// Processing
.processing-badge {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  color: var(--warning);
}
.pulse-dot {
  width: 7px; height: 7px;
  background: var(--warning);
  border-radius: 50%;
  animation: pulse 1.2s ease-in-out infinite;
}
@keyframes pulse {
  0%, 100% { opacity: 1; transform: scale(1); }
  50%       { opacity: .5; transform: scale(.7); }
}

.processing-view {
  padding: 8px 0;
  .processing-stage { font-size: 15px; font-weight: 600; margin-bottom: 14px; }
  .processing-message { font-size: 12px; color: var(--text-secondary); }
}

.status-banner {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  margin-top: 14px;
  padding: 10px 14px;
  border-radius: 8px;
  font-size: 13px;
  line-height: 1.5;
  &.info    { background: var(--info-dim);    color: var(--info);    border: 1px solid rgba(56,189,248,.2); }
  &.success { background: var(--success-dim); color: var(--success); border: 1px solid rgba(34,197,94,.2); }
}

// Info list
.info-list { display: flex; flex-direction: column; gap: 0; }
.info-row {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 12px;
  padding: 8px 0;
  border-bottom: 1px solid var(--border-faint);
  &:last-child { border-bottom: none; }
}
.info-label { font-size: 12px; color: var(--text-muted); flex-shrink: 0; }
.info-value { font-size: 13px; color: var(--text-primary); text-align: right; word-break: break-all; }
.path-value { font-size: 11px; color: var(--text-secondary); font-family: monospace; }

// Benchmark
.benchmark-card {}
.benchmark-empty, .benchmark-ok {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  padding: 4px 0;
}
.benchmark-actions {
  display: flex;
  gap: 8px;
  margin-bottom: 12px;
}
.benchmark-metrics {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  margin-top: 10px;
  font-size: 12px;
  color: var(--text-secondary);
}
.benchmark-note-title { font-size: 13px; font-weight: 600; color: var(--text-primary); margin-bottom: 4px; }
.benchmark-note-desc  { font-size: 12px; color: var(--text-secondary); line-height: 1.5; }
.readiness-summary { margin-bottom: 10px; }
.readiness-flags {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 10px;
  margin-bottom: 14px;
}
.readiness-blockers {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.readiness-blocker {
  display: flex;
  gap: 10px;
  padding: 10px 12px;
  border-radius: 8px;
  background: rgba(239, 68, 68, .08);
  border: 1px solid rgba(239, 68, 68, .16);
  color: var(--text-primary);
  font-size: 12px;
  line-height: 1.5;
}
.readiness-blocker-code {
  font-family: monospace;
  color: var(--danger);
  flex-shrink: 0;
}

// Quick nav
.quicknav-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
  margin-bottom: 16px;
}

.quicknav-btn {
  display: flex;
  align-items: center;
  gap: 7px;
  padding: 9px 12px;
  background: var(--bg-elevated);
  border: 1px solid var(--border-subtle);
  border-radius: 8px;
  color: var(--text-secondary);
  font-size: 12.5px;
  font-weight: 500;
  cursor: pointer;
  text-decoration: none;
  transition: all .15s;
  &:hover:not(.disabled) { background: var(--bg-hover); color: var(--text-primary); border-color: var(--border-default); }
  &.disabled { opacity: .4; pointer-events: none; cursor: default; }
}

.export-section {
  border-top: 1px solid var(--border-faint);
  padding-top: 14px;
}
.export-label {
  font-size: 11px;
  font-weight: 700;
  color: var(--text-muted);
  letter-spacing: .6px;
  text-transform: uppercase;
  margin-bottom: 8px;
}
.export-row {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}
.export-btn {
  padding: 5px 10px;
  background: var(--bg-elevated);
  border: 1px solid var(--border-subtle);
  border-radius: 6px;
  color: var(--text-secondary);
  font-size: 12px;
  cursor: pointer;
  transition: all .15s;
  &:hover:not(:disabled) { background: var(--bg-overlay); color: var(--text-primary); }
  &:disabled { opacity: .35; cursor: default; }
}

.creative-section {
  border-top: 1px solid var(--border-faint);
  margin-top: 14px;
  padding-top: 14px;
}
.creative-config-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: 8px;
}
.creative-actions {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  margin-top: 10px;
}
.creative-summary-text {
  font-size: 12px;
  color: var(--text-muted);
}
.creative-preview {
  margin-top: 10px;
  padding: 10px;
  border-radius: 8px;
  background: var(--bg-elevated);
  border: 1px solid var(--border-faint);
  min-height: 120px;
}
.creative-badges {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  margin-bottom: 10px;
}
.plan-badge,
.plan-tag {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 2px 8px;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 700;
  text-transform: uppercase;
}
.plan-badge.exact,
.plan-tag.exact {
  background: rgba(34, 197, 94, .14);
  color: #3ba96f;
}
.plan-badge.inferred,
.plan-tag.inferred {
  background: rgba(245, 158, 11, .14);
  color: #b7791f;
}
.plan-badge.fallback,
.plan-tag.fallback {
  background: rgba(239, 68, 68, .14);
  color: #c24141;
}
.creative-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.creative-row {
  padding: 8px 10px;
  border-radius: 8px;
  background: var(--card-bg);
  border: 1px solid var(--border-faint);
}
.creative-row-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 8px;
  margin-bottom: 6px;
}
.creative-row-index {
  font-size: 12px;
  font-weight: 700;
  color: var(--text-secondary);
}
.creative-row-text {
  font-size: 13px;
  color: var(--text-primary);
  line-height: 1.4;
}
.creative-row-meta,
.creative-empty {
  margin-top: 4px;
  font-size: 12px;
  color: var(--text-muted);
}
</style>
