<template>
  <div class="editor-page" v-loading="projectStore.loading">

    <!-- Toolbar -->
    <div class="editor-toolbar">
      <div class="toolbar-left">
        <div class="view-tabs">
          <button class="view-tab" :class="{ active: viewMode === 'segments' }" @click="viewMode = 'segments'">片段视图</button>
          <button class="view-tab" :class="{ active: viewMode === 'script' }" @click="viewMode = 'script'">完整文案</button>
        </div>
      </div>
      <div class="toolbar-right">
        <el-button size="small" type="primary" :loading="projectStore.processing" @click="handleRematchProject">
          重匹配未跳过片段
        </el-button>
      </div>
    </div>

    <!-- Script view -->
    <div v-if="viewMode === 'script'" class="script-view">
      <div class="script-header">
        <div class="script-stats">
          共 <strong>{{ scriptSegments.length }}</strong> 段 &nbsp;·&nbsp; 待复核 <strong class="warn-text">{{ reviewCount }}</strong> 段
        </div>
        <div class="script-actions">
          <el-button size="small" @click="handleResegment">重切段</el-button>
          <el-button size="small" type="success" :loading="projectStore.processing" @click="handleBatchTTS">一键TTS</el-button>
          <el-button size="small" type="primary" :loading="savingScript" @click="saveAllScript">保存全部</el-button>
        </div>
      </div>

      <div class="script-list">
        <div v-for="segment in scriptSegments" :key="segment.id" class="script-item">
          <div class="script-item-head">
            <span class="si-index">#{{ segment.index + 1 }}</span>
            <span class="si-tag" :class="getStatusClass(segment)">{{ getAlignmentText(segment.alignment_status) }}</span>
            <span class="si-time">{{ formatTime(segment.narration_start) }} – {{ formatTime(segment.narration_end) }}</span>
            <span class="si-conf">{{ percent(segment.match_confidence) }}</span>
          </div>
          <div class="script-item-body">
            <div class="script-field">
              <div class="field-label">原文</div>
              <el-input v-model="scriptEdits[segment.id].original_text" type="textarea" :autosize="{ minRows: 1, maxRows: 4 }" />
            </div>
            <div class="script-field">
              <div class="field-label">润色</div>
              <el-input v-model="scriptEdits[segment.id].polished_text" type="textarea" :autosize="{ minRows: 1, maxRows: 4 }" />
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Segments view -->
    <template v-else>
      <!-- Filter bar -->
      <div class="filter-bar">
        <div class="filter-chips">
          <button class="filter-chip" :class="{ active: filter === 'all' }" @click="filter = 'all'">全部 ({{ segments.length }})</button>
          <button class="filter-chip warn" :class="{ active: filter === 'review' }" @click="filter = 'review'">待复核 ({{ reviewCount }})</button>
          <button class="filter-chip" :class="{ active: filter === 'skipped' }" @click="filter = 'skipped'">已跳过 ({{ skippedCount }})</button>
          <button class="filter-chip danger" :class="{ active: filter === 'audio_risk' }" @click="filter = 'audio_risk'">音频风险 ({{ audioRiskCount }})</button>
          <button class="filter-chip" :class="{ active: filter === 'has_narration' }" @click="filter = 'has_narration'">有解说</button>
          <button class="filter-chip" :class="{ active: filter === 'no_narration' }" @click="filter = 'no_narration'">无解说</button>
          <button class="filter-chip" :class="{ active: filter === 'non_movie' }" @click="filter = 'non_movie'">非电影</button>
        </div>
      </div>

      <!-- Batch toolbar -->
      <div v-if="selectedIds.length" class="batch-bar">
        <span class="batch-count">已选 {{ selectedIds.length }} 段</span>
        <button class="batch-btn" @click="batchSetUse(true)">启用</button>
        <button class="batch-btn" @click="batchSetUse(false)">禁用</button>
        <button class="batch-btn" @click="batchSkipMatching(true)">跳过匹配</button>
        <button class="batch-btn" @click="batchSkipMatching(false)">恢复匹配</button>
        <button class="batch-btn" @click="batchSetPolished(true)">用润色</button>
        <button class="batch-btn" @click="batchSetPolished(false)">用原文</button>
        <button class="batch-btn muted" @click="clearSelection">清空</button>
      </div>

      <!-- Segment cards -->
      <div class="segments-list">
        <div
          v-for="segment in filteredSegments"
          :key="segment.id"
          class="seg-card"
          :class="{
            selected: selectedIds.includes(segment.id),
            review: segment.review_required,
            skipped: segment.skip_matching
          }"
        >
          <!-- Card header -->
          <div class="seg-head">
            <div class="seg-title-row">
              <el-checkbox :model-value="selectedIds.includes(segment.id)" @change="toggleSelect(segment.id)" />
              <span class="seg-idx">#{{ segment.index + 1 }}</span>
              <span class="seg-tag" :class="getStatusClass(segment)">{{ getAlignmentText(segment.alignment_status) }}</span>
              <span v-if="segment.segment_type === 'no_narration'" class="seg-tag info">无解说</span>
              <span v-if="segment.audio_activity_label && segment.audio_activity_label !== 'unknown'" class="seg-tag info">
                音频 {{ activityLabel(segment.audio_activity_label) }}
              </span>
              <span v-if="(segment.stability_score || 0) < 0.55" class="seg-tag warn">低稳定</span>
              <span v-if="segment.skip_matching" class="seg-tag muted">已跳过</span>
              <span v-if="segment.review_required" class="seg-tag warn">待复核</span>
            </div>
            <div class="seg-conf">置信度 <strong>{{ percent(segment.match_confidence) }}</strong></div>
          </div>

          <!-- Card body -->
          <div class="seg-body">
            <!-- Thumbnail + times -->
            <div class="seg-visual">
              <img class="seg-thumb" :src="getThumbnail(segment.id)" alt="thumb" />
              <div class="seg-times">
                <div class="time-row"><span class="tl">解说</span><span>{{ formatTime(segment.narration_start) }} – {{ formatTime(segment.narration_end) }}</span></div>
                <div class="time-row"><span class="tl">电影</span><span>{{ formatTime(segment.movie_start) }} – {{ formatTime(segment.movie_end) }}</span></div>
                <div class="time-row"><span class="tl">误差</span><span>{{ segment.estimated_boundary_error != null ? `${segment.estimated_boundary_error.toFixed(1)}s` : '--' }}</span></div>
              </div>
            </div>

            <!-- Text + metrics -->
            <div class="seg-info">
              <div class="seg-text-pair">
                <div class="text-block">
                  <div class="text-label">原文</div>
                  <div class="text-body">{{ segment.original_text || '(无文案)' }}</div>
                </div>
                <div class="text-block">
                  <div class="text-label">润色</div>
                  <div class="text-body">{{ segment.polished_text || '(未润色)' }}</div>
                </div>
              </div>
              <div class="seg-metrics">
                <span>visual {{ percent(segment.visual_confidence) }}</span>
                <span>audio {{ percent(segment.audio_confidence) }}</span>
                <span>temporal {{ percent(segment.temporal_confidence) }}</span>
                <span>stability {{ percent(segment.stability_score) }}</span>
                <span>speech {{ percent(segment.speech_likelihood) }}</span>
                <span>gap {{ segment.duration_gap?.toFixed(1) || '0.0' }}s</span>
              </div>
              <div class="seg-reason">{{ segment.match_reason || '暂无匹配说明' }}</div>
            </div>
          </div>

          <!-- Options -->
          <div class="seg-options">
            <label class="opt-check"><el-checkbox v-model="segment.use_segment" @change="updateSegment(segment.id, { use_segment: $event })" />使用此段</label>
            <label class="opt-check"><el-checkbox v-model="segment.skip_matching" @change="toggleSkipMatching(segment, $event)" />跳过匹配</label>
            <label class="opt-check"><el-checkbox v-model="segment.use_polished_text" @change="updateSegment(segment.id, { use_polished_text: $event })" />使用润色</label>
            <label class="opt-check"><el-checkbox v-model="segment.keep_bgm" @change="updateSegment(segment.id, { keep_bgm: $event })" />保留BGM</label>
            <label class="opt-check"><el-checkbox v-model="segment.keep_movie_audio" @change="updateSegment(segment.id, { keep_movie_audio: $event })" />保留原声</label>
            <label class="opt-check"><el-checkbox v-model="segment.mute_movie_audio" @change="updateSegment(segment.id, { mute_movie_audio: $event })" />电影静音</label>
          </div>

          <!-- Candidates -->
          <div v-if="segment.match_candidates?.length && !segment.skip_matching" class="candidates-section">
            <div class="cand-header">
              <span>候选匹配 ({{ segment.match_candidates.length }})</span>
              <el-button size="small" @click="handleRematch(segment)">重匹配</el-button>
            </div>
            <div class="cand-list">
              <div
                v-for="candidate in segment.match_candidates"
                :key="candidate.id"
                class="cand-item"
                :class="{ active: candidate.id === segment.selected_candidate_id }"
              >
                <div class="cand-info">
                  <div class="cand-row">
                    <strong>#{{ candidate.rank }}</strong>
                    <span>{{ formatTime(candidate.start) }} – {{ formatTime(candidate.end) }}</span>
                    <span class="cand-conf">{{ percent(candidate.confidence) }}</span>
                  </div>
                  <div class="cand-metrics">
                    <span>v {{ percent(candidate.visual_confidence) }}</span>
                    <span>a {{ percent(candidate.audio_confidence) }}</span>
                    <span>t {{ percent(candidate.temporal_confidence) }}</span>
                    <span>s {{ percent(candidate.stability_score) }}</span>
                  </div>
                  <div class="cand-reason">{{ candidate.reason }}</div>
                </div>
                <el-button
                  size="small"
                  :type="candidate.id === segment.selected_candidate_id ? 'success' : ''"
                  @click="applyCandidate(segment, candidate)"
                >
                  {{ candidate.id === segment.selected_candidate_id ? '当前' : '采用' }}
                </el-button>
              </div>
            </div>
          </div>

          <!-- Actions -->
          <div class="seg-actions">
            <el-button size="small" @click="showEditDialog(segment)">编辑文案</el-button>
            <el-button size="small" @click="previewTTS(segment)">试听TTS</el-button>
            <el-button size="small" @click="repolish(segment)">重润色</el-button>
            <el-button size="small" :disabled="segment.skip_matching" @click="handleRematch(segment)">重匹配</el-button>
            <el-button size="small" @click="$router.push(`/project/${projectId}/preview`)">预览</el-button>
          </div>
        </div>
      </div>
    </template>

    <!-- Edit dialog -->
    <el-dialog v-model="editDialogVisible" title="编辑文案" width="600px">
      <el-form v-if="editingSegment" label-width="48px">
        <el-form-item label="原文">
          <el-input v-model="editForm.original_text" type="textarea" :rows="4" />
        </el-form-item>
        <el-form-item label="润色">
          <el-input v-model="editForm.polished_text" type="textarea" :rows="4" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="editDialogVisible = false">取消</el-button>
        <el-button type="primary" @click="saveEdit">保存</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { computed, onMounted, ref, watch } from 'vue'
import { useRoute } from 'vue-router'
import { ElMessage } from 'element-plus'
import { previewApi, processApi } from '@/api'
import { useProjectStore } from '@/stores/project'

const route = useRoute()
const projectStore = useProjectStore()

const projectId = computed(() => route.params.id)
const segments = computed(() => projectStore.segments)
const viewMode = ref('segments')
const filter = ref('all')
const selectedIds = ref([])
const savingScript = ref(false)
const editDialogVisible = ref(false)
const editingSegment = ref(null)
const editForm = ref({ original_text: '', polished_text: '' })
const scriptEdits = ref({})

const reviewCount = computed(() => segments.value.filter(segment => segment.review_required).length)
const skippedCount = computed(() => segments.value.filter(segment => segment.skip_matching).length)
const audioRiskCount = computed(() => segments.value.filter(isAudioRiskSegment).length)
const filteredSegments = computed(() => {
  if (filter.value === 'review') return segments.value.filter(segment => segment.review_required)
  if (filter.value === 'skipped') return segments.value.filter(segment => segment.skip_matching)
  if (filter.value === 'audio_risk') return segments.value.filter(isAudioRiskSegment)
  if (filter.value === 'all') return segments.value
  return segments.value.filter(segment => segment.segment_type === filter.value)
})
const scriptSegments = computed(() => segments.value.filter(segment => segment.original_text || segment.polished_text))

watch(scriptSegments, value => {
  const next = {}
  value.forEach(segment => {
    next[segment.id] = scriptEdits.value[segment.id] || {
      original_text: segment.original_text || '',
      polished_text: segment.polished_text || ''
    }
  })
  scriptEdits.value = next
}, { immediate: true })

onMounted(() => {
  if (!projectStore.currentProject || projectStore.currentProject.id !== projectId.value) {
    projectStore.fetchProject(projectId.value)
  }
})

function percent(value) {
  return `${((value || 0) * 100).toFixed(0)}%`
}

function formatTime(seconds) {
  if (seconds == null) return '--:--'
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`
}

function getThumbnail(segmentId) {
  return previewApi.getThumbnail(projectId.value, segmentId)
}

function getAlignmentText(status) {
  return {
    auto_accepted: '自动通过',
    needs_review: '待复核',
    unmatched: '未匹配',
    skipped: '已跳过',
    non_movie: '非电影',
    manual: '手动',
    rematched: '重匹配',
    pending: '待处理'
  }[status] || status
}

function getStatusClass(segment) {
  if (segment.alignment_status === 'auto_accepted') return 'success'
  if (segment.alignment_status === 'needs_review') return 'warn'
  if (segment.alignment_status === 'non_movie') return 'danger'
  return 'muted'
}

function activityLabel(value) {
  return { active: '活跃', weak: '较弱', silent: '静音' }[value] || value
}

function toggleSelect(id) {
  const index = selectedIds.value.indexOf(id)
  if (index > -1) selectedIds.value.splice(index, 1)
  else selectedIds.value.push(id)
}

function clearSelection() {
  selectedIds.value = []
}

function isAudioRiskSegment(segment) {
  return (
    segment.segment_type === 'no_narration' ||
    segment.audio_activity_label === 'silent' ||
    segment.audio_activity_label === 'weak' ||
    (segment.speech_likelihood || 0) < 0.25
  )
}

async function updateSegment(segmentId, data) {
  await projectStore.updateSegment(projectId.value, segmentId, data)
}

async function batchSetUse(value) {
  await projectStore.batchUpdateSegments(projectId.value, selectedIds.value, { use_segment: value })
  ElMessage.success('批量更新成功')
}

async function batchSkipMatching(value) {
  await projectStore.batchUpdateSegments(projectId.value, selectedIds.value, { skip_matching: value })
  ElMessage.success(value ? '已将选中片段设为跳过匹配' : '已恢复选中片段参与匹配')
}

async function batchSetPolished(value) {
  await projectStore.batchUpdateSegments(projectId.value, selectedIds.value, { use_polished_text: value })
  ElMessage.success('批量更新成功')
}

async function toggleSkipMatching(segment, value) {
  const updated = await projectStore.updateSegment(projectId.value, segment.id, { skip_matching: value })
  Object.assign(segment, updated)
  ElMessage.success(value ? '该片段已跳过匹配' : '该片段已恢复为可匹配')
}

function showEditDialog(segment) {
  editingSegment.value = segment
  editForm.value = { original_text: segment.original_text, polished_text: segment.polished_text }
  editDialogVisible.value = true
}

async function saveEdit() {
  await updateSegment(editingSegment.value.id, editForm.value)
  editDialogVisible.value = false
  ElMessage.success('保存成功')
}

async function applyCandidate(segment, candidate) {
  await updateSegment(segment.id, {
    movie_start: candidate.start,
    movie_end: candidate.end,
    match_confidence: candidate.confidence,
    visual_confidence: candidate.visual_confidence,
    audio_confidence: candidate.audio_confidence,
    temporal_confidence: candidate.temporal_confidence,
    stability_score: candidate.stability_score,
    duration_gap: candidate.duration_gap,
    match_reason: candidate.reason,
    alignment_status: 'manual',
    review_required: false,
    selected_candidate_id: candidate.id,
    is_manual_match: true
  })
  ElMessage.success('已切换候选')
}

async function handleRematch(segment) {
  if (segment.skip_matching) {
    ElMessage.warning('该片段已标记为跳过匹配，请先取消勾选后再重匹配')
    return
  }
  await projectStore.rematchSegment(projectId.value, segment.id)
  ElMessage.success('重匹配完成')
}

async function handleRematchProject() {
  await projectStore.rematchProject(projectId.value, { preserve_manual_matches: true })
}

async function previewTTS(segment) {
  const audio = new Audio(previewApi.getSegmentAudio(projectId.value, segment.id, 'tts'))
  audio.onerror = () => ElMessage.error('TTS 音频不存在或加载失败')
  audio.play().catch(() => ElMessage.error('播放失败'))
}

async function saveAllScript() {
  savingScript.value = true
  try {
    const tasks = scriptSegments.value.map(segment => {
      const edits = scriptEdits.value[segment.id]
      if (edits.original_text !== segment.original_text || edits.polished_text !== segment.polished_text) {
        return updateSegment(segment.id, edits)
      }
      return Promise.resolve()
    })
    await Promise.all(tasks)
    ElMessage.success('文案已保存')
  } finally {
    savingScript.value = false
  }
}

async function handleResegment() {
  const result = await projectStore.resegment(projectId.value)
  ElMessage.success(`已重切段，共 ${result.segments?.length || 0} 段`)
}

async function handleBatchTTS() {
  await saveAllScript()
  await projectStore.batchGenerateTTS(projectId.value)
}

async function repolish(segment) {
  const result = await processApi.repolish(projectId.value, segment.id, { style_preset: 'movie_pro' })
  segment.polished_text = result.polished_text
  ElMessage.success('润色完成')
}
</script>

<style lang="scss" scoped>
.editor-page {
  max-width: 1400px;
  margin: 0 auto;
}

// ── Toolbar ───────────────────────────────────────────────────────────────────
.editor-toolbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.view-tabs {
  display: flex;
  background: var(--bg-elevated);
  border-radius: 8px;
  padding: 3px;
  gap: 2px;
  border: 1px solid var(--border-faint);
}

.view-tab {
  padding: 5px 16px;
  border-radius: 6px;
  font-size: 13px;
  font-weight: 500;
  color: var(--text-secondary);
  background: transparent;
  border: none;
  cursor: pointer;
  transition: all .15s;
  &.active {
    background: var(--accent-dim);
    color: var(--accent);
  }
  &:hover:not(.active) { color: var(--text-primary); }
}

// ── Filter bar ────────────────────────────────────────────────────────────────
.filter-bar {
  margin-bottom: 12px;
}

.filter-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.filter-chip {
  padding: 4px 12px;
  border-radius: 20px;
  font-size: 12px;
  font-weight: 500;
  border: 1px solid var(--border-faint);
  background: var(--bg-elevated);
  color: var(--text-secondary);
  cursor: pointer;
  transition: all .15s;
  &.active {
    background: var(--accent-dim);
    color: var(--accent);
    border-color: rgba(91,108,248,.4);
  }
  &.warn.active { background: rgba(240,160,32,.12); color: #f0a020; border-color: rgba(240,160,32,.3); }
  &.danger.active { background: rgba(245,108,108,.12); color: #f56c6c; border-color: rgba(245,108,108,.3); }
  &:hover:not(.active) { color: var(--text-primary); background: var(--bg-hover); }
}

// ── Batch bar ─────────────────────────────────────────────────────────────────
.batch-bar {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
  padding: 10px 14px;
  background: var(--bg-elevated);
  border: 1px solid var(--border-faint);
  border-radius: 8px;
  margin-bottom: 12px;
}

.batch-count {
  font-size: 12px;
  font-weight: 600;
  color: var(--accent);
  margin-right: 4px;
}

.batch-btn {
  padding: 4px 10px;
  font-size: 12px;
  border-radius: 5px;
  border: 1px solid var(--border-faint);
  background: var(--bg-surface);
  color: var(--text-secondary);
  cursor: pointer;
  transition: all .15s;
  &:hover { color: var(--text-primary); background: var(--bg-hover); }
  &.muted { color: var(--text-muted); }
}

// ── Script view ───────────────────────────────────────────────────────────────
.script-view { display: flex; flex-direction: column; gap: 0; }

.script-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 16px;
  background: var(--bg-elevated);
  border: 1px solid var(--border-faint);
  border-radius: 8px 8px 0 0;
  border-bottom: none;
}

.script-stats { font-size: 13px; color: var(--text-secondary); }
.warn-text { color: #f0a020; }
.script-actions { display: flex; gap: 8px; }

.script-list {
  border: 1px solid var(--border-faint);
  border-radius: 0 0 8px 8px;
  overflow: hidden;
}

.script-item {
  padding: 14px 16px;
  border-bottom: 1px solid var(--border-faint);
  &:last-child { border-bottom: none; }
}

.script-item-head {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 10px;
  flex-wrap: wrap;
}

.si-index { font-size: 13px; font-weight: 600; color: var(--text-primary); }
.si-time, .si-conf { font-size: 11px; color: var(--text-muted); }

.script-item-body { display: flex; flex-direction: column; gap: 8px; }

.script-field { display: flex; flex-direction: column; gap: 4px; }

.field-label {
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: .5px;
  color: var(--text-muted);
}

// ── Segment cards ─────────────────────────────────────────────────────────────
.segments-list { display: flex; flex-direction: column; gap: 12px; }

.seg-card {
  background: var(--card-bg);
  border: 1px solid var(--border-faint);
  border-radius: 10px;
  padding: 14px 16px;
  transition: border-color .15s;

  &.review { border-color: rgba(240,160,32,.4); }
  &.selected { border-color: rgba(91,108,248,.5); background: rgba(91,108,248,.04); }
  &.skipped { opacity: .65; }
}

.seg-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 12px;
}

.seg-title-row {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
}

.seg-idx { font-size: 13px; font-weight: 700; color: var(--text-primary); }

.seg-tag {
  font-size: 11px;
  font-weight: 600;
  padding: 2px 8px;
  border-radius: 4px;
  &.success { background: rgba(103,194,58,.15); color: #67c23a; }
  &.warn { background: rgba(240,160,32,.15); color: #f0a020; }
  &.danger { background: rgba(245,108,108,.15); color: #f56c6c; }
  &.info { background: var(--bg-elevated); color: var(--text-secondary); }
  &.muted { background: var(--bg-elevated); color: var(--text-muted); }
}

.seg-conf { font-size: 12px; color: var(--text-muted); }

.seg-body {
  display: flex;
  gap: 16px;
  margin-bottom: 12px;
}

.seg-visual {
  width: 200px;
  flex-shrink: 0;
}

.seg-thumb {
  width: 100%;
  aspect-ratio: 16/9;
  border-radius: 6px;
  background: #000;
  object-fit: cover;
  display: block;
  margin-bottom: 8px;
}

.seg-times { display: flex; flex-direction: column; gap: 4px; }

.time-row {
  display: flex;
  gap: 6px;
  font-size: 11px;
  color: var(--text-secondary);
}

.tl {
  font-weight: 600;
  color: var(--text-muted);
  width: 28px;
  flex-shrink: 0;
}

.seg-info {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 10px;
  min-width: 0;
}

.seg-text-pair { display: flex; flex-direction: column; gap: 6px; }

.text-block {
  background: var(--bg-elevated);
  border-radius: 6px;
  padding: 8px 10px;
}

.text-label {
  font-size: 10px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: .5px;
  color: var(--text-muted);
  margin-bottom: 4px;
}

.text-body {
  font-size: 13px;
  color: var(--text-secondary);
  line-height: 1.6;
}

.seg-metrics {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  font-size: 11px;
  color: var(--text-muted);
  font-variant-numeric: tabular-nums;
}

.seg-reason {
  font-size: 11px;
  color: var(--text-muted);
  line-height: 1.5;
  font-style: italic;
}

// ── Options row ───────────────────────────────────────────────────────────────
.seg-options {
  display: flex;
  flex-wrap: wrap;
  gap: 14px;
  padding: 10px 0;
  border-top: 1px solid var(--border-faint);
}

.opt-check {
  display: flex;
  align-items: center;
  gap: 5px;
  font-size: 12px;
  color: var(--text-secondary);
  cursor: pointer;
}

// ── Candidates ────────────────────────────────────────────────────────────────
.candidates-section {
  border-top: 1px dashed var(--border-faint);
  padding-top: 12px;
  margin-top: 4px;
}

.cand-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 12px;
  font-weight: 600;
  color: var(--text-secondary);
  margin-bottom: 8px;
}

.cand-list { display: flex; flex-direction: column; gap: 6px; }

.cand-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 10px;
  padding: 8px 10px;
  border-radius: 6px;
  background: var(--bg-elevated);
  border: 1px solid transparent;
  &.active { border-color: rgba(103,194,58,.4); }
}

.cand-info { flex: 1; display: flex; flex-direction: column; gap: 3px; }

.cand-row {
  display: flex;
  gap: 10px;
  align-items: center;
  font-size: 12px;
  color: var(--text-primary);
}

.cand-conf { color: var(--accent); font-weight: 600; }

.cand-metrics {
  display: flex;
  gap: 8px;
  font-size: 11px;
  color: var(--text-muted);
}

.cand-reason {
  font-size: 11px;
  color: var(--text-muted);
  font-style: italic;
}

// ── Actions row ───────────────────────────────────────────────────────────────
.seg-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  padding-top: 10px;
  border-top: 1px solid var(--border-faint);
  margin-top: 4px;
}

@media (max-width: 900px) {
  .seg-body { flex-direction: column; }
  .seg-visual { width: 100%; }
}
</style>
