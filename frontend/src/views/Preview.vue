<template>
  <div class="preview-page" v-loading="projectStore.loading">

    <!-- Video panels -->
    <div class="video-row" v-if="currentSegment">
      <div class="video-panel">
        <div class="vp-label">
          <span class="vp-dot narration"></span>
          解说视频
        </div>
        <div class="vp-screen">
          <video ref="narrationVideo" controls :src="narrationVideoUrl" />
        </div>
      </div>

      <div class="video-panel">
        <div class="vp-label">
          <span class="vp-dot movie"></span>
          匹配电影片段
        </div>
        <div class="vp-screen" :class="{ empty: !hasMovieMatch }">
          <video v-if="hasMovieMatch" ref="movieVideo" controls :src="movieVideoUrl" />
          <div v-else class="no-match">
            <svg width="40" height="40" viewBox="0 0 24 24" fill="none"><path d="M12 9v4m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>
            <p>该片段暂未匹配到电影片段</p>
          </div>
        </div>
      </div>
    </div>

    <!-- Empty state -->
    <div v-else class="empty-state">
      <div class="empty-icon">
        <svg width="48" height="48" viewBox="0 0 24 24" fill="none"><path d="M15 10l4.553-2.277A1 1 0 0121 8.723v6.554a1 1 0 01-1.447.894L15 14M3 8a2 2 0 012-2h10a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2V8z" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>
      </div>
      <p>暂无可预览的片段</p>
    </div>

    <!-- Content: segment list + detail -->
    <div class="content-row">

      <!-- Segment list -->
      <div class="seg-list-panel">
        <div class="slp-head">
          <span class="slp-title">片段列表</span>
          <div class="slp-nav">
            <button class="nav-btn" :disabled="currentIndex <= 0" @click="prevSegment">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none"><path d="M15 18l-6-6 6-6" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>
            </button>
            <span class="nav-counter">{{ currentIndex + 1 }} / {{ selectedSegments.length }}</span>
            <button class="nav-btn" :disabled="currentIndex >= selectedSegments.length - 1" @click="nextSegment">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none"><path d="M9 18l6-6-6-6" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>
            </button>
          </div>
        </div>
        <div class="slp-scroll">
          <div
            v-for="(segment, index) in selectedSegments"
            :key="segment.id"
            class="slp-item"
            :class="{ active: index === currentIndex, review: segment.review_required }"
            @click="goToSegment(index)"
          >
            <div class="sli-num">#{{ segment.index + 1 }}</div>
            <div class="sli-content">
              <div class="sli-text">{{ truncateText(segment.polished_text || segment.original_text) }}</div>
              <div class="sli-meta">
                <span class="conf-badge" :class="getConfClass(segment.match_confidence)">{{ percent(segment.match_confidence) }}</span>
                <span>{{ getAlignmentText(segment.alignment_status) }}</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Segment detail -->
      <div v-if="currentSegment" class="seg-detail-panel">
        <div class="sdp-head">
          <span class="sdp-title">片段 #{{ currentSegment.index + 1 }}</span>
          <div class="sdp-actions">
            <el-button size="small" @click="playTTS">试听TTS</el-button>
            <el-button size="small" :disabled="currentSegment.skip_matching" @click="handleRematch(currentSegment)">重匹配</el-button>
            <el-button size="small" @click="$router.push(`/project/${projectId}/editor`)">去编辑</el-button>
          </div>
        </div>

        <!-- Text pair -->
        <div class="text-pair">
          <div class="text-block">
            <div class="tb-label">原文</div>
            <div class="tb-body">{{ currentSegment.original_text || '(无)' }}</div>
          </div>
          <div class="text-block">
            <div class="tb-label">润色</div>
            <div class="tb-body">{{ currentSegment.polished_text || '(未润色)' }}</div>
          </div>
        </div>

        <!-- Metrics -->
        <div class="metrics-row">
          <div class="metric-chip">
            <span class="mc-label">visual</span>
            <span class="mc-val">{{ percent(currentSegment.visual_confidence) }}</span>
          </div>
          <div class="metric-chip">
            <span class="mc-label">audio</span>
            <span class="mc-val">{{ percent(currentSegment.audio_confidence) }}</span>
          </div>
          <div class="metric-chip">
            <span class="mc-label">temporal</span>
            <span class="mc-val">{{ percent(currentSegment.temporal_confidence) }}</span>
          </div>
          <div class="metric-chip">
            <span class="mc-label">stability</span>
            <span class="mc-val">{{ percent(currentSegment.stability_score) }}</span>
          </div>
          <div class="metric-chip">
            <span class="mc-label">speech</span>
            <span class="mc-val">{{ percent(currentSegment.speech_likelihood) }}</span>
          </div>
          <div class="metric-chip" v-if="currentSegment.audio_activity_label && currentSegment.audio_activity_label !== 'unknown'">
            <span class="mc-label">音频</span>
            <span class="mc-val">{{ activityLabel(currentSegment.audio_activity_label) }}</span>
          </div>
          <div class="metric-chip">
            <span class="mc-label">gap</span>
            <span class="mc-val">{{ currentSegment.duration_gap?.toFixed(1) || '0.0' }}s</span>
          </div>
          <div class="metric-chip">
            <span class="mc-label">boundary</span>
            <span class="mc-val">{{ currentSegment.estimated_boundary_error != null ? `${currentSegment.estimated_boundary_error.toFixed(1)}s` : '--' }}</span>
          </div>
        </div>

        <div class="match-reason">{{ currentSegment.match_reason || '暂无匹配说明' }}</div>

        <!-- Candidates -->
        <div v-if="currentSegment.match_candidates?.length" class="cand-section">
          <div class="cand-title">候选匹配</div>
          <div class="cand-list">
            <div
              v-for="candidate in currentSegment.match_candidates"
              :key="candidate.id"
              class="cand-item"
              :class="{ active: candidate.id === currentSegment.selected_candidate_id }"
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
                :type="candidate.id === currentSegment.selected_candidate_id ? 'success' : ''"
                @click="applyCandidate(currentSegment, candidate)"
              >
                {{ candidate.id === currentSegment.selected_candidate_id ? '当前' : '采用' }}
              </el-button>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted, ref, watch } from 'vue'
import { useRoute } from 'vue-router'
import { ElMessage } from 'element-plus'
import { previewApi } from '@/api'
import { useProjectStore } from '@/stores/project'

const route = useRoute()
const projectStore = useProjectStore()

const projectId = computed(() => route.params.id)
const selectedSegments = computed(() => projectStore.selectedSegments)
const narrationVideo = ref(null)
const movieVideo = ref(null)
const currentIndex = ref(0)

const currentSegment = computed(() => selectedSegments.value[currentIndex.value])
const narrationVideoUrl = computed(() => currentSegment.value ? previewApi.getSegmentVideo(projectId.value, currentSegment.value.id, 'narration') : '')
const hasMovieMatch = computed(() => currentSegment.value?.movie_start != null && currentSegment.value?.movie_end != null)
const movieVideoUrl = computed(() => hasMovieMatch.value ? previewApi.getSegmentVideo(projectId.value, currentSegment.value.id, 'movie') : '')

onMounted(() => {
  if (!projectStore.currentProject || projectStore.currentProject.id !== projectId.value) {
    projectStore.fetchProject(projectId.value)
  }
})

watch(currentIndex, () => {
  narrationVideo.value?.load()
  movieVideo.value?.load()
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

function truncateText(text, maxLength = 48) {
  if (!text) return '(无文案)'
  return text.length > maxLength ? `${text.slice(0, maxLength)}...` : text
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

function activityLabel(value) {
  return { active: '活跃', weak: '较弱', silent: '静音' }[value] || value
}

function getConfClass(conf) {
  if ((conf || 0) >= 0.8) return 'high'
  if ((conf || 0) >= 0.5) return 'mid'
  return 'low'
}

function goToSegment(index) {
  currentIndex.value = index
}

function prevSegment() {
  if (currentIndex.value > 0) currentIndex.value -= 1
}

function nextSegment() {
  if (currentIndex.value < selectedSegments.value.length - 1) currentIndex.value += 1
}

async function applyCandidate(segment, candidate) {
  await projectStore.updateSegment(projectId.value, segment.id, {
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
    ElMessage.warning('该片段已标记为跳过匹配，请先去编辑页恢复')
    return
  }
  await projectStore.rematchSegment(projectId.value, segment.id)
  ElMessage.success('重匹配完成')
}

function playTTS() {
  if (!currentSegment.value) return
  const audio = new Audio(previewApi.getSegmentAudio(projectId.value, currentSegment.value.id, 'tts'))
  audio.onerror = () => ElMessage.error('TTS 音频不存在或加载失败')
  audio.play().catch(() => ElMessage.error('播放失败'))
}
</script>

<style lang="scss" scoped>
.preview-page {
  max-width: 1400px;
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

// ── Video row ─────────────────────────────────────────────────────────────────
.video-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
}

.video-panel {
  background: var(--card-bg);
  border: 1px solid var(--border-faint);
  border-radius: 10px;
  overflow: hidden;
}

.vp-label {
  display: flex;
  align-items: center;
  gap: 7px;
  padding: 10px 14px;
  font-size: 13px;
  font-weight: 600;
  color: var(--text-secondary);
  border-bottom: 1px solid var(--border-faint);
  background: var(--bg-elevated);
}

.vp-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  flex-shrink: 0;
  &.narration { background: #e6a23c; }
  &.movie { background: var(--accent); }
}

.vp-screen {
  background: #000;
  video {
    width: 100%;
    aspect-ratio: 16/9;
    display: block;
  }
  &.empty {
    background: var(--bg-elevated);
    display: flex;
    align-items: center;
    justify-content: center;
    aspect-ratio: 16/9;
  }
}

.no-match {
  text-align: center;
  color: var(--text-muted);
  p { margin-top: 10px; font-size: 13px; }
}

// ── Empty state ───────────────────────────────────────────────────────────────
.empty-state {
  text-align: center;
  padding: 80px 24px;
  color: var(--text-muted);
  .empty-icon { margin-bottom: 16px; opacity: .4; }
  p { font-size: 14px; }
}

// ── Content row ───────────────────────────────────────────────────────────────
.content-row {
  display: grid;
  grid-template-columns: 280px 1fr;
  gap: 16px;
  align-items: start;
}

// ── Segment list panel ────────────────────────────────────────────────────────
.seg-list-panel {
  background: var(--card-bg);
  border: 1px solid var(--border-faint);
  border-radius: 10px;
  overflow: hidden;
}

.slp-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 14px;
  border-bottom: 1px solid var(--border-faint);
  background: var(--bg-elevated);
}

.slp-title { font-size: 13px; font-weight: 600; color: var(--text-primary); }

.slp-nav {
  display: flex;
  align-items: center;
  gap: 6px;
}

.nav-btn {
  width: 26px;
  height: 26px;
  border-radius: 5px;
  border: 1px solid var(--border-faint);
  background: var(--bg-surface);
  color: var(--text-secondary);
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all .15s;
  &:hover:not(:disabled) { background: var(--bg-hover); color: var(--text-primary); }
  &:disabled { opacity: .4; cursor: not-allowed; }
}

.nav-counter { font-size: 12px; color: var(--text-muted); }

.slp-scroll {
  max-height: 460px;
  overflow-y: auto;
}

.slp-item {
  display: flex;
  gap: 10px;
  padding: 10px 14px;
  border-bottom: 1px solid var(--border-faint);
  cursor: pointer;
  transition: background .12s;
  &:last-child { border-bottom: none; }
  &:hover { background: var(--bg-hover); }
  &.active { background: rgba(91,108,248,.08); border-left: 2px solid var(--accent); }
  &.review { background: rgba(240,160,32,.05); }
}

.sli-num {
  font-size: 12px;
  font-weight: 700;
  color: var(--text-muted);
  width: 32px;
  flex-shrink: 0;
  padding-top: 1px;
}

.sli-content { flex: 1; min-width: 0; }

.sli-text {
  font-size: 12px;
  color: var(--text-secondary);
  line-height: 1.5;
  margin-bottom: 4px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.sli-meta {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 11px;
  color: var(--text-muted);
}

.conf-badge {
  font-size: 10px;
  font-weight: 700;
  padding: 1px 5px;
  border-radius: 3px;
  &.high { background: rgba(103,194,58,.15); color: #67c23a; }
  &.mid { background: rgba(230,162,60,.15); color: #e6a23c; }
  &.low { background: rgba(245,108,108,.15); color: #f56c6c; }
}

// ── Segment detail panel ──────────────────────────────────────────────────────
.seg-detail-panel {
  background: var(--card-bg);
  border: 1px solid var(--border-faint);
  border-radius: 10px;
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.sdp-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.sdp-title { font-size: 14px; font-weight: 700; color: var(--text-primary); }
.sdp-actions { display: flex; gap: 8px; }

// ── Text pair ─────────────────────────────────────────────────────────────────
.text-pair { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }

.text-block {
  background: var(--bg-elevated);
  border-radius: 7px;
  padding: 10px 12px;
}

.tb-label {
  font-size: 10px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: .5px;
  color: var(--text-muted);
  margin-bottom: 6px;
}

.tb-body {
  font-size: 13px;
  color: var(--text-secondary);
  line-height: 1.6;
  min-height: 60px;
}

// ── Metrics ───────────────────────────────────────────────────────────────────
.metrics-row {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.metric-chip {
  background: var(--bg-elevated);
  border: 1px solid var(--border-faint);
  border-radius: 5px;
  padding: 4px 8px;
  display: flex;
  gap: 4px;
  align-items: center;
}

.mc-label { font-size: 10px; color: var(--text-muted); }
.mc-val { font-size: 11px; font-weight: 600; color: var(--text-primary); font-variant-numeric: tabular-nums; }

.match-reason {
  font-size: 12px;
  color: var(--text-muted);
  font-style: italic;
  line-height: 1.5;
}

// ── Candidates ────────────────────────────────────────────────────────────────
.cand-section { display: flex; flex-direction: column; gap: 8px; }

.cand-title {
  font-size: 12px;
  font-weight: 600;
  color: var(--text-secondary);
  padding-bottom: 6px;
  border-bottom: 1px solid var(--border-faint);
}

.cand-list { display: flex; flex-direction: column; gap: 6px; }

.cand-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 10px;
  padding: 8px 10px;
  border-radius: 7px;
  background: var(--bg-elevated);
  border: 1px solid transparent;
  &.active { border-color: rgba(103,194,58,.4); }
}

.cand-info { flex: 1; display: flex; flex-direction: column; gap: 3px; }

.cand-row {
  display: flex;
  gap: 8px;
  align-items: center;
  font-size: 12px;
  color: var(--text-primary);
}

.cand-conf { color: var(--accent); font-weight: 700; }

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

@media (max-width: 1100px) {
  .video-row { grid-template-columns: 1fr; }
  .content-row { grid-template-columns: 1fr; }
  .text-pair { grid-template-columns: 1fr; }
}
</style>
