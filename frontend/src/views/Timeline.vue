<template>
  <div class="timeline-page" v-loading="projectStore.loading">

    <!-- Toolbar -->
    <div class="tl-toolbar">
      <div class="tl-info">
        <span class="tl-stat">{{ selectedSegments.length }} 片段</span>
        <span class="tl-sep">·</span>
        <span class="tl-stat">{{ formatTime(totalDuration) }} 总时长</span>
      </div>
      <div class="tl-controls">
        <span class="zoom-label">缩放</span>
        <el-slider
          v-model="zoom"
          :min="0.5"
          :max="3"
          :step="0.1"
          :format-tooltip="val => `${(val * 100).toFixed(0)}%`"
          style="width: 130px"
        />
        <span class="zoom-val">{{ (zoom * 100).toFixed(0) }}%</span>
      </div>
    </div>

    <!-- Track legend -->
    <div class="track-legend">
      <span class="legend-item movie">主视频</span>
      <span class="legend-item audio">TTS音频</span>
      <span class="legend-item narration">解说视频</span>
    </div>

    <!-- Timeline canvas -->
    <div class="tl-canvas-wrap">
      <div class="tl-canvas" :style="{ width: timelineWidth + 80 + 'px' }">

        <!-- Label column -->
        <div class="tl-labels">
          <div class="ruler-spacer"></div>
          <div class="track-label">主视频</div>
          <div class="track-label">TTS音频</div>
          <div class="track-label">解说视频</div>
        </div>

        <!-- Scrollable track area -->
        <div class="tl-scroll-area" ref="scrollArea">
          <!-- Ruler -->
          <div class="tl-ruler" :style="{ width: timelineWidth + 'px' }">
            <div
              v-for="tick in timeTicks"
              :key="tick.time"
              class="ruler-tick"
              :style="{ left: timeToPosition(tick.time) + 'px' }"
            >
              <span class="tick-label">{{ formatTime(tick.time) }}</span>
            </div>
          </div>

          <!-- Tracks -->
          <div class="tracks-wrap" :style="{ width: timelineWidth + 'px' }">
            <!-- Movie track -->
            <div class="tl-track">
              <div
                v-for="segment in selectedSegments"
                :key="segment.id"
                class="tl-seg movie"
                :class="[selectedSegmentId === segment.id ? 'active' : '', segment.match_type || 'exact']"
                :style="getSegmentStyle(segment, 'movie')"
                @click="selectSegment(segment)"
                :title="`#${segment.index + 1} · ${formatTime(segment.movie_start)}–${formatTime(segment.movie_end)}`"
              >
                <span class="seg-num">{{ segment.index + 1 }}</span>
              </div>
            </div>

            <!-- TTS track -->
            <div class="tl-track">
              <div
                v-for="segment in selectedSegments"
                :key="segment.id"
                class="tl-seg audio"
                :style="getSegmentStyle(segment, 'tts')"
                :title="`#${segment.index + 1} · TTS ${segment.tts_duration?.toFixed(1) || '?'}s`"
              >
                <span class="seg-num">{{ segment.index + 1 }}</span>
              </div>
            </div>

            <!-- Narration track -->
            <div class="tl-track">
              <div
                v-for="segment in projectStore.segments"
                :key="segment.id"
                class="tl-seg narration"
                :style="getSegmentStyle(segment, 'narration')"
                :title="`#${segment.index + 1} · 解说 ${formatTime(segment.narration_start)}–${formatTime(segment.narration_end)}`"
              >
                <span class="seg-num">{{ segment.index + 1 }}</span>
              </div>
            </div>

            <!-- Playhead -->
            <div class="playhead" :style="{ left: timeToPosition(currentTime) + 'px' }" />
          </div>
        </div>
      </div>
    </div>

    <!-- Detail panel -->
    <div v-if="selectedSegment" class="detail-panel">
      <div class="detail-head">
        <span class="detail-title">片段 #{{ selectedSegment.index + 1 }}</span>
        <el-button size="small" @click="$router.push(`/project/${projectId}/editor`)">去编辑</el-button>
      </div>
      <div class="detail-grid">
        <div class="detail-item">
          <div class="di-label">电影时间</div>
          <div class="di-value">{{ formatTime(selectedSegment.movie_start) }} – {{ formatTime(selectedSegment.movie_end) }}</div>
        </div>
        <div class="detail-item">
          <div class="di-label">解说时间</div>
          <div class="di-value">{{ formatTime(selectedSegment.narration_start) }} – {{ formatTime(selectedSegment.narration_end) }}</div>
        </div>
        <div class="detail-item">
          <div class="di-label">TTS时长</div>
          <div class="di-value">{{ selectedSegment.tts_duration?.toFixed(2) || '--' }} 秒</div>
        </div>
        <div class="detail-item">
          <div class="di-label">置信度</div>
          <div class="di-value accent">{{ ((selectedSegment.match_confidence || 0) * 100).toFixed(0) }}%</div>
        </div>
        <div class="detail-item">
          <div class="di-label">Match Type</div>
          <div class="di-value">{{ selectedSegment.match_type || 'exact' }}</div>
        </div>
        <div class="detail-item detail-wide">
          <div class="di-label">Evidence</div>
          <div class="di-value">{{ selectedSegment.evidence_summary || selectedSegment.match_reason || '--' }}</div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useRoute } from 'vue-router'
import { useProjectStore } from '@/stores/project'

const route = useRoute()
const projectStore = useProjectStore()

const projectId = computed(() => route.params.id)
const zoom = ref(1)
const currentTime = ref(0)
const selectedSegmentId = ref(null)

const selectedSegments = computed(() => projectStore.selectedSegments.filter(segment => segment.segment_type !== 'non_movie'))

const selectedSegment = computed(() => {
  if (!selectedSegmentId.value) return null
  return projectStore.segments.find(s => s.id === selectedSegmentId.value)
})

const totalDuration = computed(() => {
  let duration = 0
  selectedSegments.value.forEach(seg => {
    duration += seg.tts_duration || (seg.movie_end - seg.movie_start) || 0
  })
  return duration
})

const timelineWidth = computed(() => {
  return Math.max(totalDuration.value * 10 * zoom.value, 1000)
})

const timeTicks = computed(() => {
  const ticks = []
  const interval = zoom.value > 1.5 ? 5 : zoom.value > 0.8 ? 10 : 30
  for (let t = 0; t <= totalDuration.value; t += interval) {
    ticks.push({ time: t })
  }
  return ticks
})

onMounted(() => {
  if (!projectStore.currentProject || projectStore.currentProject.id !== projectId.value) {
    projectStore.fetchProject(projectId.value)
  }
})

function formatTime(seconds) {
  if (seconds == null) return '--:--'
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${m}:${s.toString().padStart(2, '0')}`
}

function timeToPosition(time) {
  return time * 10 * zoom.value
}

function getSegmentStyle(segment, type) {
  let start, duration

  if (type === 'narration') {
    start = segment.narration_start
    duration = segment.narration_end - segment.narration_start
  } else if (type === 'tts') {
    let pos = 0
    for (const seg of selectedSegments.value) {
      if (seg.id === segment.id) break
      pos += seg.tts_duration || (seg.movie_end - seg.movie_start) || 0
    }
    start = pos
    duration = segment.tts_duration || 3
  } else {
    let pos = 0
    for (const seg of selectedSegments.value) {
      if (seg.id === segment.id) break
      pos += seg.tts_duration || (seg.movie_end - seg.movie_start) || 0
    }
    start = pos
    duration = segment.tts_duration || (segment.movie_end - segment.movie_start) || 3
  }

  return {
    left: timeToPosition(start) + 'px',
    width: Math.max(timeToPosition(duration), 20) + 'px'
  }
}

function selectSegment(segment) {
  selectedSegmentId.value = segment.id
}
</script>

<style lang="scss" scoped>
.timeline-page {
  max-width: 100%;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

// ── Toolbar ───────────────────────────────────────────────────────────────────
.tl-toolbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 16px;
  background: var(--card-bg);
  border: 1px solid var(--border-faint);
  border-radius: 8px;
}

.tl-info { display: flex; align-items: center; gap: 8px; }
.tl-stat { font-size: 13px; font-weight: 500; color: var(--text-secondary); }
.tl-sep { color: var(--text-muted); }

.tl-controls {
  display: flex;
  align-items: center;
  gap: 10px;
}

.zoom-label { font-size: 12px; color: var(--text-muted); }
.zoom-val { font-size: 12px; color: var(--text-secondary); width: 36px; }

// ── Legend ────────────────────────────────────────────────────────────────────
.track-legend {
  display: flex;
  gap: 16px;
  padding: 0 4px;
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  color: var(--text-muted);
  &::before {
    content: '';
    width: 12px;
    height: 12px;
    border-radius: 3px;
  }
  &.movie::before { background: rgba(91,108,248,.5); }
  &.audio::before { background: rgba(103,194,58,.5); }
  &.narration::before { background: rgba(230,162,60,.5); }
}

// ── Canvas ────────────────────────────────────────────────────────────────────
.tl-canvas-wrap {
  background: var(--card-bg);
  border: 1px solid var(--border-faint);
  border-radius: 10px;
  overflow: hidden;
}

.tl-canvas {
  display: flex;
  min-width: 100%;
}

.tl-labels {
  width: 80px;
  flex-shrink: 0;
  border-right: 1px solid var(--border-faint);
  background: var(--bg-elevated);
}

.ruler-spacer {
  height: 32px;
  border-bottom: 1px solid var(--border-faint);
}

.track-label {
  height: 52px;
  display: flex;
  align-items: center;
  padding: 0 10px;
  font-size: 11px;
  font-weight: 600;
  color: var(--text-muted);
  border-bottom: 1px solid var(--border-faint);
  &:last-child { border-bottom: none; }
}

.tl-scroll-area {
  flex: 1;
  overflow-x: auto;
}

// ── Ruler ─────────────────────────────────────────────────────────────────────
.tl-ruler {
  position: relative;
  height: 32px;
  border-bottom: 1px solid var(--border-faint);
}

.ruler-tick {
  position: absolute;
  top: 0;
  height: 100%;
  border-left: 1px solid var(--border-faint);
}

.tick-label {
  position: absolute;
  top: 6px;
  left: 4px;
  font-size: 10px;
  color: var(--text-muted);
  white-space: nowrap;
}

// ── Tracks ────────────────────────────────────────────────────────────────────
.tracks-wrap {
  position: relative;
}

.tl-track {
  position: relative;
  height: 52px;
  border-bottom: 1px solid var(--border-faint);
  background: var(--bg-elevated);
  &:last-child { border-bottom: none; }
}

// ── Segments ──────────────────────────────────────────────────────────────────
.tl-seg {
  position: absolute;
  top: 8px;
  bottom: 8px;
  border-radius: 4px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
  transition: filter .15s, box-shadow .15s;

  &:hover { filter: brightness(1.2); }

  &.movie {
    background: rgba(91,108,248,.35);
    border: 1px solid rgba(91,108,248,.6);
    &.active { box-shadow: 0 0 0 2px var(--accent); }
    &.exact { background: rgba(34,197,94,.30); border-color: rgba(34,197,94,.58); }
    &.inferred { background: rgba(245,158,11,.28); border-color: rgba(245,158,11,.58); }
    &.fallback { background: rgba(239,68,68,.24); border-color: rgba(239,68,68,.56); }
  }

  &.audio {
    background: rgba(103,194,58,.3);
    border: 1px solid rgba(103,194,58,.6);
  }

  &.narration {
    background: rgba(230,162,60,.3);
    border: 1px solid rgba(230,162,60,.6);
  }
}

.seg-num {
  font-size: 10px;
  font-weight: 700;
  color: rgba(255,255,255,.85);
  text-shadow: 0 1px 2px rgba(0,0,0,.5);
}

// ── Playhead ──────────────────────────────────────────────────────────────────
.playhead {
  position: absolute;
  top: 0;
  bottom: 0;
  width: 2px;
  background: #f56c6c;
  pointer-events: none;
  z-index: 10;

  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: -5px;
    border-left: 6px solid transparent;
    border-right: 6px solid transparent;
    border-top: 8px solid #f56c6c;
  }
}

// ── Detail panel ──────────────────────────────────────────────────────────────
.detail-panel {
  background: var(--card-bg);
  border: 1px solid var(--border-faint);
  border-radius: 10px;
  padding: 16px;
}

.detail-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 14px;
}

.detail-title {
  font-size: 14px;
  font-weight: 600;
  color: var(--text-primary);
}

.detail-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 12px;
}

.detail-item {
  background: var(--bg-elevated);
  border-radius: 7px;
  padding: 10px 12px;
}
.detail-item.detail-wide {
  grid-column: span 2;
}

.di-label {
  font-size: 11px;
  color: var(--text-muted);
  margin-bottom: 4px;
}

.di-value {
  font-size: 13px;
  font-weight: 600;
  color: var(--text-primary);
  font-variant-numeric: tabular-nums;
  &.accent { color: var(--accent); }
}

@media (max-width: 900px) {
  .detail-grid { grid-template-columns: repeat(2, 1fr); }
  .detail-item.detail-wide { grid-column: span 2; }
}
</style>
