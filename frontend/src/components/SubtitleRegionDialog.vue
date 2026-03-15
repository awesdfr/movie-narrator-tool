<template>
  <el-dialog
    v-model="dialogVisible"
    title="框选字幕 / 水印区域"
    width="1220px"
    top="3vh"
    destroy-on-close
  >
    <div class="subtitle-dialog">
      <el-alert
        type="info"
        :closable="false"
        show-icon
        class="dialog-tip"
        title="手动屏蔽容易干扰匹配的区域"
        description="可以分别给解说视频和原电影框出字幕、台标、水印区域。每个区域还能限制生效时间段，用来应对前后字幕位置变化的素材。"
      />

      <div class="toolbar">
        <el-radio-group v-model="activeSource" size="small">
          <el-radio-button label="narration">解说视频</el-radio-button>
          <el-radio-button label="movie">原电影</el-radio-button>
        </el-radio-group>

        <el-select v-model="maskMode" class="mask-mode-select">
          <el-option label="自动检测 + 手动画框" value="hybrid" />
          <el-option label="仅手动画框" value="manual_only" />
          <el-option label="仅自动检测" value="auto_only" />
        </el-select>

        <el-radio-group v-model="previewMode" size="small">
          <el-radio-button label="original">原图</el-radio-button>
          <el-radio-button label="masked">遮罩预览</el-radio-button>
        </el-radio-group>

        <el-button @click="loadFrame">刷新当前帧</el-button>
        <el-button :disabled="!currentRegions.length" @click="copyToOtherSource">复制到另一源</el-button>
        <el-button :disabled="!currentRegions.length" @click="clearCurrentSource">清空当前源</el-button>
      </div>

      <div class="workspace">
        <div class="frame-panel">
          <div class="frame-controls">
            <div class="frame-time">
              <span class="control-label">{{ sourceLabel(activeSource) }}时间</span>
              <el-slider
                v-model="currentTime"
                :min="0"
                :max="currentDuration"
                :step="0.1"
                :show-tooltip="true"
                :disabled="currentDuration <= 0"
                @change="loadFrame"
              />
            </div>

            <el-input-number
              v-model="currentTime"
              :min="0"
              :max="currentDuration"
              :step="0.1"
              :precision="1"
              controls-position="right"
              @change="loadFrame"
            />
          </div>

          <div class="source-meta">
            <span>{{ currentInfo.width || '--' }} x {{ currentInfo.height || '--' }}</span>
            <span>{{ formatDuration(currentDuration) }}</span>
            <span>{{ currentInfo.fps ? `${currentInfo.fps.toFixed(2)} fps` : '--' }}</span>
            <span>{{ previewMode === 'masked' ? '当前显示遮罩后画面' : '当前显示原始画面' }}</span>
          </div>

          <div class="preset-row">
            <span class="control-label">快速预设</span>
            <el-button size="small" @click="applyPreset('bottom_narrow')">底部字幕 12%</el-button>
            <el-button size="small" @click="applyPreset('bottom_wide')">底部字幕 18%</el-button>
            <el-button size="small" @click="applyPreset('top_logo')">顶部台标 10%</el-button>
            <el-button size="small" @click="applyPreset('left_watermark')">左上水印</el-button>
          </div>

          <div class="frame-stage">
            <div
              class="frame-wrapper"
              :class="{ drawing: interaction.type === 'draw', loading: frameLoading }"
              @mousedown.prevent="startDrawing"
            >
              <img
                v-if="frameUrl"
                ref="frameImage"
                :src="frameUrl"
                class="frame-image"
                @load="handleImageLoaded"
                @error="handleImageError"
              >

              <div class="frame-overlay">
                <div
                  v-for="(region, index) in currentRegions"
                  :key="region.id"
                  class="region-box"
                  :class="{
                    disabled: region.enabled === false,
                    selected: selectedRegionId === region.id,
                    inactive: !isRegionActiveAtCurrentTime(region)
                  }"
                  :style="regionStyle(region)"
                  @mousedown.stop.prevent="startMove(region.id, $event)"
                  @click.stop="selectedRegionId = region.id"
                >
                  <span class="region-label">
                    {{ region.label || `区域 ${index + 1}` }}
                    <small v-if="!isRegionActiveAtCurrentTime(region)">未生效</small>
                  </span>
                  <span
                    v-for="handle in resizeHandles"
                    :key="handle"
                    class="resize-handle"
                    :class="`handle-${handle}`"
                    @mousedown.stop.prevent="startResize(region.id, handle, $event)"
                  />
                </div>

                <div
                  v-if="draftRegion"
                  class="region-box draft"
                  :style="regionStyle(draftRegion)"
                />
              </div>

              <div v-if="frameLoading" class="frame-placeholder">正在加载帧...</div>
              <div v-else-if="frameError" class="frame-placeholder">{{ frameError }}</div>
              <div v-else-if="!frameUrl" class="frame-placeholder">没有可预览的视频帧</div>
            </div>
          </div>
        </div>

        <div class="region-panel">
          <div class="panel-section">
            <div class="panel-title">当前源区域</div>
            <div class="panel-description">
              {{ sourceLabel(activeSource) }}已配置 {{ currentRegions.length }} 个区域，其中当前时刻生效 {{ activeRegionCount }}
              个。
            </div>
          </div>

          <div v-if="selectedRegion" class="editor-card">
            <div class="panel-title">区域设置</div>

            <el-input
              :model-value="selectedRegion.label || ''"
              placeholder="区域名称，例如：字幕、台标、水印"
              @update:model-value="updateSelectedRegion({ label: $event || null })"
            />

            <div class="range-toggle">
              <span class="control-label">时间范围</span>
              <el-switch
                :model-value="selectedRegionHasRange"
                inline-prompt
                active-text="分段"
                inactive-text="全程"
                @change="toggleSelectedRegionRange"
              />
            </div>

            <template v-if="selectedRegionHasRange">
              <div class="range-inputs">
                <div class="range-field">
                  <span class="control-label">开始时间</span>
                  <el-input-number
                    :model-value="selectedRegion.start_time ?? 0"
                    :min="0"
                    :max="selectedRegion.end_time ?? currentDuration"
                    :step="0.1"
                    :precision="1"
                    controls-position="right"
                    @update:model-value="updateSelectedRegionRange('start_time', $event)"
                  />
                </div>

                <div class="range-field">
                  <span class="control-label">结束时间</span>
                  <el-input-number
                    :model-value="selectedRegion.end_time ?? currentDuration"
                    :min="selectedRegion.start_time ?? 0"
                    :max="currentDuration"
                    :step="0.1"
                    :precision="1"
                    controls-position="right"
                    @update:model-value="updateSelectedRegionRange('end_time', $event)"
                  />
                </div>
              </div>

              <div class="range-shortcuts">
                <el-button size="small" @click="setSelectedRegionWindow(15)">当前前后 15s</el-button>
                <el-button size="small" @click="setSelectedRegionWindow(30)">当前前后 30s</el-button>
                <el-button size="small" @click="setSelectedRegionFromCurrent">从当前到结尾</el-button>
              </div>
            </template>

            <div class="region-meta">
              <span>X {{ percent(selectedRegion.x) }}</span>
              <span>Y {{ percent(selectedRegion.y) }}</span>
              <span>W {{ percent(selectedRegion.width) }}</span>
              <span>H {{ percent(selectedRegion.height) }}</span>
            </div>
          </div>

          <div v-if="currentRegions.length" class="region-list">
            <div
              v-for="(region, index) in currentRegions"
              :key="region.id"
              class="region-item"
              :class="{
                selected: selectedRegionId === region.id,
                inactive: !isRegionActiveAtCurrentTime(region)
              }"
              @click="selectedRegionId = region.id"
            >
              <div class="region-item-header">
                <span>{{ region.label || `区域 ${index + 1}` }}</span>
                <el-switch
                  :model-value="region.enabled !== false"
                  inline-prompt
                  active-text="启用"
                  inactive-text="停用"
                  @change="toggleRegionEnabled(region.id, $event)"
                />
              </div>

              <div class="region-item-meta">
                <span>X {{ percent(region.x) }}</span>
                <span>Y {{ percent(region.y) }}</span>
                <span>{{ regionTimeText(region) }}</span>
              </div>

              <div class="region-actions">
                <el-button text size="small" @click.stop="focusRegion(region.id)">选中</el-button>
                <el-button text size="small" @click.stop="setRegionRangeToFull(region.id)">全程</el-button>
                <el-button text size="small" type="danger" @click.stop="removeRegion(region.id)">删除</el-button>
              </div>
            </div>
          </div>
          <el-empty v-else description="当前源还没有手动画框" :image-size="84" />

          <div class="panel-section summary-section">
            <div class="panel-title">总览</div>
            <div class="summary-item">解说视频：{{ regionsState.narration.length }} 个区域</div>
            <div class="summary-item">原电影：{{ regionsState.movie.length }} 个区域</div>
            <div class="summary-item">遮罩模式：{{ maskModeText }}</div>
            <div class="summary-item">支持字幕、台标、水印和时间段遮罩</div>
          </div>
        </div>
      </div>
    </div>

    <template #footer>
      <div class="dialog-footer">
        <el-button @click="dialogVisible = false">取消</el-button>
        <el-button :loading="saving === 'save'" @click="saveRegions(false)">保存区域</el-button>
        <el-button type="primary" :loading="saving === 'rematch'" @click="saveRegions(true)">
          保存并重匹配
        </el-button>
      </div>
    </template>
  </el-dialog>
</template>

<script setup>
import { computed, onBeforeUnmount, reactive, ref, watch } from 'vue'
import { ElMessage } from 'element-plus'
import { previewApi } from '@/api'
import { useProjectStore } from '@/stores/project'

const PRESETS = {
  bottom_narrow: { label: '底部字幕 12%', x: 0.05, y: 0.84, width: 0.9, height: 0.12 },
  bottom_wide: { label: '底部字幕 18%', x: 0.02, y: 0.8, width: 0.96, height: 0.18 },
  top_logo: { label: '顶部台标', x: 0.68, y: 0.02, width: 0.28, height: 0.1 },
  left_watermark: { label: '左上水印', x: 0.02, y: 0.02, width: 0.22, height: 0.12 }
}

const resizeHandles = ['nw', 'ne', 'sw', 'se']

const props = defineProps({
  modelValue: {
    type: Boolean,
    default: false
  },
  project: {
    type: Object,
    default: null
  }
})

const emit = defineEmits(['update:modelValue', 'saved'])

const projectStore = useProjectStore()

const dialogVisible = computed({
  get: () => props.modelValue,
  set: value => emit('update:modelValue', value)
})

const activeSource = ref('narration')
const previewMode = ref('original')
const maskMode = ref('hybrid')
const frameLoading = ref(false)
const frameError = ref('')
const frameRevision = ref(0)
const imageReady = ref(false)
const saving = ref('')
const frameImage = ref(null)
const selectedRegionId = ref(null)
const draftRegion = ref(null)

const interaction = reactive({
  type: '',
  regionId: null,
  handle: '',
  startPoint: null,
  originRegion: null
})

const sourceInfo = reactive({
  narration: emptySourceInfo(),
  movie: emptySourceInfo()
})

const sourceTimes = reactive({
  narration: 0,
  movie: 0
})

const loadedSourceTimes = reactive({
  narration: 0,
  movie: 0
})

const loadedPreviewState = reactive({
  narration: emptyPreviewState(),
  movie: emptyPreviewState()
})

const regionsState = reactive({
  narration: [],
  movie: []
})

const currentInfo = computed(() => sourceInfo[activeSource.value] || emptySourceInfo())
const currentDuration = computed(() => currentInfo.value.duration || 0)
const currentRegions = computed(() => regionsState[activeSource.value] || [])
const currentTime = computed({
  get: () => sourceTimes[activeSource.value] || 0,
  set: value => {
    sourceTimes[activeSource.value] = clampNumber(value, 0, currentDuration.value || 0)
  }
})

const selectedRegion = computed(() => currentRegions.value.find(region => region.id === selectedRegionId.value) || null)
const selectedRegionHasRange = computed(() => Boolean(selectedRegion.value && (
  selectedRegion.value.start_time != null || selectedRegion.value.end_time != null
)))
const activeRegionCount = computed(() => currentRegions.value.filter(region => isRegionActiveAtCurrentTime(region)).length)

const frameUrl = computed(() => {
  if (!props.project?.id) {
    return ''
  }
  const time = Number(loadedSourceTimes[activeSource.value] || 0).toFixed(2)
  const previewState = loadedPreviewState[activeSource.value]
  return `${previewApi.getFrame(props.project.id, activeSource.value, time, {
    masked: previewState.previewMode === 'masked',
    maskMode: previewState.maskMode,
    manualRegions: previewState.manualRegions
  })}&v=${frameRevision.value}`
})

const maskModeText = computed(() => ({
  hybrid: '自动检测 + 手动画框',
  manual_only: '仅手动画框',
  auto_only: '仅自动检测'
}[maskMode.value] || maskMode.value))

watch(
  () => props.modelValue,
  async visible => {
    if (!visible) {
      cleanupInteraction()
      return
    }
    try {
      resetStateFromProject()
      await Promise.all([ensureSourceInfo('narration'), ensureSourceInfo('movie')])
      await loadFrame()
    } catch (error) {
      frameLoading.value = false
      frameError.value = '读取视频信息失败，请确认项目视频可以正常访问'
    }
  }
)

watch(activeSource, async () => {
  cleanupInteraction()
  selectedRegionId.value = null
  if (dialogVisible.value) {
    try {
      await ensureSourceInfo(activeSource.value)
      await loadFrame()
    } catch (error) {
      frameLoading.value = false
      frameError.value = '切换视频源失败，请稍后重试'
    }
  }
})

watch(previewMode, async () => {
  if (dialogVisible.value) {
    await loadFrame()
  }
})

watch(maskMode, async () => {
  if (dialogVisible.value && previewMode.value === 'masked') {
    await loadFrame()
  }
})

onBeforeUnmount(() => {
  cleanupInteraction()
})

function emptySourceInfo() {
  return {
    path: '',
    duration: 0,
    fps: 0,
    width: 0,
    height: 0,
    loaded: false
  }
}

function emptyPreviewState() {
  return {
    previewMode: 'original',
    maskMode: 'hybrid',
    manualRegions: []
  }
}

function resetStateFromProject() {
  maskMode.value = props.project?.subtitle_mask_mode || 'hybrid'
  previewMode.value = 'original'
  regionsState.narration = cloneRegions(props.project?.narration_subtitle_regions || [])
  regionsState.movie = cloneRegions(props.project?.movie_subtitle_regions || [])
  sourceInfo.narration = emptySourceInfo()
  sourceInfo.movie = emptySourceInfo()
  sourceTimes.narration = 0
  sourceTimes.movie = 0
  loadedSourceTimes.narration = 0
  loadedSourceTimes.movie = 0
  loadedPreviewState.narration = emptyPreviewState()
  loadedPreviewState.movie = emptyPreviewState()
  activeSource.value = 'narration'
  selectedRegionId.value = null
  frameRevision.value += 1
  frameError.value = ''
  imageReady.value = false
}

function cloneRegions(regions) {
  return (regions || []).map((region, index) => ({
    id: region.id || `region_${index}`,
    x: clampNumber(region.x, 0, 1),
    y: clampNumber(region.y, 0, 1),
    width: clampNumber(region.width, 0.01, 1),
    height: clampNumber(region.height, 0.01, 1),
    enabled: region.enabled !== false,
    label: region.label || null,
    start_time: normalizeOptionalTime(region.start_time),
    end_time: normalizeOptionalTime(region.end_time)
  }))
}

function normalizeOptionalTime(value) {
  if (value == null || value === '') {
    return null
  }
  const numeric = Number(value)
  return Number.isNaN(numeric) ? null : Math.max(0, numeric)
}

async function ensureSourceInfo(source) {
  if (sourceInfo[source]?.loaded || !props.project?.id) {
    return
  }
  const info = await previewApi.getSourceInfo(props.project.id, source)
  sourceInfo[source] = {
    ...info,
    loaded: true
  }
  if ((sourceTimes[source] || 0) <= 0 && info.duration > 0) {
    sourceTimes[source] = Math.min(30, Math.max(0, Number((info.duration * 0.2).toFixed(1))))
  }
}

async function loadFrame() {
  if (!props.project?.id) {
    return
  }
  loadedSourceTimes[activeSource.value] = clampNumber(currentTime.value, 0, currentDuration.value || 0)
  loadedPreviewState[activeSource.value] = {
    previewMode: previewMode.value,
    maskMode: maskMode.value,
    manualRegions: cloneRegions(currentRegions.value)
  }
  frameLoading.value = true
  frameError.value = ''
  imageReady.value = false
  frameRevision.value += 1
}

function handleImageLoaded() {
  frameLoading.value = false
  frameError.value = ''
  imageReady.value = true
}

function handleImageError() {
  frameLoading.value = false
  imageReady.value = false
  frameError.value = '当前时间点无法加载视频帧，请换一个时间再试'
}

function regionStyle(region) {
  return {
    left: `${region.x * 100}%`,
    top: `${region.y * 100}%`,
    width: `${region.width * 100}%`,
    height: `${region.height * 100}%`
  }
}

function sourceLabel(source) {
  return source === 'movie' ? '原电影' : '解说视频'
}

function percent(value) {
  return `${(Number(value || 0) * 100).toFixed(1)}%`
}

function formatDuration(seconds) {
  if (!seconds) return '--'
  const total = Math.max(0, Number(seconds))
  const minutes = Math.floor(total / 60)
  const secs = Math.floor(total % 60)
  const tenths = Math.floor((total - Math.floor(total)) * 10)
  if (minutes >= 60) {
    const hours = Math.floor(minutes / 60)
    const restMinutes = minutes % 60
    return `${hours}:${String(restMinutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}`
  }
  return `${minutes}:${String(secs).padStart(2, '0')}.${tenths}`
}

function regionTimeText(region) {
  if (region.start_time == null && region.end_time == null) {
    return '全片生效'
  }
  const start = region.start_time == null ? '0.0s' : `${Number(region.start_time).toFixed(1)}s`
  const end = region.end_time == null ? '结尾' : `${Number(region.end_time).toFixed(1)}s`
  return `${start} - ${end}`
}

function clampNumber(value, min, max) {
  const numeric = Number(value)
  if (Number.isNaN(numeric)) {
    return min
  }
  return Math.min(max, Math.max(min, numeric))
}

function pointerToNormalized(event, allowOutside = false) {
  const imageEl = frameImage.value
  if (!imageEl || !imageReady.value) {
    return null
  }
  const rect = imageEl.getBoundingClientRect()
  if (!rect.width || !rect.height) {
    return null
  }
  let x = (event.clientX - rect.left) / rect.width
  let y = (event.clientY - rect.top) / rect.height
  if (allowOutside) {
    x = clampNumber(x, 0, 1)
    y = clampNumber(y, 0, 1)
  } else if (x < 0 || x > 1 || y < 0 || y > 1) {
    return null
  }
  return { x, y }
}

function isRegionActiveAtTime(region, timeValue) {
  const time = Number(timeValue || 0)
  if (region.start_time != null && time < Number(region.start_time)) {
    return false
  }
  if (region.end_time != null && time > Number(region.end_time)) {
    return false
  }
  return true
}

function isRegionActiveAtCurrentTime(region) {
  return isRegionActiveAtTime(region, currentTime.value)
}

function startDrawing(event) {
  if (
    event.button !== 0 ||
    frameLoading.value ||
    !imageReady.value ||
    maskMode.value === 'auto_only'
  ) {
    return
  }
  const point = pointerToNormalized(event)
  if (!point) {
    return
  }
  selectedRegionId.value = null
  interaction.type = 'draw'
  interaction.startPoint = point
  draftRegion.value = {
    id: 'draft',
    x: point.x,
    y: point.y,
    width: 0,
    height: 0,
    enabled: true,
    label: null,
    start_time: null,
    end_time: null
  }
  bindWindowEvents()
}

function startMove(regionId, event) {
  if (maskMode.value === 'auto_only') {
    return
  }
  const point = pointerToNormalized(event, true)
  const region = findRegion(regionId)
  if (!point || !region) {
    return
  }
  selectedRegionId.value = regionId
  interaction.type = 'move'
  interaction.regionId = regionId
  interaction.startPoint = point
  interaction.originRegion = { ...region }
  bindWindowEvents()
}

function startResize(regionId, handle, event) {
  if (maskMode.value === 'auto_only') {
    return
  }
  const point = pointerToNormalized(event, true)
  const region = findRegion(regionId)
  if (!point || !region) {
    return
  }
  selectedRegionId.value = regionId
  interaction.type = 'resize'
  interaction.regionId = regionId
  interaction.handle = handle
  interaction.startPoint = point
  interaction.originRegion = { ...region }
  bindWindowEvents()
}

function bindWindowEvents() {
  window.addEventListener('mousemove', handlePointerMove)
  window.addEventListener('mouseup', handlePointerUp)
}

function handlePointerMove(event) {
  if (!interaction.type || !interaction.startPoint) {
    return
  }
  const point = pointerToNormalized(event, true)
  if (!point) {
    return
  }

  if (interaction.type === 'draw') {
    const x = Math.min(interaction.startPoint.x, point.x)
    const y = Math.min(interaction.startPoint.y, point.y)
    draftRegion.value = {
      id: 'draft',
      x,
      y,
      width: Math.abs(point.x - interaction.startPoint.x),
      height: Math.abs(point.y - interaction.startPoint.y),
      enabled: true,
      label: null,
      start_time: null,
      end_time: null
    }
    return
  }

  const origin = interaction.originRegion
  if (!origin) {
    return
  }

  if (interaction.type === 'move') {
    const deltaX = point.x - interaction.startPoint.x
    const deltaY = point.y - interaction.startPoint.y
    updateRegion(interaction.regionId, {
      x: clampNumber(origin.x + deltaX, 0, 1 - origin.width),
      y: clampNumber(origin.y + deltaY, 0, 1 - origin.height)
    })
    return
  }

  if (interaction.type === 'resize') {
    updateRegion(interaction.regionId, resizeRegion(origin, point, interaction.handle))
  }
}

function resizeRegion(origin, point, handle) {
  const minSize = 0.01
  let left = origin.x
  let top = origin.y
  let right = origin.x + origin.width
  let bottom = origin.y + origin.height

  if (handle.includes('n')) {
    top = clampNumber(point.y, 0, bottom - minSize)
  }
  if (handle.includes('s')) {
    bottom = clampNumber(point.y, top + minSize, 1)
  }
  if (handle.includes('w')) {
    left = clampNumber(point.x, 0, right - minSize)
  }
  if (handle.includes('e')) {
    right = clampNumber(point.x, left + minSize, 1)
  }

  return {
    x: left,
    y: top,
    width: clampNumber(right - left, minSize, 1),
    height: clampNumber(bottom - top, minSize, 1)
  }
}

function handlePointerUp() {
  if (interaction.type === 'draw' && draftRegion.value) {
    const region = draftRegion.value
    if (region.width >= 0.01 && region.height >= 0.01) {
      const nextRegion = {
        id: createRegionId(),
        x: clampNumber(region.x, 0, 1),
        y: clampNumber(region.y, 0, 1),
        width: clampNumber(region.width, 0.01, 1),
        height: clampNumber(region.height, 0.01, 1),
        enabled: true,
        label: `区域 ${currentRegions.value.length + 1}`,
        start_time: null,
        end_time: null
      }
      regionsState[activeSource.value] = [...currentRegions.value, nextRegion]
      selectedRegionId.value = nextRegion.id
    }
  }
  cleanupInteraction()
  refreshMaskedPreviewIfNeeded()
}

function cleanupInteraction() {
  interaction.type = ''
  interaction.regionId = null
  interaction.handle = ''
  interaction.startPoint = null
  interaction.originRegion = null
  draftRegion.value = null
  window.removeEventListener('mousemove', handlePointerMove)
  window.removeEventListener('mouseup', handlePointerUp)
}

function refreshMaskedPreviewIfNeeded() {
  if (dialogVisible.value && previewMode.value === 'masked') {
    void loadFrame()
  }
}

function createRegionId() {
  return `region_${Date.now()}_${Math.random().toString(16).slice(2, 8)}`
}

function findRegion(regionId) {
  return currentRegions.value.find(region => region.id === regionId) || null
}

function updateRegion(regionId, patch) {
  regionsState[activeSource.value] = currentRegions.value.map(region => (
    region.id === regionId ? { ...region, ...patch } : region
  ))
}

function updateSelectedRegion(patch) {
  if (!selectedRegion.value) {
    return
  }
  updateRegion(selectedRegion.value.id, patch)
}

function updateSelectedRegionRange(field, nextValue) {
  if (!selectedRegion.value) {
    return
  }
  const value = normalizeOptionalTime(nextValue)
  if (field === 'start_time') {
    const end = selectedRegion.value.end_time
    updateSelectedRegion({
      start_time: value,
      end_time: end != null && value != null && end < value ? value : end
    })
  } else {
    const start = selectedRegion.value.start_time
    updateSelectedRegion({
      end_time: value,
      start_time: start != null && value != null && start > value ? value : start
    })
  }
  refreshMaskedPreviewIfNeeded()
}

function toggleSelectedRegionRange(enabled) {
  if (!selectedRegion.value) {
    return
  }
  if (!enabled) {
    updateSelectedRegion({
      start_time: null,
      end_time: null
    })
  } else {
    const start = clampNumber(currentTime.value - 15, 0, currentDuration.value || 0)
    const end = clampNumber(currentTime.value + 15, start, currentDuration.value || 0)
    updateSelectedRegion({
      start_time: Number(start.toFixed(1)),
      end_time: Number(end.toFixed(1))
    })
  }
  refreshMaskedPreviewIfNeeded()
}

function setSelectedRegionWindow(seconds) {
  if (!selectedRegion.value) {
    return
  }
  const start = clampNumber(currentTime.value - seconds, 0, currentDuration.value || 0)
  const end = clampNumber(currentTime.value + seconds, start, currentDuration.value || 0)
  updateSelectedRegion({
    start_time: Number(start.toFixed(1)),
    end_time: Number(end.toFixed(1))
  })
  refreshMaskedPreviewIfNeeded()
}

function setSelectedRegionFromCurrent() {
  if (!selectedRegion.value) {
    return
  }
  updateSelectedRegion({
    start_time: Number(currentTime.value.toFixed(1)),
    end_time: Number(currentDuration.value.toFixed(1))
  })
  refreshMaskedPreviewIfNeeded()
}

function setRegionRangeToFull(regionId) {
  updateRegion(regionId, {
    start_time: null,
    end_time: null
  })
  refreshMaskedPreviewIfNeeded()
}

function focusRegion(regionId) {
  selectedRegionId.value = regionId
}

function removeRegion(regionId) {
  regionsState[activeSource.value] = currentRegions.value.filter(region => region.id !== regionId)
  if (selectedRegionId.value === regionId) {
    selectedRegionId.value = null
  }
  refreshMaskedPreviewIfNeeded()
}

function toggleRegionEnabled(regionId, enabled) {
  updateRegion(regionId, { enabled: Boolean(enabled) })
  refreshMaskedPreviewIfNeeded()
}

function clearCurrentSource() {
  regionsState[activeSource.value] = []
  selectedRegionId.value = null
  refreshMaskedPreviewIfNeeded()
}

function copyToOtherSource() {
  const target = activeSource.value === 'movie' ? 'narration' : 'movie'
  const now = Date.now()
  regionsState[target] = currentRegions.value.map((region, index) => ({
    ...region,
    id: `${target}_${now}_${index}`
  }))
  ElMessage.success(`已复制到${sourceLabel(target)}`)
}

function applyPreset(name) {
  const preset = PRESETS[name]
  if (!preset) {
    return
  }
  const next = {
    id: createRegionId(),
    x: preset.x,
    y: preset.y,
    width: preset.width,
    height: preset.height,
    enabled: true,
    label: preset.label,
    start_time: null,
    end_time: null
  }
  regionsState[activeSource.value] = [...currentRegions.value, next]
  selectedRegionId.value = next.id
  refreshMaskedPreviewIfNeeded()
}

async function saveRegions(rematchAfterSave = false) {
  if (!props.project?.id) {
    return
  }
  saving.value = rematchAfterSave ? 'rematch' : 'save'
  try {
    const updated = await projectStore.updateSubtitleRegions(props.project.id, {
      subtitle_mask_mode: maskMode.value,
      narration_subtitle_regions: regionsState.narration,
      movie_subtitle_regions: regionsState.movie
    })
    emit('saved', updated)
    dialogVisible.value = false
    ElMessage.success(rematchAfterSave ? '区域已保存，开始重匹配' : '区域已保存')
    if (rematchAfterSave) {
      await projectStore.rematchProject(props.project.id, { preserve_manual_matches: true })
    }
  } finally {
    saving.value = ''
  }
}
</script>

<style scoped lang="scss">
.subtitle-dialog {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.dialog-tip {
  margin-bottom: 4px;
}

.toolbar,
.preset-row,
.range-toggle,
.range-shortcuts,
.region-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  align-items: center;
}

.mask-mode-select {
  width: 220px;
}

.workspace {
  display: grid;
  grid-template-columns: minmax(0, 1.85fr) minmax(320px, 0.9fr);
  gap: 18px;
}

.frame-panel,
.region-panel {
  border: 1px solid var(--el-border-color-light);
  border-radius: 12px;
  background: var(--el-bg-color);
}

.frame-panel {
  padding: 16px;
}

.region-panel {
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.frame-controls {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 170px;
  gap: 16px;
  align-items: center;
}

.frame-time,
.range-field {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.control-label {
  font-size: 13px;
  color: var(--el-text-color-secondary);
}

.source-meta,
.region-meta,
.region-item-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  color: var(--el-text-color-secondary);
  font-size: 13px;
}

.source-meta {
  margin-top: 10px;
}

.preset-row {
  margin-top: 12px;
}

.frame-stage {
  margin-top: 16px;
  display: flex;
  justify-content: center;
}

.frame-wrapper {
  position: relative;
  width: 100%;
  max-width: 800px;
  min-height: 240px;
  border-radius: 14px;
  overflow: hidden;
  background: linear-gradient(135deg, rgba(20, 28, 42, 0.95), rgba(48, 63, 88, 0.88));
  border: 1px solid rgba(255, 255, 255, 0.1);
  user-select: none;
}

.frame-wrapper.drawing {
  cursor: crosshair;
}

.frame-wrapper.loading {
  cursor: progress;
}

.frame-image {
  display: block;
  width: 100%;
  height: auto;
}

.frame-overlay {
  position: absolute;
  inset: 0;
}

.region-box {
  position: absolute;
  border: 2px solid rgba(255, 129, 82, 0.95);
  background: rgba(255, 129, 82, 0.14);
  box-shadow: 0 0 0 1px rgba(255, 255, 255, 0.08) inset;
  pointer-events: auto;
  cursor: move;
}

.region-box.selected {
  border-color: rgba(59, 130, 246, 0.95);
  background: rgba(59, 130, 246, 0.2);
}

.region-box.disabled,
.region-box.inactive {
  border-color: rgba(160, 176, 201, 0.88);
  background: rgba(160, 176, 201, 0.1);
}

.region-box.draft {
  border-style: dashed;
  background: rgba(59, 130, 246, 0.16);
  pointer-events: none;
}

.region-label {
  position: absolute;
  top: -1px;
  left: -1px;
  display: inline-flex;
  gap: 6px;
  padding: 2px 8px;
  font-size: 12px;
  line-height: 1.4;
  color: #fff;
  background: rgba(16, 24, 40, 0.88);
  border-bottom-right-radius: 8px;
  pointer-events: none;
}

.region-label small {
  opacity: 0.8;
}

.resize-handle {
  position: absolute;
  width: 10px;
  height: 10px;
  border-radius: 999px;
  background: #fff;
  border: 2px solid rgba(16, 24, 40, 0.92);
}

.handle-nw {
  left: -6px;
  top: -6px;
  cursor: nwse-resize;
}

.handle-ne {
  right: -6px;
  top: -6px;
  cursor: nesw-resize;
}

.handle-sw {
  left: -6px;
  bottom: -6px;
  cursor: nesw-resize;
}

.handle-se {
  right: -6px;
  bottom: -6px;
  cursor: nwse-resize;
}

.frame-placeholder {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 24px;
  color: rgba(255, 255, 255, 0.88);
  text-align: center;
  backdrop-filter: blur(4px);
}

.panel-section,
.editor-card {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.panel-title {
  font-size: 15px;
  font-weight: 700;
}

.panel-description,
.summary-item {
  color: var(--el-text-color-secondary);
  font-size: 13px;
}

.editor-card {
  padding: 14px;
  border-radius: 12px;
  background: var(--el-fill-color-light);
}

.range-inputs {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
}

.region-list {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.region-item {
  padding: 12px;
  border: 1px solid var(--el-border-color-lighter);
  border-radius: 10px;
  background: var(--el-fill-color-blank);
  cursor: pointer;
}

.region-item.selected {
  border-color: var(--el-color-primary);
  box-shadow: 0 0 0 1px rgba(64, 158, 255, 0.18);
}

.region-item.inactive {
  opacity: 0.78;
}

.region-item-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 10px;
}

.summary-section {
  padding-top: 8px;
  border-top: 1px solid var(--el-border-color-lighter);
}

.dialog-footer {
  display: flex;
  justify-content: flex-end;
  gap: 12px;
}

@media (max-width: 960px) {
  .workspace,
  .range-inputs,
  .frame-controls {
    grid-template-columns: 1fr;
  }
}
</style>
