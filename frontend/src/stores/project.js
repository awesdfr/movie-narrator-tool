import { defineStore } from 'pinia'
import { computed, ref } from 'vue'
import { processApi, projectApi } from '@/api'
import wsService from '@/api/websocket'

const TERMINAL_STAGES = ['completed', 'error', 'ready_for_tts', 'ready_for_polish']
const ACTIVE_PROJECT_STATUSES = ['analyzing', 'matching', 'recognizing', 'polishing', 'generating_tts']

export const useProjectStore = defineStore('project', () => {
  const projects = ref([])
  const currentProject = ref(null)
  const segments = ref([])
  const loading = ref(false)
  const processing = ref(false)
  const progress = ref({ stage: '', progress: 0, message: '' })

  let progressPollTimer = null
  let removeProgressListener = null

  const segmentsByType = computed(() => {
    const result = {
      has_narration: [],
      no_narration: [],
      non_movie: []
    }
    segments.value.forEach(segment => {
      if (result[segment.segment_type]) {
        result[segment.segment_type].push(segment)
      }
    })
    return result
  })

  const selectedSegments = computed(() => segments.value.filter(segment => segment.use_segment))

  async function fetchProjects() {
    loading.value = true
    try {
      projects.value = await projectApi.list()
    } finally {
      loading.value = false
    }
  }

  async function fetchProject(id) {
    loading.value = true
    try {
      currentProject.value = await projectApi.get(id)
      segments.value = currentProject.value.segments || []
      syncProgressFromProject(currentProject.value)
      if (processing.value) {
        await connectProgress(id, currentProject.value.progress?.stage, currentProject.value.progress?.message)
      }
    } finally {
      loading.value = false
    }
  }

  async function createProject(data) {
    const project = await projectApi.create(data)
    projects.value.unshift(project)
    return project
  }

  async function deleteProject(id) {
    await projectApi.delete(id)
    projects.value = projects.value.filter(project => project.id !== id)
    if (currentProject.value?.id === id) {
      currentProject.value = null
      segments.value = []
      stopProgressTracking()
    }
  }

  async function updateSubtitleRegions(projectId, data) {
    const updated = await projectApi.updateSubtitleRegions(projectId, data)
    if (currentProject.value?.id === projectId) {
      currentProject.value = updated
      segments.value = updated.segments || []
      syncProgressFromProject(updated)
    }
    projects.value = projects.value.map(project => (project.id === projectId ? { ...project, ...updated } : project))
    return updated
  }

  async function connectProgress(projectId, initialStage, initialMessage) {
    if (removeProgressListener && wsService.projectId === projectId && processing.value) {
      return
    }

    stopProgressTracking()
    processing.value = true
    progress.value = {
      stage: initialStage || currentProject.value?.progress?.stage || '',
      progress: currentProject.value?.progress?.progress || 0,
      message: initialMessage || currentProject.value?.progress?.message || ''
    }

    await wsService.connect(projectId)

    removeProgressListener = wsService.on('progress', data => {
      applyProgressState(projectId, {
        stage: data.stage,
        progress: data.progress,
        message: data.message
      })

      if (TERMINAL_STAGES.includes(data.stage)) {
        stopProgressTracking()
        fetchProject(projectId)
      }
    })

    startProgressPolling(projectId)
  }

  async function startProcessing(projectId) {
    await connectProgress(projectId, 'starting', 'Starting processing...')
    await processApi.start(projectId)
  }

  async function stopProcessing(projectId) {
    await processApi.stop(projectId)
    stopProgressTracking()
    await fetchProject(projectId)
  }

  async function updateSegment(projectId, segmentId, data) {
    const updated = await processApi.updateSegment(projectId, segmentId, data)
    const index = segments.value.findIndex(segment => segment.id === segmentId)
    if (index > -1) {
      segments.value[index] = { ...segments.value[index], ...updated }
    }
    return updated
  }

  async function batchUpdateSegments(projectId, segmentIds, data) {
    const result = await processApi.batchUpdateSegments(projectId, {
      segment_ids: segmentIds,
      ...data
    })
    const updatedSegments = result.segments || []
    segments.value.forEach(segment => {
      const updated = updatedSegments.find(item => item.id === segment.id)
      if (updated) {
        Object.assign(segment, updated)
      } else if (segmentIds.includes(segment.id)) {
        Object.assign(segment, data)
      }
    })
    return result
  }

  async function rematchProject(projectId, data = { preserve_manual_matches: true }) {
    await connectProgress(projectId, 'matching', 'Rematching eligible segments...')
    await processApi.rematchProject(projectId, data)
  }

  async function resegment(projectId, data = { preserve_manual_matches: true }) {
    const result = await processApi.resegment(projectId, data)
    segments.value = result.segments || []
    if (currentProject.value?.id === projectId) {
      currentProject.value = { ...currentProject.value, segments: segments.value }
      syncProgressFromProject(currentProject.value)
    }
    return result
  }

  async function rematchSegment(projectId, segmentId, data = {}) {
    const updated = await processApi.rematchSegment(projectId, segmentId, data)
    const index = segments.value.findIndex(segment => segment.id === segmentId)
    if (index > -1) {
      segments.value[index] = { ...segments.value[index], ...updated }
    }
    return updated
  }

  async function startPolishing(projectId, stylePreset = 'movie_pro') {
    await connectProgress(projectId, 'polishing', 'Polishing narration...')
    await processApi.startPolish(projectId, { style_preset: stylePreset })
  }

  async function batchGenerateTTS(projectId) {
    await connectProgress(projectId, 'generating_tts', 'Generating TTS...')
    await processApi.batchGenerateTTS(projectId)
  }

  function clearCurrentProject() {
    currentProject.value = null
    segments.value = []
    stopProgressTracking()
  }

  function syncProgressFromProject(project) {
    const status = project?.status || ''
    const currentProgress = project?.progress || { stage: '', progress: 0, message: '' }
    progress.value = {
      stage: currentProgress.stage || status,
      progress: currentProgress.progress || 0,
      message: currentProgress.message || ''
    }
    processing.value = ACTIVE_PROJECT_STATUSES.includes(status)
  }

  function applyProgressState(projectId, nextProgress, status = nextProgress.stage) {
    progress.value = {
      stage: nextProgress.stage || '',
      progress: nextProgress.progress || 0,
      message: nextProgress.message || ''
    }
    processing.value = ACTIVE_PROJECT_STATUSES.includes(status)

    if (currentProject.value?.id === projectId) {
      currentProject.value = {
        ...currentProject.value,
        status,
        progress: { ...progress.value }
      }
    }
  }

  function startProgressPolling(projectId) {
    stopProgressPolling()
    progressPollTimer = window.setInterval(async () => {
      try {
        const response = await processApi.getProgress(projectId)
        applyProgressState(projectId, response.progress || {}, response.status)

        if (TERMINAL_STAGES.includes(response.status) || TERMINAL_STAGES.includes(response.progress?.stage)) {
          stopProgressTracking()
          await fetchProject(projectId)
        }
      } catch (error) {
        console.warn('Progress polling failed', error)
      }
    }, 3000)
  }

  function stopProgressPolling() {
    if (progressPollTimer != null) {
      window.clearInterval(progressPollTimer)
      progressPollTimer = null
    }
  }

  function stopProgressTracking() {
    processing.value = false
    stopProgressPolling()
    if (removeProgressListener) {
      removeProgressListener()
      removeProgressListener = null
    }
    wsService.disconnect()
  }

  return {
    projects,
    currentProject,
    segments,
    loading,
    processing,
    progress,
    segmentsByType,
    selectedSegments,
    fetchProjects,
    fetchProject,
    createProject,
    deleteProject,
    updateSubtitleRegions,
    startProcessing,
    stopProcessing,
    updateSegment,
    batchUpdateSegments,
    rematchProject,
    resegment,
    rematchSegment,
    startPolishing,
    batchGenerateTTS,
    clearCurrentProject
  }
})
