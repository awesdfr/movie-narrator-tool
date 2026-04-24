import axios from 'axios'
import { ElMessage } from 'element-plus'

const api = axios.create({
  baseURL: '/api',
  timeout: 60000,
  headers: {
    'Content-Type': 'application/json'
  }
})

api.interceptors.request.use(
  config => config,
  error => Promise.reject(error)
)

api.interceptors.response.use(
  response => response.data,
  error => {
    const message = error.response?.data?.detail || error.message || 'Request failed'
    ElMessage.error(message)
    return Promise.reject(error)
  }
)

export const projectApi = {
  list: () => api.get('/project/list'),
  create: data => api.post('/project/create', data),
  get: id => api.get(`/project/${id}`),
  update: (id, data) => api.put(`/project/${id}`, data),
  updateSubtitleRegions: (id, data) => api.put(`/project/${id}/subtitle-regions`, data),
  delete: id => api.delete(`/project/${id}`),
  duplicate: (id, newName) => api.post(`/project/${id}/duplicate`, null, { params: { new_name: newName } })
}

export const processApi = {
  start: projectId => api.post(`/process/${projectId}/start`),
  stop: projectId => api.post(`/process/${projectId}/stop`),
  getProgress: projectId => api.get(`/process/${projectId}/progress`),
  getSegments: projectId => api.get(`/process/${projectId}/segments`),
  updateSegment: (projectId, segmentId, data) => api.put(`/process/${projectId}/segments/${segmentId}`, data),
  batchUpdateSegments: (projectId, data) => api.post(`/process/${projectId}/segments/batch`, data),
  rematchProject: (projectId, data = {}) => api.post(`/process/${projectId}/rematch`, data),
  rematchWeakProject: (projectId, data = {}) => api.post(`/process/${projectId}/rematch-weak`, data),
  resegment: (projectId, data = {}) => api.post(`/process/${projectId}/resegment`, data),
  rematchSegment: (projectId, segmentId, data = {}) => api.post(`/process/${projectId}/segments/${segmentId}/rematch`, data),
  regenerateTTS: (projectId, segmentId) => api.post(`/process/${projectId}/segments/${segmentId}/regenerate-tts`),
  repolish: (projectId, segmentId, data = {}) => api.post(`/process/${projectId}/segments/${segmentId}/repolish`, data),
  startPolish: (projectId, data = { style_preset: 'movie_pro' }) => api.post(`/process/${projectId}/start-polish`, data),
  batchGenerateTTS: projectId => api.post(`/process/${projectId}/generate-tts`)
}

export const previewApi = {
  getThumbnail: (projectId, segmentId) => `/api/preview/${projectId}/thumbnail/${segmentId}`,
  getFrame: (projectId, source, time, options = {}) => {
    const params = new URLSearchParams({
      source,
      time
    })
    if (options.masked) {
      params.set('masked', 'true')
      if (options.maskMode) {
        params.set('mask_mode', options.maskMode)
      }
      if (options.manualRegions) {
        params.set('manual_regions', JSON.stringify(options.manualRegions))
      }
    }
    return `/api/preview/${projectId}/frame?${params.toString()}`
  },
  getSourceInfo: (projectId, source) => api.get(`/preview/${projectId}/source-info?source=${source}`),
  getSegmentAudio: (projectId, segmentId, source = 'tts') => `/api/preview/${projectId}/segment/${segmentId}/audio?source=${source}`,
  getSegmentVideo: (projectId, segmentId, source = 'narration') => `/api/preview/${projectId}/segment/${segmentId}/video?source=${source}`,
  getCreativePlan: projectId => api.get(`/preview/${projectId}/creative-plan`),
  getMatchReportUrl: projectId => `/api/preview/${projectId}/export/report`,
  getDaVinciXmlUrl: projectId => `/api/preview/${projectId}/export/davinci`,
  evaluateBenchmark: (projectId, manifestPath = null) =>
    api.post(`/preview/${projectId}/benchmark/evaluate`, { manifest_path: manifestPath }),
  getBenchmarkReportUrl: projectId => `/api/preview/${projectId}/benchmark/report`,
  runVisualAudit: (projectId, data = {}) => api.post(`/preview/${projectId}/audit/visual`, data),
  getVisualAuditReportUrl: projectId => `/api/preview/${projectId}/audit/report`,
  getCommercialReadiness: projectId => api.get(`/preview/${projectId}/commercial-readiness`),
  exportJianying: (projectId, mode = 'creative_draft') =>
    api.post(`/preview/${projectId}/export/jianying`, null, { params: { mode } }),
  exportMaterialBasket: projectId => api.post(`/preview/${projectId}/export/material-basket`),
  exportSubtitle: (projectId, format = 'srt') => api.post(`/preview/${projectId}/export/subtitle?format=${format}`)
}

export const settingsApi = {
  get: () => api.get('/settings'),
  update: data => api.put('/settings', data),
  getAI: () => api.get('/settings/ai'),
  updateAI: data => api.put('/settings/ai', data),
  getTTS: () => api.get('/settings/tts'),
  updateTTS: data => api.put('/settings/tts', data),
  testAI: () => api.post('/settings/ai/test'),
  testTTS: () => api.post('/settings/tts/test'),
  checkFFmpeg: () => api.get('/settings/check_ffmpeg')
}

export const filesApi = {
  listMovies: () => api.get('/files/movies'),
  listNarrations: () => api.get('/files/narrations'),
  listReferenceAudio: () => api.get('/files/reference_audio'),
  listSubtitles: () => api.get('/files/subtitles'),
  openFolder: folderType => api.post('/files/open_folder', null, { params: { folder_type: folderType } }),
  validateVideo: path => api.post('/files/validate_video', null, { params: { path } }),
  getVideosDir: () => api.get('/files/videos_dir')
}

export default api
