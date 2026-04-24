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
  rematchProject: (projectId, data = {}) => api.post(`/process/${projectId}/rematch`, data),
  rematchWeakProject: (projectId, data = {}) => api.post(`/process/${projectId}/rematch-weak`, data),
  resegment: (projectId, data = {}) => api.post(`/process/${projectId}/resegment`, data),
  rematchSegment: (projectId, segmentId, data = {}) => api.post(`/process/${projectId}/segments/${segmentId}/rematch`, data)
}

export const matchApi = {
  createJob: data => api.post('/match/jobs', data),
  getJob: projectId => api.get(`/match/jobs/${projectId}`),
  getSegments: projectId => api.get(`/match/jobs/${projectId}/segments`),
  exportJianying: projectId => api.post(`/match/jobs/${projectId}/export/jianying`)
}

export const previewApi = {
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
  getMatchReportUrl: projectId => `/api/preview/${projectId}/export/report`,
  runVisualAudit: (projectId, data = {}) => api.post(`/preview/${projectId}/audit/visual`, data),
  getVisualAuditReportUrl: projectId => `/api/preview/${projectId}/audit/report`,
  exportJianying: (projectId, mode = 'restore_draft') =>
    api.post(`/preview/${projectId}/export/jianying`, null, { params: { mode } })
}

export const filesApi = {
  listMovies: () => api.get('/files/movies'),
  listNarrations: () => api.get('/files/narrations'),
  listSubtitles: () => api.get('/files/subtitles'),
  openFolder: folderType => api.post('/files/open_folder', null, { params: { folder_type: folderType } }),
}

export default api
