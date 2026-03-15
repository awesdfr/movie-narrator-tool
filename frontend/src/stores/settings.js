import { computed, ref } from 'vue'
import { defineStore } from 'pinia'
import { settingsApi } from '@/api'

function createDefaultSettings() {
  return {
    ai: {
      api_base: 'https://api.openai.com/v1',
      api_key: '',
      model: 'gpt-4o',
      max_tokens: 2000,
      temperature: 0.4,
      polish_style_preset: 'movie_pro',
      enable_de_ai_pass: true,
      enable_self_review: true,
      polish_template: ''
    },
    tts: {
      api_base: 'http://127.0.0.1:7860',
      api_endpoint: '/gradio_api/call/gen_single',
      reference_audio: '',
      speed: 1.0,
      infer_mode: '批次推理'
    },
    segmentation: {
      min_segment_duration: 1.2,
      max_segment_duration: 8.0,
      split_pause_seconds: 0.55,
      merge_gap_seconds: 0.35,
      sentence_snap_tolerance: 0.4,
      enable_scene_snap: true,
      prefer_word_timestamps: true
    },
    match: {
      frame_match_threshold: 0.65,
      phash_threshold: 8,
      phash_strict_threshold: 5,
      phash_loose_threshold: 10,
      scene_threshold: 30,
      use_deep_learning: true,
      sample_interval: 5,
      index_sample_fps: 8,
      fast_mode: false,
      use_multi_scale_hash: true,
      use_sequence_alignment: true,
      use_dynamic_sampling: true,
      use_prefilter: true,
      high_confidence_threshold: 0.86,
      medium_confidence_threshold: 0.72,
      low_confidence_threshold: 0.55,
      candidate_top_k: 4,
      allow_non_sequential: true,
      use_lis_filter: false,
      rerank_low_confidence: true,
      use_multimodal_rerank: false,
      global_backtrack_penalty: 1.4,
      duplicate_scene_penalty: 0.5
    },
    voiceprint: {
      threshold: 0.75,
      min_speech_duration: 0.5
    },
    whisper: {
      model: 'medium',
      device: 'cuda',
      language: 'zh',
      word_timestamps: true
    },
    export: {
      jianying_drafts_dir: 'D:\\my_video_project\\jianyingcaogao\\JianyingPro Drafts',
      output_fps: 0,
      output_resolution: 'original',
      audio_source: 'original',
      export_subtitles: true,
      subtitle_format: 'srt',
      min_playback_speed: 0.5,
      max_playback_speed: 2.0
    },
    ui: {
      language: 'zh-CN',
      theme: 'light'
    },
    concurrency: {
      polish_concurrency: 5,
      tts_concurrency: 5,
      match_concurrency: 4
    }
  }
}

function mergeSettings(target, source) {
  Object.entries(source || {}).forEach(([key, value]) => {
    if (value && typeof value === 'object' && !Array.isArray(value) && target[key]) {
      target[key] = { ...target[key], ...value }
      return
    }
    target[key] = value
  })
  return target
}

export const useSettingsStore = defineStore('settings', () => {
  const settings = ref(createDefaultSettings())
  const loading = ref(false)

  async function fetchSettings() {
    loading.value = true
    try {
      settings.value = mergeSettings(createDefaultSettings(), await settingsApi.get())
    } finally {
      loading.value = false
    }
  }

  async function updateSettings(data) {
    loading.value = true
    try {
      settings.value = mergeSettings(createDefaultSettings(), await settingsApi.update(data))
    } finally {
      loading.value = false
    }
  }

  async function updateAISettings(data) {
    settings.value.ai = await settingsApi.updateAI(data)
  }

  async function updateTTSSettings(data) {
    settings.value.tts = await settingsApi.updateTTS(data)
  }

  async function testAIConnection() {
    return settingsApi.testAI()
  }

  async function testTTSConnection() {
    return settingsApi.testTTS()
  }

  const language = computed(() => settings.value.ui.language)
  const theme = computed(() => settings.value.ui.theme)

  return {
    settings,
    loading,
    language,
    theme,
    fetchSettings,
    updateSettings,
    updateAISettings,
    updateTTSSettings,
    testAIConnection,
    testTTSConnection
  }
})
