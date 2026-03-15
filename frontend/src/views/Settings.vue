<template>
  <div class="settings-page" v-loading="settingsStore.loading">

    <div class="page-header">
      <div>
        <h1 class="page-title">设置</h1>
        <p class="page-subtitle">配置分段、匹配策略、润色和导出参数</p>
      </div>
      <el-button type="primary" size="large" @click="saveSettings" :loading="saving">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" style="margin-right:6px">
          <path d="M19 21H5a2 2 0 01-2-2V5a2 2 0 012-2h11l5 5v11a2 2 0 01-2 2z"
            stroke="currentColor" stroke-width="2" stroke-linejoin="round"/>
          <polyline points="17 21 17 13 7 13 7 21" stroke="currentColor" stroke-width="2"/>
          <polyline points="7 3 7 8 15 8" stroke="currentColor" stroke-width="2"/>
        </svg>
        保存设置
      </el-button>
    </div>

    <!-- Environment status -->
    <div class="env-card">
      <div class="env-header">
        <span class="env-title">运行环境</span>
        <button class="text-btn" @click="refreshFFmpeg" :disabled="ffmpegLoading">
          {{ ffmpegLoading ? '检测中...' : '重新检测' }}
        </button>
      </div>
      <div class="env-body">
        <div class="env-item">
          <span class="env-label">FFmpeg</span>
          <span v-if="ffmpegLoading" class="env-pending">检测中…</span>
          <span v-else-if="ffmpegStatus.installed" class="env-ok">
            ✓ 已就绪 {{ ffmpegStatus.version || '' }}
          </span>
          <span v-else class="env-err">✗ 未安装或不可用</span>
        </div>
        <div v-if="ffmpegStatus.path" class="env-item">
          <span class="env-label">路径</span>
          <span class="env-path">{{ ffmpegStatus.path }}</span>
        </div>
      </div>
    </div>

    <!-- Tabs -->
    <el-tabs v-model="activeTab" class="settings-tabs">

      <!-- AI Polish -->
      <el-tab-pane label="AI 润色" name="ai">
        <div class="tab-grid">
          <el-card>
            <template #header><span>API 配置</span></template>
            <el-form label-width="140px" label-position="left">
              <el-form-item label="API 地址">
                <el-input v-model="settings.ai.api_base" placeholder="https://api.openai.com/v1" />
              </el-form-item>
              <el-form-item label="API Key">
                <el-input v-model="settings.ai.api_key" type="password" show-password placeholder="sk-..." />
              </el-form-item>
              <el-form-item label="模型">
                <el-select v-model="settings.ai.model" style="width:100%" filterable allow-create>
                  <el-option label="deepseek-chat"      value="deepseek-chat" />
                  <el-option label="gpt-4o"             value="gpt-4o" />
                  <el-option label="gpt-4.1"            value="gpt-4.1" />
                  <el-option label="claude-3-7-sonnet"  value="claude-3-7-sonnet" />
                </el-select>
              </el-form-item>
              <el-form-item label="温度">
                <el-slider v-model="settings.ai.temperature" :min="0" :max="1.2" :step="0.05" show-input />
              </el-form-item>
              <el-form-item label="最大 tokens">
                <el-input-number v-model="settings.ai.max_tokens" :min="200" :max="8000" />
              </el-form-item>
              <el-form-item>
                <el-button type="primary" @click="testAI" :loading="testingAI">测试 AI 连接</el-button>
                <span v-if="aiTestResult" :class="['test-result', aiTestResult.success ? 'ok' : 'fail']">
                  {{ aiTestResult.message }}
                </span>
              </el-form-item>
            </el-form>
          </el-card>

          <el-card>
            <template #header><span>润色配置</span></template>
            <el-form label-width="140px" label-position="left">
              <el-form-item label="风格 preset">
                <el-select v-model="settings.ai.polish_style_preset" style="width:100%">
                  <el-option label="movie_pro"          value="movie_pro" />
                  <el-option label="natural_story"      value="natural_story" />
                  <el-option label="tight_commentary"   value="tight_commentary" />
                </el-select>
                <div class="form-tip">推荐 movie_pro，贴近专业电影解说口吻</div>
              </el-form-item>
              <el-form-item label="去 AI 味预处理">
                <el-switch v-model="settings.ai.enable_de_ai_pass" />
                <div class="form-tip">清理模板腔和过度工整句式，再做正式润色</div>
              </el-form-item>
              <el-form-item label="润色后自检">
                <el-switch v-model="settings.ai.enable_self_review" />
                <div class="form-tip">检查重复短语、时长超预算等问题，自动重写一次</div>
              </el-form-item>
              <el-form-item label="润色模板">
                <el-input v-model="settings.ai.polish_template" type="textarea" :rows="6"
                  placeholder="使用 {text} 作为原文占位符" />
              </el-form-item>
            </el-form>
          </el-card>
        </div>
      </el-tab-pane>

      <!-- Segmentation / ASR -->
      <el-tab-pane label="分段 / ASR" name="segmentation">
        <div class="tab-grid">
          <el-card>
            <template #header><span>分段策略</span></template>
            <el-form label-width="150px" label-position="left">
              <el-form-item label="最短片段时长">
                <el-slider v-model="settings.segmentation.min_segment_duration" :min="0.4" :max="4" :step="0.1" show-input />
              </el-form-item>
              <el-form-item label="最长片段时长">
                <el-slider v-model="settings.segmentation.max_segment_duration" :min="4" :max="16" :step="0.5" show-input />
              </el-form-item>
              <el-form-item label="停顿切分阈值">
                <el-slider v-model="settings.segmentation.split_pause_seconds" :min="0.15" :max="2" :step="0.05" show-input />
              </el-form-item>
              <el-form-item label="碎片合并间隔">
                <el-slider v-model="settings.segmentation.merge_gap_seconds" :min="0" :max="1.5" :step="0.05" show-input />
              </el-form-item>
              <el-form-item label="句边界吸附容差">
                <el-slider v-model="settings.segmentation.sentence_snap_tolerance" :min="0" :max="1.5" :step="0.05" show-input />
              </el-form-item>
              <el-form-item label="场景切换吸附">
                <el-switch v-model="settings.segmentation.enable_scene_snap" />
                <div class="form-tip">边界优先吸附到镜头切点，减少一段解说跨两个画面的问题</div>
              </el-form-item>
              <el-form-item label="优先词级时间戳">
                <el-switch v-model="settings.segmentation.prefer_word_timestamps" />
              </el-form-item>
            </el-form>
          </el-card>

          <el-card>
            <template #header><span>Whisper / 声纹</span></template>
            <el-form label-width="150px" label-position="left">
              <el-form-item label="Whisper 模型">
                <el-select v-model="settings.whisper.model" style="width:100%">
                  <el-option label="small"    value="small" />
                  <el-option label="medium"   value="medium" />
                  <el-option label="large-v3" value="large-v3" />
                </el-select>
              </el-form-item>
              <el-form-item label="设备">
                <el-select v-model="settings.whisper.device" style="width:100%" allow-create filterable>
                  <el-option label="cuda" value="cuda" />
                  <el-option label="cpu"  value="cpu" />
                </el-select>
              </el-form-item>
              <el-form-item label="解说语言">
                <el-select v-model="settings.whisper.language" style="width:100%" allow-create filterable placeholder="选择解说语言">
                  <el-option label="中文 (zh)" value="zh" />
                  <el-option label="English (en)" value="en" />
                  <el-option label="日本語 (ja)" value="ja" />
                  <el-option label="한국어 (ko)" value="ko" />
                  <el-option label="Français (fr)" value="fr" />
                  <el-option label="Deutsch (de)" value="de" />
                  <el-option label="Español (es)" value="es" />
                  <el-option label="Русский (ru)" value="ru" />
                  <el-option label="Italiano (it)" value="it" />
                  <el-option label="Português (pt)" value="pt" />
                  <el-option label="العربية (ar)" value="ar" />
                  <el-option label="ภาษาไทย (th)" value="th" />
                  <el-option label="Tiếng Việt (vi)" value="vi" />
                  <el-option label="自动检测 (auto)" value="auto" />
                </el-select>
                <div class="form-tip">影响语音识别(ASR)和文案润色语言</div>
              </el-form-item>
              <el-form-item label="词级时间戳">
                <el-switch v-model="settings.whisper.word_timestamps" />
              </el-form-item>
              <el-divider content-position="left" style="margin:16px 0">声纹</el-divider>
              <el-form-item label="声纹阈值">
                <el-slider v-model="settings.voiceprint.threshold" :min="0.4" :max="0.95" :step="0.01" show-input />
              </el-form-item>
              <el-form-item label="最小语音时长">
                <el-slider v-model="settings.voiceprint.min_speech_duration" :min="0.2" :max="2" :step="0.05" show-input />
              </el-form-item>
            </el-form>
          </el-card>
        </div>
      </el-tab-pane>

      <!-- Matching -->
      <el-tab-pane label="匹配策略" name="match">
        <div class="tab-grid">
          <el-card>
            <template #header><span>置信度阈值</span></template>
            <el-alert type="warning" :closable="false" show-icon class="inner-alert"
              title="准确率优先"
              description="推荐先保持默认，再基于 benchmark 微调。" />
            <el-form label-width="150px" label-position="left" style="margin-top:16px">
              <el-form-item label="自动通过阈值">
                <el-slider v-model="settings.match.high_confidence_threshold" :min="0.5" :max="0.98" :step="0.01" show-input />
              </el-form-item>
              <el-form-item label="待复核阈值">
                <el-slider v-model="settings.match.medium_confidence_threshold" :min="0.3" :max="0.95" :step="0.01" show-input />
              </el-form-item>
              <el-form-item label="低置信截止">
                <el-slider v-model="settings.match.low_confidence_threshold" :min="0.1" :max="0.9" :step="0.01" show-input />
              </el-form-item>
              <el-form-item label="每段候选数">
                <el-input-number v-model="settings.match.candidate_top_k" :min="1" :max="10" />
                <div class="form-tip">推荐 6。越多精度上限越高，但速度略慢</div>
              </el-form-item>
              <el-form-item label="全局回跳惩罚">
                <el-slider v-model="settings.match.global_backtrack_penalty" :min="0" :max="5" :step="0.1" show-input />
              </el-form-item>
              <el-form-item label="重复场景惩罚">
                <el-slider v-model="settings.match.duplicate_scene_penalty" :min="0" :max="3" :step="0.05" show-input />
              </el-form-item>
            </el-form>
          </el-card>

          <el-card>
            <template #header><span>特征提取</span></template>
            <el-form label-width="150px" label-position="left">
              <el-form-item label="视觉匹配阈值">
                <el-slider v-model="settings.match.frame_match_threshold" :min="0.4" :max="1" :step="0.01" show-input />
              </el-form-item>
              <el-form-item label="pHash 阈值">
                <el-input-number v-model="settings.match.phash_threshold" :min="1" :max="30" />
              </el-form-item>
              <el-form-item label="严格 pHash">
                <el-input-number v-model="settings.match.phash_strict_threshold" :min="1" :max="20" />
              </el-form-item>
              <el-form-item label="宽松 pHash">
                <el-input-number v-model="settings.match.phash_loose_threshold" :min="1" :max="30" />
              </el-form-item>
              <el-form-item label="场景阈值">
                <el-slider v-model="settings.match.scene_threshold" :min="10" :max="80" :step="1" show-input />
              </el-form-item>
              <el-form-item label="采样间隔 (s)">
                <el-slider v-model="settings.match.sample_interval" :min="1" :max="30" :step="1" show-input />
                <div class="form-tip">默认 1 秒/帧，值越小精度越高但索引越慢</div>
              </el-form-item>
              <el-form-item label="深度特征">
                <el-switch v-model="settings.match.use_deep_learning" />
              </el-form-item>
              <el-form-item label="多尺度哈希">
                <el-switch v-model="settings.match.use_multi_scale_hash" />
              </el-form-item>
              <el-form-item label="序列对齐精排">
                <el-switch v-model="settings.match.use_sequence_alignment" />
              </el-form-item>
              <el-form-item label="低置信二次精排">
                <el-switch v-model="settings.match.rerank_low_confidence" />
              </el-form-item>
              <el-form-item label="多模态 rerank">
                <el-switch v-model="settings.match.use_multimodal_rerank" />
                <div class="form-tip">仅对低置信片段启用，需配置多模态 API</div>
              </el-form-item>
              <el-form-item label="允许非顺序">
                <el-switch v-model="settings.match.allow_non_sequential" />
              </el-form-item>
              <el-form-item label="快速模式">
                <el-switch v-model="settings.match.fast_mode" />
                <div class="form-tip">临时预跑时开启，追求准确率请关闭</div>
              </el-form-item>
            </el-form>
          </el-card>
        </div>
      </el-tab-pane>

      <!-- TTS -->
      <el-tab-pane label="TTS" name="tts">
        <div class="tab-single">
          <el-card>
            <template #header><span>TTS 服务配置</span></template>
            <el-form label-width="140px" label-position="left">
              <el-form-item label="服务地址">
                <el-input v-model="settings.tts.api_base" placeholder="http://127.0.0.1:7860" />
              </el-form-item>
              <el-form-item label="API 端点">
                <el-select v-model="settings.tts.api_endpoint" style="width:100%" filterable allow-create>
                  <el-option label="/gradio_api/call/gen_single" value="/gradio_api/call/gen_single" />
                  <el-option label="/gradio_api/call/gen_batch"  value="/gradio_api/call/gen_batch" />
                  <el-option label="/tts"                       value="/tts" />
                  <el-option label="/v1/audio/speech"           value="/v1/audio/speech" />
                </el-select>
              </el-form-item>
              <el-form-item label="参考音频">
                <el-select v-model="settings.tts.reference_audio" style="width:100%"
                  filterable allow-create clearable placeholder="选择或直接输入路径">
                  <el-option v-for="item in referenceAudioOptions" :key="item.path"
                    :label="`${item.name} (${item.size_display})`" :value="item.path" />
                </el-select>
                <button class="text-btn" style="margin-top:6px" @click="loadReferenceAudios">刷新列表</button>
              </el-form-item>
              <el-form-item label="语速">
                <el-slider v-model="settings.tts.speed" :min="0.5" :max="2" :step="0.05" show-input />
              </el-form-item>
              <el-form-item label="推理模式">
                <el-radio-group v-model="settings.tts.infer_mode">
                  <el-radio label="普通推理">普通推理</el-radio>
                  <el-radio label="批次推理">批次推理</el-radio>
                </el-radio-group>
              </el-form-item>
              <el-form-item>
                <el-button type="primary" @click="testTTS" :loading="testingTTS">测试 TTS 连接</el-button>
                <span v-if="ttsTestResult" :class="['test-result', ttsTestResult.success ? 'ok' : 'fail']">
                  {{ ttsTestResult.message }}
                </span>
              </el-form-item>
            </el-form>
          </el-card>
        </div>
      </el-tab-pane>

      <!-- Export -->
      <el-tab-pane label="导出" name="export">
        <div class="tab-single">
          <el-card>
            <template #header><span>导出配置</span></template>
            <el-form label-width="140px" label-position="left">
              <el-form-item label="剪映草稿目录">
                <el-input v-model="settings.export.jianying_drafts_dir" placeholder="留空则使用系统默认路径" />
              </el-form-item>
              <el-form-item label="输出帧率">
                <el-select v-model="settings.export.output_fps" style="width:220px">
                  <el-option label="保持原始帧率" :value="0" />
                  <el-option label="24 fps"       :value="24" />
                  <el-option label="25 fps"       :value="25" />
                  <el-option label="30 fps"       :value="30" />
                  <el-option label="60 fps"       :value="60" />
                </el-select>
              </el-form-item>
              <el-form-item label="输出分辨率">
                <el-select v-model="settings.export.output_resolution" style="width:220px">
                  <el-option label="保持原始" value="original" />
                  <el-option label="3840×2160" value="3840x2160" />
                  <el-option label="1920×1080" value="1920x1080" />
                  <el-option label="1280×720"  value="1280x720" />
                </el-select>
              </el-form-item>
              <el-form-item label="音频来源">
                <el-radio-group v-model="settings.export.audio_source">
                  <el-radio label="original">原始解说音频</el-radio>
                  <el-radio label="tts">TTS 生成音频</el-radio>
                </el-radio-group>
              </el-form-item>
              <el-form-item label="最小播放速度">
                <el-slider v-model="settings.export.min_playback_speed" :min="0.25" :max="1" :step="0.05" show-input />
              </el-form-item>
              <el-form-item label="最大播放速度">
                <el-slider v-model="settings.export.max_playback_speed" :min="1" :max="4" :step="0.1" show-input />
              </el-form-item>
              <el-form-item label="导出字幕">
                <el-switch v-model="settings.export.export_subtitles" />
              </el-form-item>
              <el-form-item label="字幕格式" v-if="settings.export.export_subtitles">
                <el-radio-group v-model="settings.export.subtitle_format">
                  <el-radio label="srt">SRT</el-radio>
                  <el-radio label="ass">ASS</el-radio>
                </el-radio-group>
              </el-form-item>
            </el-form>
          </el-card>
        </div>
      </el-tab-pane>

      <!-- Concurrency / UI -->
      <el-tab-pane label="系统" name="system">
        <div class="tab-single">
          <el-card>
            <template #header><span>并发 &amp; 界面</span></template>
            <el-form label-width="140px" label-position="left">
              <el-form-item label="AI 润色并发">
                <el-slider v-model="settings.concurrency.polish_concurrency" :min="1" :max="20" :step="1" show-input />
              </el-form-item>
              <el-form-item label="TTS 并发">
                <el-slider v-model="settings.concurrency.tts_concurrency" :min="1" :max="10" :step="1" show-input />
              </el-form-item>
              <el-form-item label="匹配并发">
                <el-slider v-model="settings.concurrency.match_concurrency" :min="1" :max="10" :step="1" show-input />
              </el-form-item>
              <el-divider content-position="left" style="margin:20px 0 16px">界面</el-divider>
              <el-form-item label="语言">
                <el-select v-model="settings.ui.language" style="width:220px">
                  <el-option label="简体中文" value="zh-CN" />
                  <el-option label="English"  value="en" />
                </el-select>
              </el-form-item>
            </el-form>
          </el-card>
        </div>
      </el-tab-pane>

    </el-tabs>
  </div>
</template>

<script setup>
import { onMounted, reactive, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { filesApi, settingsApi } from '@/api'
import { useSettingsStore } from '@/stores/settings'

const settingsStore = useSettingsStore()

const activeTab    = ref('ai')
const saving       = ref(false)
const testingAI    = ref(false)
const testingTTS   = ref(false)
const aiTestResult  = ref(null)
const ttsTestResult = ref(null)
const ffmpegLoading = ref(false)
const ffmpegStatus  = ref({ installed: false, version: null, path: null })
const referenceAudioOptions = ref([])

function createDefaultSettings() {
  return {
    ai: { api_base:'https://api.openai.com/v1', api_key:'', model:'gpt-4o', max_tokens:2000,
      temperature:0.4, polish_style_preset:'movie_pro', enable_de_ai_pass:true,
      enable_self_review:true, polish_template:'' },
    tts: { api_base:'http://127.0.0.1:7860', api_endpoint:'/gradio_api/call/gen_single',
      reference_audio:'', speed:1.0, infer_mode:'批次推理' },
    segmentation: { min_segment_duration:1.2, max_segment_duration:8.0, split_pause_seconds:0.55,
      merge_gap_seconds:0.35, sentence_snap_tolerance:0.4, enable_scene_snap:true, prefer_word_timestamps:true },
    match: { frame_match_threshold:0.65, phash_threshold:8, phash_strict_threshold:5, phash_loose_threshold:10,
      scene_threshold:30, use_deep_learning:true, sample_interval:1, index_sample_fps:8, fast_mode:false,
      use_multi_scale_hash:true, use_sequence_alignment:true, use_dynamic_sampling:true, use_prefilter:true,
      high_confidence_threshold:0.86, medium_confidence_threshold:0.72, low_confidence_threshold:0.55,
      candidate_top_k:6, allow_non_sequential:true, use_lis_filter:false, rerank_low_confidence:true,
      use_multimodal_rerank:false, global_backtrack_penalty:1.4, duplicate_scene_penalty:0.5 },
    voiceprint: { threshold:0.75, min_speech_duration:0.5 },
    whisper:    { model:'medium', device:'cuda', language:'zh', word_timestamps:true },
    export:     { jianying_drafts_dir:'', output_fps:0, output_resolution:'original', audio_source:'original',
      export_subtitles:true, subtitle_format:'srt', min_playback_speed:0.5, max_playback_speed:2.0 },
    ui:          { language:'zh-CN', theme:'dark' },
    concurrency: { polish_concurrency:5, tts_concurrency:5, match_concurrency:4 },
  }
}

const settings = reactive(createDefaultSettings())

function mergeSettings(target, source) {
  Object.entries(source || {}).forEach(([key, value]) => {
    if (value && typeof value === 'object' && !Array.isArray(value) && target[key]) {
      Object.assign(target[key], value)
    } else { target[key] = value }
  })
}

function fillNullsWithDefaults(target, defaults) {
  Object.entries(defaults).forEach(([key, value]) => {
    if (target[key] == null) { target[key] = value; return }
    if (value && typeof value === 'object' && !Array.isArray(value) && typeof target[key] === 'object') {
      fillNullsWithDefaults(target[key], value)
    }
  })
}

function normalizeOptionalText(v)        { return typeof v === 'string' && v.trim() === '' ? null : v }
function normalizeReferenceAudioValue(v) {
  if (!v) return null
  if (typeof v === 'string') return normalizeOptionalText(v)
  if (typeof v === 'object' && typeof v.path === 'string') return normalizeOptionalText(v.path)
  return null
}

function serializeSettings() {
  const payload = JSON.parse(JSON.stringify(settings))
  fillNullsWithDefaults(payload, createDefaultSettings())
  payload.ai.api_key             = normalizeOptionalText(payload.ai.api_key)
  payload.tts.reference_audio    = normalizeReferenceAudioValue(payload.tts.reference_audio)
  return payload
}

async function refreshFFmpeg() {
  ffmpegLoading.value = true
  try { ffmpegStatus.value = await settingsApi.checkFFmpeg() }
  catch { ffmpegStatus.value = { installed: false, version: null, path: null } }
  finally { ffmpegLoading.value = false }
}

async function loadReferenceAudios() {
  try {
    referenceAudioOptions.value = await filesApi.listReferenceAudio()
    settings.tts.reference_audio = normalizeReferenceAudioValue(settings.tts.reference_audio)
  } catch { referenceAudioOptions.value = [] }
}

onMounted(async () => {
  await settingsStore.fetchSettings()
  mergeSettings(settings, settingsStore.settings)
  await Promise.all([refreshFFmpeg(), loadReferenceAudios()])
})

async function saveSettings() {
  saving.value = true
  try { await settingsStore.updateSettings(serializeSettings()); ElMessage.success('设置已保存') }
  finally { saving.value = false }
}

async function testAI() {
  testingAI.value = true; aiTestResult.value = null
  try { aiTestResult.value = await settingsStore.testAIConnection() }
  catch { aiTestResult.value = { success: false, message: 'AI 连接测试失败' } }
  finally { testingAI.value = false }
}

async function testTTS() {
  testingTTS.value = true; ttsTestResult.value = null
  try { ttsTestResult.value = await settingsStore.testTTSConnection() }
  catch { ttsTestResult.value = { success: false, message: 'TTS 连接测试失败' } }
  finally { testingTTS.value = false }
}
</script>

<style lang="scss" scoped>
.settings-page { max-width: 1100px; }

// Env card
.env-card {
  background: var(--bg-surface);
  border: 1px solid var(--border-faint);
  border-radius: 10px;
  margin-bottom: 24px;
  overflow: hidden;
}
.env-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 18px;
  background: var(--bg-elevated);
  border-bottom: 1px solid var(--border-faint);
}
.env-title { font-size: 12px; font-weight: 700; color: var(--text-muted); letter-spacing: .5px; text-transform: uppercase; }
.env-body  { padding: 14px 18px; display: flex; flex-direction: column; gap: 8px; }
.env-item  { display: flex; align-items: center; gap: 16px; }
.env-label { font-size: 12px; color: var(--text-muted); width: 60px; flex-shrink: 0; }
.env-ok    { font-size: 13px; color: var(--success); font-weight: 500; }
.env-err   { font-size: 13px; color: var(--danger);  font-weight: 500; }
.env-pending { font-size: 13px; color: var(--text-secondary); }
.env-path  { font-size: 12px; color: var(--text-secondary); font-family: monospace; word-break: break-all; }

// Text button
.text-btn {
  background: none;
  border: none;
  color: var(--accent);
  font-size: 12px;
  cursor: pointer;
  padding: 2px 0;
  &:hover { color: var(--accent-light); }
  &:disabled { opacity: .4; cursor: default; }
}

// Tabs
.settings-tabs {
  :deep(.el-tabs__content) { padding-top: 18px; }
}

.tab-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
  align-items: start;
}
.tab-single { max-width: 640px; }

.inner-alert { margin-bottom: 16px; }

// Test result
.test-result {
  margin-left: 12px;
  font-size: 13px;
  font-weight: 500;
  &.ok   { color: var(--success); }
  &.fail { color: var(--danger); }
}
</style>
