import { createI18n } from 'vue-i18n'

const messages = {
  'zh-CN': {
    common: {
      confirm: '确认',
      cancel: '取消',
      save: '保存',
      delete: '删除',
      edit: '编辑',
      create: '创建',
      search: '搜索',
      loading: '加载中...',
      success: '操作成功',
      error: '操作失败',
      warning: '警告',
      info: '提示'
    },
    nav: {
      home: '首页',
      settings: '设置'
    },
    home: {
      title: '项目管理',
      newProject: '新建项目',
      recentProjects: '最近项目',
      noProjects: '暂无项目，点击上方按钮创建',
      projectName: '项目名称',
      moviePath: '原电影路径',
      narrationPath: '解说视频路径',
      referenceAudio: '参考音频（声纹）'
    },
    project: {
      status: {
        created: '已创建',
        importing: '导入中',
        analyzing: '分析中',
        matching: '匹配中',
        recognizing: '识别中',
        ready_for_polish: '待确认润色',
        polishing: '润色中',
        generating_tts: '生成TTS',
        completed: '已完成',
        error: '错误'
      }
    },
    editor: {
      title: '片段编辑',
      filter: {
        all: '全部',
        hasNarration: '有解说',
        noNarration: '无解说',
        nonMovie: '非电影'
      },
      segment: {
        originalText: '原解说词',
        polishedText: 'AI润色',
        ttsDuration: 'TTS预估',
        movieTime: '原片段',
        options: {
          useSegment: '使用此片段',
          keepBgm: '保留解说视频背景音乐',
          keepMovieAudio: '保留电影原声',
          muteMovieAudio: '电影原声静音',
          usePolishedText: '使用AI润色文案'
        },
        actions: {
          editText: '编辑文案',
          previewTTS: '试听TTS',
          adjustTime: '调整时间',
          manualMatch: '手动匹配'
        }
      }
    },
    settings: {
      title: '设置',
      ai: {
        title: 'AI配置',
        apiBase: 'API地址',
        apiKey: 'API密钥',
        model: '模型',
        maxTokens: '最大Token数',
        temperature: '温度',
        testConnection: '测试连接',
        polishTemplate: '润色模板'
      },
      tts: {
        title: 'TTS配置',
        apiBase: '服务地址',
        referenceAudio: '参考音频',
        speed: '语速',
        testConnection: '测试连接'
      },
      match: {
        title: '匹配参数',
        threshold: '匹配阈值',
        phashThreshold: 'pHash阈值',
        sceneThreshold: '场景阈值',
        useDeepLearning: '使用深度学习'
      },
      export: {
        title: '导出设置',
        jianyingPath: '剪映草稿路径',
        minPlaybackSpeed: '最小播放速度',
        maxPlaybackSpeed: '最大播放速度',
        exportSubtitles: '导出字幕',
        subtitleFormat: '字幕格式'
      },
      ui: {
        title: '界面设置',
        language: '语言',
        theme: '主题'
      }
    }
  },
  'en': {
    common: {
      confirm: 'Confirm',
      cancel: 'Cancel',
      save: 'Save',
      delete: 'Delete',
      edit: 'Edit',
      create: 'Create',
      search: 'Search',
      loading: 'Loading...',
      success: 'Success',
      error: 'Error',
      warning: 'Warning',
      info: 'Info'
    },
    nav: {
      home: 'Home',
      settings: 'Settings'
    },
    home: {
      title: 'Project Management',
      newProject: 'New Project',
      recentProjects: 'Recent Projects',
      noProjects: 'No projects yet, click the button above to create one',
      projectName: 'Project Name',
      moviePath: 'Movie Path',
      narrationPath: 'Narration Video Path',
      referenceAudio: 'Reference Audio (Voiceprint)'
    },
    project: {
      status: {
        created: 'Created',
        importing: 'Importing',
        analyzing: 'Analyzing',
        matching: 'Matching',
        recognizing: 'Recognizing',
        ready_for_polish: 'Ready for Polish',
        polishing: 'Polishing',
        generating_tts: 'Generating TTS',
        completed: 'Completed',
        error: 'Error'
      }
    }
  }
}

const i18n = createI18n({
  legacy: false,
  locale: 'zh-CN',
  fallbackLocale: 'en',
  messages
})

export default i18n
