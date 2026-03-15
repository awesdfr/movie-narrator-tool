# 电影解说视频制作工具

基于 WebUI 的电影解说视频制作工具，实现解说视频与原电影的帧级画面匹配、AI 文案润色、TTS 语音合成，并导出剪映草稿文件。

## 功能特性

- **高精度帧匹配**: 多尺度注意力哈希 + Smith-Waterman 序列比对 + 动态采样 + 失真归一化，精确匹配解说视频与原电影画面
- **字幕智能去除**: PaddleOCR 检测字幕区域并遮罩，消除字幕对匹配的干扰（可选依赖）
- **语音识别**: 基于 Whisper 进行高精度语音转文字
- **声纹识别**: 使用 Resemblyzer 区分解说人声和电影对白
- **AI 润色**: 支持多种 AI 服务（OpenAI / 第三方 API），自动润色解说文案
- **TTS 合成**: 集成 index-tts2，支持音色克隆
- **非电影检测**: 自动识别并标记非电影画面（纯色背景、文字等）
- **剪映导出**: 直接导出为剪映 Pro 草稿格式
- **WebUI 界面**: 可视化操作，支持片段编辑、时间轴、预览等
- **GPU 加速**: 支持 CUDA 加速帧哈希计算和深度学习推理

## 技术栈

### 后端
- Python 3.10+
- FastAPI + WebSocket
- OpenCV + FFmpeg
- PyTorch + ResNet（深度特征提取）
- FAISS（向量检索）
- Whisper（语音识别）
- PaddleOCR（字幕检测，可选）

### 前端
- Vue 3 + Vite
- Element Plus
- Pinia
- Vue Router
- Video.js
- WaveSurfer.js

## 目录结构

```
movie-narrator-tool/
├── backend/                 # 后端服务
│   ├── main.py             # FastAPI 入口
│   ├── config.py           # 配置管理
│   ├── api/                # API 路由
│   │   ├── routes/         # REST 接口
│   │   └── websocket.py    # WebSocket 实时通信
│   ├── core/               # 核心处理模块
│   │   ├── video_processor/  # 帧匹配 / 视频处理
│   │   ├── audio_processor/  # 语音识别 / 声纹
│   │   ├── ai_service/       # AI 文案润色
│   │   ├── tts_service/      # TTS 语音合成
│   │   ├── exporter/         # 剪映导出
│   │   ├── matcher/          # 混合匹配策略
│   │   └── project/          # 项目管理
│   ├── models/             # 数据模型
│   ├── utils/              # 工具函数
│   └── requirements.txt
├── frontend/               # Vue 3 前端项目
│   ├── src/
│   │   ├── views/          # 页面组件
│   │   ├── components/     # 通用组件
│   │   ├── stores/         # Pinia 状态管理
│   │   ├── api/            # API 调用
│   │   ├── i18n/           # 国际化（中/英）
│   │   └── styles/         # 全局样式
│   ├── package.json
│   └── vite.config.js
├── videos/                 # 视频素材存放（不上传）
├── models/                 # 预训练模型存放（不上传）
└── temp/                   # 临时文件（不上传）
```

## 快速开始

### 环境要求

- Python 3.10+
- Node.js 18+（仅构建前端时需要，运行时不需要）
- FFmpeg（添加到系统 PATH）
- CUDA（可选，用于 GPU 加速）

### 一键安装和运行

**首次使用：**

1. 双击 `install.bat`，等待安装完成
2. 双击 `start.bat` 运行程序
3. 浏览器会自动打开 http://127.0.0.1:8000

**日常使用：**

直接双击 `start.bat` 即可

### 可用脚本

| 脚本 | 用途 |
|------|------|
| `install.bat` | 首次安装：检查环境、安装依赖、构建前端 |
| `start.bat` | 启动程序（自动打开浏览器） |
| `build.bat` | 重新构建前端 |
| `stop.bat` | 停止所有服务 |

### 可选依赖

```bash
# 字幕检测去除（提升帧匹配精度）
pip install paddlepaddle paddleocr

# 声纹识别
pip install resemblyzer webrtcvad-wheels
```

---

## 手动安装（开发者）

### 后端

```bash
cd backend

# 创建虚拟环境（推荐）
python -m venv venv
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 配置
cp .env.example .env
# 编辑 .env 填写 API Key 等配置

# 启动服务
python main.py
```

后端服务默认运行在 http://127.0.0.1:8000

### 前端

```bash
cd frontend

# 安装依赖
npm install

# 开发模式
npm run dev

# 生产构建（输出到 backend/static）
npm run build
```

前端开发服务运行在 http://localhost:3000

### React + Tailwind（可选）

仓库内额外提供了一个独立的 React + Tailwind UI（不影响原有 Vue 前端），目录为 `frontend-react/`。

```bash
cd frontend-react
npm install
npm run dev
```

默认运行在 http://localhost:3001 ，也可以直接双击 `start-frontend-react.bat`。

## 使用流程

1. **创建项目** — 在首页点击"新建项目"，输入项目名称、原电影路径、解说视频路径
2. **开始处理** — 点击"开始处理"，系统自动执行：视频分析 → 语音识别 → 声纹识别 → 帧匹配 → AI 润色
3. **编辑片段** — 查看和筛选片段，编辑文案，试听 TTS 效果
4. **预览和调整** — 左右分屏对比解说视频和原电影，在时间轴上微调
5. **导出** — 导出到剪映草稿或 SRT/ASS 字幕文件

## 配置说明

### AI 服务

支持 OpenAI API、Azure OpenAI、第三方 API 中转（兼容 OpenAI 格式）。在设置页面或 `backend/.env` 中配置。

### TTS 服务

设计为与 [index-tts2](https://github.com/index-tts/index-tts2) 配合使用，支持音色克隆。

### 剪映导出

默认草稿路径在 `backend/.env` 中配置（`JIANYING_DRAFTS_DIR`），导出后可在剪映 Pro 中直接打开。

## API 文档

启动后端后，访问 http://127.0.0.1:8000/docs 查看 Swagger API 文档。

## 注意事项

- **GPU 加速**: 建议使用 NVIDIA GPU，可显著提升 Whisper 语音识别和帧匹配速度
- **内存**: 处理长视频建议 16GB 以上内存
- **FFmpeg**: 必须安装并添加到系统 PATH
- **视频格式**: 支持常见格式（MP4、MKV、AVI 等）

## 许可证

[MIT License](LICENSE)
