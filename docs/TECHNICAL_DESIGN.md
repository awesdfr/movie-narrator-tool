# Technical Design Document

## 1. 文档信息

- 项目名称：Movie Narrator Tool
- 文档类型：技术实施文档
- 文档版本：v1.0
- 文档日期：2026-04-19
- 对应 PRD：`docs/PRD.md`
- 当前底座：`movie-narrator-tool-main`

## 2. 技术目标

本方案的技术目标是基于现有项目底座，构建一套可交付的高精度视频画面匹配系统，满足以下要求：

- 支持外部成品识别模式和来源可控模式
- 输出 `exact / inferred / fallback` 三层结果
- 最终生成剪映草稿
- 支持 benchmark 回归与阈值校准

## 3. 总体架构

推荐采用以下架构：

- 桌面壳：Electron
- 前端：Vue 3 + Vite + Element Plus
- 本地服务：FastAPI + WebSocket
- 算法层：Python + PyTorch + OpenCV + FAISS
- 导出层：剪映草稿生成器

系统分为 6 层：

1. Shell 层：Electron 启动本地前后端并提供桌面能力
2. UI 层：项目页、处理页、结果页、预览页、设置页
3. API 层：项目、处理、预览、导出、设置接口
4. Pipeline 层：预处理、建库、检索、对齐、验真、补全
5. Storage 层：项目文件、缓存、索引、benchmark、导出草稿
6. Integrations 层：FFmpeg、剪映、模型权重、后续 AI/TTS 服务

## 4. 双模式技术方案

### 4.1 模式一：来源可控模式

核心思路：

- 在生成解说视频时记录来源片段
- 对每个片段保存原片起止、播放速度、裁切、镜像、转场等信息
- 最终导出时直接读取 `mapping.json`

优点：

- 几乎不依赖反向识别
- 准确率最高
- 运行速度最快

适用场景：

- 自建解说生产流水线
- 自有剪辑工具链

### 4.2 模式二：外部成品识别模式

核心思路：

- 对解说成片先生成低干扰代理视频
- 对代理视频执行视觉召回、时序对齐、局部验真和自动补全

优点：

- 可处理外部成品
- 适合当前 MVP 需求

## 5. 现有项目改造策略

### 5.1 建议保留的模块

- `backend/main.py`
- `backend/config.py`
- `backend/api/routes/project.py`
- `backend/api/routes/process_v2.py`
- `backend/api/routes/preview.py`
- `backend/core/project/project_manager.py`
- `backend/core/exporter/jianying_exporter.py`
- `backend/core/video_processor/subtitle_masker.py`
- `backend/core/video_processor/analysis_video.py`
- `backend/models/project.py`
- `backend/models/segment.py`
- `frontend/src/views/Process.vue`
- `frontend/src/views/Timeline.vue`
- `backend/cli/benchmark_cli.py`

### 5.2 建议替换的模块

- `backend/core/video_processor/frame_matcher.py`
- `backend/core/matcher/hybrid_matcher.py`
- `backend/core/video_processor/scene_detector.py`
- 旧的纯 pHash 主流程

### 5.3 建议新增的模块

- `backend/core/matching_proxy/`
- `backend/core/shot_detector/`
- `backend/core/embed_index/`
- `backend/core/retrieval/`
- `backend/core/temporal_align/`
- `backend/core/verifier/`
- `backend/core/auto_complete/`
- `backend/core/controlled_mapping/`
- `backend/core/benchmarking/`

## 6. 核心算法链路

### 6.1 matching_proxy

职责：

- 生成仅用于匹配的代理视频或代理关键帧集合
- 降低字幕、水印、黑边、镜像、调色等干扰

处理步骤：

1. 检查视频容器和分析代理需求
2. 去黑边
3. 固定屏蔽水印、台标、角标
4. 字幕区域自动检测和遮罩
5. 镜像检测和纠正
6. 强转场、强模糊、闪白帧过滤
7. 关键帧质量评分和抽样

输出：

- 代理视频路径
- 关键帧列表
- 屏蔽区域信息
- 帧质量评分

### 6.2 shot_detector

建议方案：

- V1 使用 `TransNetV2`
- 兼容保留现有简化切分器作 fallback

输入：

- 原片视频
- 解说视频或代理视频

输出：

- 镜头边界列表
- 镜头级关键帧索引

### 6.3 embed_index

建议方案：

- 使用 `DINOv2` 提取视觉向量
- 使用 `FAISS` 建立原片索引

索引粒度：

- 帧级索引
- 镜头级聚合索引

输出：

- `faiss.index`
- 帧时间戳到向量 ID 映射
- 镜头级摘要

### 6.4 retrieval

职责：

- 对解说关键帧查询原片候选片段

规则：

- 每个查询帧返回 Top-K 候选
- 同时记录第一候选与第二候选分差
- 仅做粗召回，不做最终判定

### 6.5 temporal_align

建议方案：

- 使用 `DP / DTW / Viterbi`
- 强制路径单调前进

目标：

- 过滤时序跳变的伪匹配
- 将离散帧候选合并为连续候选片段

输出：

- 候选时间窗
- 对齐路径
- 对齐稳定性分数

### 6.6 verifier

建议方案：

- 默认使用 `LightGlue + RANSAC + SSIM`
- 难样本复核使用 `OmniGlue`

验证维度：

- 局部特征匹配
- RANSAC 内点数
- 内点率
- 几何稳定性
- 变换后结构相似度
- 多帧连续一致性

输出：

- 精验真置信度
- 验真证据
- 最佳候选段

### 6.7 auto_complete

目标：

- 避免因为少数难片段导致整条时间线断裂

策略：

1. `exact`：直接使用高置信匹配结果
2. `inferred`：当前片段落在前后 `exact` 锚点之间时，只在局部时间窗搜索并推断补位
3. `fallback`：后续结合文案找画面或替代镜头生成补段

### 6.8 controlled_mapping

职责：

- 支持来源可控模式

记录字段：

- `source_video_id`
- `source_start`
- `source_end`
- `playback_speed`
- `crop`
- `mirror`
- `filters`
- `transition_type`

输出：

- `mapping.json`

## 7. 模块目录建议

```text
backend/
  core/
    matching_proxy/
      proxy_builder.py
      mask_detector.py
      frame_quality.py
    shot_detector/
      transnet_adapter.py
    embed_index/
      dinov2_encoder.py
      faiss_index.py
    retrieval/
      candidate_retriever.py
    temporal_align/
      monotonic_aligner.py
    verifier/
      lightglue_verifier.py
      omniglue_verifier.py
      confidence_gate.py
    auto_complete/
      inferred_filler.py
      fallback_builder.py
    controlled_mapping/
      mapping_recorder.py
      mapping_loader.py
```

## 8. 数据模型设计

### 8.1 Project 扩展字段

建议新增：

- `match_mode`: `controlled` 或 `recognition`
- `pipeline_version`
- `export_profile`
- `benchmark_report_path`
- `proxy_cache_dir`

### 8.2 Segment 扩展字段

建议新增：

- `match_type`: `exact / inferred / fallback`
- `evidence_summary`
- `candidate_gap_score`
- `frame_support_count`
- `proxy_frame_ids`
- `source_mapping_id`

### 8.3 MatchCandidate 扩展字段

建议新增：

- `rank_gap`
- `geometric_inliers`
- `geometric_inlier_ratio`
- `warp_ssim`
- `alignment_path_score`
- `verification_source`

### 8.4 新增 MappingRecord

字段建议：

- `id`
- `segment_id`
- `movie_path`
- `movie_start`
- `movie_end`
- `speed`
- `crop`
- `mirror`
- `effect_meta`

## 9. 接口设计

### 9.1 项目相关

- `POST /api/project/create`
- `GET /api/project/{id}`
- `PUT /api/project/{id}`
- `PUT /api/project/{id}/subtitle-regions`

### 9.2 处理相关

- `POST /api/process/{project_id}/start`
- `POST /api/process/{project_id}/stop`
- `GET /api/process/{project_id}/progress`
- `GET /api/process/{project_id}/segments`
- `POST /api/process/{project_id}/rematch`
- `POST /api/process/{project_id}/segments/{segment_id}/rematch`

### 9.3 新增建议接口

- `POST /api/process/{project_id}/build-proxy`
- `POST /api/process/{project_id}/build-index`
- `POST /api/process/{project_id}/verify`
- `POST /api/process/{project_id}/auto-complete`
- `POST /api/process/{project_id}/controlled-mapping/import`
- `GET /api/process/{project_id}/benchmark`

### 9.4 导出相关

- `POST /api/preview/{project_id}/export/jianying`
- `GET /api/preview/{project_id}/export/report`

## 10. 前端改造方案

### 10.1 保留现有页面

- `Process.vue`
- `Timeline.vue`
- 现有项目页和设置页

### 10.2 需要新增的信息展示

- 片段匹配类型标签
- 置信度和证据摘要
- exact / inferred / fallback 筛选
- 代理帧预览
- benchmark 指标展示

### 10.3 Electron 桌面化

建议方案：

- Electron 主进程负责启动 FastAPI
- Electron 窗口加载本地前端页面
- 前端保持现有 API 调用方式
- 仅增加桌面专有功能，如打开目录、日志查看、缓存清理

## 11. 缓存与文件布局

建议缓存结构：

```text
temp/
  analysis_video/
  proxy_frames/
  frame_index/
  faiss_index/
  verification/
  exports/
projects/
  <project_id>.json
benchmarks/
  manifests/
  reports/
```

建议项目输出：

- `project.json`
- `matching_report.json`
- `mapping.json`
- `benchmark_report.json`

## 12. 置信度与决策规则

### 12.1 exact 判定

需同时满足：

- 视觉召回分数过阈
- 第一候选与第二候选有足够分差
- 时序对齐稳定
- 多帧局部匹配通过
- RANSAC 内点与内点率达标
- SSIM 达标

### 12.2 inferred 判定

需满足：

- 当前片段位于已命中锚点之间
- 局部时间窗内候选稳定
- 与前后片段时间关系合理

### 12.3 fallback 判定

进入条件：

- 无法稳定构造 exact
- 无法稳定构造 inferred
- 需要保持草稿完整

## 13. benchmark 与验收

### 13.1 benchmark 设计

沿用现有 `benchmark_cli.py`，增加：

- `exact precision`
- `false_match_rate`
- `boundary_error`
- `timeline_completeness`
- `match_type_breakdown`

### 13.2 benchmark 样本要求

必须覆盖：

- 字幕遮挡
- 裁切
- 缩放
- 镜像
- 调色
- 变速
- 重复镜头
- 相似场景
- 插入 B-roll
- 强转场

### 13.3 回归要求

- 所有主分支合入前需跑 benchmark
- 阈值调整要附带回归报告

## 14. 实施排期

### 阶段 1：底座整理

- 梳理现有项目结构
- 建立新模块目录
- 接入 `matching_proxy`
- 打通代理视频生成

### 阶段 2：新匹配引擎

- 接入 `DINOv2 + FAISS`
- 接入 `TransNetV2`
- 完成候选检索和时序对齐

### 阶段 3：精验真

- 接入 `LightGlue + RANSAC + SSIM`
- 完成 confidence gate
- 实现 `exact`

### 阶段 4：自动补全

- 实现 `inferred`
- 预留 `fallback`
- 导出完整草稿

### 阶段 5：桌面化与回归

- Electron 封装
- benchmark 数据集整理
- 指标校准和发布

## 15. 风险与对策

### 风险 1：匹配精度不足

对策：

- 使用代理视频降低干扰
- 使用多阶段验证而非单阶段决策
- 用 benchmark 做阈值校准

### 风险 2：速度过慢

对策：

- 原片建库缓存
- 镜头级先召回再验真
- 只对候选片段做重验证

### 风险 3：完整性不足

对策：

- 引入 `inferred`
- 预留 `fallback`
- 引入来源可控模式

### 风险 4：许可风险

对策：

- NarratoAI 仅参考设计和工作流
- 不直接纳入非商业许可代码

## 16. 技术结论

本项目最合理的工程路线不是从零重写整套应用，而是基于现有 `movie-narrator-tool-main` 的 Web 应用底座，替换匹配核心，补齐代理视频、视觉检索、时序对齐、几何验真和自动补全，再通过 Electron 封装为桌面产品。这样可以最大化复用现有项目管理、预览、时间轴和剪映导出能力，同时把核心价值集中在新匹配引擎上。
