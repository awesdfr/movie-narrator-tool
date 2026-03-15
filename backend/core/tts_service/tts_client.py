"""TTS客户端模块

与TTS服务进行HTTP通信，支持标准REST API和Gradio API（IndexTTS）
"""
import asyncio
import hashlib
import json
import wave
from pathlib import Path
from typing import Optional
import httpx
from loguru import logger

from config import settings


class TTSClient:
    """TTS客户端

    调用TTS HTTP API进行语音合成，支持多种TTS服务：
    - 标准REST API
    - Gradio API（IndexTTS）
    """

    def __init__(
        self,
        api_base: Optional[str] = None,
        api_endpoint: Optional[str] = None,
        reference_audio: Optional[str] = None,
        timeout: float = 120.0,
        infer_mode: str = "批次推理"
    ):
        """初始化

        Args:
            api_base: TTS服务地址
            api_endpoint: API端点路径（如 /tts, /gradio_api/call/gen_single）
            reference_audio: 参考音频路径
            timeout: 请求超时时间
            infer_mode: 推理模式（"普通推理"或"批次推理"）
        """
        self.api_base = (api_base or settings.tts_api_base).rstrip("/")
        self.api_endpoint = api_endpoint or "/tts"
        self.reference_audio = reference_audio or settings.tts_reference_audio
        self.timeout = timeout
        self.infer_mode = infer_mode

        self._client: Optional[httpx.AsyncClient] = None

        # 检测是否是Gradio API
        self._is_gradio = "gradio" in self.api_endpoint.lower() or "gradio" in self.api_base.lower()

        # 缓存已上传的参考音频路径，避免重复上传 {本地路径: 服务端路径}
        self._uploaded_ref_cache: dict[str, str] = {}

    async def _get_client(self) -> httpx.AsyncClient:
        """获取HTTP客户端"""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                limits=httpx.Limits(
                    max_connections=50,
                    max_keepalive_connections=30
                )
            )
        return self._client

    async def generate(
        self,
        text: str,
        output_name: Optional[str] = None,
        output_dir: Optional[Path] = None,
        reference_audio: Optional[str] = None
    ) -> Path:
        """生成TTS音频

        Args:
            text: 要合成的文本
            output_name: 输出文件名（不含扩展名）
            output_dir: 输出目录
            reference_audio: 参考音频（用于克隆音色）

        Returns:
            生成的音频文件路径
        """
        if not text or not text.strip():
            raise ValueError("文本不能为空")

        # 生成输出路径
        if not output_name:
            text_hash = hashlib.md5(text.encode()).hexdigest()[:12]
            output_name = f"tts_{text_hash}"

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{output_name}.wav"
        else:
            output_path = settings.temp_dir / f"{output_name}.wav"
            output_path.parent.mkdir(parents=True, exist_ok=True)

        # 如果已存在且文本相同，直接返回
        if output_path.exists():
            return output_path

        # 根据API类型选择不同的生成方式
        if self._is_gradio:
            return await self._generate_gradio(text, output_path, reference_audio)
        else:
            return await self._generate_rest(text, output_path, reference_audio)

    async def _generate_rest(
        self,
        text: str,
        output_path: Path,
        reference_audio: Optional[str] = None
    ) -> Path:
        """使用标准REST API生成TTS"""
        client = await self._get_client()
        ref_audio = reference_audio or self.reference_audio

        try:
            # 构建multipart请求
            data = {"text": text}
            files = {}

            # 发送请求（使用配置的端点）
            endpoint = self.api_endpoint if self.api_endpoint.startswith("/") else f"/{self.api_endpoint}"
            
            # 使用上下文管理器确保文件正确关闭
            file_handle = None
            try:
                if ref_audio and Path(ref_audio).exists():
                    file_handle = open(ref_audio, "rb")
                    files["reference_audio"] = (
                        "reference.wav",
                        file_handle,
                        "audio/wav"
                    )
                
                response = await client.post(
                    f"{self.api_base}{endpoint}",
                    data=data,
                    files=files if files else None
                )
            finally:
                # 确保文件句柄被关闭
                if file_handle:
                    file_handle.close()

            if response.status_code == 200:
                # 保存音频
                with open(output_path, "wb") as f:
                    f.write(response.content)

                logger.debug(f"TTS生成成功: {output_path}")
                return output_path
            else:
                error_msg = response.text
                try:
                    error_json = response.json()
                    error_msg = error_json.get("error", error_msg)
                except Exception:
                    pass
                raise RuntimeError(f"TTS生成失败: {response.status_code} - {error_msg}")

        except httpx.ConnectError:
            raise RuntimeError(f"无法连接到TTS服务: {self.api_base}")
        except httpx.TimeoutException:
            raise RuntimeError("TTS请求超时")
        except Exception as e:
            logger.error(f"TTS生成错误: {e}")
            raise

    async def _generate_gradio(
        self,
        text: str,
        output_path: Path,
        reference_audio: Optional[str] = None
    ) -> Path:
        """使用Gradio API生成TTS（IndexTTS），失败自动重试"""
        ref_audio = reference_audio or self.reference_audio

        # IndexTTS必须有参考音频
        if not ref_audio:
            raise RuntimeError("IndexTTS需要参考音频，请在设置中配置'参考音频'路径")

        if not Path(ref_audio).exists():
            raise RuntimeError(f"参考音频文件不存在: {ref_audio}")

        max_retries = 3
        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                return await self._gradio_call(text, output_path, ref_audio)
            except httpx.ConnectError:
                raise RuntimeError(f"无法连接到TTS服务: {self.api_base}")
            except httpx.TimeoutException:
                raise RuntimeError("TTS请求超时")
            except Exception as e:
                last_error = e
                # 重试前清除参考音频缓存，可能是服务端重启导致上传路径失效
                self._uploaded_ref_cache.pop(ref_audio, None)
                if attempt < max_retries:
                    wait = attempt * 2
                    logger.warning(f"Gradio TTS第{attempt}次失败: {e}, {wait}秒后重试")
                    await asyncio.sleep(wait)
                else:
                    logger.error(f"Gradio TTS重试{max_retries}次仍失败: {e}")

        raise last_error

    async def _gradio_call(
        self,
        text: str,
        output_path: Path,
        ref_audio: str
    ) -> Path:
        """单次 Gradio TTS 调用"""
        client = await self._get_client()

        # 步骤1：上传参考音频（缓存命中则跳过）
        if ref_audio in self._uploaded_ref_cache:
            uploaded_file_path = self._uploaded_ref_cache[ref_audio]
            logger.debug(f"复用已上传的参考音频: {uploaded_file_path}")
        else:
            uploaded_file_path = await self._gradio_upload_file(client, ref_audio)
            self._uploaded_ref_cache[ref_audio] = uploaded_file_path
            logger.debug(f"Gradio上传参考音频: {uploaded_file_path}")

        # 步骤2：调用TTS生成API
        # 新版 Gradio 要求文件参数是 FileData 对象格式（带 meta 字段）
        file_data = {
            "path": uploaded_file_path,
            "meta": {"_type": "gradio.FileData"}
        }
        api_data = {
            "data": [
                file_data,           # prompt - 参考音频（FileData格式）
                text,                # text - 要合成的文本
                self.infer_mode,     # infer_mode - 推理模式
                120,                 # max_text_tokens_per_sentence
                4,                   # sentences_bucket_max_size
                True,                # do_sample
                0.8,                 # top_p
                30,                  # top_k
                1.0,                 # temperature
                0.0,                 # length_penalty
                3,                   # num_beams
                10.0,                # repetition_penalty
                600,                 # max_mel_tokens
            ]
        }

        # 确定API端点
        if "call" in self.api_endpoint:
            call_endpoint = self.api_endpoint if self.api_endpoint.startswith("/") else f"/{self.api_endpoint}"
        else:
            call_endpoint = "/gradio_api/call/gen_single"

        logger.debug(f"Gradio API调用: {self.api_base}{call_endpoint}")
        logger.debug(f"请求数据: {api_data}")

        # 发送POST请求启动任务
        try:
            response = await client.post(
                f"{self.api_base}{call_endpoint}",
                json=api_data
            )
        except httpx.HTTPError as e:
            raise RuntimeError(f"Gradio API 连接失败: {e}")

        if response.status_code != 200:
            error_msg = response.text
            if response.status_code == 422:
                # 参数验证错误，尝试解析具体的错误信息
                try:
                    error_json = response.json()
                    detail = error_json.get("detail", [])
                    if isinstance(detail, list):
                        error_msg = f"参数验证失败: {json.dumps(detail, ensure_ascii=False)}"
                    else:
                        error_msg = f"参数错误: {detail}"
                except:
                    pass
            elif response.status_code == 500:
                error_msg = f"服务器内部错误: {response.text}"
                
            logger.error(f"Gradio调用失败: {error_msg}")
            
            # 尝试获取 /info 以帮助调试
            try:
                info_res = await client.get(f"{self.api_base}/gradio_api/info")
                if info_res.status_code == 200:
                    logger.info(f"服务端API定义: {json.dumps(info_res.json(), ensure_ascii=False)}")
            except:
                pass
                
            raise RuntimeError(f"Gradio API调用失败: {response.status_code} - {error_msg}")

        # 解析响应获取event_id
        result = response.json()
        event_id = result.get("event_id")

        if not event_id:
            raise RuntimeError(f"Gradio API未返回event_id: {result}")

        logger.debug(f"Gradio任务ID: {event_id}")

        # 步骤3：通过SSE获取结果
        audio_url = await self._gradio_get_result(client, call_endpoint, event_id)

        if not audio_url:
            raise RuntimeError("Gradio API未返回音频文件")

        # 步骤4：下载音频文件
        await self._download_file(client, audio_url, output_path)

        logger.debug(f"TTS生成成功(Gradio): {output_path}")
        return output_path

    async def _gradio_upload_file(self, client: httpx.AsyncClient, file_path: str) -> str:
        """上传文件到Gradio服务器"""
        path = Path(file_path)

        with open(path, "rb") as f:
            files = {"files": (path.name, f, "audio/wav")}
            response = await client.post(
                f"{self.api_base}/gradio_api/upload",
                files=files
            )

        if response.status_code != 200:
            raise RuntimeError(f"文件上传失败: {response.status_code} - {response.text}")

        result = response.json()
        # Gradio返回上传文件的路径列表，可能是字符串列表或对象列表
        if isinstance(result, list) and len(result) > 0:
            item = result[0]
            if isinstance(item, str):
                return item
            elif isinstance(item, dict) and "name" in item:
                return item["name"]
            elif isinstance(item, dict) and "path" in item:
                return item["path"]
            
        logger.warning(f"无法解析上传响应: {result}，尝试直接使用预期路径")
        if isinstance(result, list) and len(result) > 0:
            item = result[0]
            if isinstance(item, dict):
                return item.get("path") or item.get("name") or str(item)
            return str(item)
        return str(result)

    async def _gradio_get_result(self, client: httpx.AsyncClient, call_endpoint: str, event_id: str) -> str:
        """通过SSE获取Gradio任务结果"""
        # 构建SSE端点
        sse_url = f"{self.api_base}{call_endpoint}/{event_id}"

        logger.debug(f"Gradio SSE URL: {sse_url}")

        # 使用stream方式读取SSE
        async with client.stream("GET", sse_url) as response:
            if response.status_code != 200:
                raise RuntimeError(f"Gradio SSE请求失败: {response.status_code}")

            audio_path = None
            current_event = None

            async for line in response.aiter_lines():
                line = line.strip()
                if not line:
                    current_event = None
                    continue

                logger.debug(f"SSE行: {line}")

                if line.startswith("event:"):
                    current_event = line[6:].strip()
                    continue

                if line.startswith("data:"):
                    data_str = line[5:].strip()

                    # 处理 error 事件
                    if current_event == "error":
                        error_detail = data_str if data_str and data_str != "null" else "未知错误（服务端未返回详情）"
                        raise RuntimeError(f"Gradio服务端错误: {error_detail}")

                    try:
                        data = json.loads(data_str)

                        # 检查是否完成
                        if isinstance(data, list) and len(data) > 0:
                            # IndexTTS返回音频文件路径
                            result = data[0]
                            audio_path = None

                            if isinstance(result, dict):
                                # 情况1: Gradio 4.x 组件更新格式 {"value": {"path": ...}}
                                if "value" in result and isinstance(result["value"], dict):
                                    # 优先使用url，因为path可能是服务端本地绝对路径，会导致下载URL拼接错误
                                    audio_path = result["value"].get("url") or result["value"].get("path")
                                
                                # 情况2: 扁平格式 {"path": ...}
                                if not audio_path:
                                    audio_path = result.get("url") or result.get("path")
                            elif isinstance(result, str):
                                # 情况3: 直接返回字符串
                                audio_path = result

                            if audio_path:
                                logger.debug(f"获取到音频路径: {audio_path}")
                                return audio_path

                    except json.JSONDecodeError:
                        continue

        return audio_path

    async def _download_file(self, client: httpx.AsyncClient, url: str, output_path: Path):
        """下载文件"""
        # 如果是相对路径，添加base_url
        if not url.startswith("http"):
            if url.startswith("/"):
                url = f"{self.api_base}{url}"
            else:
                url = f"{self.api_base}/{url}"

        logger.debug(f"下载音频: {url}")

        response = await client.get(url)
        if response.status_code != 200:
            raise RuntimeError(f"音频下载失败: {response.status_code}")

        content = response.content
        await asyncio.to_thread(self._write_file, output_path, content)

    @staticmethod
    def _write_file(path: Path, data: bytes):
        """同步写入文件（由 to_thread 调度到线程池，避免阻塞事件循环）"""
        with open(path, "wb") as f:
            f.write(data)

    async def estimate_duration(self, text: str) -> float:
        """预估TTS时长

        基于文本长度估算，中文约4字/秒

        Args:
            text: 文本内容

        Returns:
            预估时长(秒)
        """
        if not text:
            return 0.0

        # 统计字符数（中文为主）
        char_count = len(text)

        # 估算：中文约4字/秒，考虑标点停顿
        punctuation_count = sum(1 for c in text if c in "，。！？、；：""''（）")
        pause_time = punctuation_count * 0.2  # 标点停顿约0.2秒

        duration = char_count / 4.0 + pause_time

        return round(duration, 2)

    async def get_duration(self, audio_path: str) -> float:
        """获取音频实际时长

        Args:
            audio_path: 音频文件路径

        Returns:
            时长(秒)
        """
        path = Path(audio_path)
        if not path.exists():
            return 0.0

        def _get_duration():
            try:
                with wave.open(str(path), 'rb') as w:
                    frames = w.getnframes()
                    rate = w.getframerate()
                    return frames / rate
            except Exception as e:
                logger.warning(f"读取音频时长失败: {e}")
                return 0.0

        return await asyncio.to_thread(_get_duration)

    async def test_connection(self) -> dict:
        """测试TTS服务连接

        Returns:
            {success, message, latency_ms, api_type}
        """
        try:
            import time
            client = await self._get_client()

            start_time = time.time()
            api_type = "gradio" if self._is_gradio else "rest"

            if self._is_gradio:
                # Gradio API测试 - 尝试访问API信息
                try:
                    response = await client.get(f"{self.api_base}/gradio_api/info")
                    if response.status_code == 200:
                        latency_ms = (time.time() - start_time) * 1000
                        message = "Gradio TTS服务连接成功 (IndexTTS)"
                        warning = None

                        # 检查并发限制
                        try:
                            info = response.json()
                            concurrency_limit = info.get("default_concurrency_limit")
                            if concurrency_limit is not None and concurrency_limit <= 1:
                                warning = (
                                    f"Gradio服务并发限制为 {concurrency_limit}，"
                                    "多片段同时生成TTS时会排队等待。"
                                    "建议在启动TTS服务时设置 --default-concurrency-limit 参数提高并发数。"
                                )
                                message += f" [警告: 并发限制={concurrency_limit}]"
                                logger.warning(f"TTS服务并发限制较低: {concurrency_limit}")
                        except Exception:
                            pass

                        result = {
                            "success": True,
                            "message": message,
                            "latency_ms": round(latency_ms, 2),
                            "api_type": api_type
                        }
                        if warning:
                            result["warning"] = warning
                        return result
                except Exception:
                    pass

                # 备选：尝试访问根路径
                response = await client.get(self.api_base)
            else:
                # 标准REST API测试
                try:
                    response = await client.get(f"{self.api_base}/health")
                except Exception:
                    response = await client.get(self.api_base)

            latency_ms = (time.time() - start_time) * 1000

            if response.status_code in [200, 404]:  # 404也说明服务在运行
                return {
                    "success": True,
                    "message": f"TTS服务连接成功 ({api_type})",
                    "latency_ms": round(latency_ms, 2),
                    "api_type": api_type
                }
            else:
                return {
                    "success": False,
                    "message": f"TTS服务响应异常: {response.status_code}",
                    "latency_ms": round(latency_ms, 2),
                    "api_type": api_type
                }

        except httpx.ConnectError:
            return {
                "success": False,
                "message": f"无法连接到TTS服务: {self.api_base}",
                "latency_ms": None,
                "api_type": "unknown"
            }
        except httpx.TimeoutException:
            return {
                "success": False,
                "message": "连接超时",
                "latency_ms": None,
                "api_type": "unknown"
            }
        except Exception as e:
            return {
                "success": False,
                "message": str(e),
                "latency_ms": None,
                "api_type": "unknown"
            }

    async def close(self):
        """关闭客户端"""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
