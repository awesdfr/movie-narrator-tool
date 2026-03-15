"""Connectivity checks for AI endpoints."""
from __future__ import annotations

from typing import Optional

import httpx
from loguru import logger


class APITester:
    async def test_ai_api(self, api_base: str, api_key: Optional[str] = None, model: str = 'gpt-4o-mini') -> dict:
        api_base = api_base.rstrip('/')
        headers = {'Content-Type': 'application/json'}
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'

        payload = {
            'model': model,
            'messages': [{'role': 'user', 'content': "Reply with exactly: test ok"}],
            'max_tokens': 16,
            'temperature': 0,
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                import time

                started = time.time()
                response = await client.post(f'{api_base}/chat/completions', headers=headers, json=payload)
                latency_ms = round((time.time() - started) * 1000, 2)

                if response.status_code == 200:
                    data = response.json()
                    content = data['choices'][0]['message']['content']
                    return {
                        'success': True,
                        'message': f'连接成功，AI 回复: {content}',
                        'latency_ms': latency_ms,
                        'model_info': {
                            'model': data.get('model', model),
                            'usage': data.get('usage', {}),
                        },
                    }

                detail = response.text
                try:
                    detail = response.json().get('error', {}).get('message', detail)
                except Exception:
                    pass
                return {
                    'success': False,
                    'message': f'API 返回错误: {response.status_code} - {detail}',
                    'latency_ms': latency_ms,
                    'model_info': None,
                }
        except httpx.ConnectError:
            return {'success': False, 'message': f'连接失败: 无法连接到 {api_base}', 'latency_ms': None, 'model_info': None}
        except httpx.TimeoutException:
            return {'success': False, 'message': '连接超时', 'latency_ms': None, 'model_info': None}
        except Exception as exc:
            logger.error(f'AI API test failed: {exc}')
            return {'success': False, 'message': f'测试失败: {exc}', 'latency_ms': None, 'model_info': None}

    async def test_models_list(self, api_base: str, api_key: Optional[str] = None) -> dict:
        api_base = api_base.rstrip('/')
        headers = {'Authorization': f'Bearer {api_key}'} if api_key else {}
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f'{api_base}/models', headers=headers)
                if response.status_code == 200:
                    data = response.json()
                    return {'success': True, 'models': [item['id'] for item in data.get('data', [])]}
                return {'success': False, 'models': [], 'message': response.text}
        except Exception as exc:
            logger.error(f'List models failed: {exc}')
            return {'success': False, 'models': [], 'message': str(exc)}
