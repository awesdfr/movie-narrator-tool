"""
Mock webrtcvad module
由于 Python 3.14 无法编译 webrtcvad，这里提供一个兼容层。
警告：这将禁用语音活动检测（VAD），所有音频帧都将被视为语音。
"""

class Vad:
    def __init__(self, mode=None):
        pass

    def set_mode(self, mode):
        pass

    def is_speech(self, buf, sample_rate, length=None):
        # 始终返回 True (假设都是语音)
        return True
