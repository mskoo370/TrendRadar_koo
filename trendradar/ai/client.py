# coding=utf-8
"""
AI 客户端模块

基于 LiteLLM 的统一 AI 模型接口
支持 100+ AI 提供商（OpenAI、DeepSeek、Gemini、Claude、国内模型等）
"""

import os
from typing import Any, Dict, List

from litellm import completion


class AIClient:
    """统一的 AI 客户端（基于 LiteLLM）"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化 AI 客户端

        Args:
            config: AI 配置字典
                - MODEL: 模型标识（格式: provider/model_name）
                - API_KEY: API 密钥
                - API_BASE: API 基础 URL（可选）
                - TEMPERATURE: 采样温度
                - MAX_TOKENS: 最大生成 token 数
                - TIMEOUT: 请求超时时间（秒）
                - NUM_RETRIES: 重试次数（可选）
                - FALLBACK_MODELS: 备用模型列表（可选）
        """
        self.model = config.get("MODEL", config.get("model", "deepseek/deepseek-chat"))
        self.api_key = config.get("API_KEY", config.get("api_key")) or os.environ.get("AI_API_KEY", "")
        self.api_base = config.get("API_BASE", config.get("api_base", ""))
        self.temperature = config.get("TEMPERATURE", config.get("temperature", 1.0))
        self.max_tokens = config.get("MAX_TOKENS", config.get("max_tokens", 5000))
        self.timeout = config.get("TIMEOUT", config.get("timeout", 120))
        self.num_retries = config.get("NUM_RETRIES", config.get("num_retries", 2))
        self.fallback_models = config.get("FALLBACK_MODELS", config.get("fallback_models", []))
        
        # LiteLLM 은 Gemini 에 대해 GEMINI_API_KEY 환경 변수를 사용함
        if self.api_key and "gemini" in self.model.lower():
            os.environ["GEMINI_API_KEY"] = self.api_key
        
        # LiteLLM 은 Groq 에 대해 GROQ_API_KEY 환경 변수를 사용함
        if self.api_key and "groq" in self.model.lower():
            os.environ["GROQ_API_KEY"] = self.api_key
        
        # 打印初始化状态
        masked_key = f"{self.api_key[:6]}******" if self.api_key else "None"
        print(f"[AI] Client 초기화: model={self.model}, api_key={masked_key}")

    def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """
        调用 AI 模型进行对话

        Args:
            messages: 消息列表，格式: [{"role": "system/user/assistant", "content": "..."}]
            **kwargs: 额外参数，会覆盖默认配置

        Returns:
            str: AI 响应内容

        Raises:
            Exception: API 调用失败时抛出异常
        """
        # 构建请求参数
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "timeout": kwargs.get("timeout", self.timeout),
            "num_retries": kwargs.get("num_retries", self.num_retries),
        }

        # 添加 API Key
        if self.api_key:
            params["api_key"] = self.api_key

        # 添加 API Base（如果配置了）
        if self.api_base:
            params["api_base"] = self.api_base

        # 添加 max_tokens（如果配置了且不为 0）
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        if max_tokens and max_tokens > 0:
            params["max_tokens"] = max_tokens

        # 添加 fallback 模型（如果配置了）
        if self.fallback_models:
            params["fallbacks"] = self.fallback_models

        # 合并其他额外参数
        for key, value in kwargs.items():
            if key not in params:
                params[key] = value

        # 调用 LiteLLM (带自动重试和模型名容错)
        model_variants = [self.model]
        if "groq" in self.model.lower():
            model_variants = [
                self.model,
                "groq/llama-3.3-70b-versatile",
                "groq/llama3-70b-8192",
                "groq/mixtral-8x7b-32768",
                "groq/llama-3.1-8b-instant",
                "groq/llama3-8b-8192",
            ]
            model_variants = list(dict.fromkeys(model_variants))

        last_error = None
        
        # LiteLLM fallback
        for variant in model_variants:
            max_attempts = 2 if "groq" in variant else 1
            for attempt in range(max_attempts):
                try:
                    current_params = params.copy()
                    current_params["model"] = variant
                    current_params["num_retries"] = 0
                    response = completion(**current_params)
                    content = response.choices[0].message.content
                    if isinstance(content, list):
                        content = "\n".join(
                            item.get("text", str(item)) if isinstance(item, dict) else str(item)
                            for item in content
                        )
                    return content or ""
                except Exception as e:
                    last_error = e
                    err_str = str(e)
                    print(f"[AI] 모델 {variant} 호출 실패 (시도 {attempt+1}/{max_attempts}): {err_str[:100]}...")
                    if "RateLimit" in err_str or "rate limit" in err_str.lower():
                        if attempt < max_attempts - 1:
                            print("[AI] Rate Limit 초과. Groq TPM 한도 리셋을 위해 60초 대기 후 재시도합니다...")
                            import time
                            time.sleep(60)
                            continue
                    break

        raise last_error

    def validate_config(self) -> tuple[bool, str]:
        """
        验证配置是否有效

        Returns:
            tuple: (是否有效, 错误信息)
        """
        if not self.model:
            return False, "未配置 AI 模型（model）"

        if not self.api_key:
            return False, "未配置 AI API Key，请在 config.yaml 或环境变量 AI_API_KEY 中设置"

        # 验证模型格式（应该包含 provider/model）
        if "/" not in self.model:
            return False, f"模型格式错误: {self.model}，应为 'provider/model' 格式（如 'deepseek/deepseek-chat'）"

        return True, ""
