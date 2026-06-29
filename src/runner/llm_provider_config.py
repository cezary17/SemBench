"""Shared LLM provider configuration for SemBench runners."""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional


ENV_LLM_PROVIDER = "SEMBENCH_LLM_PROVIDER"
ENV_LLM_BASE_URL = "SEMBENCH_LLM_BASE_URL"
ENV_LLM_API_KEY = "SEMBENCH_LLM_API_KEY"
ENV_LLM_PARAMS = "SEMBENCH_LLM_PARAMS"

DEFAULT_PROVIDER = "default"
LOCAL_PROVIDER = "local"
SENSITIVE_PARAM_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "openai_api_key",
    "token",
    "access_token",
}


@dataclass(frozen=True)
class LLMProviderConfig:
    """Normalized LLM provider settings shared by all system runners."""

    provider: str = DEFAULT_PROVIDER
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_local(self) -> bool:
        return self.provider == LOCAL_PROVIDER

    def merge_kwargs(
        self,
        defaults: Optional[Dict[str, Any]] = None,
        endpoint_fields: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Merge runner defaults, endpoint fields, then user LLM params."""
        merged = dict(defaults or {})
        merged.update({
            key: value
            for key, value in (endpoint_fields or {}).items()
            if value is not None
        })
        merged.update(self.params)
        return merged

    def to_cli_args(self) -> list[str]:
        """Return CLI arguments for forwarding this config to a worker."""
        args = ["--llm-provider", self.provider]
        if self.base_url is not None:
            args.extend(["--llm-base-url", self.base_url])
        if self.api_key is not None:
            args.extend(["--llm-api-key", self.api_key])
        if self.params:
            args.extend(["--llm-params", json.dumps(self.params)])
        return args

    def public_params(self) -> Dict[str, Any]:
        """Return params safe for metrics/logging."""
        return _redact_sensitive_params(self.params)

    def litellm_model_name(self, model_name: str) -> str:
        """Return the model name LiteLLM expects for local endpoints."""
        if not self.is_local:
            return model_name
        if "custom_llm_provider" in self.params:
            return model_name
        if model_name.startswith("openai/"):
            return model_name
        return f"openai/{model_name}"


def parse_llm_params(raw_params: Optional[str]) -> Dict[str, Any]:
    """Parse the JSON value supplied to --llm-params."""
    if raw_params is None or raw_params == "":
        return {}

    try:
        parsed = json.loads(raw_params)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in --llm-params: {exc.msg}") from exc

    if not isinstance(parsed, dict):
        raise ValueError("--llm-params must be a JSON object")

    return parsed


def build_llm_provider_config(
    provider: str = DEFAULT_PROVIDER,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    params_json: Optional[str] = None,
) -> LLMProviderConfig:
    """Validate and normalize CLI or environment LLM provider settings."""
    provider = provider or DEFAULT_PROVIDER
    if provider not in {DEFAULT_PROVIDER, LOCAL_PROVIDER}:
        raise ValueError(
            "--llm-provider must be one of: default, local"
        )

    params = parse_llm_params(params_json)

    if provider == LOCAL_PROVIDER:
        if not base_url:
            raise ValueError(
                "--llm-base-url is required when --llm-provider local is used"
            )
        if not api_key:
            api_key = "local"

    return LLMProviderConfig(
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        params=params,
    )


def llm_provider_config_from_args(args: Any) -> LLMProviderConfig:
    return build_llm_provider_config(
        provider=getattr(args, "llm_provider", DEFAULT_PROVIDER),
        base_url=getattr(args, "llm_base_url", None),
        api_key=getattr(args, "llm_api_key", None),
        params_json=getattr(args, "llm_params", None),
    )


def llm_provider_config_from_env() -> LLMProviderConfig:
    return build_llm_provider_config(
        provider=os.environ.get(ENV_LLM_PROVIDER, DEFAULT_PROVIDER),
        base_url=os.environ.get(ENV_LLM_BASE_URL) or None,
        api_key=os.environ.get(ENV_LLM_API_KEY) or None,
        params_json=os.environ.get(ENV_LLM_PARAMS, "{}"),
    )


def set_llm_provider_env(config: LLMProviderConfig) -> None:
    """Mirror normalized LLM config into the process environment."""
    os.environ[ENV_LLM_PROVIDER] = config.provider
    os.environ[ENV_LLM_PARAMS] = json.dumps(config.params)

    _set_or_clear_env(ENV_LLM_BASE_URL, config.base_url)
    _set_or_clear_env(ENV_LLM_API_KEY, config.api_key)


def validate_no_bigquery_local(
    systems: Iterable[str], config: LLMProviderConfig
) -> None:
    if config.is_local and any(system == "bigquery" for system in systems):
        raise ValueError("BigQuery does not support local provider mode")


def _set_or_clear_env(name: str, value: Optional[str]) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


def _redact_sensitive_params(value: Any) -> Any:
    if isinstance(value, dict):
        redacted = {}
        for key, item in value.items():
            if str(key).lower() in SENSITIVE_PARAM_KEYS:
                redacted[key] = "<redacted>"
            else:
                redacted[key] = _redact_sensitive_params(item)
        return redacted
    if isinstance(value, list):
        return [_redact_sensitive_params(item) for item in value]
    return value
