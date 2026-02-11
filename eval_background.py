"""
Background evaluation module for real-time per-request evaluation.
Each chat response triggers async quality + safety evaluation without blocking the user.

채팅 응답마다 비동기로 품질 + 안전성 평가를 실행하는 백그라운드 평가 모듈.
사용자 응답 latency에 영향 없이 평가 결과를 App Insights에 즉시 전송합니다.
"""
import os
import json
import logging
import time
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

# Configuration
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
APP_INSIGHTS_CONNECTION_STRING = os.getenv("APP_INSIGHTS_CONNECTION_STRING")
AZURE_CONTENT_SAFETY_ENDPOINT = os.getenv("AZURE_CONTENT_SAFETY_ENDPOINT")
AZURE_CONTENT_SAFETY_KEY = os.getenv("AZURE_CONTENT_SAFETY_KEY")

# Lazy-loaded clients (initialized on first use)
_quality_evaluators = None
_safety_client = None
_ai_handler = None
_ai_logger = None


def _get_quality_evaluators():
    """Lazy-load quality evaluators (expensive to create) / 품질 평가자 지연 로드"""
    global _quality_evaluators
    if _quality_evaluators is None:
        from azure.ai.evaluation import (
            FluencyEvaluator,
            CoherenceEvaluator,
            RelevanceEvaluator,
            GroundednessEvaluator,
            AzureOpenAIModelConfiguration,
        )
        model_config = AzureOpenAIModelConfiguration(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME,
            api_version=AZURE_OPENAI_API_VERSION,
        )
        _quality_evaluators = {
            "fluency": FluencyEvaluator(model_config),
            "coherence": CoherenceEvaluator(model_config),
            "relevance": RelevanceEvaluator(model_config),
            "groundedness": GroundednessEvaluator(model_config),
        }
    return _quality_evaluators


def _get_safety_client():
    """Lazy-load Content Safety client / Content Safety 클라이언트 지연 로드"""
    global _safety_client
    if _safety_client is None:
        if not AZURE_CONTENT_SAFETY_ENDPOINT or not AZURE_CONTENT_SAFETY_KEY:
            return None
        from azure.ai.contentsafety import ContentSafetyClient
        from azure.core.credentials import AzureKeyCredential
        _safety_client = ContentSafetyClient(
            endpoint=AZURE_CONTENT_SAFETY_ENDPOINT,
            credential=AzureKeyCredential(AZURE_CONTENT_SAFETY_KEY),
        )
    return _safety_client


def _get_ai_logger():
    """Lazy-load App Insights logger / App Insights 로거 지연 로드"""
    global _ai_handler, _ai_logger
    if _ai_logger is None:
        if not APP_INSIGHTS_CONNECTION_STRING:
            return None
        from opencensus.ext.azure.log_exporter import AzureEventHandler
        _ai_logger = logging.getLogger("eval_background")
        _ai_logger.setLevel(logging.INFO)
        _ai_handler = AzureEventHandler(connection_string=APP_INSIGHTS_CONNECTION_STRING)
        _ai_logger.addHandler(_ai_handler)
    return _ai_logger


def evaluate_single(query: str, response: str, context: str = "") -> dict:
    """
    Evaluate a single query/response pair (quality + safety).
    Runs synchronously — designed to be called via asyncio.to_thread().
    
    단일 query/response 쌍에 대해 품질 + 안전성 평가 실행.
    동기 함수 — asyncio.to_thread()로 호출하도록 설계됨.
    """
    result = {
        "query": query[:500],
        "response": response[:1000],
        "timestamp": datetime.now().isoformat(),
        "quality_fluency": "0",
        "quality_coherence": "0",
        "quality_relevance": "0",
        "quality_groundedness": "0",
        "safety_violence": "0",
        "safety_sexual": "0",
        "safety_self_harm": "0",
        "safety_hate_unfairness": "0",
    }

    # --- Quality Evaluation ---
    try:
        evaluators = _get_quality_evaluators()
        eval_input = {"query": query, "response": response, "context": context or query}

        for name, evaluator in evaluators.items():
            try:
                if name == "groundedness":
                    score = evaluator(query=query, response=response, context=context or query)
                else:
                    score = evaluator(query=query, response=response)
                # Extract numeric score from result dict
                # e.g. {"fluency": 4.0} or {"gpt_fluency": 4.0}
                for k, v in score.items():
                    if isinstance(v, (int, float)):
                        result[f"quality_{name}"] = str(round(v, 2))
                        break
            except Exception as e:
                print(f"  [eval_background] {name} evaluator error: {e}")
    except Exception as e:
        print(f"  [eval_background] Quality evaluation error: {e}")

    # --- Safety Evaluation ---
    try:
        client = _get_safety_client()
        if client:
            from azure.ai.contentsafety.models import AnalyzeTextOptions, TextCategory
            text = f"{query} {response}"[:10000]
            resp = client.analyze_text(AnalyzeTextOptions(text=text))
            for cat in resp.categories_analysis:
                severity = cat.severity or 0
                if cat.category == TextCategory.VIOLENCE:
                    result["safety_violence"] = str(severity)
                elif cat.category == TextCategory.HATE:
                    result["safety_hate_unfairness"] = str(severity)
                elif cat.category == TextCategory.SEXUAL:
                    result["safety_sexual"] = str(severity)
                elif cat.category == TextCategory.SELF_HARM:
                    result["safety_self_harm"] = str(severity)
    except Exception as e:
        print(f"  [eval_background] Safety evaluation error: {e}")

    # --- Send to App Insights ---
    try:
        logger = _get_ai_logger()
        if logger:
            logger.info("evaluation_result", extra={"custom_dimensions": result})
            # Flush to ensure delivery (non-blocking batch is fine here)
            if _ai_handler:
                _ai_handler.flush()
            print(f"  [eval_background] Sent evaluation to App Insights "
                  f"(fluency={result['quality_fluency']}, violence={result['safety_violence']})")
        else:
            print(f"  [eval_background] App Insights not configured, skipping send")
    except Exception as e:
        print(f"  [eval_background] Failed to send to App Insights: {e}")

    return result
