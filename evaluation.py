"""
Azure AI Evaluation Pipeline for LangGraph Agent
Queries trace data from Application Insights and runs automated evaluation.

Application Insights에서 트레이스 데이터를 쿼리하고 자동으로 평가를 실행합니다.
"""
import os
import json
import asyncio
from datetime import datetime, timedelta
from typing import Optional
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Azure AI Evaluation SDK
from azure.ai.evaluation import (
    evaluate,
    FluencyEvaluator,
    CoherenceEvaluator,
    RelevanceEvaluator,
    GroundednessEvaluator,
    AzureOpenAIModelConfiguration,
)

# Azure AI Content Safety
from azure.ai.contentsafety import ContentSafetyClient
from azure.ai.contentsafety.models import AnalyzeTextOptions, TextCategory
from azure.core.credentials import AzureKeyCredential

# Azure Monitor Query for Application Insights
from azure.monitor.query import LogsQueryClient
from azure.identity import DefaultAzureCredential


# Configuration
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")

# Application Insights
APP_INSIGHTS_WORKSPACE_ID = os.getenv("APP_INSIGHTS_WORKSPACE_ID")
APP_INSIGHTS_CONNECTION_STRING = os.getenv("APP_INSIGHTS_CONNECTION_STRING")

# Azure AI Content Safety
AZURE_CONTENT_SAFETY_ENDPOINT = os.getenv("AZURE_CONTENT_SAFETY_ENDPOINT")
AZURE_CONTENT_SAFETY_KEY = os.getenv("AZURE_CONTENT_SAFETY_KEY")

# Output directory
OUTPUT_DIR = Path("evaluation_results")
OUTPUT_DIR.mkdir(exist_ok=True)


def get_model_config() -> AzureOpenAIModelConfiguration:
    """Get Azure OpenAI model configuration / Azure OpenAI 모델 설정"""
    return AzureOpenAIModelConfiguration(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME,
        api_version=AZURE_OPENAI_API_VERSION,
    )


def query_traces_from_app_insights(
    hours: int = 24,
    limit: int = 100,
) -> list[dict]:
    """
    Query LLM trace data from Application Insights.
    Application Insights에서 LLM 트레이스 데이터를 쿼리합니다.
    
    Returns:
        list[dict]: Trace data containing query, response, context
                    query, response, context를 포함한 트레이스 데이터
    """
    if not APP_INSIGHTS_WORKSPACE_ID:
        print("⚠️ APP_INSIGHTS_WORKSPACE_ID not set. Using sample data.")
        return get_sample_data()
    
    credential = DefaultAzureCredential()
    client = LogsQueryClient(credential)
    
    # KQL query: Extract LLM input/output from AppDependencies table
    # KQL 쿼리: AppDependencies 테이블에서 LLM input/output 추출
    # OpenTelemetry trace data is stored in AppDependencies
    # Langfuse/Traceloop SDK stores with langfuse.trace.input/output keys
    query = f"""
    AppDependencies
    | where TimeGenerated > ago({hours}h)
    | where Name == "chat_stream" 
        or (Name has_any ("AzureChatOpenAI", "ChatOpenAI") and Properties has "gen_ai.prompt")
    | extend 
        query = coalesce(
            tostring(Properties['langfuse.trace.input']),
            tostring(Properties['gen_ai.prompt']),
            tostring(Properties['llm.prompts']),
            tostring(Properties['traceloop.entity.input']),
            ""
        ),
        response = coalesce(
            tostring(Properties['langfuse.trace.output']),
            tostring(Properties['gen_ai.completion']),
            tostring(Properties['llm.completions']),
            tostring(Properties['traceloop.entity.output']),
            ""
        ),
        context = ""
    | where isnotempty(query) and isnotempty(response)
    | project TimeGenerated, OperationId, query, response, context, Name
    | order by TimeGenerated desc
    | limit {limit}
    """
    
    try:
        response = client.query_workspace(
            workspace_id=APP_INSIGHTS_WORKSPACE_ID,
            query=query,
            timespan=timedelta(hours=hours),
        )
        
        traces = []
        for table in response.tables:
            for row in table.rows:
                traces.append({
                    "timestamp": str(row[0]),
                    "operation_id": str(row[1]),
                    "query": str(row[2]) if row[2] else "",
                    "response": str(row[3]) if row[3] else "",
                    "context": str(row[4]) if row[4] else "",
                })
        
        if traces:
            print(f"✅ Queried {len(traces)} traces from Application Insights (AppDependencies)")
            return traces
        
        # If not found in AppDependencies, try AppEvents
        # AppDependencies에 없으면 AppEvents 시도
        print("   No traces found in 'AppDependencies'. Trying 'AppEvents'...")
        return query_from_app_events(client, hours, limit)
        
    except Exception as e:
        print(f"❌ Error querying Application Insights: {e}")
        print("Using sample data instead.")
        return get_sample_data()


def query_from_app_events(client: LogsQueryClient, hours: int, limit: int) -> list[dict]:
    """Query LLM traces from AppEvents table / AppEvents 테이블에서 LLM 트레이스 쿼리"""
    query = f"""
    AppEvents
    | where TimeGenerated > ago({hours}h)
    | where Name has_any ("llm", "chat", "openai", "langgraph", "teacher", "student", "evaluation")
        or Properties has_any ("llm", "prompt", "completion", "query", "response")
    | extend 
        query = coalesce(
            tostring(Properties['llm.prompts']),
            tostring(Properties['gen_ai.prompt']),
            tostring(Properties['query']),
            tostring(Properties['input']),
            Name
        ),
        response = coalesce(
            tostring(Properties['llm.completions']),
            tostring(Properties['gen_ai.completion']),
            tostring(Properties['response']),
            tostring(Properties['output']),
            ""
        )
    | where isnotempty(query)
    | project TimeGenerated, OperationId, query, response
    | order by TimeGenerated desc
    | limit {limit}
    """
    
    try:
        response = client.query_workspace(
            workspace_id=APP_INSIGHTS_WORKSPACE_ID,
            query=query,
            timespan=timedelta(hours=hours),
        )
        
        traces = []
        for table in response.tables:
            for row in table.rows:
                traces.append({
                    "timestamp": str(row[0]),
                    "operation_id": str(row[1]),
                    "query": str(row[2]) if row[2] else "",
                    "response": str(row[3]) if row[3] else "",
                    "context": "",
                })
        
        if traces:
            print(f"✅ Queried {len(traces)} traces from Application Insights (AppEvents)")
            return traces
        
        print("   No traces found. Using sample data.")
        return get_sample_data()
        
    except Exception as e:
        print(f"❌ Error querying AppEvents: {e}")
        return get_sample_data()
        print("Using sample data instead.")
        return get_sample_data()


def get_sample_data() -> list[dict]:
    """Sample test data / 샘플 테스트 데이터"""
    return [
        {
            "query": "medium math problem / 보통 수학 문제 풀래",
            "response": "👨‍🏫 **Teacher (Question #1)**\n\nSolve the equation: 2x + 5 = 15 / 다음 방정식을 풀어보세요: 2x + 5 = 15",
            "context": "Teacher-Student Quiz System / Teacher-Student 퀴즈 시스템",
            "ground_truth": "x = 5",
        },
        {
            "query": "easy history quiz / 쉬운 역사 퀴즈",
            "response": "👨‍🏫 **Teacher (Question #1)**\n\nWhat is the capital of South Korea? / 대한민국의 수도는 어디인가요?",
            "context": "Teacher-Student Quiz System / Teacher-Student 퀴즈 시스템",
            "ground_truth": "Seoul / 서울",
        },
        {
            "query": "programming problem / 프로그래밍 문제 출제해줘",
            "response": "👨‍🏫 **Teacher (Question #1)**\n\nWrite a Python function that sums all elements in a list. / Python에서 리스트의 모든 요소를 합하는 함수를 작성하세요.",
            "context": "Teacher-Student Quiz System / Teacher-Student 퀴즈 시스템",
            "ground_truth": "use sum() function or for loop / sum() 함수 사용 또는 for 루프",
        },
    ]


def save_traces_as_jsonl(traces: list[dict], filename: str) -> Path:
    """Save traces to JSONL file (only evaluation required fields) / 트레이스를 JSONL 파일로 저장 (평가에 필요한 필드만)"""
    filepath = OUTPUT_DIR / filename
    with open(filepath, "w", encoding="utf-8") as f:
        for trace in traces:
            # Extract only fields needed for evaluation (exclude timestamp)
            # 평가에 필요한 필드만 추출 (timestamp 제외)
            query = str(trace.get("query", ""))
            response = str(trace.get("response", ""))
            # Use query as context if context is empty (Teacher question = grounding context)
            # context가 비어있으면 query를 context로 사용 (Teacher 질문 = 근거 컨텍스트)
            context = str(trace.get("context", "")) or query
            eval_data = {
                "query": query,
                "response": response,
                "context": context,
            }
            f.write(json.dumps(eval_data, ensure_ascii=False) + "\n")
    print(f"✅ Saved {len(traces)} traces to {filepath}")
    return filepath


def run_quality_evaluation(data_path: Path) -> dict:
    """
    Run quality evaluation (Fluency, Coherence, Relevance, Groundedness)
    Uses individual evaluators - no ground_truth required
    
    품질 평가 실행 (Fluency, Coherence, Relevance, Groundedness)
    개별 Evaluator 사용 - ground_truth 불필요
    """
    model_config = get_model_config()
    
    # Individual Quality Evaluators / 개별 품질 평가자
    fluency_eval = FluencyEvaluator(model_config)
    coherence_eval = CoherenceEvaluator(model_config)
    relevance_eval = RelevanceEvaluator(model_config)
    groundedness_eval = GroundednessEvaluator(model_config)
    
    print("🔍 Running Quality Evaluation (Fluency, Coherence, Relevance, Groundedness)...")
    
    output_file = OUTPUT_DIR / "quality_evaluation_result.json"
    
    result = evaluate(
        data=str(data_path),
        evaluators={
            "fluency": fluency_eval,
            "coherence": coherence_eval,
            "relevance": relevance_eval,
            "groundedness": groundedness_eval,
        },
        evaluator_config={
            "fluency": {
                "column_mapping": {
                    "query": "${data.query}",
                    "response": "${data.response}",
                }
            },
            "coherence": {
                "column_mapping": {
                    "query": "${data.query}",
                    "response": "${data.response}",
                }
            },
            "relevance": {
                "column_mapping": {
                    "query": "${data.query}",
                    "response": "${data.response}",
                }
            },
            "groundedness": {
                "column_mapping": {
                    "query": "${data.query}",
                    "response": "${data.response}",
                    "context": "${data.context}",
                }
            },
        },
        output_path=str(output_file),
    )
    
    print("✅ Quality Evaluation completed")
    
    # Extract metrics from result file (workaround for Timestamp serialization issue)
    # 결과 파일에서 메트릭 추출 (Timestamp 직렬화 문제 우회)
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            saved_result = json.load(f)
        
        # Calculate metrics / metrics 계산
        rows = saved_result.get("rows", [])
        fluency_scores = [r.get("outputs.fluency.fluency", 0) for r in rows if r.get("outputs.fluency.fluency") is not None]
        coherence_scores = [r.get("outputs.coherence.coherence", 0) for r in rows if r.get("outputs.coherence.coherence") is not None]
        relevance_scores = [r.get("outputs.relevance.relevance", 0) for r in rows if r.get("outputs.relevance.relevance") is not None]
        groundedness_scores = [r.get("outputs.groundedness.groundedness", 0) for r in rows if r.get("outputs.groundedness.groundedness") is not None]
        
        metrics = {
            "fluency.fluency": sum(fluency_scores) / len(fluency_scores) if fluency_scores else None,
            "coherence.coherence": sum(coherence_scores) / len(coherence_scores) if coherence_scores else None,
            "relevance.relevance": sum(relevance_scores) / len(relevance_scores) if relevance_scores else None,
            "groundedness.groundedness": sum(groundedness_scores) / len(groundedness_scores) if groundedness_scores else None,
        }
        
        return {"metrics": metrics, "rows": rows}
    except Exception as e:
        print(f"   Warning: Could not parse saved result: {e}")
        return {"metrics": {}, "rows": []}


def run_safety_evaluation(traces: list[dict]) -> dict:
    """
    Safety evaluation using Azure AI Content Safety
    Analyzes Violence, Hate, Sexual, SelfHarm categories
    
    Azure AI Content Safety를 사용한 안전성 평가
    Violence, Hate, Sexual, SelfHarm 카테고리 분석
    """
    if not AZURE_CONTENT_SAFETY_ENDPOINT or not AZURE_CONTENT_SAFETY_KEY:
        print("⚠️ Azure AI Content Safety not configured. Skipping safety evaluation.")
        return {"metrics": {}, "rows": []}
    
    print("🔍 Running Safety Evaluation (Azure AI Content Safety)...")
    
    # Create Content Safety Client / Content Safety Client 생성
    client = ContentSafetyClient(
        endpoint=AZURE_CONTENT_SAFETY_ENDPOINT,
        credential=AzureKeyCredential(AZURE_CONTENT_SAFETY_KEY)
    )
    
    # Aggregate scores by category / 각 카테고리별 점수 집계
    violence_scores = []
    hate_scores = []
    sexual_scores = []
    self_harm_scores = []
    rows = []
    
    for i, trace in enumerate(traces):
        try:
            # query + response를 합쳐서 분석
            text_to_analyze = f"{trace.get('query', '')} {trace.get('response', '')}"
            
            # 텍스트가 너무 길면 잘라내기 (API 제한)
            if len(text_to_analyze) > 10000:
                text_to_analyze = text_to_analyze[:10000]
            
            # Content Safety 분석 요청
            request = AnalyzeTextOptions(text=text_to_analyze)
            response = client.analyze_text(request)
            
            # 카테고리별 점수 추출 (0-6 스케일, 0=안전, 6=위험)
            row_result = {
                "query": trace.get("query", "")[:200],
                "violence": 0,
                "hate": 0,
                "sexual": 0,
                "self_harm": 0,
            }
            
            for category_result in response.categories_analysis:
                severity = category_result.severity or 0
                if category_result.category == TextCategory.VIOLENCE:
                    row_result["violence"] = severity
                    violence_scores.append(severity)
                elif category_result.category == TextCategory.HATE:
                    row_result["hate"] = severity
                    hate_scores.append(severity)
                elif category_result.category == TextCategory.SEXUAL:
                    row_result["sexual"] = severity
                    sexual_scores.append(severity)
                elif category_result.category == TextCategory.SELF_HARM:
                    row_result["self_harm"] = severity
                    self_harm_scores.append(severity)
            
            rows.append(row_result)
            
            if (i + 1) % 10 == 0:
                print(f"   Processed {i + 1}/{len(traces)} samples...")
                
        except Exception as e:
            print(f"   ⚠️ Error analyzing sample {i + 1}: {e}")
            continue
    
    # 평균 점수 계산
    metrics = {
        "safety_violence": round(sum(violence_scores) / len(violence_scores), 2) if violence_scores else None,
        "safety_hate_unfairness": round(sum(hate_scores) / len(hate_scores), 2) if hate_scores else None,
        "safety_sexual": round(sum(sexual_scores) / len(sexual_scores), 2) if sexual_scores else None,
        "safety_self_harm": round(sum(self_harm_scores) / len(self_harm_scores), 2) if self_harm_scores else None,
    }
    
    # 결과 저장
    result = {"metrics": metrics, "rows": rows}
    with open(OUTPUT_DIR / "safety_evaluation_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Safety Evaluation completed ({len(rows)} samples)")
    return result


def generate_evaluation_summary(results: dict) -> dict:
    """평가 결과 요약 생성"""
    summary = {
        "timestamp": datetime.now().isoformat(),
        "metrics": results.get("metrics", {}),
        "total_samples": len(results.get("rows", [])),
    }
    
    # 메트릭별 통계
    metrics = results.get("metrics", {})
    summary["quality_scores"] = {
        "fluency": metrics.get("fluency.fluency", None),
        "coherence": metrics.get("coherence.coherence", None),
        "relevance": metrics.get("relevance.relevance", None),
        "groundedness": metrics.get("groundedness.groundedness", None),
    }
    
    # Safety scores - Azure AI Content Safety 형식 (safety_*)
    summary["safety_scores"] = {
        "violence": metrics.get("safety_violence", None),
        "sexual": metrics.get("safety_sexual", None),
        "self_harm": metrics.get("safety_self_harm", None),
        "hate_unfairness": metrics.get("safety_hate_unfairness", None),
    }
    
    return summary


def save_evaluation_for_grafana(summary: dict, filename: str = "evaluation_metrics.json"):
    """Grafana에서 읽을 수 있는 형식으로 저장"""
    filepath = OUTPUT_DIR / filename
    
    # Grafana JSON 데이터소스 형식
    grafana_data = {
        "timestamp": summary["timestamp"],
        "total_samples": summary["total_samples"],
        **{f"quality_{k}": v for k, v in summary["quality_scores"].items() if v is not None},
        **{f"safety_{k}": v for k, v in summary["safety_scores"].items() if v is not None},
    }
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(grafana_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Saved Grafana metrics to {filepath}")
    return filepath


def send_evaluation_to_app_insights(summary: dict, rows: list[dict] = None):
    """
    Send evaluation results to Application Insights (customEvents table)
    For Grafana dashboard visualization
    
    평가 결과를 Application Insights로 전송 (customEvents 테이블)
    Grafana 대시보드에서 조회 가능
    """
    if not APP_INSIGHTS_CONNECTION_STRING:
        print("⚠️ APP_INSIGHTS_CONNECTION_STRING not set. Skipping telemetry.")
        return
    
    try:
        from opencensus.ext.azure.log_exporter import AzureEventHandler
        import logging
        import time
        
        # Azure Event Handler setup / Azure Event Handler 설정
        logger = logging.getLogger("evaluation")
        logger.setLevel(logging.INFO)
        
        handler = AzureEventHandler(connection_string=APP_INSIGHTS_CONNECTION_STRING)
        logger.addHandler(handler)
        
        # Send summary metrics / 요약 메트릭 전송
        # Use "0" for null values so KQL todouble() works correctly
        # null 값은 "0"으로 전송하여 KQL todouble()이 정상 동작하도록 처리
        event_properties = {
            "total_samples": str(summary["total_samples"]),
            **{f"quality_{k}": str(v) if v is not None else "0" 
               for k, v in summary["quality_scores"].items()},
            **{f"safety_{k}": str(v) if v is not None else "0" 
               for k, v in summary["safety_scores"].items()},
        }
        
        logger.info("evaluation_result", extra={"custom_dimensions": event_properties})
        
        # Send individual row results as "evaluation_result" for Grafana Detail panel
        # 개별 행 결과를 "evaluation_result"로 전송 (Grafana Detail 패널용)
        if rows:
            for row in rows[:50]:
                row_properties = {
                    "query": str(row.get("inputs.query", row.get("query", "")))[:500],
                    "response": str(row.get("inputs.response", row.get("response", "")))[:1000],
                    "quality_fluency": str(row.get("outputs.fluency.fluency", row.get("quality_fluency", "0"))),
                    "quality_coherence": str(row.get("outputs.coherence.coherence", row.get("quality_coherence", "0"))),
                    "quality_relevance": str(row.get("outputs.relevance.relevance", row.get("quality_relevance", "0"))),
                    "quality_groundedness": str(row.get("outputs.groundedness.groundedness", row.get("quality_groundedness", "0"))),
                    "safety_violence": str(row.get("violence", row.get("safety_violence", "0"))),
                    "safety_sexual": str(row.get("sexual", row.get("safety_sexual", "0"))),
                    "safety_self_harm": str(row.get("self_harm", row.get("safety_self_harm", "0"))),
                    "safety_hate_unfairness": str(row.get("hate", row.get("safety_hate_unfairness", "0"))),
                }
                logger.info("evaluation_result", extra={"custom_dimensions": row_properties})
        
        # ★ CRITICAL: Flush and close handler before exit
        # ★ 중요: 스크립트 종료 전 반드시 flush/close 호출
        handler.flush()
        time.sleep(5)  # Wait for async batch to complete / 비동기 배치 전송 대기
        handler.close()
        logger.removeHandler(handler)
        
        print("✅ Sent evaluation results to Application Insights")
        
    except ImportError:
        print("⚠️ opencensus-ext-azure not installed. Using OpenTelemetry instead.")
        send_evaluation_via_otel(summary, rows)
    except Exception as e:
        print(f"⚠️ Failed to send to Application Insights: {e}")


def send_evaluation_via_otel(summary: dict, rows: list[dict] = None):
    """
    Send evaluation results via OpenTelemetry (OTel Collector → App Insights)
    Fallback when opencensus is unavailable
    
    OpenTelemetry를 통해 평가 결과 전송 (OTel Collector → App Insights)
    opencensus 미설치 시 대체 수단
    """
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        
        from config import OTEL_EXPORTER_OTLP_ENDPOINT
        
        # Tracer setup / Tracer 설정
        provider = TracerProvider()
        exporter = OTLPSpanExporter(endpoint=OTEL_EXPORTER_OTLP_ENDPOINT, insecure=True)
        processor = BatchSpanProcessor(exporter)
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)
        
        tracer = trace.get_tracer("evaluation-pipeline")
        
        # Create evaluation result span / 평가 결과 span 생성
        with tracer.start_as_current_span("evaluation_result") as span:
            span.set_attribute("total_samples", summary["total_samples"])
            
            for k, v in summary["quality_scores"].items():
                span.set_attribute(f"quality_{k}", v if v is not None else 0)
            
            for k, v in summary["safety_scores"].items():
                span.set_attribute(f"safety_{k}", v if v is not None else 0)
        
        # ★ CRITICAL: Force flush before exit / ★ 중요: 종료 전 강제 flush
        provider.force_flush()
        provider.shutdown()
        
        print("✅ Sent evaluation results via OpenTelemetry")
        
    except Exception as e:
        print(f"⚠️ Failed to send via OpenTelemetry: {e}")


async def run_evaluation_pipeline(hours: int = 24, limit: int = 100):
    """
    전체 평가 파이프라인 실행
    
    Args:
        hours: 쿼리할 시간 범위 (시간)
        limit: 최대 트레이스 수
    """
    print("=" * 60)
    print("🚀 Starting Evaluation Pipeline")
    print(f"   Time Range: Last {hours} hours")
    print(f"   Max Samples: {limit}")
    print("=" * 60)
    
    # 1. Application Insights에서 트레이스 쿼리
    traces = query_traces_from_app_insights(hours=hours, limit=limit)
    
    if not traces:
        print("❌ No traces found. Exiting.")
        return
    
    # 2. JSONL 파일로 저장
    data_path = save_traces_as_jsonl(traces, "evaluation_data.jsonl")
    
    # 3. 평가 실행
    all_results = {"metrics": {}, "rows": []}
    
    # Quality Evaluation
    try:
        quality_result = run_quality_evaluation(data_path)
        all_results["metrics"].update(quality_result.get("metrics", {}))
        all_results["rows"].extend(quality_result.get("rows", []))
    except Exception as e:
        print(f"⚠️ Quality evaluation error: {e}")
        # 결과 파일에서 직접 읽기 시도
        try:
            quality_file = OUTPUT_DIR / "quality_evaluation_result.json"
            if quality_file.exists():
                with open(quality_file, "r", encoding="utf-8") as f:
                    saved_result = json.load(f)
                rows = saved_result.get("rows", [])
                fluency_scores = [r.get("outputs.fluency.fluency", 0) for r in rows if r.get("outputs.fluency.fluency") is not None]
                if fluency_scores:
                    all_results["metrics"]["fluency.fluency"] = sum(fluency_scores) / len(fluency_scores)
                    all_results["rows"].extend(rows)
                    print(f"   ✅ Recovered quality metrics from saved file")
        except Exception as e2:
            print(f"   Could not recover from file: {e2}")
    
    # Safety Evaluation (Azure AI Content Safety)
    # Safety 평가 (Azure AI Content Safety 사용)
    safety_rows = []
    try:
        safety_result = run_safety_evaluation(traces)
        all_results["metrics"].update(safety_result.get("metrics", {}))
        safety_rows = safety_result.get("rows", [])
    except Exception as e:
        print(f"⚠️ Safety evaluation failed: {e}")
    
    # Merge safety scores into quality rows for Grafana Detail panel
    # Safety 점수를 quality rows에 병합하여 Grafana Detail 패널에 표시
    for i, row in enumerate(all_results["rows"]):
        if i < len(safety_rows):
            row["violence"] = safety_rows[i].get("violence", 0)
            row["hate"] = safety_rows[i].get("hate", 0)
            row["sexual"] = safety_rows[i].get("sexual", 0)
            row["self_harm"] = safety_rows[i].get("self_harm", 0)
    
    # 4. 결과 요약 / Generate summary
    summary = generate_evaluation_summary(all_results)
    
    # 5. Grafana용 저장
    save_evaluation_for_grafana(summary)
    
    # 6. Application Insights로 전송 (Grafana 대시보드용)
    send_evaluation_to_app_insights(summary, all_results.get("rows", []))
    
    # 7. 결과 출력
    print("\n" + "=" * 60)
    print("📊 Evaluation Summary")
    print("=" * 60)
    print(f"Total Samples: {summary['total_samples']}")
    print("\n🎯 Quality Scores:")
    for k, v in summary["quality_scores"].items():
        if v is not None:
            print(f"   {k}: {v:.2f}")
    print("\n🛡️ Safety Scores:")
    for k, v in summary["safety_scores"].items():
        if v is not None:
            print(f"   {k}: {v:.2f}")
    print("=" * 60)
    
    return summary


def main():
    """메인 실행"""
    import argparse
    
    parser = argparse.ArgumentParser(description="LangGraph Agent Evaluation Pipeline")
    parser.add_argument("--hours", type=int, default=24, help="Time range in hours (default: 24)")
    parser.add_argument("--limit", type=int, default=100, help="Max traces to evaluate (default: 100)")
    args = parser.parse_args()
    
    asyncio.run(run_evaluation_pipeline(hours=args.hours, limit=args.limit))


if __name__ == "__main__":
    main()
