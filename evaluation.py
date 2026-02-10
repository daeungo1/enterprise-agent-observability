"""
Azure AI Evaluation Pipeline for LangGraph Agent
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
    QAEvaluator,
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
    """Azure OpenAI 모델 설정"""
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
    Application Insights에서 LLM 트레이스 데이터를 쿼리합니다.
    
    Returns:
        list[dict]: query, response, context를 포함한 트레이스 데이터
    """
    if not APP_INSIGHTS_WORKSPACE_ID:
        print("⚠️ APP_INSIGHTS_WORKSPACE_ID not set. Using sample data.")
        return get_sample_data()
    
    credential = DefaultAzureCredential()
    client = LogsQueryClient(credential)
    
    # KQL 쿼리: AppDependencies 테이블에서 LLM input/output 추출
    # OpenTelemetry 트레이스 데이터는 AppDependencies에 저장됨
    # Langfuse/Traceloop SDK가 langfuse.trace.input/output 키로 저장
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
        
        # AppDependencies에 없으면 AppEvents 시도
        print("   No traces found in 'AppDependencies'. Trying 'AppEvents'...")
        return query_from_app_events(client, hours, limit)
        
    except Exception as e:
        print(f"❌ Error querying Application Insights: {e}")
        print("Using sample data instead.")
        return get_sample_data()


def query_from_app_events(client: LogsQueryClient, hours: int, limit: int) -> list[dict]:
    """AppEvents 테이블에서 LLM 트레이스 쿼리"""
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
    """샘플 테스트 데이터"""
    return [
        {
            "query": "보통 수학 문제 풀래",
            "response": "👨‍🏫 **Teacher (문제 #1)**\n\n다음 방정식을 풀어보세요: 2x + 5 = 15",
            "context": "Teacher-Student 퀴즈 시스템",
            "ground_truth": "x = 5",
        },
        {
            "query": "쉬운 역사 퀴즈",
            "response": "👨‍🏫 **Teacher (문제 #1)**\n\n대한민국의 수도는 어디인가요?",
            "context": "Teacher-Student 퀴즈 시스템",
            "ground_truth": "서울",
        },
        {
            "query": "프로그래밍 문제 출제해줘",
            "response": "👨‍🏫 **Teacher (문제 #1)**\n\nPython에서 리스트의 모든 요소를 합하는 함수를 작성하세요.",
            "context": "Teacher-Student 퀴즈 시스템",
            "ground_truth": "sum() 함수 사용 또는 for 루프",
        },
    ]


def save_traces_as_jsonl(traces: list[dict], filename: str) -> Path:
    """트레이스를 JSONL 파일로 저장 (평가에 필요한 필드만)"""
    filepath = OUTPUT_DIR / filename
    with open(filepath, "w", encoding="utf-8") as f:
        for trace in traces:
            # 평가에 필요한 필드만 추출 (timestamp 제외)
            eval_data = {
                "query": str(trace.get("query", "")),
                "response": str(trace.get("response", "")),
                "context": str(trace.get("context", "")),
            }
            f.write(json.dumps(eval_data, ensure_ascii=False) + "\n")
    print(f"✅ Saved {len(traces)} traces to {filepath}")
    return filepath


def run_quality_evaluation(data_path: Path) -> dict:
    """
    품질 평가 실행 (Fluency, QA)
    """
    model_config = get_model_config()
    
    # Quality Evaluators
    fluency_eval = FluencyEvaluator(model_config)
    
    print("🔍 Running Quality Evaluation...")
    
    output_file = OUTPUT_DIR / "quality_evaluation_result.json"
    
    result = evaluate(
        data=str(data_path),
        evaluators={
            "fluency": fluency_eval,
        },
        evaluator_config={
            "fluency": {
                "column_mapping": {
                    "query": "${data.query}",
                    "response": "${data.response}",
                }
            },
        },
        output_path=str(output_file),
    )
    
    print("✅ Quality Evaluation completed")
    
    # 결과 파일에서 메트릭 추출 (Timestamp 직렬화 문제 우회)
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            saved_result = json.load(f)
        
        # metrics 계산
        rows = saved_result.get("rows", [])
        fluency_scores = [r.get("outputs.fluency.fluency", 0) for r in rows if r.get("outputs.fluency.fluency") is not None]
        
        metrics = {
            "fluency.fluency": sum(fluency_scores) / len(fluency_scores) if fluency_scores else None
        }
        
        return {"metrics": metrics, "rows": rows}
    except Exception as e:
        print(f"   Warning: Could not parse saved result: {e}")
        return {"metrics": {}, "rows": []}


def run_safety_evaluation(traces: list[dict]) -> dict:
    """
    Azure AI Content Safety를 사용한 안전성 평가
    Violence, Hate, Sexual, SelfHarm 카테고리 분석
    """
    if not AZURE_CONTENT_SAFETY_ENDPOINT or not AZURE_CONTENT_SAFETY_KEY:
        print("⚠️ Azure AI Content Safety not configured. Skipping safety evaluation.")
        return {"metrics": {}, "rows": []}
    
    print("🔍 Running Safety Evaluation (Azure AI Content Safety)...")
    
    # Content Safety Client 생성
    client = ContentSafetyClient(
        endpoint=AZURE_CONTENT_SAFETY_ENDPOINT,
        credential=AzureKeyCredential(AZURE_CONTENT_SAFETY_KEY)
    )
    
    # 각 카테고리별 점수 집계
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


def run_qa_evaluation(data_path: Path) -> dict:
    """
    QA 평가 실행 (Groundedness, Relevance, Coherence, Fluency, Similarity, F1)
    """
    model_config = get_model_config()
    
    # QA Evaluator (복합 평가자)
    qa_eval = QAEvaluator(model_config)
    
    print("🔍 Running QA Evaluation...")
    
    output_file = OUTPUT_DIR / "qa_evaluation_result.json"
    
    result = evaluate(
        data=str(data_path),
        evaluators={
            "qa": qa_eval,
        },
        evaluator_config={
            "qa": {
                "column_mapping": {
                    "query": "${data.query}",
                    "response": "${data.response}",
                    "context": "${data.context}",
                    "ground_truth": "${data.ground_truth}",
                }
            },
        },
        output_path=str(output_file),
    )
    
    print("✅ QA Evaluation completed")
    
    # 결과 파일에서 메트릭 추출 (Timestamp 직렬화 문제 우회)
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            saved_result = json.load(f)
        
        # metrics 계산
        rows = saved_result.get("rows", [])
        coherence_scores = [r.get("outputs.qa.coherence", 0) for r in rows if r.get("outputs.qa.coherence") is not None]
        relevance_scores = [r.get("outputs.qa.relevance", 0) for r in rows if r.get("outputs.qa.relevance") is not None]
        groundedness_scores = [r.get("outputs.qa.groundedness", 0) for r in rows if r.get("outputs.qa.groundedness") is not None]
        
        metrics = {
            "qa.coherence": sum(coherence_scores) / len(coherence_scores) if coherence_scores else None,
            "qa.relevance": sum(relevance_scores) / len(relevance_scores) if relevance_scores else None,
            "qa.groundedness": sum(groundedness_scores) / len(groundedness_scores) if groundedness_scores else None,
        }
        
        return {"metrics": metrics, "rows": rows}
    except Exception as e:
        print(f"   Warning: Could not parse saved result: {e}")
        return {"metrics": {}, "rows": []}


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
        "coherence": metrics.get("qa.coherence", None),
        "relevance": metrics.get("qa.relevance", None),
        "groundedness": metrics.get("qa.groundedness", None),
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
    평가 결과를 Application Insights로 전송 (customEvents)
    Grafana 대시보드에서 조회 가능
    """
    if not APP_INSIGHTS_CONNECTION_STRING:
        print("⚠️ APP_INSIGHTS_CONNECTION_STRING not set. Skipping telemetry.")
        return
    
    try:
        from opencensus.ext.azure import metrics_exporter
        from opencensus.ext.azure.log_exporter import AzureEventHandler
        import logging
        
        # Azure Event Handler 설정
        logger = logging.getLogger("evaluation")
        logger.setLevel(logging.INFO)
        
        handler = AzureEventHandler(connection_string=APP_INSIGHTS_CONNECTION_STRING)
        logger.addHandler(handler)
        
        # 요약 메트릭 전송
        event_properties = {
            "total_samples": str(summary["total_samples"]),
            **{f"quality_{k}": str(v) if v is not None else "null" 
               for k, v in summary["quality_scores"].items()},
            **{f"safety_{k}": str(v) if v is not None else "null" 
               for k, v in summary["safety_scores"].items()},
        }
        
        logger.info("evaluation_result", extra={"custom_dimensions": event_properties})
        
        # 개별 행 결과 전송 (선택적)
        if rows:
            for row in rows[:50]:  # 최대 50개까지만
                row_properties = {
                    "query": str(row.get("inputs.query", ""))[:500],
                    "response": str(row.get("inputs.response", ""))[:500],
                    **{k: str(v) for k, v in row.items() if k.startswith("outputs.")}
                }
                logger.info("evaluation_row", extra={"custom_dimensions": row_properties})
        
        print("✅ Sent evaluation results to Application Insights")
        
    except ImportError:
        print("⚠️ opencensus-ext-azure not installed. Using OpenTelemetry instead.")
        send_evaluation_via_otel(summary, rows)
    except Exception as e:
        print(f"⚠️ Failed to send to Application Insights: {e}")


def send_evaluation_via_otel(summary: dict, rows: list[dict] = None):
    """
    OpenTelemetry를 통해 평가 결과 전송 (OTel Collector → App Insights)
    """
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        
        from config import OTEL_EXPORTER_OTLP_ENDPOINT
        
        # Tracer 설정
        provider = TracerProvider()
        exporter = OTLPSpanExporter(endpoint=OTEL_EXPORTER_OTLP_ENDPOINT, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        
        tracer = trace.get_tracer("evaluation-pipeline")
        
        # 평가 결과 span 생성
        with tracer.start_as_current_span("evaluation_result") as span:
            span.set_attribute("total_samples", summary["total_samples"])
            
            for k, v in summary["quality_scores"].items():
                if v is not None:
                    span.set_attribute(f"quality_{k}", v)
            
            for k, v in summary["safety_scores"].items():
                if v is not None:
                    span.set_attribute(f"safety_{k}", v)
        
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
    
    # Safety Evaluation (Azure AI Content Safety 사용)
    try:
        safety_result = run_safety_evaluation(traces)
        all_results["metrics"].update(safety_result.get("metrics", {}))
    except Exception as e:
        print(f"⚠️ Safety evaluation failed: {e}")
    
    # QA Evaluation (ground_truth가 있는 경우)
    if any("ground_truth" in t for t in traces):
        try:
            qa_result = run_qa_evaluation(data_path)
            all_results["metrics"].update(qa_result.get("metrics", {}))
        except Exception as e:
            print(f"⚠️ QA evaluation failed: {e}")
    
    # 4. 결과 요약
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
