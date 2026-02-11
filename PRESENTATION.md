# LLM Observability with OpenTelemetry
## otel-langfuse: LangGraph Agent의 관측 가능성 확보하기

---

## Agenda

| # | 주제 | 시간 |
|---|------|------|
| 1 | 왜 LLM Observability가 필요한가? | 5분 |
| 2 | 데모 앱: Teacher-Student 퀴즈 | 5분 |
| 3 | Architecture 딥다이브 | 10분 |
| 4 | 핵심 코드 워크스루 | 10분 |
| 5 | 라이브 데모 | 10분 |
| 6 | Q&A | 5분 |

---

## 1. 왜 LLM Observability가 필요한가?

### 문제 인식

LLM 기반 앱은 기존 앱과 다릅니다:

- **비결정적 출력**: 같은 입력에도 매번 다른 응답
- **Multi-Agent 복잡성**: Agent 간 호출 체인이 블랙박스
- **품질 측정 어려움**: "잘 됐나?"를 정량적으로 판단하기 어려움
- **비용/성능 추적**: 토큰 사용량, 응답 시간을 모니터링해야 함
- **안전성**: 유해 콘텐츠 생성 여부를 자동으로 감지해야 함

### 이 프로젝트가 해결하는 것

```
"LLM 앱을 만들었는데... 잘 돌아가는 건가요?"
                    ↓
  [Tracing] + [Dashboard] + [Automated Evaluation]
                    ↓
  "수치로 보여드리겠습니다"
```

| 관점 | 방법 | 도구 |
|------|------|------|
| **Tracing** | LLM input/output 자동 수집 | Traceloop SDK + OTel Collector |
| **Visualization** | 실시간 대시보드 | Langfuse + Azure Grafana |
| **Quality** | 응답 품질 자동 평가 | Azure AI Evaluation SDK |
| **Safety** | 유해 콘텐츠 자동 탐지 | Azure AI Content Safety |

---

## 2. 데모 앱: Teacher-Student 퀴즈

### LangGraph Multi-Agent 구조

```
사용자 입력: "보통 수학 문제"

  ┌─────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
  │  Setup   │────▶│ Teacher Question│────▶│ Student Answer  │────▶│ Teacher Evaluate│
  │ (난이도/ │     │ (문제 출제)      │     │ (풀이 시도)      │     │ (평가 & 피드백)  │
  │  영역)   │     └─────────────────┘     └─────────────────┘     └─────────────────┘
  └─────────┘           GPT-4o                  GPT-4o                  GPT-4o
```

### LangGraph 워크플로우 (graph.py)

```python
# 4개 노드, 조건부 엣지 포함
graph_builder = StateGraph(State)

graph_builder.add_node("setup", setup_handler)           # 난이도/영역 파싱
graph_builder.add_node("teacher_question", teacher_question)  # 문제 출제
graph_builder.add_node("student_answer", student_answer)      # 풀이
graph_builder.add_node("teacher_evaluate", teacher_evaluate)  # 평가

graph_builder.add_edge(START, "setup")
graph_builder.add_conditional_edges("setup", route_after_setup, {
    "teacher_question": "teacher_question",
    "end": END,
})
graph_builder.add_edge("teacher_question", "student_answer")
graph_builder.add_edge("student_answer", "teacher_evaluate")
graph_builder.add_edge("teacher_evaluate", END)
```

### 퀴즈 Phase 흐름

```
SETUP → QUESTIONING → ANSWERING → EVALUATING → COMPLETE
  ↑                                                 │
  └────────── "reset" / "새로 시작" ────────────────┘
```

### 라이브 데모 포인트
- http://localhost:8000 접속
- "보통 수학" 입력 → Teacher/Student 대화 관찰
- SSE 스트리밍으로 노드별 실시간 응답 확인

---

## 3. Architecture 딥다이브

### 전체 아키텍처

```
                          ┌───────────────────────────────────────────┐
                          │          Kubernetes Cluster               │
                          │                                           │
┌─────────────────┐       │  ┌──────────────┐     ┌──────────────┐  │
│   LangGraph     │ OTLP  │  │    OTel      │OTLP │   Langfuse   │  │
│   FastAPI App   │─gRPC──┼─▶│  Collector   │─HTTP┼─▶│  (Web UI)  │  │
│  + Traceloop    │       │  │              │     │  └──────────────┘  │
└─────────────────┘       │  │              │     │                    │
   Azure OpenAI           │  │              │     │                    │
     GPT-4o               │  └──────┬───────┘     │                    │
                          │         │Azure Monitor │                    │
                          └─────────┼──────────────┘                    │
                                    ▼                                    
                          ┌──────────────────┐                          
                          │ Azure Application │◀──── evaluation.py      
                          │     Insights      │      (평가 결과 저장)    
                          └────────┬─────────┘                          
                                   │ KQL Query                          
                                   ▼                                    
                          ┌──────────────────┐                          
                          │  Azure Managed   │                          
                          │     Grafana      │                          
                          └──────────────────┘                          
```

### 데이터 흐름 상세

| 단계 | From | To | 프로토콜 | 데이터 |
|------|------|----|----------|--------|
| ① | FastAPI App | OTel Collector | OTLP/gRPC (:4317) | LLM traces (input/output/tokens) |
| ② | OTel Collector | Langfuse | OTLP/HTTP | 동일 traces → LLM 특화 UI |
| ③ | OTel Collector | App Insights | Azure Monitor exporter | 동일 traces → KQL 쿼리 가능 |
| ④ | evaluation.py | App Insights | opencensus (customEvents) | 평가 점수 (quality + safety) |
| ⑤ | App Insights | Grafana | KQL query | traces + 평가 결과 시각화 |

### 왜 OTel Collector를 사용하는가?

**Collector 없이 (직접 전송)**
```
App → Langfuse    (SDK A 필요)
App → App Insights (SDK B 필요)
App → 다른 백엔드  (SDK C 필요)
```

**Collector 있으면 (Fan-out)**
```
App → OTel Collector → Langfuse
                     → App Insights
                     → (미래에 추가할 백엔드)
```

- 앱 코드 변경 없이 백엔드 추가/제거 가능
- 배치 처리, 메모리 제한 등 프로세싱을 Collector에서 처리
- 앱은 OTLP 하나만 알면 됨

### OTel Collector 설정 핵심 (k8s/otel-collector-values.yaml)

```yaml
config:
  receivers:
    otlp:                          # 앱에서 OTLP로 수신
      protocols:
        grpc:
          endpoint: 0.0.0.0:4317

  exporters:
    otlphttp/langfuse:             # Langfuse로 전달
      endpoint: "http://langfuse-web:3000/api/public/otel"
    azuremonitor:                  # App Insights로 전달
      connection_string: "InstrumentationKey=..."

  service:
    pipelines:
      traces:
        receivers: [otlp]
        processors: [memory_limiter, batch]
        exporters: [otlphttp/langfuse, azuremonitor]  # Fan-out!
```

---

## 4. 핵심 코드 워크스루

### 4-1. Traceloop 초기화 (main.py)

```python
from traceloop.sdk import Traceloop
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# OTLP Exporter → OTel Collector로 전송
otlp_exporter = OTLPSpanExporter(
    endpoint=OTEL_EXPORTER_OTLP_ENDPOINT,  # "http://collector:4317"
    insecure=True,
)

# Traceloop 초기화 → LangChain/OpenAI 자동 계측
Traceloop.init(
    app_name="teacher-student-quiz",
    disable_batch=False,
    exporter=otlp_exporter,
)
```

**포인트**: Traceloop SDK가 LangChain/OpenAI 호출을 자동으로 계측합니다.
코드 변경 없이 `gen_ai.prompt`, `gen_ai.completion`, `llm.usage.total_tokens` 등이 수집됩니다.

### 4-2. 수동 Span 추가 (main.py - chat_stream)

```python
# 자동 계측 외에 비즈니스 메타데이터를 수동으로 추가
with tracer.start_as_current_span("chat_stream") as span:
    # Langfuse가 인식하는 attribute 키
    span.set_attribute("langfuse.trace.name", "langgraph-session")
    span.set_attribute("langfuse.session.id", session_id)
    span.set_attribute("langfuse.trace.input", user_input)

    # LangGraph 실행
    async for event in graph.astream(invoke_state, config=config, stream_mode="updates"):
        # ... 스트리밍 처리 ...

    # 최종 응답도 기록
    span.set_attribute("langfuse.trace.output", final_output)
```

**포인트**: `langfuse.*` prefix attribute를 설정하면 Langfuse UI에서 trace/session 단위로 그룹핑됩니다.

### 4-3. Evaluation Pipeline (evaluation.py)

```python
# Step 1: App Insights에서 트레이스 쿼리 (KQL)
traces = query_traces_from_app_insights(hours=24, limit=100)
#   → AppDependencies 테이블에서 gen_ai.prompt, gen_ai.completion 추출

# Step 2: 품질 평가 (Azure AI Evaluation SDK)
result = evaluate(
    data=str(data_path),
    evaluators={
        "fluency": FluencyEvaluator(model_config),
        "coherence": CoherenceEvaluator(model_config),
        "relevance": RelevanceEvaluator(model_config),
        "groundedness": GroundednessEvaluator(model_config),
    },
)

# Step 3: 안전성 평가 (Azure AI Content Safety)
client = ContentSafetyClient(endpoint, credential)
response = client.analyze_text(AnalyzeTextOptions(text=text))
# → Violence, Hate, Sexual, SelfHarm 카테고리별 0-6 점수

# Step 4: 결과를 App Insights customEvents로 전송
logger.info("evaluation_result", extra={"custom_dimensions": event_properties})
# → Grafana에서 KQL로 조회 가능
```

### 4-4. Evaluation ↔ Observability 연결 고리

```
                        App Insights
                    ┌───────────────────┐
OTel Collector ────▶│ AppDependencies   │ ← LLM traces (자동)
                    │  (traces 저장)     │
                    │                   │      ┌──────────────────┐
evaluation.py ─────▶│ customEvents      │─────▶│ Grafana Dashboard│
  ↑  query (KQL)    │  (평가 결과 저장)  │      │  - Quality Scores│
  └─────────────────│                   │      │  - Safety Scores │
                    └───────────────────┘      │  - Trends        │
                                               └──────────────────┘
```

---

## 5. 라이브 데모 시나리오

### Demo 1: 퀴즈 앱 실행 & Tracing 확인

```bash
# 서버 시작
uv run main.py

# 브라우저에서 퀴즈 실행
# http://localhost:8000
# "보통 수학" 입력
```

**확인할 것:**
- Teacher → Student → Teacher 3단계 실행 관찰
- SSE 스트리밍으로 노드별 실시간 응답

### Demo 2: Langfuse에서 Trace 확인

**확인할 것:**
- `langgraph-session` trace 클릭
- 각 LLM 호출의 input/output 원문
- 토큰 사용량, latency
- session_id로 그룹핑된 대화 흐름

### Demo 3: Grafana 대시보드

**Grafana 패널 구성:**

| 패널 | 데이터 소스 | 설명 |
|------|------------|------|
| LangGraph Agent Summary | AppDependencies | 총 trace 수, LLM 호출, 토큰, 성공률 |
| Agent Execution Trends | AppDependencies | 시간별 실행 추이 (성공/실패) |
| LLM Call Trends | AppDependencies | 시간별 LLM 호출 수 + 토큰 |
| Node Performance | AppDependencies | 노드별 평균 처리시간, 호출 수 |
| Quality Evaluation Scores | customEvents | Fluency, Coherence, Relevance, Groundedness |
| Safety Evaluation Scores | customEvents | Violence, Sexual, SelfHarm, HateUnfairness |
| Score Trends | customEvents | 시간별 품질/안전성 점수 추이 |

### Demo 4: Evaluation Pipeline 실행

```bash
# 최근 24시간 트레이스에 대해 평가 실행
uv run python evaluation.py --hours 24 --limit 100
```

**확인할 것:**
- App Insights에서 트레이스 쿼리
- 품질 평가 점수 (1-5점 스케일)
- 안전성 평가 점수 (0-6점, 0=안전)
- 결과가 Grafana에 반영되는 과정

---

## 6. 기술 스택 요약

| 레이어 | 기술 | 역할 |
|--------|------|------|
| **Application** | FastAPI + LangGraph | Web server + Multi-Agent workflow |
| **LLM** | Azure OpenAI GPT-4o | Teacher/Student Agent의 두뇌 |
| **Auto-Instrumentation** | Traceloop SDK | LangChain/OpenAI 호출 자동 계측 |
| **Trace Transport** | OTel Collector (K8s) | OTLP 수신 → Langfuse + App Insights fan-out |
| **LLM Observability** | Langfuse (K8s) | LLM 특화 trace UI |
| **Trace Storage** | Azure Application Insights | 범용 trace 저장소 + KQL 쿼리 |
| **Dashboard** | Azure Managed Grafana | 커스텀 대시보드 (KQL 기반) |
| **Quality Eval** | Azure AI Evaluation SDK | Fluency, Coherence, Relevance, Groundedness |
| **Safety Eval** | Azure AI Content Safety | Violence, Sexual, SelfHarm, Hate 탐지 |

---

## 핵심 Takeaway

1. **OpenTelemetry는 LLM 앱에도 적용 가능하다**
   - Traceloop SDK로 LangChain/OpenAI 자동 계측
   - 기존 OTel 인프라 재활용 (Collector, App Insights, Grafana)

2. **OTel Collector = 만능 라우터**
   - 앱 코드 수정 없이 여러 백엔드로 fan-out
   - Langfuse(LLM 특화) + App Insights(범용) 동시 전송

3. **Observability + Evaluation = 완전한 모니터링**
   - Tracing만으로는 부족 → 자동화된 품질/안전성 평가 필수
   - 평가 결과를 다시 App Insights에 저장 → Grafana에서 통합 조회

4. **프로덕션 적용 시 고려사항**
   - `OTEL_ATTRIBUTE_VALUE_LENGTH_LIMIT` 늘리기 (LLM 메시지가 김)
   - `TRACELOOP_TRACE_CONTENT=true` 설정 (기본은 input/output 비캡처)
   - Evaluation은 배치로 주기적 실행 (CI/CD 또는 cron)

---

## 참고 링크

- [OpenTelemetry](https://opentelemetry.io/)
- [Traceloop SDK](https://github.com/traceloop/openllmetry)
- [LangGraph](https://langchain-ai.github.io/langgraph/)
- [Langfuse](https://langfuse.com/)
- [Azure AI Evaluation SDK](https://learn.microsoft.com/azure/ai-studio/how-to/develop/evaluate-sdk)
- [Azure AI Content Safety](https://learn.microsoft.com/azure/ai-services/content-safety/)
- [Azure Managed Grafana](https://learn.microsoft.com/azure/managed-grafana/)
