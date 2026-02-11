"""FastAPI Chat Agent with LangGraph - Teacher-Student Quiz System with OpenTelemetry"""
import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, AsyncGenerator
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# OpenTelemetry + Traceloop for LLM tracing
from traceloop.sdk import Traceloop
from opentelemetry import trace

from langchain_core.messages import HumanMessage

from config import AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT_NAME, OTEL_EXPORTER_OTLP_ENDPOINT
from graph import (
    create_graph, 
    QuizPhase,
)
from eval_background import evaluate_single

# Global
graph = None
# Session state storage per session (phase, difficulty, subject, etc.) / 세션별 상태 저장
session_states = {}


# OpenTelemetry tracer
tracer = None

def setup_opentelemetry():
    """Initialize OpenTelemetry + Traceloop (LLM input/output capture) / OpenTelemetry + Traceloop 초기화"""
    global tracer
    
    import os
    # Increase attribute length limit (default is too small for LLM messages) / Attribute 길이 제한 늘리기
    os.environ.setdefault("OTEL_ATTRIBUTE_VALUE_LENGTH_LIMIT", "65535")
    # Enable Traceloop content capture / Traceloop content 캡처 활성화
    os.environ.setdefault("TRACELOOP_TRACE_CONTENT", "true")
    
    # Initialize Traceloop - auto-instrument LangChain, OpenAI, etc. / Traceloop 초기화
    # Create exporter to send to OTel Collector / exporter를 직접 생성하여 OTel Collector로 전송
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    
    otlp_exporter = OTLPSpanExporter(
        endpoint=OTEL_EXPORTER_OTLP_ENDPOINT,
        insecure=True,
    )
    
    Traceloop.init(
        app_name="teacher-student-quiz",
        disable_batch=False,
        exporter=otlp_exporter,
    )
    
    tracer = trace.get_tracer(__name__)
    
    print(f"✅ OpenTelemetry + Traceloop initialized!")
    print(f"   OTLP Endpoint: {OTEL_EXPORTER_OTLP_ENDPOINT}")
    
    return tracer


@asynccontextmanager
async def lifespan(app: FastAPI):
    global graph, tracer
    try:
        # Initialize OpenTelemetry / OpenTelemetry 초기화
        tracer = setup_opentelemetry()
        
        graph = create_graph()
        print("✅ LangGraph Teacher-Student Quiz Agent initialized!")
        print(f"   Endpoint: {AZURE_OPENAI_ENDPOINT}")
        print(f"   Deployment: {AZURE_OPENAI_DEPLOYMENT_NAME}")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        raise e
    yield
    print("Shutting down...")


app = FastAPI(
    title="LangGraph Chat Agent",
    description="Chat agent powered by LangGraph and Azure OpenAI",
    version="1.0.0",
    lifespan=lifespan
)

# Static file serving / Static 파일 서빙
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None  # Session ID (creates new if not provided) / 세션 ID


class ChatResponse(BaseModel):
    response: str
    session_id: str  # Session ID for client's next request / 클라이언트가 다음 요청에 사용할 세션 ID


@app.get("/", response_class=HTMLResponse)
async def root():
    template_path = Path(__file__).parent / "templates" / "index.html"
    return template_path.read_text(encoding="utf-8")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not graph:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        # Handle session ID (create new if not provided) / 세션 ID 처리
        session_id = request.session_id or str(uuid4())
        
        # Get or initialize session state / 세션 상태 가져오기 또는 초기화
        if session_id not in session_states:
            session_states[session_id] = {
                "phase": QuizPhase.SETUP,
                "difficulty": None,
                "subject": None,
                "round_count": 0,
            }
        
        current_state = session_states[session_id]
        user_input = request.message.strip()
        
        # Use LangGraph built-in checkpointer / LangGraph 내장 checkpointer 사용
        config = {"configurable": {"thread_id": session_id}}
        
        # Process based on current phase / 현재 phase에 따른 처리
        phase = current_state.get("phase", QuizPhase.SETUP)
        
        # Handle reset commands / 리셋 명령 처리
        if any(word in user_input.lower() for word in ["새로", "리셋", "reset", "다시", "처음"]):
            session_states[session_id] = {
                "phase": QuizPhase.SETUP,
                "difficulty": None,
                "subject": None,
                "round_count": 0,
            }
            current_state = session_states[session_id]
            phase = QuizPhase.SETUP
        
        # Handle next question commands / 다음 문제 명령 처리
        if phase == QuizPhase.COMPLETE and any(word in user_input.lower() for word in ["다음", "계속", "next", "continue", "더"]):
            phase = QuizPhase.QUESTIONING
            current_state["phase"] = phase
        
        # Prepare graph invoke state / 그래프 invoke 준비
        invoke_state = {
            "messages": [HumanMessage(content=user_input)],
            "user_input": user_input,
            "phase": phase,
            "difficulty": current_state.get("difficulty"),
            "subject": current_state.get("subject"),
            "round_count": current_state.get("round_count", 0),
        }
        
        # Execute graph / 그래프 실행
        result = graph.invoke(invoke_state, config=config)
        
        # Update session state / 세션 상태 업데이트
        session_states[session_id] = {
            "phase": result.get("phase", QuizPhase.SETUP),
            "difficulty": result.get("difficulty"),
            "subject": result.get("subject"),
            "round_count": result.get("round_count", 0),
        }
        
        # Collect all new messages / 모든 새 메시지 수집
        all_responses = []
        for msg in result["messages"]:
            if hasattr(msg, 'content') and msg.content:
                # Collect only non-HumanMessage / HumanMessage가 아닌 것만 수집
                if not isinstance(msg, HumanMessage):
                    all_responses.append(msg.content)
        
        # Return last AI response (combine if multiple) / 마지막 AI 응답 반환
        response_text = "\n\n".join(all_responses) if all_responses else "Unable to generate response. / 응답을 생성할 수 없습니다."
        
        # Background evaluation (non-blocking) / 백그라운드 평가 (비동기, latency 영향 없음)
        asyncio.create_task(
            asyncio.to_thread(evaluate_single, request.message, response_text)
        )
        
        return ChatResponse(response=response_text, session_id=session_id)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Stream agent conversation via SSE token-by-token through LangGraph / SSE 스트리밍으로 LangGraph를 통한 에이전트 대화를 토큰 단위로 실시간 전송"""
    if not graph:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            # Handle session ID / 세션 ID 처리
            session_id = request.session_id or str(uuid4())
            
            # Get or initialize session state / 세션 상태 가져오기 또는 초기화
            if session_id not in session_states:
                session_states[session_id] = {
                    "phase": QuizPhase.SETUP,
                    "difficulty": None,
                    "subject": None,
                    "round_count": 0,
                }
            
            current_state = session_states[session_id]
            user_input = request.message.strip()
            
            phase = current_state.get("phase", QuizPhase.SETUP)
            
            # Send session ID / 세션 ID 전송
            yield f"data: {json.dumps({'type': 'session', 'session_id': session_id})}\n\n"
            
            # Handle reset commands / 리셋 명령 처리
            if any(word in user_input.lower() for word in ["새로", "리셋", "reset", "다시", "처음"]):
                session_states[session_id] = {
                    "phase": QuizPhase.SETUP,
                    "difficulty": None,
                    "subject": None,
                    "round_count": 0,
                }
                current_state = session_states[session_id]
                phase = QuizPhase.SETUP
            
            # Handle next question commands / 다음 문제 명령 처리
            if phase == QuizPhase.COMPLETE and any(word in user_input.lower() for word in ["다음", "계속", "next", "continue", "더"]):
                phase = QuizPhase.QUESTIONING
                current_state["phase"] = phase
            
            # LangGraph 설정
            config = {
                "configurable": {"thread_id": session_id},
            }
            
            # OpenTelemetry span for tracing (Langfuse optimized attributes) / OpenTelemetry span으로 트레이싱
            with tracer.start_as_current_span("chat_stream") as span:
                # Langfuse Trace-Level Attributes (범용)
                span.set_attribute("langfuse.trace.name", "langgraph-session")
                span.set_attribute("langfuse.session.id", session_id)
                span.set_attribute("langfuse.trace.input", user_input)
            
                # Prepare graph invoke state / 그래프 invoke 준비
                invoke_state = {
                    "messages": [HumanMessage(content=user_input)],
                    "user_input": user_input,
                    "phase": phase,
                    "difficulty": current_state.get("difficulty"),
                    "subject": current_state.get("subject"),
                    "round_count": current_state.get("round_count", 0),
                }
                
                # Use astream for LangGraph execution (stream_mode="updates" for per-node streaming)
                # astream으로 LangGraph 실행 (stream_mode="updates"로 노드별 결과 스트리밍)
                current_node = None
                node_labels = {
                    "teacher_question": "👨‍🏫 Teacher (Question) / 👨‍🏫 Teacher (문제)",
                    "student_answer": "🧑‍🎓 Student",
                    "teacher_evaluate": "👨‍🏫 Teacher (Evaluate) / 👨‍🏫 Teacher (평가)",
                }
                
                # stream_mode="updates" for per-node result streaming
                # Note: traceloop-sdk auto-instruments LLM calls (gen_ai.prompt, gen_ai.completion)
                # Here we only add node-level metadata
                final_output = ""  # Track final output / 최종 출력 추적용
                async for event in graph.astream(invoke_state, config=config, stream_mode="updates"):
                    for node_name, node_output in event.items():
                        print(f"[DEBUG] node={node_name}, output_keys={node_output.keys() if isinstance(node_output, dict) else 'not dict'}")
                        
                        # Extract messages / 메시지 추출
                        if isinstance(node_output, dict) and "messages" in node_output:
                            for msg in node_output["messages"]:
                                if hasattr(msg, "content") and msg.content:
                                    content = msg.content
                                    final_output = content  # Save final output / 최종 출력 저장
                                    
                                    # Set node-specific labels / 노드별 라벨 설정
                                    label = node_labels.get(node_name, node_name)
                                    if node_name == "teacher_question":
                                        rc = current_state.get("round_count", 0) + 1
                                        current_state["round_count"] = rc
                                        label = f"👨‍🏫 Teacher (Question #{rc}) / 👨‍🏫 Teacher (문제 #{rc})"
                                    
                                    # Node start notification / 노드 시작 알림
                                    if node_name in node_labels:
                                        yield f"data: {json.dumps({'type': 'node_start', 'node': node_name, 'label': label}, ensure_ascii=False)}\n\n"
                                    
                                    # Send full message (typing effect on frontend) / 전체 메시지 전송
                                    yield f"data: {json.dumps({'type': 'message', 'node': node_name, 'content': content}, ensure_ascii=False)}\n\n"
                                    
                                    # Node end / 노드 종료
                                    if node_name in node_labels:
                                        yield f"data: {json.dumps({'type': 'node_end', 'node': node_name})}\n\n"
                                    
                                    # Show waiting for next node / 다음 노드 대기 표시
                                    if node_name == "setup" and "퀴즈 설정 완료" in content:
                                        yield f"data: {json.dumps({'type': 'waiting', 'message': '👨‍🏫 Teacher is preparing a question... / Teacher가 문제를 준비 중...'})}\n\n"
                                    elif node_name == "teacher_question":
                                        yield f"data: {json.dumps({'type': 'waiting', 'message': '🧑‍🎓 Student is thinking... / Student가 생각 중...'})}\n\n"
                                    elif node_name == "student_answer":
                                        yield f"data: {json.dumps({'type': 'waiting', 'message': '👨‍🏫 Teacher is evaluating... / Teacher가 평가 중...'})}\n\n"
                            
                                    await asyncio.sleep(0.1)
                
                # Set trace output (final response) / Trace output 설정 (최종 응답)
                if final_output:
                    span.set_attribute("langfuse.trace.output", final_output[:10000] if len(final_output) > 10000 else final_output)
            
            # Get final state / 최종 상태 가져오기
            final_state = graph.get_state(config)
            if final_state and final_state.values:
                session_states[session_id] = {
                    "phase": final_state.values.get("phase", QuizPhase.SETUP),
                    "difficulty": final_state.values.get("difficulty"),
                    "subject": final_state.values.get("subject"),
                    "round_count": final_state.values.get("round_count", 0),
                }
            
            # Background evaluation (non-blocking) / 백그라운드 평가 (비동기, latency 영향 없음)
            if final_output and user_input:
                asyncio.create_task(
                    asyncio.to_thread(evaluate_single, user_input, final_output)
                )
            
            # Done event / 완료 이벤트
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            
        except Exception as e:
            print(f"Streaming Error: {e}")
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@app.get("/health")
async def health_check():
    return {"status": "healthy", "graph_initialized": graph is not None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
