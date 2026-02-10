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

# Global
graph = None
# 세션별 상태 저장 (phase, difficulty, subject 등)
session_states = {}


# OpenTelemetry tracer
tracer = None

def setup_opentelemetry():
    """OpenTelemetry + Traceloop 초기화 (LLM input/output 캡처)"""
    global tracer
    
    import os
    # Attribute 길이 제한 늘리기 (기본값이 작아서 LLM 메시지가 잘림)
    os.environ.setdefault("OTEL_ATTRIBUTE_VALUE_LENGTH_LIMIT", "65535")
    # Traceloop content 캡처 활성화
    os.environ.setdefault("TRACELOOP_TRACE_CONTENT", "true")
    
    # Traceloop 초기화 - LangChain, OpenAI 등 자동 계측
    # exporter를 직접 생성하여 OTel Collector로 전송
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
        # OpenTelemetry 초기화
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

# Static 파일 서빙
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
    session_id: Optional[str] = None  # 세션 ID (없으면 새로 생성)


class ChatResponse(BaseModel):
    response: str
    session_id: str  # 클라이언트가 다음 요청에 사용할 세션 ID


@app.get("/", response_class=HTMLResponse)
async def root():
    template_path = Path(__file__).parent / "templates" / "index.html"
    return template_path.read_text(encoding="utf-8")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not graph:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        # 세션 ID 처리 (없으면 새로 생성)
        session_id = request.session_id or str(uuid4())
        
        # 세션 상태 가져오기 또는 초기화
        if session_id not in session_states:
            session_states[session_id] = {
                "phase": QuizPhase.SETUP,
                "difficulty": None,
                "subject": None,
                "round_count": 0,
            }
        
        current_state = session_states[session_id]
        user_input = request.message.strip()
        
        # LangGraph 내장 checkpointer 사용
        config = {"configurable": {"thread_id": session_id}}
        
        # 현재 phase에 따른 처리
        phase = current_state.get("phase", QuizPhase.SETUP)
        
        # 리셋 명령 처리
        if any(word in user_input.lower() for word in ["새로", "리셋", "reset", "다시", "처음"]):
            session_states[session_id] = {
                "phase": QuizPhase.SETUP,
                "difficulty": None,
                "subject": None,
                "round_count": 0,
            }
            current_state = session_states[session_id]
            phase = QuizPhase.SETUP
        
        # 다음 문제 명령 처리
        if phase == QuizPhase.COMPLETE and any(word in user_input.lower() for word in ["다음", "계속", "next", "continue", "더"]):
            phase = QuizPhase.QUESTIONING
            current_state["phase"] = phase
        
        # 그래프 invoke 준비
        invoke_state = {
            "messages": [HumanMessage(content=user_input)],
            "user_input": user_input,
            "phase": phase,
            "difficulty": current_state.get("difficulty"),
            "subject": current_state.get("subject"),
            "round_count": current_state.get("round_count", 0),
        }
        
        # 그래프 실행
        result = graph.invoke(invoke_state, config=config)
        
        # 세션 상태 업데이트
        session_states[session_id] = {
            "phase": result.get("phase", QuizPhase.SETUP),
            "difficulty": result.get("difficulty"),
            "subject": result.get("subject"),
            "round_count": result.get("round_count", 0),
        }
        
        # 모든 새 메시지 수집
        all_responses = []
        for msg in result["messages"]:
            if hasattr(msg, 'content') and msg.content:
                # HumanMessage가 아닌 것만 수집
                if not isinstance(msg, HumanMessage):
                    all_responses.append(msg.content)
        
        # 마지막 AI 응답 반환 (여러 개면 합침)
        response_text = "\n\n".join(all_responses) if all_responses else "응답을 생성할 수 없습니다."
        
        return ChatResponse(response=response_text, session_id=session_id)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """SSE 스트리밍으로 LangGraph를 통한 에이전트 대화를 토큰 단위로 실시간 전송"""
    if not graph:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            # 세션 ID 처리
            session_id = request.session_id or str(uuid4())
            
            # 세션 상태 가져오기 또는 초기화
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
            
            # 세션 ID 전송
            yield f"data: {json.dumps({'type': 'session', 'session_id': session_id})}\n\n"
            
            # 리셋 명령 처리
            if any(word in user_input.lower() for word in ["새로", "리셋", "reset", "다시", "처음"]):
                session_states[session_id] = {
                    "phase": QuizPhase.SETUP,
                    "difficulty": None,
                    "subject": None,
                    "round_count": 0,
                }
                current_state = session_states[session_id]
                phase = QuizPhase.SETUP
            
            # 다음 문제 명령 처리
            if phase == QuizPhase.COMPLETE and any(word in user_input.lower() for word in ["다음", "계속", "next", "continue", "더"]):
                phase = QuizPhase.QUESTIONING
                current_state["phase"] = phase
            
            # LangGraph 설정
            config = {
                "configurable": {"thread_id": session_id},
            }
            
            # OpenTelemetry span으로 트레이싱 (Langfuse 최적화 속성 사용)
            with tracer.start_as_current_span("chat_stream") as span:
                # Langfuse Trace-Level Attributes (범용)
                span.set_attribute("langfuse.trace.name", "langgraph-session")
                span.set_attribute("langfuse.session.id", session_id)
                span.set_attribute("langfuse.trace.input", user_input)
            
                # 그래프 invoke 준비
                invoke_state = {
                    "messages": [HumanMessage(content=user_input)],
                    "user_input": user_input,
                    "phase": phase,
                    "difficulty": current_state.get("difficulty"),
                    "subject": current_state.get("subject"),
                    "round_count": current_state.get("round_count", 0),
                }
                
                # astream으로 LangGraph 실행 (stream_mode="updates"로 노드별 결과 스트리밍)
                current_node = None
                node_labels = {
                    "teacher_question": "👨‍🏫 Teacher (문제)",
                    "student_answer": "🧑‍🎓 Student",
                    "teacher_evaluate": "👨‍🏫 Teacher (평가)",
                }
                
                # stream_mode="updates"로 노드별 결과 스트리밍
                # Note: traceloop-sdk가 LLM 호출(gen_ai.prompt, gen_ai.completion)을 자동 계측
                # 여기서는 노드 레벨 메타데이터만 추가
                final_output = ""  # 최종 출력 추적용
                async for event in graph.astream(invoke_state, config=config, stream_mode="updates"):
                    for node_name, node_output in event.items():
                        print(f"[DEBUG] node={node_name}, output_keys={node_output.keys() if isinstance(node_output, dict) else 'not dict'}")
                        
                        # 메시지 추출
                        if isinstance(node_output, dict) and "messages" in node_output:
                            for msg in node_output["messages"]:
                                if hasattr(msg, "content") and msg.content:
                                    content = msg.content
                                    final_output = content  # 최종 출력 저장
                                    
                                    # 노드별 라벨 설정
                                    label = node_labels.get(node_name, node_name)
                                    if node_name == "teacher_question":
                                        rc = current_state.get("round_count", 0) + 1
                                        current_state["round_count"] = rc
                                        label = f"👨‍🏫 Teacher (문제 #{rc})"
                                    
                                    # 노드 시작 알림
                                    if node_name in node_labels:
                                        yield f"data: {json.dumps({'type': 'node_start', 'node': node_name, 'label': label}, ensure_ascii=False)}\n\n"
                                    
                                    # 전체 메시지 전송 (타이핑 효과는 프론트에서)
                                    yield f"data: {json.dumps({'type': 'message', 'node': node_name, 'content': content}, ensure_ascii=False)}\n\n"
                                    
                                    # 노드 종료
                                    if node_name in node_labels:
                                        yield f"data: {json.dumps({'type': 'node_end', 'node': node_name})}\n\n"
                                    
                                    # 다음 노드 대기 표시
                                    if node_name == "setup" and "퀴즈 설정 완료" in content:
                                        yield f"data: {json.dumps({'type': 'waiting', 'message': '👨‍🏫 Teacher가 문제를 준비 중...'})}\n\n"
                                    elif node_name == "teacher_question":
                                        yield f"data: {json.dumps({'type': 'waiting', 'message': '🧑‍🎓 Student가 생각 중...'})}\n\n"
                                    elif node_name == "student_answer":
                                        yield f"data: {json.dumps({'type': 'waiting', 'message': '👨‍🏫 Teacher가 평가 중...'})}\n\n"
                            
                                    await asyncio.sleep(0.1)
                
                # Trace output 설정 (최종 응답)
                if final_output:
                    span.set_attribute("langfuse.trace.output", final_output[:10000] if len(final_output) > 10000 else final_output)
            
            # 최종 상태 가져오기
            final_state = graph.get_state(config)
            if final_state and final_state.values:
                session_states[session_id] = {
                    "phase": final_state.values.get("phase", QuizPhase.SETUP),
                    "difficulty": final_state.values.get("difficulty"),
                    "subject": final_state.values.get("subject"),
                    "round_count": final_state.values.get("round_count", 0),
                }
            
            # 완료 이벤트
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
