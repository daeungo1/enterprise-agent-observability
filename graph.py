"""LangGraph Multi-Agent: Teacher-Student Quiz System with OpenTelemetry Tracing"""
from typing import Annotated, TypedDict, Optional, Callable
from enum import Enum

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from config import (
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_DEPLOYMENT_NAME,
    AZURE_OPENAI_API_VERSION,
)


class QuizPhase(str, Enum):
    """Quiz progress phases / 퀴즈 진행 단계"""
    SETUP = "setup"              # Waiting for difficulty/subject setup / 난이도/영역 설정 대기
    QUESTIONING = "questioning"  # Teacher creates question / Teacher가 문제 출제
    ANSWERING = "answering"      # Student answers / Student가 답변
    EVALUATING = "evaluating"    # Teacher evaluates / Teacher가 평가
    COMPLETE = "complete"        # Round complete / 한 라운드 완료


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    phase: str
    difficulty: Optional[str]       # easy, medium, hard / 쉬움, 보통, 어려움
    subject: Optional[str]          # math, science, history, etc. / 수학, 과학, 역사 등
    current_question: Optional[str]
    student_answer: Optional[str]
    round_count: int
    user_input: Optional[str]       # User input storage / 사용자 입력 저장


# Memory checkpointer / 메모리 체크포인터
memory = MemorySaver()

# Streaming callback storage (per session) / 스트리밍 콜백 저장소 (세션별)
streaming_callbacks: dict[str, Callable] = {}


def set_streaming_callback(session_id: str, callback: Callable):
    """Set streaming callback / 스트리밍 콜백 설정"""
    streaming_callbacks[session_id] = callback


def clear_streaming_callback(session_id: str):
    """Clear streaming callback / 스트리밍 콜백 제거"""
    if session_id in streaming_callbacks:
        del streaming_callbacks[session_id]


def create_llm(streaming: bool = False):
    """Create LLM instance / LLM 인스턴스 생성"""
    return AzureChatOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME,
        api_version=AZURE_OPENAI_API_VERSION,
        streaming=streaming,
    )


def create_graph():
    """Create LangGraph workflow for Teacher-Student Quiz"""
    
    llm = create_llm(streaming=True)

    # ========== Node Definitions / 노드 정의 ==========
    
    def setup_handler(state: State) -> State:
        """Parse user input to set difficulty and subject / 사용자 입력을 파싱하여 난이도와 영역 설정"""
        user_input = state.get("user_input", "")
        
        # Try to extract settings from user input / 사용자 입력에서 설정 추출 시도
        difficulty = None
        subject = None
        
        # Parse difficulty / 난이도 파싱
        if "쉬움" in user_input or "쉬운" in user_input or "easy" in user_input.lower():
            difficulty = "easy / 쉬움"
        elif "보통" in user_input or "중간" in user_input or "medium" in user_input.lower():
            difficulty = "medium / 보통"
        elif "어려움" in user_input or "어려운" in user_input or "hard" in user_input.lower():
            difficulty = "hard / 어려움"
        
        # Parse subject / 영역 파싱
        subjects_map = {
            "수학": "math / 수학", "math": "math / 수학",
            "과학": "science / 과학", "science": "science / 과학",
            "역사": "history / 역사", "history": "history / 역사",
            "영어": "english / 영어", "english": "english / 영어",
            "일반상식": "general knowledge / 일반상식", "general": "general knowledge / 일반상식",
            "프로그래밍": "programming / 프로그래밍", "programming": "programming / 프로그래밍",
            "지리": "geography / 지리", "geography": "geography / 지리",
        }
        for key, value in subjects_map.items():
            if key in user_input.lower():
                subject = value
                break
        
        if difficulty and subject:
            welcome_msg = f"🎓 **Quiz Setup Complete! / 퀴즈 설정 완료!**\n\n📊 Difficulty / 난이도: {difficulty}\n📚 Subject / 영역: {subject}\n\nTeacher will now create a question! / 이제 Teacher가 문제를 출제합니다!"
            return {
                "messages": [AIMessage(content=welcome_msg)],
                "phase": QuizPhase.QUESTIONING,
                "difficulty": difficulty,
                "subject": subject,
                "round_count": 0,
            }
        else:
            # Setup guide message / 설정 안내 메시지
            guide_msg = """🎓 **Welcome to Teacher-Student Quiz! / Teacher-Student 퀴즈에 오신 것을 환영합니다!**

To start the quiz, please tell us the **difficulty** and **subject**.
퀴즈를 시작하려면 **난이도**와 **영역**을 알려주세요.

📊 **Difficulty / 난이도**: easy / medium / hard (쉬움 / 보통 / 어려움)
📚 **Subject / 영역**: math / science / history / english / general / programming / geography
                        (수학 / 과학 / 역사 / 영어 / 일반상식 / 프로그래밍 / 지리)

Example / 예시: "medium math" or "easy history quiz" / "보통 수학 문제" 또는 "쉬운 역사 퀴즈"
"""
            return {
                "messages": [AIMessage(content=guide_msg)],
                "phase": QuizPhase.SETUP,
            }

    def teacher_question(state: State) -> State:
        """Teacher Agent: Create question / 문제 출제"""
        difficulty = state.get("difficulty", "medium / 보통")
        subject = state.get("subject", "general knowledge / 일반상식")
        round_count = state.get("round_count", 0) + 1
        
        messages = get_teacher_question_prompt(difficulty, subject, round_count)
        response = llm.invoke(messages)
        
        formatted_msg = f"👨‍🏫 **Teacher (Question #{round_count} / 문제 #{round_count})**\n\n{response.content}"
        
        return {
            "messages": [AIMessage(content=formatted_msg)],
            "current_question": response.content,
            "phase": QuizPhase.ANSWERING,
            "round_count": round_count,
        }

    def student_answer(state: State) -> State:
        """Student Agent: Answer question / 문제 풀이"""
        question = state.get("current_question", "")
        difficulty = state.get("difficulty", "medium / 보통")
        
        messages = get_student_answer_prompt(question, difficulty)
        response = llm.invoke(messages)
        
        formatted_msg = f"🧑‍🎓 **Student**\n\n{response.content}"
        
        return {
            "messages": [AIMessage(content=formatted_msg)],
            "student_answer": response.content,
            "phase": QuizPhase.EVALUATING,
        }

    def teacher_evaluate(state: State) -> State:
        """Teacher Agent: Evaluate answer and provide feedback / 답변 평가 및 피드백"""
        question = state.get("current_question", "")
        student_answer = state.get("student_answer", "")
        
        messages = get_teacher_evaluate_prompt(question, student_answer)
        response = llm.invoke(messages)
        
        formatted_msg = f"👨‍🏫 **Teacher (Evaluation / 평가)**\n\n{response.content}\n\n---\n💡 *For next question, type 'next' or 'continue'. / 다음 문제를 원하시면 '다음' 또는 '계속'을 입력하세요.*\n*For new settings, type 'reset'. / 새로운 설정을 원하시면 '새로 시작'을 입력하세요.*"
        
        return {
            "messages": [AIMessage(content=formatted_msg)],
            "phase": QuizPhase.COMPLETE,
        }

    def route_after_setup(state: State) -> str:
        """Determine next step after setup / setup 후 다음 단계 결정"""
        if state.get("difficulty") and state.get("subject"):
            return "teacher_question"
        return "end"

    # ========== Graph Construction / 그래프 구성 ==========
    
    graph_builder = StateGraph(State)
    
    # Add nodes / 노드 추가
    graph_builder.add_node("setup", setup_handler)
    graph_builder.add_node("teacher_question", teacher_question)
    graph_builder.add_node("student_answer", student_answer)
    graph_builder.add_node("teacher_evaluate", teacher_evaluate)
    
    # Add edges / 엣지 추가
    graph_builder.add_edge(START, "setup")
    graph_builder.add_conditional_edges("setup", route_after_setup, {
        "teacher_question": "teacher_question",
        "end": END,
    })
    graph_builder.add_edge("teacher_question", "student_answer")
    graph_builder.add_edge("student_answer", "teacher_evaluate")
    graph_builder.add_edge("teacher_evaluate", END)
    
    return graph_builder.compile(checkpointer=memory)


def get_teacher_question_prompt(difficulty: str, subject: str, round_count: int) -> list:
    """Generate Teacher question prompt / Teacher 문제 출제 프롬프트 생성"""
    teacher_prompt = f"""You are a kind and encouraging teacher (Teacher Agent).
You need to create a {difficulty} difficulty question in {subject} for the student.

당신은 친절하고 격려하는 선생님(Teacher Agent)입니다.
학생에게 {subject} 분야의 {difficulty} 난이도 문제를 출제해야 합니다.

Rules / 규칙:
1. The question must be clear and have a definite answer / 문제는 명확하고 답이 있는 것이어야 합니다
2. Match the difficulty level / 난이도에 맞게 출제하세요:
   - easy: Basic concepts, simple calculations / 기초적인 개념, 간단한 계산
   - medium: Questions requiring some thinking / 약간의 사고력이 필요한 문제
   - hard: Deep understanding and application / 깊은 이해와 응용력이 필요한 문제
3. Only give the question, not the answer / 문제만 출제하고, 답은 말하지 마세요
4. Keep a friendly and encouraging tone / 친근하고 격려하는 톤을 유지하세요

This is question #{round_count}. / 현재 {round_count}번째 문제입니다.
"""
    return [
        SystemMessage(content=teacher_prompt),
        HumanMessage(content=f"Please create a {difficulty} question in {subject}. / {subject} 분야의 {difficulty} 난이도 문제를 출제해주세요.")
    ]


def get_student_answer_prompt(question: str, difficulty: str) -> list:
    """Generate Student answer prompt / Student 답변 프롬프트 생성"""
    if "easy" in difficulty.lower() or "쉬" in difficulty:
        student_persona = "a hardworking elementary student who usually solves problems well but sometimes makes mistakes / 열심히 공부하는 초등학생으로, 대부분의 문제를 잘 풀지만 가끔 실수합니다"
    elif "medium" in difficulty.lower() or "보통" in difficulty:
        student_persona = "a curious middle school student who actively shows the solution process with about 70% accuracy / 호기심 많은 중학생으로, 적극적으로 풀이 과정을 보여주며 약 70% 정도의 정답률을 보입니다"
    else:
        student_persona = "a challenging high school student who approaches difficult problems logically but may not be perfect / 도전적인 고등학생으로, 어려운 문제도 논리적으로 접근하지만 완벽하지 않을 수 있습니다"
    
    student_prompt = f"""You are {student_persona}.
You must answer the teacher's question.

Rules / 규칙:
1. Show your solution process / 풀이 과정을 보여주세요
2. Do your best, but if unsure, honestly say "I'm not sure" / 최선을 다해 답하되, 확실하지 않으면 "잘 모르겠어요"라고 솔직히 말해도 됩니다
3. Use natural student-like language / 학생답게 자연스러운 말투를 사용하세요
4. Wait for teacher feedback after answering / 답변 후 선생님의 피드백을 기다리세요
"""
    return [
        SystemMessage(content=student_prompt),
        HumanMessage(content=f"Teacher's question: {question}\n\nPlease answer this question. / 선생님 문제: {question}\n\n이 문제에 답해보세요.")
    ]


def get_teacher_evaluate_prompt(question: str, student_answer: str) -> list:
    """Generate Teacher evaluation prompt / Teacher 평가 프롬프트 생성"""
    eval_prompt = f"""You are a kind and encouraging teacher (Teacher Agent).
You must evaluate the student's answer and provide feedback.

당신은 친절하고 격려하는 선생님(Teacher Agent)입니다.
학생의 답변을 평가하고 피드백을 제공해야 합니다.

Question / 문제: {question}
Student's answer / 학생 답변: {student_answer}

Rules / 규칙:
1. Clearly indicate if correct (⭕ Correct / ❌ Wrong / 🔺 Partial) / 먼저 정답 여부를 명확히 알려주세요
2. If correct: Praise and provide additional explanation / 정답인 경우: 칭찬하고 추가 설명을 해주세요
3. If wrong: Encourage and explain the correct answer / 오답인 경우: 격려하며 올바른 답과 설명을 알려주세요
4. Briefly explain key concepts or tips / 핵심 개념이나 팁을 짧게 설명해주세요
5. Maintain a kind and educational tone / 친절하고 교육적인 톤을 유지하세요
"""
    return [
        SystemMessage(content=eval_prompt),
        HumanMessage(content="Please evaluate the student's answer. / 학생의 답변을 평가해주세요.")
    ]
