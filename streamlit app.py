"""
CSV/TXT 데이터 분석 에이전트 - Streamlit 프론트엔드

agent.py의 graph를 그대로 재사용합니다. 파일은 세션별 임시 폴더에 저장한 뒤,
경로 기반 도구(resolve_data_path_tool 등)를 기존 그대로 호출합니다.
"""

import os
import glob
import shutil
import tempfile

import streamlit as st

# Streamlit Cloud에 배포할 때는 .env 파일이 없으므로, st.secrets에 등록한 키를
# agent.py가 import되기 *전에* 환경변수로 옮겨줘야 한다(agent.py의 load_dotenv()는
# Cloud 환경에서는 아무 효과가 없다). 로컬 개발 중에는 secrets.toml이 없을 수 있으므로
# 예외를 무시하고 .env 쪽 로딩에 맡긴다.
if "OPENAI_API_KEY" not in os.environ:
    try:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass

from agent import graph  # noqa: E402  (환경변수 설정 이후에 import해야 함)
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage  # noqa: E402

st.set_page_config(page_title="데이터 분석 에이전트", page_icon="📊")
st.title("📊 데이터 분석 & 그래프 생성 에이전트")
st.caption(
    "CSV 또는 TXT 파일을 업로드하면 AI가 데이터를 검토하고 몇 가지 질문을 한 뒤 "
    "그래프를 그려드립니다. 그래프가 마음에 안 들면 채팅으로 수정 요청을 이어가세요."
)

# ---------------------------------------------------------------------------
# 세션 상태 초기화
# ---------------------------------------------------------------------------
if "temp_dir" not in st.session_state:
    # 사용자(세션)마다 별도 폴더를 써서 동시 접속자끼리 파일이 섞이지 않게 한다.
    st.session_state.temp_dir = tempfile.mkdtemp(prefix="data_agent_")
if "lc_messages" not in st.session_state:
    st.session_state.lc_messages = []
if "uploaded_path" not in st.session_state:
    st.session_state.uploaded_path = None


def reset_session():
    """업로드한 파일과 대화 기록을 모두 지우고 처음부터 다시 시작한다."""
    shutil.rmtree(st.session_state.temp_dir, ignore_errors=True)
    for key in ("temp_dir", "lc_messages", "uploaded_path"):
        st.session_state.pop(key, None)


def run_graph(new_human_text: str):
    """사용자 메시지를 누적된 대화 기록과 함께 graph에 보내고, 반환된 전체 메시지
    리스트로 세션 상태를 갱신한다. (별도 체크포인터 없이 세션 상태로 직접 기록을 관리)"""
    st.session_state.lc_messages.append(("user", new_human_text))
    try:
        result = graph.invoke({"messages": st.session_state.lc_messages})
    except Exception as e:
        st.error(f"에이전트 실행 중 오류가 발생했습니다: {repr(e)}")
        return
    st.session_state.lc_messages = result["messages"]


def render_message(msg):
    """메시지 하나를 채팅 버블로 그린다. 도구 호출/결과는 펼쳐보기로 따로 보여준다."""
    if isinstance(msg, tuple):
        role, content = msg
        if role == "user":
            with st.chat_message("user"):
                st.markdown(content)
        return

    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        if msg.content:
            with st.chat_message("assistant"):
                st.markdown(msg.content)
        if msg.tool_calls:
            tool_names = ", ".join(tc["name"] for tc in msg.tool_calls)
            with st.expander(f"🔧 도구 호출: {tool_names}"):
                for tc in msg.tool_calls:
                    st.json(tc["args"])
    elif isinstance(msg, ToolMessage):
        with st.expander(f"📋 도구 결과: {msg.name}"):
            st.text(msg.content)


def show_outputs():
    """세션 임시 폴더에 생성된 그래프(png)와 분석 코드(py)를 보여준다.
    수정 요청으로 덮어써도 매번 디스크에서 새로 읽으므로 항상 최신 내용이 표시된다."""
    png_files = sorted(glob.glob(os.path.join(st.session_state.temp_dir, "*.png")))
    py_files = sorted(
        p for p in glob.glob(os.path.join(st.session_state.temp_dir, "*.py"))
    )

    for png_path in png_files:
        st.image(png_path, caption=os.path.basename(png_path))

    for py_path in py_files:
        with st.expander(f"📄 분석 코드: {os.path.basename(py_path)}"):
            with open(py_path, "r", encoding="utf-8") as f:
                code = f.read()
            st.code(code, language="python")
            st.download_button(
                "코드 다운로드",
                data=code,
                file_name=os.path.basename(py_path),
                mime="text/x-python",
                key=f"download_{os.path.basename(py_path)}",
            )


# ---------------------------------------------------------------------------
# 사이드바: 새로 시작하기
# ---------------------------------------------------------------------------
with st.sidebar:
    st.subheader("세션 관리")
    if st.session_state.uploaded_path:
        st.caption(f"현재 파일: {os.path.basename(st.session_state.uploaded_path)}")
    if st.button("🔄 새 파일로 다시 시작"):
        reset_session()
        st.rerun()

# ---------------------------------------------------------------------------
# 1) 파일 업로드 (아직 업로드 전이면 여기서 멈춤)
# ---------------------------------------------------------------------------
if st.session_state.uploaded_path is None:
    uploaded_file = st.file_uploader(
        "CSV 또는 TXT 파일을 업로드하세요", type=["csv", "txt"]
    )
    if uploaded_file is None:
        st.info("먼저 분석할 데이터 파일을 업로드해주세요.")
        st.stop()

    save_path = os.path.join(st.session_state.temp_dir, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.session_state.uploaded_path = save_path

    with st.spinner("데이터를 검토하는 중입니다..."):
        run_graph(f"'{save_path}' 파일을 분석해서 그래프를 그려주세요.")
    st.rerun()

# ---------------------------------------------------------------------------
# 2) 대화 기록 + 결과물 표시
# ---------------------------------------------------------------------------
for msg in st.session_state.lc_messages:
    render_message(msg)

show_outputs()

# ---------------------------------------------------------------------------
# 3) 후속 메시지 입력 (질문 답변, 수정 요청 등)
# ---------------------------------------------------------------------------
user_input = st.chat_input("답변을 입력하거나, 그래프 수정 요청을 입력하세요...")
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.spinner("처리 중입니다..."):
        run_graph(user_input)
    st.rerun()
