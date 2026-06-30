from agent_tools import (
    python_exec_tool,
    file_write_tool,
    csv_inspect_tool,
    inspect_csv,
    resolve_data_path_tool,
    clean_path_input,
    diagnose_missing_file,
)
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import os

load_dotenv()

tools = [resolve_data_path_tool, csv_inspect_tool, python_exec_tool, file_write_tool]

SYSTEM_PROMPT = """당신은 사용자가 제공한 CSV 또는 TXT 데이터 파일을 분석하고 그래프로 시각화하는 작업을 돕는 파이썬 코드 작성 에이전트입니다. 이 대화는 한 번으로 끝나지 않을 수 있습니다. 사용자가 채팅으로 직접 요청할 수도 있고, 이미 사전 처리된 메시지를 받을 수도 있습니다.

다음 순서를 반드시 지키세요:
0. 먼저 사용자 메시지의 형태를 판단하세요.
   - 메시지 안에 '데이터 검토 결과'와 '그래프는 반드시 이 경로에 저장하세요'라는 문구가 모두 포함되어 있다면, 사전 처리가 끝난 메시지입니다. 추가 질문 없이 바로 2번부터 진행하세요.
   - 그렇지 않다면, 사용자가 채팅으로 직접 요청한 것입니다. 다음을 따르세요.
     a. 이 대화에서 아직 파일 경로를 확인하지 않았다면, 사용자가 언급한 경로로 resolve_data_path_tool을 호출하세요. 파일을 찾지 못했다면 그 결과(비슷한 파일 후보 등)를 그대로 사용자에게 보여주고, 거기서 응답을 멈춘 뒤 올바른 경로를 다시 물어보세요.
     b. 경로를 찾았다면 csv_inspect_tool을 호출해 구조를 확인하세요. (이미 이 대화에서 같은 파일을 확인했다면 다시 호출하지 마세요.)
     c. 검토 결과만 근거로(파일 경로나 폴더명은 절대 참고하지 말고) 데이터 종류를 잠정적으로 추정하고, 실제 컬럼명을 언급하며 그 의미를 묻는 질문을 하고, 컬럼 구조에 맞는 그래프 스타일 후보를 2~4개 제안하세요. 이 내용을 채팅 응답으로 작성한 뒤 거기서 턴을 마치고 사용자의 답변을 기다리세요. 같은 턴에서 바로 python_exec_tool로 넘어가지 마세요.
     d. 사용자가 이미 이전 턴에서 데이터 종류·컬럼 의미·그래프 스타일에 대해 답했다면, 그 답변을 데이터 해석의 기준으로 삼아 2번으로 진행하세요.
1. 사용자 메시지에는 보통 데이터 검토 결과(컬럼명, 데이터 타입, 결측치, 미리보기)가 이미 포함되어 있습니다. 이 검토 결과와 함께 제공된 데이터 종류, 행/열 설명, 원하는 그래프 스타일을 데이터 해석의 기준으로 삼으세요. 검토 결과가 이미 있다면 csv_inspect_tool을 다시 호출하지 마세요. 검토 결과가 없거나 불충분하다고 표시되어 있다면 그때 csv_inspect_tool을 호출해 실제 컬럼명, 데이터 타입, 결측치, 샘플 데이터를 확인하세요. 파일 경로, 폴더명, 파일명에 등장하는 단어만으로 데이터의 종류나 측정 방식을 추측하지 마세요. 사용자가 데이터 종류를 알려주지 않았다면 임의로 단정짓지 말고, 확인된 사실(컬럼명, 데이터 타입, 값의 범위)에만 근거해 설명하세요. 컬럼명을 추측하지 마세요.
2. 1번에서 확인한 정보를 바탕으로 python_exec_tool을 사용해 pandas와 matplotlib으로 데이터를 불러오고 그래프를 그리는 코드를 작성·실행하세요. 사용자가 그래프 스타일을 지정했다면(예: 2D 매핑, 3D plot, 산점도 등) 반드시 그 스타일을 따르세요. 3D plot은 mpl_toolkits.mplot3d를, 2D 매핑/히트맵은 plt.pcolormesh나 plt.imshow를 사용하세요.
   - imports에는 반드시 `import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; import matplotlib.font_manager as fm; import pandas as pd`를 포함하세요. (화면 출력이 없는 환경이므로 Agg 백엔드가 필요합니다)
   - 그래프 안에 한글이 들어가면, 아래 코드로 시스템에 설치된 한글 지원 폰트를 자동으로 찾아 등록한 뒤 사용하세요. 폰트 이름을 임의로 하드코딩하지 마세요(환경마다 설치된 폰트가 다릅니다):
     ```
     for font_path in fm.findSystemFonts():
         if any(k in font_path for k in ['Nanum', 'Malgun', 'NotoSansCJK', 'NotoSerifCJK', 'AppleSDGothic']):
             fm.fontManager.addfont(font_path)
             plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
             break
     plt.rcParams['axes.unicode_minus'] = False
     ```
   - 코드 마지막에는 plt.savefig(...)로 결과를 저장하되, 저장 경로는 사용자 메시지나 resolve_data_path_tool 결과에 포함된 정확한 경로를 그대로 사용하세요. 다른 폴더나 임의의 파일명을 만들어내지 마세요. 저장 경로를 print()로 출력해 실행 결과에서 확인할 수 있게 하세요.
3. python_exec_tool 실행이 실패하면 에러 메시지를 보고 코드를 수정해 다시 시도하세요. 같은 에러가 3회 이상 반복되면 사용자에게 실패 원인을 설명하고 중단하세요.
4. 그래프 생성에 성공했다면, file_write_tool로 실행한 코드를 정확한 경로에 저장하세요.
5. 작업이 끝나면 어떤 그래프를 그렸는지, 저장된 이미지 파일과 코드 파일의 경로를 사용자에게 명확히 요약해서 알려주세요.

코드를 작성할 때는 항상 1번에서 확인한 실제 컬럼명만 사용하고, 존재하지 않는 컬럼을 임의로 만들어내지 마세요.

그래프를 보고한 뒤에도 대화는 끝난 것이 아닙니다. 사용자가 그래프나 코드 수정을 요청하면 resolve_data_path_tool이나 csv_inspect_tool을 다시 호출하지 말고, 이미 알고 있는 파일 경로·컬럼 정보·저장 경로를 그대로 사용해 python_exec_tool로 수정된 코드를 다시 실행하고, file_write_tool로 같은 경로에 다시 저장한 뒤 결과를 보고하세요."""

llm = ChatOpenAI(model='gpt-4o')
graph = create_agent(llm, tools, system_prompt=SYSTEM_PROMPT)


class ReviewSuggestion(BaseModel):
    """검토 결과를 본 LLM이 내놓는 제안. data_type_guess는 어디까지나 잠정적인 추측이며,
    사용자가 그대로 확정하기 전까지는 사실로 취급하지 않는다."""

    data_type_guess: str = Field(
        description="검토 결과(컬럼명, 타입, 미리보기)만 근거로 한 데이터 종류 잠정 추정. "
        "파일 경로나 폴더명은 절대 참고하지 말 것. 확신하지 말고 '~로 보입니다' 같은 "
        "추측 표현을 사용할 것."
    )
    column_question: str = Field(
        description="실제 컬럼명을 직접 언급하며 각 컬럼이 무엇을 의미하는지 사용자에게 "
        "물어보는 질문 한 문장."
    )
    graph_style_options: list[str] = Field(
        description="이 데이터의 컬럼 개수와 타입에 맞는 그래프 스타일 후보 2~4개. "
        "구체적으로 작성할 것 (예: '파장 대 세기 선 그래프', 'x,y 좌표 기반 2D 히트맵')."
    )


def generate_review_suggestion(inspect_result: str) -> ReviewSuggestion:
    """검토 결과를 LLM에게 보여주고, 데이터 종류 추정·컬럼 질문·그래프 스타일 후보를
    구조화된 형태로 받아온다. 도구 호출이 필요 없는 단발성 요청이라 무거운 분석 Agent(graph)
    대신 llm을 with_structured_output으로 직접 호출한다."""
    prompt = (
        "다음은 데이터 파일을 검토한 결과입니다. 이 정보만 근거로 답변하고, "
        "파일 경로나 폴더명은 절대 참고하지 마세요.\n\n"
        f"{inspect_result}\n\n"
        "1) 이 데이터가 어떤 종류일 가능성이 있는지 한 문장으로 잠정 추정하세요.\n"
        "2) 실제 컬럼명을 언급하며 각 컬럼의 의미를 물어볼 질문을 한 문장으로 작성하세요.\n"
        "3) 이 데이터의 컬럼 구조에 맞는 그래프 스타일 후보를 2~4개 제시하세요."
    )
    structured_llm = llm.with_structured_output(ReviewSuggestion)
    return structured_llm.invoke(prompt)


GRAPH_STYLE_MENU = """
원하는 그래프 스타일을 선택하세요.
  1. 자동 (AI가 데이터에 맞게 판단)
  2. 2D 라인/바 차트
  3. 2D 산점도 (scatter)
  4. 2D 히트맵/매핑 (heatmap)
  5. 3D plot
  6. 직접 입력
"""

GRAPH_STYLE_MAP = {
    "1": "AI가 데이터에 적합한 그래프 종류를 자율적으로 판단",
    "2": "2D 라인 또는 바 차트",
    "3": "2D 산점도(scatter plot)",
    "4": "2D 히트맵/매핑(heatmap)",
    "5": "3D plot",
}


def build_user_message(
    data_path: str,
    inspect_success: bool,
    inspect_result: str,
    data_description: str,
    column_description: str,
    graph_style: str,
    output_image_path: str,
    output_script_path: str,
) -> str:
    message_parts = [f"'{data_path}' 파일을 분석해서 그래프를 그려주세요."]
    if inspect_success:
        message_parts.append(f"데이터 검토 결과 (csv_inspect_tool 재호출 불필요):\n{inspect_result}")
    else:
        message_parts.append(
            f"사전 검토 시도 중 오류가 발생했습니다: {inspect_result}\n"
            "필요하다면 csv_inspect_tool을 직접 호출해 확인하세요."
        )
    if data_description:
        message_parts.append(f"데이터 종류: {data_description}")
    else:
        message_parts.append(
            "데이터 종류: 사용자가 알려주지 않았습니다. 파일 경로로 추측하지 말고, "
            "위 검토 결과에서 확인되는 사실에만 근거해 설명하세요."
        )
    if column_description:
        message_parts.append(f"행/열 설명: {column_description}")
    message_parts.append(f"원하는 그래프 스타일: {graph_style}")
    message_parts.append(f"그래프는 반드시 이 경로에 저장하세요: {output_image_path}")
    message_parts.append(f"분석 코드는 반드시 이 경로에 저장하세요: {output_script_path}")
    return "\n".join(message_parts)


def main():
    while True:
        raw_input_path = input("분석할 데이터 파일의 경로를 입력하세요 (CSV 또는 TXT): ")
        data_path = clean_path_input(raw_input_path)
        if os.path.isfile(data_path):
            break
        print(f"\n파일을 찾을 수 없습니다: '{data_path}'")
        print(diagnose_missing_file(data_path))
        print("\n경로를 다시 확인해주세요.\n")

    ext = os.path.splitext(data_path)[1].lower()
    if ext not in (".csv", ".txt"):
        print(f"\n참고: 확장자 '{ext}'는 CSV/TXT가 아니지만, 일단 CSV 형식으로 읽기를 시도합니다.\n")

    # 모델이 저장 경로를 잘못 계산하지 않도록, 원본 파일과 같은 폴더의
    # 절대경로를 미리 계산해서 그대로 지정해준다.
    data_path = os.path.abspath(data_path)
    data_dir = os.path.dirname(data_path)
    data_base = os.path.splitext(os.path.basename(data_path))[0]
    output_image_path = os.path.join(data_dir, f"{data_base}_chart.png")
    output_script_path = os.path.join(data_dir, f"{data_base}_analysis.py")

    print("\n데이터를 검토하는 중입니다...\n")
    inspect_success, inspect_result, df = inspect_csv(data_path)
    print(inspect_result)

    column_names = list(df.columns) if inspect_success else []

    if not inspect_success:
        proceed = input(
            "\n검토에 실패했습니다. 그래도 진행할까요? (헤더 형식이 특이한 파일일 수 있습니다) (y/N): "
        ).strip().lower()
        if proceed != "y":
            print("종료합니다.")
            return

    suggestion = None
    if inspect_success:
        print("AI가 데이터를 살펴보고 질문을 준비하는 중입니다...\n")
        try:
            suggestion = generate_review_suggestion(inspect_result)
        except Exception as e:
            print(f"AI 제안 생성에 실패했습니다 ({repr(e)}). 기본 질문으로 진행합니다.\n")
            suggestion = None

    print("\n데이터에 대해 알려주시면 더 정확한 분석이 가능합니다. 모르면 Enter로 넘어가도 됩니다.\n")

    if suggestion is not None:
        print(f"AI 추정: {suggestion.data_type_guess}")
        data_description = input("이 추정이 맞다면 Enter, 아니라면 직접 입력해주세요: ").strip()
        if not data_description:
            data_description = suggestion.data_type_guess

        column_description = input(f"{suggestion.column_question} ").strip()

        options = list(suggestion.graph_style_options) + ["AI가 자율적으로 판단", "직접 입력"]
        print("\n원하는 그래프 스타일을 선택하세요.")
        for i, opt in enumerate(options, start=1):
            print(f"  {i}. {opt}")
        default_choice = len(options) - 1  # "AI가 자율적으로 판단"
        choice = input(f"선택 (1-{len(options)}, 기본값 {default_choice}): ").strip() or str(default_choice)
        try:
            idx = int(choice) - 1
            if not 0 <= idx < len(options):
                raise ValueError
        except ValueError:
            idx = default_choice - 1
        if options[idx] == "직접 입력":
            graph_style = input("원하는 그래프 스타일을 직접 입력하세요: ").strip()
        else:
            graph_style = options[idx]
    else:
        # 검토 실패 또는 AI 제안 생성 실패 시, 고정 질문으로 폴백한다.
        data_description = input(
            "이 데이터는 어떤 종류의 데이터인가요? (예: PL 분광 측정 데이터, 월별 매출 데이터 등): "
        ).strip()

        if column_names:
            column_prompt = (
                f"위에서 확인된 컬럼({', '.join(column_names)})이 각각 무엇을 의미하나요? "
                "(예: 첫 컬럼=측정 지점, 둘째 컬럼=파장(nm) 등): "
            )
        else:
            column_prompt = "각 행과 열이 무엇을 의미하는지 알려주세요: "
        column_description = input(column_prompt).strip()

        print(GRAPH_STYLE_MENU)
        style_choice = input("선택 (1-6, 기본값 1): ").strip() or "1"
        if style_choice == "6":
            graph_style = input("원하는 그래프 스타일을 직접 입력하세요: ").strip()
        else:
            graph_style = GRAPH_STYLE_MAP.get(style_choice, GRAPH_STYLE_MAP["1"])

    user_message = build_user_message(
        data_path,
        inspect_success,
        inspect_result,
        data_description,
        column_description,
        graph_style,
        output_image_path,
        output_script_path,
    )

    response = graph.stream(
        {"messages": [("user", user_message)]},
        stream_mode="updates",
    )

    for chunk in response:
        for node, value in chunk.items():
            if node:
                print("---", node, "---")
            if "messages" in value:
                print(value['messages'][0].content)


if __name__ == "__main__":
    main()