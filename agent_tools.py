from pydantic import Field
from langchain.tools import tool

import io
import os
import csv
import difflib
import contextlib
import contextvars
import pandas as pd


@tool
def python_exec_tool(
    imports: str = Field(description="임포트 구문"),
    code: str = Field(description="임포트 구문을 제외한 코드 블록"),
) -> str:
    """
        파이썬 코드를 실행하는 도구
    """
    exec_namespace = {}

    try:
        exec(imports, exec_namespace)
    except Exception as e:
        return f"모듈을 임포트하는데 실패했습니다. ERROR: {repr(e)}"

    output_buffer = io.StringIO()

    try:
        with contextlib.redirect_stdout(output_buffer):
            exec(code, exec_namespace)
    except Exception as e:
        return f"코드 실행에 실패했습니다. ERROR: {repr(e)}"

    captured_output = output_buffer.getvalue()

    result_str = (
        f"성공적으로 코드가 실행되었습니다.\n"
        f"실행된 코드:\n'''python\n{code}\n'''\n"
        f"실제 출력 결과:\n{captured_output if captured_output else '(출력 없음 - print()를 사용했는지 확인하세요)'}"
    )
    return result_str


@tool
def file_write_tool(
        file_path: str = Field(description="생성/수정할 파일의 경로"),
        content: str = Field(description="파일에 작성할 내용")
) -> str:
    """
    파일을 생성하거나 내용을 작성하는 도구입니다.
    """

    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"파일 '{file_path}'에 성공적으로 작성했습니다."
    except Exception as e:
        return f"파일 작성 실패 {repr(e)}"


def _detect_delimiter(file_path: str, encoding: str) -> str:
    """TXT 파일의 구분자를 추정한다. csv.Sniffer로 먼저 시도하고, 실패하면 등장 빈도가
    가장 높은 구분자를 쓰고, 그마저도 없으면 공백 여러 개(정규식 \\s+)를 구분자로 본다."""
    with open(file_path, "r", encoding=encoding, errors="ignore") as f:
        sample = f.read(4096)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
        return dialect.delimiter
    except csv.Error:
        pass
    counts = {sep: sample.count(sep) for sep in ["\t", ",", ";", "|"]}
    best_sep, best_count = max(counts.items(), key=lambda kv: kv[1])
    if best_count > 0:
        return best_sep
    return r"\s+"


def _header_looks_like_data(df: pd.DataFrame) -> bool:
    """컬럼명이 전부 숫자로 변환되면, 첫 행이 사실 헤더가 아니라 데이터였다고 판단한다.
    헤더 없이 raw 숫자만 내보내는 분광기 등 계측 장비의 txt export에서 흔하다."""
    try:
        for col in df.columns:
            float(str(col))
        return True
    except (ValueError, TypeError):
        return False


def _read_data_file(file_path: str):
    """확장자에 맞춰 CSV/TXT를 읽는다. 인코딩은 utf-8 -> cp949 순서로 시도하고,
    헤더가 없는 raw 데이터도 자동으로 감지해 재시도한다.
    반환: (DataFrame, 적용한 처리 내역을 담은 notes 리스트)
    """
    ext = os.path.splitext(file_path)[1].lower()
    notes = []
    last_error = None

    for encoding in ("utf-8", "cp949"):
        try:
            if ext == ".txt":
                delimiter = _detect_delimiter(file_path, encoding)
                df = pd.read_csv(file_path, encoding=encoding, sep=delimiter, engine="python")
                notes.append(f"TXT 파일로 인식하여 구분자 '{delimiter}'로 읽었습니다.")
            else:
                df = pd.read_csv(file_path, encoding=encoding)
                if ext != ".csv":
                    notes.append(f"확장자 '{ext}'를 CSV 형식(콤마 구분)으로 시도했습니다.")
        except Exception as e:
            last_error = e
            continue

        if _header_looks_like_data(df):
            try:
                if ext == ".txt":
                    df = pd.read_csv(
                        file_path, encoding=encoding, sep=delimiter, engine="python", header=None
                    )
                else:
                    df = pd.read_csv(file_path, encoding=encoding, header=None)
                df.columns = [f"col_{i}" for i in range(len(df.columns))]
                notes.append(
                    "첫 행이 헤더가 아니라 데이터로 보여, 헤더 없이 다시 읽고 "
                    "컬럼명을 col_0, col_1, ...로 지정했습니다."
                )
            except Exception as e:
                last_error = e
                continue

        return df, notes

    raise last_error if last_error else RuntimeError("알 수 없는 오류로 파일을 읽지 못했습니다.")


def inspect_csv(file_path: str, preview_rows: int = 5):
    """CSV 또는 TXT 파일 구조를 확인하는 핵심 로직. csv_inspect_tool과 agent.py의 main()에서
    함께 재사용한다. (성공여부, 결과 문자열, DataFrame 또는 None)을 반환한다."""
    if not os.path.isfile(file_path):
        return False, f"파일을 찾을 수 없습니다. 경로를 확인하세요: '{file_path}'", None

    try:
        df, notes = _read_data_file(file_path)
    except Exception as e:
        return False, f"파일을 읽는데 실패했습니다. ERROR: {repr(e)}", None

    n_rows, n_cols = df.shape
    columns_info = "\n".join(
        f"- {col} ({dtype}, 결측치 {df[col].isna().sum()}개)"
        for col, dtype in zip(df.columns, df.dtypes)
    )
    preview = df.head(preview_rows).to_string(index=False)

    note_block = ("\n".join(notes) + "\n\n") if notes else ""

    result_str = (
        f"파일 '{file_path}' 확인 결과\n"
        f"{note_block}"
        f"전체 크기: {n_rows}행 x {n_cols}열\n\n"
        f"컬럼 정보:\n{columns_info}\n\n"
        f"상위 {preview_rows}행 미리보기:\n{preview}"
    )
    return True, result_str, df


def clean_path_input(raw: str) -> str:
    """터미널/채팅창에 경로를 따옴표로 감싸서 붙여넣거나, Finder 드래그앤드롭으로
    공백이 백슬래시로 이스케이프된 채 입력되는 경우를 정리한다."""
    s = raw.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        s = s[1:-1]
    s = s.replace('\\ ', ' ')
    return s.strip()


def diagnose_missing_file(data_path: str) -> str:
    """경로를 찾지 못했을 때, 같은 폴더의 CSV/TXT 목록과 비슷한 이름의 파일을 찾아 안내한다.
    오타나 보이지 않는 특수문자(전각 공백 등)로 인한 불일치를 사용자가 직접 비교해볼 수 있게 한다."""
    directory = os.path.dirname(data_path) or "."
    target_name = os.path.basename(data_path)

    lines = [f"입력값 그대로(repr): {repr(data_path)}"]

    if not os.path.isdir(directory):
        lines.append(f"'{directory}' 폴더 자체를 찾을 수 없습니다. 상위 경로를 다시 확인해주세요.")
        return "\n".join(lines)

    try:
        entries = os.listdir(directory)
    except Exception as e:
        lines.append(f"폴더 내용을 읽는 중 오류가 발생했습니다: {repr(e)}")
        return "\n".join(lines)

    data_entries = sorted(e for e in entries if e.lower().endswith((".csv", ".txt")))
    if not data_entries:
        lines.append(f"'{directory}' 폴더에 CSV/TXT 파일이 없습니다.")
        return "\n".join(lines)

    close_matches = difflib.get_close_matches(target_name, data_entries, n=3, cutoff=0.8)

    lines.append(f"'{directory}' 폴더에서 찾은 CSV/TXT 파일 ({len(data_entries)}개):")
    for entry in data_entries[:15]:
        marker = "  <- 입력하신 이름과 가장 비슷함" if entry in close_matches else ""
        lines.append(f"  - {entry}{marker}")
    if len(data_entries) > 15:
        lines.append(f"  ... 외 {len(data_entries) - 15}개")

    if close_matches:
        lines.append(
            "\n비슷한 파일이 있다면, 위 목록에서 그대로 복사해서 다시 입력해보세요 "
            "(직접 타이핑하면 공백이나 특수문자가 다르게 들어갈 수 있습니다)."
        )
    return "\n".join(lines)


_last_resolved_path_info: contextvars.ContextVar = contextvars.ContextVar(
    "last_resolved_path_info", default=None
)
# 동시 접속자(Streamlit 여러 세션, langgraph dev의 여러 thread)끼리 서로 값을
# 덮어쓰지 않도록, 일반 모듈 전역변수 대신 contextvars로 실행 컨텍스트별로 격리한다.


def resolve_data_path(raw_path: str):
    """경로 문자열을 정리하고 존재 여부를 확인하는 핵심 로직. agent.py의 main()과
    resolve_data_path_tool이 함께 재사용한다.
    반환: (성공여부, 결과 문자열, 경로 정보 dict 또는 None)
    경로 정보 dict 키: data_path, output_image_path, output_script_path
    """
    data_path = clean_path_input(raw_path)

    if not os.path.isfile(data_path):
        diagnosis = diagnose_missing_file(data_path)
        return False, f"파일을 찾을 수 없습니다: '{data_path}'\n\n{diagnosis}", None

    data_path = os.path.abspath(data_path)
    data_dir = os.path.dirname(data_path)
    data_base = os.path.splitext(os.path.basename(data_path))[0]
    output_image_path = os.path.join(data_dir, f"{data_base}_chart.png")
    output_script_path = os.path.join(data_dir, f"{data_base}_analysis.py")

    result_str = (
        "파일을 찾았습니다.\n"
        f"절대경로: {data_path}\n"
        f"그래프 저장 경로 (반드시 이 경로 그대로 사용): {output_image_path}\n"
        f"분석 코드 저장 경로 (반드시 이 경로 그대로 사용): {output_script_path}"
    )
    path_info = {
        "data_path": data_path,
        "output_image_path": output_image_path,
        "output_script_path": output_script_path,
    }
    _last_resolved_path_info.set(path_info)
    return True, result_str, path_info


@tool
def resolve_data_path_tool(
    raw_path: str = Field(description="사용자가 채팅으로 보낸 파일 경로 원문 (따옴표 등 그대로)"),
) -> str:
    """
    사용자가 메시지로 보낸 데이터 파일 경로를 정리하고 존재 여부를 확인하는 도구입니다.
    따옴표로 감싸진 경로나 공백이 백슬래시로 이스케이프된 경로도 자동으로 정리합니다.
    파일이 있으면 절대경로와 함께 그래프·분석 코드를 저장할 정확한 경로도 알려주므로,
    이후 csv_inspect_tool, python_exec_tool, file_write_tool을 호출할 때 그대로 사용하세요.
    파일을 못 찾으면 같은 폴더의 비슷한 파일 목록을 보여주니, 그 내용을 사용자에게 그대로
    전달하고 올바른 경로를 다시 물어보세요. 사용자가 채팅으로 경로를 보냈을 때(이미 절대경로와
    검토 결과가 메시지에 포함되어 있지 않을 때) 가장 먼저 호출해야 하는 도구입니다.
    """
    _, result_str, _ = resolve_data_path(raw_path)
    return result_str


@tool
def get_current_data_path_tool() -> str:
    """
    이 프로세스에서 가장 최근에 resolve_data_path_tool로 확인했던 파일 경로·저장 경로를
    다시 가져오는 도구입니다. 인자가 없습니다.

    사용자가 그래프나 코드 수정을 요청하는 등 이미 경로를 확인한 뒤 후속 턴에서
    python_exec_tool이나 file_write_tool을 다시 호출해야 할 때는, 대화 기록에서 경로를
    직접 기억해 타이핑하지 말고 항상 이 도구를 먼저 호출해 정확한 경로를 새로 받아오세요.
    특히 한글처럼 비-ASCII 문자가 포함된 긴 경로를 대화 맥락에서 다시 떠올려 타이핑하면
    글자가 깨질 수 있으므로, 이 도구로 매번 새로 확인하는 것이 안전합니다.
    """
    info = _last_resolved_path_info.get()
    if info is None:
        return "아직 확인된 경로가 없습니다. 먼저 resolve_data_path_tool을 호출하세요."
    return (
        "최근 확인된 경로 정보\n"
        f"절대경로: {info['data_path']}\n"
        f"그래프 저장 경로 (반드시 이 경로 그대로 사용): {info['output_image_path']}\n"
        f"분석 코드 저장 경로 (반드시 이 경로 그대로 사용): {info['output_script_path']}"
    )



@tool
def csv_inspect_tool(
    file_path: str = Field(description="확인할 CSV 또는 TXT 파일의 경로"),
    preview_rows: int = Field(default=5, description="미리보기로 보여줄 행 수"),
) -> str:
    """
    CSV 또는 TXT 데이터 파일의 구조(행/열 수, 컬럼명과 타입, 결측치 개수, 상위 N행 미리보기)를
    확인하는 도구입니다. 확장자에 따라 자동으로 처리 방식을 맞춥니다: CSV는 콤마 구분으로,
    TXT는 tab/콤마/공백 등 구분자를 자동 감지해서 읽고, 헤더가 없는 raw 데이터도 자동으로
    인식합니다. 그래프를 그리는 코드를 작성하기 전에 실제 컬럼명과 데이터 타입을 먼저 확인할 때
    사용하세요.
    """
    _, result_str, _ = inspect_csv(file_path, preview_rows)
    return result_str