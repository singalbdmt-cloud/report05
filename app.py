"""
app.py
------
Streamlit 메인 앱.

구조:
- model.py      : 딥러닝(ResNet18) 이미지 분류 클래스
- llm_service.py: Claude API를 이용한 결과 설명(LLM) 클래스
- app.py        : 위 두 클래스를 연결하는 UI (이 파일)

흐름:
1. 사용자가 이미지를 업로드한다.
2. ImageClassifier가 딥러닝 추론을 수행해 Top-3 클래스를 예측한다.
3. LLMExplainer(Claude)가 예측 결과를 쉬운 말로 설명해준다.
4. 사용자는 추가 질문도 할 수 있다.
"""

import streamlit as st
from PIL import Image

from model import ImageClassifier
from llm_service import LLMExplainer


st.set_page_config(page_title="딥러닝 + LLM 이미지 설명 서비스", page_icon="")
st.title("딥러닝 + LLM 이미지 분류/설명 서비스")
st.caption("ResNet18(딥러닝)로 이미지를 분류하고, Claude(LLM)가 결과를 쉽게 설명해줍니다.")


# ---------- 사이드바: API Key 입력 ----------
with st.sidebar:
    st.header("설정")
    api_key = st.text_input("API Key", type="password")
    st.caption("Key는 저장되지 않고 이 세션에서만 사용됩니다.")


# ---------- 딥러닝 모델 로드 (캐싱: 최초 1회만 로드) ----------
@st.cache_resource
def load_model():
    return ImageClassifier()


classifier = load_model()


# ---------- 메인 UI ----------
uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"])
user_question = st.text_input("궁금한 점이 있다면 입력하세요 (선택)", "")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="업로드한 이미지", use_container_width=True)

    if st.button("분석 시작"):
        # 1) 딥러닝 추론
        with st.spinner("딥러닝 모델이 이미지를 분석 중입니다..."):
            predictions = classifier.predict(image, top_k=3)

        st.subheader("📊 딥러닝 예측 결과 (Top-3)")
        for p in predictions:
            st.write(f"- **{p['label']}** : {p['score']*100:.1f}%")

        # 2) LLM 설명
        if not api_key:
            st.warning("LLM 설명을 보려면 사이드바에 API Key를 입력하세요.")
        else:
            with st.spinner("Claude가 결과를 설명하는 중입니다..."):
                explainer = LLMExplainer(api_key=api_key)
                explanation = explainer.explain(predictions, user_question)

            st.subheader("LLM 설명")
            st.write(explanation)
else:
    st.info("먼저 이미지를 업로드해주세요.")
