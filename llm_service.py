"""
llm_service.py
--------------
LLM 파트: Anthropic Claude API를 이용해
딥러닝 모델(ResNet18)의 예측 결과를 사람이 이해하기 쉬운 설명으로 바꿔주는 서비스 클래스.
"""

import anthropic


class LLMExplainer:
    """딥러닝 예측 결과를 자연어로 설명해주는 LLM 서비스 클래스"""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6"):
        """
        Args:
            api_key (str): Anthropic API Key
            model (str): 사용할 Claude 모델명
        """
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def explain(self, predictions: list, user_question: str = "") -> str:
        """
        딥러닝 모델의 Top-K 예측 결과를 받아 자연어 설명을 생성한다.

        Args:
            predictions (list[dict]): [{"label": str, "score": float}, ...]
            user_question (str): 사용자가 추가로 궁금해하는 질문 (선택)

        Returns:
            str: LLM이 생성한 설명 텍스트
        """
        pred_text = "\n".join(
            f"- {p['label']}: {p['score']*100:.1f}%" for p in predictions
        )

        prompt = f"""다음은 딥러닝 이미지 분류 모델(ResNet18)이 예측한 결과입니다.

[예측 결과 (Top-{len(predictions)})]
{pred_text}

위 결과를 바탕으로 이미지에 무엇이 있을지 일반인도 이해하기 쉽게 한국어로 2~4문장 정도로 설명해줘.
전문 용어는 최소화하고, 1등 예측을 중심으로 자연스럽게 설명해줘.
"""

        if user_question:
            prompt += f"\n추가로 사용자가 다음 질문을 했어: \"{user_question}\"\n이 질문에도 함께 답해줘."

        message = self.client.messages.create(
            model=self.model,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )

        return message.content[0].text
