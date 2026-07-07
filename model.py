"""
model.py
--------
딥러닝 파트: 사전학습된 ResNet18(ImageNet)을 이용한 이미지 분류 모델 클래스.

- 사용자가 업로드한 이미지를 입력받아 Top-K 클래스와 확률(confidence)을 반환한다.
- torchvision에 내장된 사전학습 가중치를 사용하므로 별도의 학습 과정 없이 바로 추론(inference)이 가능하다.
"""

import torch
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image


class ImageClassifier:
    """ResNet18 기반 이미지 분류기 클래스"""

    def __init__(self):
        # 1) 사전학습 가중치 로드 (ImageNet 1000개 클래스)
        self.weights = ResNet18_Weights.DEFAULT
        self.model = models.resnet18(weights=self.weights)
        self.model.eval()  # 추론 모드

        # 2) 클래스 이름 목록 (weights 메타데이터에서 제공)
        self.categories = self.weights.meta["categories"]

        # 3) 전처리 변환 (모델이 학습된 방식과 동일하게)
        self.preprocess = self.weights.transforms()

    def predict(self, image: Image.Image, top_k: int = 3):
        """
        이미지를 입력받아 Top-K 예측 결과를 반환한다.

        Args:
            image (PIL.Image.Image): 입력 이미지 (RGB)
            top_k (int): 반환할 상위 클래스 개수

        Returns:
            list[dict]: [{"label": str, "score": float}, ...] (score는 0~1 확률)
        """
        image = image.convert("RGB")
        input_tensor = self.preprocess(image).unsqueeze(0)  # 배치 차원 추가

        with torch.no_grad():
            output = self.model(input_tensor)
            probs = F.softmax(output[0], dim=0)

        top_probs, top_idxs = torch.topk(probs, top_k)

        results = []
        for prob, idx in zip(top_probs, top_idxs):
            results.append({
                "label": self.categories[idx.item()],
                "score": float(prob.item())
            })
        return results
