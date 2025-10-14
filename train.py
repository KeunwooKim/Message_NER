import json
import torch
import torch.nn as nn
import numpy as np
import os
from datasets import Dataset
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
    AutoModelForTokenClassification,
    AutoConfig,
    AutoTokenizer,
    BertTokenizer,
    EarlyStoppingCallback
)
from seqeval.metrics import classification_report
from typing import Dict, Any, Optional


# ----------------------------------------------------
# 1. 커스텀 Trainer 정의
# ----------------------------------------------------

class WeightedNERTrainer(Trainer):
    """
    클래스 가중치(class weights)를 적용하여 개체명 불균형 문제를 다루는 커스텀 Trainer.
    """
    def __init__(self, *args, class_weights: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(*args, **kwargs)
        # 클래스 가중치를 모델과 동일한 장치(device)로 이동
        if class_weights is not None:
            self.class_weights = class_weights.to(self.args.device)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        손실 함수 계산 시 클래스 가중치를 적용하도록 `compute_loss` 메서드를 오버라이드.
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if labels is not None:
            # CrossEntropyLoss에 클래스 가중치 적용
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)

            # loss 계산을 위해 logits과 labels의 차원 조정
            logits = logits.view(-1, self.model.config.num_labels)
            labels = labels.view(-1)

            loss = loss_fct(logits, labels)
        else:
            loss = torch.tensor(0.0, device=logits.device)

        return (loss, outputs) if return_outputs else loss


# ----------------------------------------------------
# 2. 데이터 전처리
# ----------------------------------------------------

def load_and_preprocess_data(file_path: str, tokenizer, label_map: Dict[str, int]) -> Dataset:
    """
    데이터 파일을 로드하고 토크나이저에 맞게 전처리.
    Slow tokenizer 사용 시 서브워드(subword)에 대한 레이블을 수동으로 정렬.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"'{file_path}' 파일을 찾을 수 없습니다. 내장된 예제 데이터로 실행합니다.")
        data = [
            {"kobert_tokens": ["오늘", "서울", "날씨", "##는", "맑음"], "aligned_ner_tags": ["O", "B-LOC", "O", "O", "O"]},
            {"kobert_tokens": ["부산", "해운대", "##구", "주민", "##은", "대피", "##하세", "##요"],
             "aligned_ner_tags": ["B-LOC", "I-LOC", "I-LOC", "O", "O", "O", "O", "O"]}
        ]

    # Hugging Face Dataset 형식으로 변환
    processed_for_dataset = [
        {"tokens": item["kobert_tokens"], "ner_tags": item["aligned_ner_tags"]}
        for item in data
    ]
    raw_dataset = Dataset.from_list(processed_for_dataset)
    dataset_dict = raw_dataset.train_test_split(test_size=0.1, seed=42)

    def tokenize_and_align_labels(examples: Dict[str, Any]) -> Dict[str, Any]:
        """토큰화 및 레이블 정렬을 수행하는 내부 함수"""
        tokenized_inputs = tokenizer(
            examples["tokens"],
            padding="max_length",
            truncation=True,
            max_length=128,
            is_split_into_words=True
        )

        labels = []
        # 각 문장에 대해 레이블 정렬
        for i, (input_ids_list, label_list) in enumerate(zip(tokenized_inputs["input_ids"], examples["ner_tags"])):
            label_ids = []
            word_idx = -1 # 원본 단어 인덱스
            tokens = tokenizer.convert_ids_to_tokens(input_ids_list)

            for token in tokens:
                # [CLS], [SEP], [PAD] 토큰은 loss 계산에서 제외 (-100)
                if token in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]:
                    label_ids.append(-100)
                    continue

                # 서브워드 여부 확인
                is_subword = token.startswith("##")

                # 서브워드가 아니면 새로운 단어 시작
                if not is_subword:
                    word_idx += 1

                # 단어 인덱스가 레이블 길이를 벗어나면 예외 처리
                if word_idx >= len(label_list):
                    label_ids.append(-100)
                else:
                    original_tag = label_list[word_idx]

                    # 서브워드의 경우, 태그 규칙에 따라 변환 (B-LOC -> I-LOC)
                    if is_subword:
                        if original_tag.startswith("B-"):
                            new_tag = "I-" + original_tag[2:]
                        else: # I- 또는 O 태그는 그대로 유지
                            new_tag = original_tag
                        label_ids.append(label_map.get(new_tag, -100))
                    else: # 첫 토큰은 원래 태그 사용
                        label_ids.append(label_map.get(original_tag, -100))
            
            # 패딩 등으로 길이가 맞지 않을 경우 조정
            if len(label_ids) < len(input_ids_list):
                label_ids.extend([-100] * (len(input_ids_list) - len(label_ids)))
            elif len(label_ids) > len(input_ids_list):
                label_ids = label_ids[:len(input_ids_list)]

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    # 전체 데이터셋에 함수 적용
    tokenized_datasets = dataset_dict.map(tokenize_and_align_labels, batched=True)
    if 'token_type_ids' in tokenized_datasets['train'].column_names:
        tokenized_datasets = tokenized_datasets.remove_columns(['token_type_ids'])

    return tokenized_datasets


# ----------------------------------------------------
# 3. 평가 지표 계산
# ----------------------------------------------------

def compute_metrics(p, id2label: Dict[int, str]) -> Dict[str, float]:
    """F1-score, 정밀도, 재현율을 계산"""
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # 실제 예측값과 정답값을 위해 -100 레이블(패딩 등)을 제외
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    report = classification_report(true_labels, true_predictions, output_dict=True, zero_division=0)
    
    # 전체 성능(micro avg)을 기준으로 값을 추출
    avg_metrics = report.get("micro avg", report.get("weighted avg", {}))

    return {
        "precision": avg_metrics.get("precision", 0),
        "recall": avg_metrics.get("recall", 0),
        "f1-score": avg_metrics.get("f1-score", 0),
    }


# ----------------------------------------------------
# 4. 메인 실행 함수
# ----------------------------------------------------

def main():
    """전체 학습 파이프라인을 실행"""
    # --- 설정 ---
    MODEL_PATH = "./kobert-base"
    DATA_FILE_PATH = "labeled_disaster_locations_final.json"
    OUTPUT_DIR = "./kobert-ner-weighted-final"
    FINAL_MODEL_PATH = "./my-final-kobert-ner-weighted-model"

    label_map = {"O": 0, "B-LOC": 1, "I-LOC": 2}
    id2label = {v: k for k, v in label_map.items()}

    # 논문 기반 최적 하이퍼파라미터
    BEST_LEARNING_RATE = 2.703e-05
    BEST_BATCH_SIZE = 8

    # 클래스 가중치 (O: 1.0, B-LOC: 7.0, I-LOC: 7.0)
    # 소수 클래스인 LOC 태그에 더 높은 가중치를 부여
    CLASS_WEIGHTS_TENSOR = torch.tensor([1.0, 7.0, 7.0])

    # --- 1. 모델 및 토크나이저 로드 ---
    print("1. 모델 및 토크나이저 로드 중...")
    try:
        # KoBERT는 공식적으로 Slow Tokenizer(BertTokenizer) 사용을 권장
        tokenizer = BertTokenizer.from_pretrained(MODEL_PATH, model_max_length=128, use_fast=False)
        print(f"'{MODEL_PATH}'에서 BertTokenizer(slow)를 로드했습니다.")
    except Exception as e:
        print(f"토크나이저 로드 실패: {e}. AutoTokenizer로 대체합니다.")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, model_max_length=128, use_fast=False)

    # 모델 config에 레이블 정보 추가
    config = AutoConfig.from_pretrained(MODEL_PATH)
    config.num_labels = len(label_map)
    config.id2label = id2label
    config.label2id = label_map

    model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH, config=config)

    # --- 2. 데이터셋 준비 ---
    print("2. 데이터셋 로드 및 전처리 중...")
    tokenized_datasets = load_and_preprocess_data(DATA_FILE_PATH, tokenizer, label_map)
    print("전처리 완료된 데이터셋 정보:\n", tokenized_datasets)

    # --- 3. Trainer 설정 ---
    print("3. 학습 설정 초기화 중...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=20,  # 조기 종료를 고려해 epochs는 충분히 설정
        per_device_train_batch_size=BEST_BATCH_SIZE,
        per_device_eval_batch_size=BEST_BATCH_SIZE,
        warmup_steps=500,
        weight_decay=0.01,
        learning_rate=BEST_LEARNING_RATE,
        logging_dir="./logs_weighted",
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,      # 훈련 종료 후 가장 성능이 좋았던 모델을 로드
        metric_for_best_model="f1-score", # F1-score를 기준으로 최적 모델 판단
        greater_is_better=True,
        save_total_limit=2,
        report_to="none"
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    # 3 epoch 동안 성능 개선이 없으면 훈련 조기 종료
    early_stopping = EarlyStoppingCallback(early_stopping_patience=3)

    # 가중 손실 함수를 적용한 커스텀 Trainer 초기화
    trainer = WeightedNERTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(p, id2label),
        callbacks=[early_stopping],
        class_weights=CLASS_WEIGHTS_TENSOR # 클래스 가중치 전달
    )

    # --- 4. 모델 학습 시작 ---
    print("4. 모델 학습을 시작합니다.")
    train_result = trainer.train()

    # --- 5. 최종 평가 ---
    print("5. 학습 완료된 모델을 평가합니다.")
    eval_results = trainer.evaluate()
    print("최종 평가 결과:", eval_results)

    # 학습 및 평가 결과 로그 저장
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_metrics("eval", eval_results)

    # --- 6. 모델 저장 ---
    print(f"6. 최적 모델을 '{FINAL_MODEL_PATH}' 경로에 저장합니다.")
    trainer.save_model(FINAL_MODEL_PATH)
    print("저장 완료.")


if __name__ == "__main__":
    main()
