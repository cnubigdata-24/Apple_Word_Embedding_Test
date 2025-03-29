# Comparing Word Embeddings in Different Contexts

# Import required libraries
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 1. 단어 수준 임베딩을 추출하기 위해 BERT 모델 사용
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# 2. 문맥이 다른 두 문장 정의
sentence_1 = "I ate an apple for lunch yesterday."
sentence_2 = "Apple Inc. released a new iPhone model today."

# 3. 토큰화 및 임베딩 추출 함수
def get_word_embedding(sentence, target_word):
    # 입력 토큰화
    inputs = tokenizer(sentence, return_tensors="pt")
    
    # 모델 통과
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 마지막 히든 스테이트 가져오기 (단어 임베딩)
    word_embeddings = outputs.last_hidden_state[0]
    
    # 토큰 ID 리스트
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    # 타겟 단어의 인덱스 찾기 (대소문자 구분 없이)
    target_indices = [i for i, token in enumerate(tokens) 
                     if token.lower() == target_word.lower() or 
                     token.lower() == '##' + target_word.lower()]
    
    if not target_indices:
        # BERT는 단어를 하위 토큰으로 나눌 수 있음
        # 예: 'apple'이 'app'과 '##le'로 분할될 수 있음
        print(f"Warning: '{target_word}' not found exactly. Tokenized as: {tokens}")
        # 가장 유사한 토큰 찾기
        target_indices = [i for i, token in enumerate(tokens) 
                         if target_word.lower() in token.lower()]
    
    if not target_indices:
        raise ValueError(f"Target word '{target_word}' not found in tokens: {tokens}")
    
    # 타겟 단어의 임베딩 반환
    return word_embeddings[target_indices[0]].numpy(), tokens

# 4. 두 문장에서 'apple' 단어의 임베딩 추출
try:
    apple_embedding_1, tokens_1 = get_word_embedding(sentence_1, "apple")
    apple_embedding_2, tokens_2 = get_word_embedding(sentence_2, "apple")
    
    # 5. 코사인 유사도 계산
    cosine_sim = cosine_similarity([apple_embedding_1], [apple_embedding_2])[0][0]
    
    # 6. 결과 출력
    print("문장 1:", sentence_1)
    print("문장 2:", sentence_2)
    print("\n문장 1 토큰:", tokens_1)
    print("문장 2 토큰:", tokens_2)
    print("\n'apple' 임베딩 1 (first 10 dimensions):", apple_embedding_1[:10])  
    print("'apple' 임베딩 2 (first 10 dimensions):", apple_embedding_2[:10])
    print("\n'apple' 단어의 문맥 간 코사인 유사도:", cosine_sim)
    
    # 7. 임베딩 시각화를 위한 PCA
    combined_embeddings = np.vstack([apple_embedding_1, apple_embedding_2])
    pca = PCA(n_components=2)
    result = pca.fit_transform(combined_embeddings)
    
    # 8. 시각화
    plt.figure(figsize=(10, 6))
    plt.scatter(result[0, 0], result[0, 1], c='blue', label="'apple' in sentence 1 (fruit)")
    plt.scatter(result[1, 0], result[1, 1], c='red', label="'apple' in sentence 2 (company)")
    
    # 원점에서 각 점까지 선 그리기
    plt.arrow(0, 0, result[0, 0], result[0, 1], head_width=0.05, head_length=0.05, fc='blue', ec='blue', alpha=0.5)
    plt.arrow(0, 0, result[1, 0], result[1, 1], head_width=0.05, head_length=0.05, fc='red', ec='red', alpha=0.5)
    
    plt.grid(True)
    plt.legend()
    plt.title("PCA Visualization of 'apple' Embeddings in Different Contexts")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.show()
    
    # 9. 추가: 문맥별 유사도 비교를 위한 예제 확장
    fruit_context = "The tree produces sweet red fruits. I like eating them."
    company_context = "The tech giant produces smartphones. Their products are popular."
    
    fruit_embedding, _ = get_word_embedding(fruit_context, "fruit")
    company_embedding, _ = get_word_embedding(company_context, "tech")
    
    # 각 문맥의 'apple'과 일반적인 과일/회사 개념 사이의 유사도
    apple1_fruit_sim = cosine_similarity([apple_embedding_1], [fruit_embedding])[0][0]
    apple1_company_sim = cosine_similarity([apple_embedding_1], [company_embedding])[0][0]
    
    apple2_fruit_sim = cosine_similarity([apple_embedding_2], [fruit_embedding])[0][0]
    apple2_company_sim = cosine_similarity([apple_embedding_2], [company_embedding])[0][0]
    
    print("\n문맥 비교:")
    print(f"문장 1의 'apple'과 '과일' 문맥 유사도: {apple1_fruit_sim:.4f}")
    print(f"문장 1의 'apple'과 '회사' 문맥 유사도: {apple1_company_sim:.4f}")
    print(f"문장 2의 'apple'과 '과일' 문맥 유사도: {apple2_fruit_sim:.4f}")
    print(f"문장 2의 'apple'과 '회사' 문맥 유사도: {apple2_company_sim:.4f}")
    
except ValueError as e:
    print(e)
