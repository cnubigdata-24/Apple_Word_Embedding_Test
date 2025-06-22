# pip install sentence-transformers

from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

sentence1 = "I ate an apple for breakfast."
sentence2 = "Apple released a new product yesterday."

# Get sentence embedding ([CLS])
def get_sentence_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        return outputs.last_hidden_state[0][0].numpy()

# Get word embedding
def get_word_embedding(sentence, target_word):
    inputs = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        embeddings = outputs.last_hidden_state[0]
        for i, token in enumerate(tokens):
            if target_word.lower() in token.lower():
                return embeddings[i].numpy(), tokens
        raise ValueError(f"'{target_word}' not found in tokens: {tokens}")

sentence_emb_1 = get_sentence_embedding(sentence1)
sentence_emb_2 = get_sentence_embedding(sentence2)

apple_emb_1, tokens_1 = get_word_embedding(sentence1, "apple")
apple_emb_2, tokens_2 = get_word_embedding(sentence2, "apple")

# Cosine similarity
sentence_sim = cosine_similarity([sentence_emb_1], [sentence_emb_2])[0][0]
apple_sim = cosine_similarity([apple_emb_1], [apple_emb_2])[0][0]

np.set_printoptions(precision=4, suppress=True)

def short_embed(embed, n=7):
    head = np.round(embed[:n], 4)
    return f"[{' '.join(map(str, head))} ...]"

print("Sentence 1\n---------------------------")
print(f"Sentence: {sentence1}")
print(f"Token: {tokens_1}")
print(f"문장 임베딩 (총 {len(sentence_emb_1)}차원): {short_embed(sentence_emb_1)}")
print(f"apple 임베딩 (총 {len(apple_emb_1)}차원): {short_embed(apple_emb_1)}")

print("\nSentence 2\n---------------------------")
print(f"Sentence: {sentence2}")
print(f"Token: {tokens_2}")
print(f"문장 임베딩 (총 {len(sentence_emb_2)}차원): {short_embed(sentence_emb_2)}")
print(f"apple 임베딩 (총 {len(apple_emb_2)}차원): {short_embed(apple_emb_2)}")

print("\nCosine similarity\n---------------------------")
print(f"Similarity between sentences: {round(sentence_sim, 4)}")
print(f"Apple similarity : {round(apple_sim, 4)}")
