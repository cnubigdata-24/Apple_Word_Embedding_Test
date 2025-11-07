# pip install sentence-transformers

print("Example 1 ###############################################\n")

from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np

sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

sentence1 = "I ate an apple for breakfast."
sentence2 = "Apple released a new product yesterday."

# Get Sentence-BERT embedding
def get_sbert_embedding(sentence):
    return sbert_model.encode(sentence)

# Get word embedding from BERT
def get_word_embedding(sentence, target_word):
    inputs = tokenizer(sentence, return_tensors="pt")
    
    with torch.no_grad():
        outputs = bert_model(**inputs)
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        embeddings = outputs.last_hidden_state[0]
        
        for i, token in enumerate(tokens):
            if target_word.lower() in token.lower():
                return embeddings[i].numpy(), tokens
        raise ValueError(f"'{target_word}' not found in tokens: {tokens}")

sbert_emb_1 = get_sbert_embedding(sentence1)
sbert_emb_2 = get_sbert_embedding(sentence2)

apple_emb_1, tokens_1 = get_word_embedding(sentence1, "apple")
apple_emb_2, tokens_2 = get_word_embedding(sentence2, "apple")

sbert_sim = cosine_similarity([sbert_emb_1], [sbert_emb_2])[0][0]
apple_sim = cosine_similarity([apple_emb_1], [apple_emb_2])[0][0]

np.set_printoptions(precision=4, suppress=True)

# Display shortened embedding vector
def short_embed(embed, n=7):
    head = np.round(embed[:n], 4)
    return f"[{' '.join(map(str, head))} ...]"

print("Sentence 1\n---------------------------")
print(f"Sentence: {sentence1}")
print(f"Token: {tokens_1}")
print(f"Sentence embedding (total {len(sbert_emb_1)} dimensions): {short_embed(sbert_emb_1)}")
print(f"Apple embedding (total {len(apple_emb_1)} dimensions): {short_embed(apple_emb_1)}")

print("\nSentence 2\n---------------------------")
print(f"Sentence: {sentence2}")
print(f"Token: {tokens_2}")
print(f"Sentence embedding (total {len(sbert_emb_2)} dimensions): {short_embed(sbert_emb_2)}")
print(f"Apple embedding (total {len(apple_emb_2)} dimensions): {short_embed(apple_emb_2)}")

print("\nCosine similarity\n---------------------------")
print(f"Similarity between sentences: {round(sbert_sim, 4)}")
print(f"Apple similarity: {round(apple_sim, 4)}")

print("\nExample 2 ###############################################\n")

from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

# Test the same word "apple" in different contexts
sentences = [
    "I ate an apple for breakfast.",           # fruit
    "Apple released a new product yesterday.", # company
    "This apple is very sweet.",               # fruit
    "Apple stock went up today."               # company
]

def get_word_embedding(sentence, target_word):
    inputs = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = bert_model(**inputs)
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        embeddings = outputs.last_hidden_state[0]
        for i, token in enumerate(tokens):
            if target_word.lower() in token.lower():
                return embeddings[i].numpy(), tokens
    return None, None

embeddings = []
for sent in sentences:
    emb, tokens = get_word_embedding(sent, "apple")
    if emb is not None:
        embeddings.append(emb)
        print(f"Sentence: {sent}")
        print(f"Tokens: {tokens}")
        print(f"Apple embedding (first 5): {emb[:5]}")
        print()

# Similarity matrix
print("Apple word similarity matrix:")
print("-" * 50)

for i in range(len(embeddings)):
    for j in range(len(embeddings)):
        sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
        print(f"Sentence {i+1} vs {j+1}: {round(sim, 4)}", end=" / ")
    print()

print("\nAnalysis:")
print("- Sentence 1 vs 3: Both mean 'fruit' -> Expected high similarity")
print("- Sentence 2 vs 4: Both mean 'company' -> Expected high similarity")
print("- Sentence 1 vs 2: 'fruit' vs 'company' -> Expected low similarity")
