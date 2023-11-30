from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize


def get_classic_ref(text):

    # Токенизация текста на предложения
    sentences = sent_tokenize(text)

    # Вычисление TF-IDF для предложений
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sentences)

    # Вычисление матрицы сходства косинусного угла между предложениями
    cosine_sim_matrix = cosine_similarity(X, X)

    # Вычисление среднего значения сходства косинусного угла для каждого предложения
    sentence_scores = cosine_sim_matrix.mean(axis=1)

    # Выбор N наиболее информативных предложений для реферата
    n = 5  # количество предложений в реферате
    top_indices = sentence_scores.argsort()[-n:][::-1]
    top_sentences = [sentences[i] for i in top_indices]

    summary = '\n'.join(top_sentences)
    return summary
