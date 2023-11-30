from transformers import BartForConditionalGeneration, BartTokenizer
from pathlib import Path

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer


def get_ml_ref(text, min, max):
    # Загрузка предварительно обученной модели BART и токенизатора
    model_name = "facebook/bart-large-cnn"
    model = BartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = BartTokenizer.from_pretrained(model_name)

    # Исходный текст, который вы хотите сжать до резюме
    input_text = """
    Это исходный текст, который вы хотите сжать до резюме. 
    Добавьте здесь всю необходимую информацию и детали.
    """
    # input_text = Path('text.txt', encoding="UTF-8", errors='ignore').read_text(encoding="UTF-8", errors='ignore')
    # print(input_text)
    # Токенизация и кодирование текста
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True, padding=True)

    # Генерация резюме
    summary_ids = model.generate(inputs["input_ids"], max_length=max, min_length=min, length_penalty=1.0, num_beams=8,
                                 early_stopping=True)

    # Декодирование и вывод резюме
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    print("Резюме:")
    print(summary)
    return summary


# эта функция использует LSA(latent semantic analysis)
# https://cyberleninka.ru/article/n/sravnenie-nekotoryh-metodov-mashinnogo-obucheniya-dlya-analiza-tekstovyh-dokumentov/viewer -
# стр 4 сразу под рисунком - подтверждение, что LSA - метод машинного обучения
# https://ru.wikipedia.org/wiki/%D0%92%D0%B5%D1%80%D0%BE%D1%8F%D1%82%D0%BD%D0%BE%D1%81%D1%82%D0%BD%D1%8B%D0%B9_%D0%BB%D0%B0%D1%82%D0%B5%D0%BD%D1%82%D0%BD%D0%BE-%D1%81%D0%B5%D0%BC%D0%B0%D0%BD%D1%82%D0%B8%D1%87%D0%B5%D1%81%D0%BA%D0%B8%D0%B9_%D0%B0%D0%BD%D0%B0%D0%BB%D0%B8%D0%B7
# ВЛСА - наследник ЛСА - применяется в машинном обучении
def summary_extraction(text):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    stemmer = Stemmer("english")

    summarizer = Summarizer(stemmer)

    summary = summarizer(parser.document, 3)
    return ' '.join([str(sentence) for sentence in summary])
