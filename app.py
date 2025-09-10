
from flask import Flask, render_template, request, jsonify
import pandas as pd
import os

# Normalize (remove accents) helper
try:
    from unidecode import unidecode
except Exception:
    def unidecode(x): return str(x)

def normalize(s: str) -> str:
    return unidecode(str(s)).lower().strip()

DATA_BASENAME = "content"  # content.csv or content.xlsx

def load_dataframe():
    csv_path = f"{DATA_BASENAME}.csv"
    xls_path = f"{DATA_BASENAME}.xlsx"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, header=None, encoding="utf-8",delimiter=";")
    elif os.path.exists(xls_path):
        df = pd.read_excel(xls_path, header=None)
    else:
        # Create a sample file for first run
        sample = [
            ["Môn học này là gì?", "Đây là môn An toàn và Bảo mật Thông tin."],
            ["Giảng viên là ai?", "Giảng viên là TS. Nguyễn Văn A."],
            ["Có bao nhiêu chương?", "Môn học gồm 6 chương."],
            ["Hình thức thi thế nào?", "Thi viết và trắc nghiệm cuối kỳ."],
            ["Điều kiện qua môn là gì?", "Điểm trung bình >= 5.0 và không bị điểm liệt."],
        ]
        pd.DataFrame(sample).to_csv(csv_path, index=False, header=False, encoding="utf-8",sep=";")
        df = pd.DataFrame(sample)
    df = df.iloc[:, :2].copy()
    df.columns = ["question", "answer"]
    df.dropna(how="any", inplace=True)
    return df

df = load_dataframe()
QUESTIONS = df["question"].astype(str).tolist()
ANSWERS = df["answer"].astype(str).tolist()

# Try to use scikit-learn TF-IDF; fallback to difflib if not installed
USE_SKLEARN = True
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    VECT = TfidfVectorizer()
    MATRIX = VECT.fit_transform([normalize(q) for q in QUESTIONS])
except Exception:
    USE_SKLEARN = False
    from difflib import SequenceMatcher

def top_matches(user_text: str, k=4):
    q = normalize(user_text)
    if not q:
        return []
    if USE_SKLEARN:
        vec = VECT.transform([q])
        sims = cosine_similarity(vec, MATRIX)[0]
        idx = sims.argsort()[::-1][:k]
        return [{"question": QUESTIONS[i], "answer": ANSWERS[i], "score": float(sims[i])} for i in idx]
    else:
        scored = []
        for i, qq in enumerate(QUESTIONS):
            s = SequenceMatcher(None, q, normalize(qq)).ratio()
            scored.append((i, s))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [{"question": QUESTIONS[i], "answer": ANSWERS[i], "score": float(s)} for i, s in scored[:k]]

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', suggestions=QUESTIONS[:6])

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json(silent=True) or {}
    msg = (data.get('message') or '').strip()
    if not msg:
        return jsonify({'reply': 'Bạn hãy nhập câu hỏi nhé!', 'suggestions': []})
    matches = top_matches(msg, k=4)
    reply = "Xin lỗi, mình chưa rõ câu hỏi của bạn. Bạn muốn biết thông tin gì về môn Lập Trình Mạng này?."
    if matches and matches[0]['score'] >= 0.45:
        reply = matches[0]['answer']
    suggestions = [m['question'] for m in matches[1:4]]
    return jsonify({'reply': reply, 'suggestions': suggestions})

@app.route('/reload', methods=['POST'])
def reload():
    global df, QUESTIONS, ANSWERS, USE_SKLEARN, VECT, MATRIX
    df = load_dataframe()
    QUESTIONS = df['question'].astype(str).tolist()
    ANSWERS = df['answer'].astype(str).tolist()
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        VECT = TfidfVectorizer()
        MATRIX = VECT.fit_transform([normalize(q) for q in QUESTIONS])
        USE_SKLEARN = True
    except Exception:
        USE_SKLEARN = False
    return jsonify({'ok': True, 'count': len(QUESTIONS), 'use_sklearn': USE_SKLEARN})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
