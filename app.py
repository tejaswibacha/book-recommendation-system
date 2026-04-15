from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process

app = Flask(__name__)

# =========================
# 📚 LOAD DATA
# =========================
books = pd.read_csv("Books.csv", low_memory=False)
ratings = pd.read_csv("Ratings.csv", low_memory=False)

books = books.fillna("")
ratings = ratings.fillna("")

# =========================
# 🔗 MERGE DATA
# =========================
data = books.merge(ratings, on="ISBN")

# =========================
# ⭐ RATING CALCULATION
# =========================
ratings_count = data.groupby("Book-Title")["Book-Rating"].count().reset_index()
ratings_count.rename(columns={"Book-Rating": "Num-Ratings"}, inplace=True)

ratings_avg = data.groupby("Book-Title")["Book-Rating"].mean().reset_index()
ratings_avg.rename(columns={"Book-Rating": "Avg-Rating"}, inplace=True)

books = books.merge(ratings_count, on="Book-Title")
books = books.merge(ratings_avg, on="Book-Title")

# Filter good books
books = books[books["Num-Ratings"] > 50]
books = books.drop_duplicates("Book-Title")
books = books.head(2000)
books = books.reset_index(drop=True)

# =========================
# 🧠 FEATURES
# =========================
books["Genre"] = books["Publisher"]
books["Description"] = books["Book-Title"] + " " + books["Publisher"]

tfidf_genre = TfidfVectorizer(stop_words="english")
genre_matrix = tfidf_genre.fit_transform(books["Genre"])

tfidf_author = TfidfVectorizer(stop_words="english")
author_matrix = tfidf_author.fit_transform(books["Book-Author"])

tfidf_desc = TfidfVectorizer(stop_words="english")
desc_matrix = tfidf_desc.fit_transform(books["Description"])

# =========================
# 🖼 IMAGE FUNCTION
# =========================
def get_image(row):
    isbn = str(row["ISBN"]).strip()

    if isbn and isbn != "nan":
        return f"https://covers.openlibrary.org/b/isbn/{isbn}-M.jpg"

    img = row.get("Image-URL-M", "")
    if isinstance(img, str) and img.startswith("http"):
        return img

    return "https://via.placeholder.com/150x220?text=No+Image"

# =========================
# 🧠 TYPO CORRECTION (FUZZYWUZZY)
# =========================
def correct_book_name(query):
    choices = books["Book-Title"].dropna().tolist()
    best = process.extractOne(query, choices)

    if best and best[1] > 70:
        return best[0]
    return query

# =========================
# 🔥 TRENDING BOOKS
# =========================
trending_books = books.sort_values("Num-Ratings", ascending=False).head(10)

# =========================
# 🏠 HOME PAGE
# =========================
@app.route("/")
def home():
    return render_template("index.html")

# =========================
# 🔍 AUTO SUGGESTIONS
# =========================
@app.route("/search")
def search():
    query = request.args.get("q", "").lower().strip()

    if query == "":
        return {"results": []}

    results = books[books["Book-Title"].str.lower().str.contains(query)]["Book-Title"].head(5).tolist()
    return {"results": results}

# =========================
# 🔥 TRENDING API
# =========================
@app.route("/trending")
def trending():
    result = []

    for _, row in trending_books.iterrows():
        result.append({
            "title": row["Book-Title"],
            "author": row["Book-Author"],
            "rating": round(row["Avg-Rating"], 1),
            "image": get_image(row)
        })

    return {"trending": result}

# =========================
# 📚 RECOMMENDATION ENGINE
# =========================
@app.route("/recommend", methods=["POST"])
def recommend():
    book_name = request.form["book"]

    # 🧠 Fix spelling mistakes
    book_name = correct_book_name(book_name)

    matches = books[books["Book-Title"].str.lower().str.contains(book_name.lower())]

    if matches.empty:
        return render_template("index.html", error="Book not found!")

    index = matches.index[0]

    # =========================
    # Recommendation function
    # =========================
    def get_books(matrix):
        sim = cosine_similarity(matrix[index], matrix).flatten()
        top = sorted(list(enumerate(sim)), key=lambda x: x[1], reverse=True)[1:5]

        result = []
        for i in top:
            row = books.iloc[i[0]]
            result.append((
                row["Book-Title"],
                row["Book-Author"],
                get_image(row),
                round(row["Avg-Rating"], 1),
                int(row["Num-Ratings"])
            ))
        return result

    genre_rec = get_books(genre_matrix)
    author_rec = get_books(author_matrix)
    desc_rec = get_books(desc_matrix)

    return render_template(
        "index.html",
        genre_rec=genre_rec,
        author_rec=author_rec,
        desc_rec=desc_rec
    )

# =========================
# 🚀 RUN APP
# =========================
if __name__ == "__main__":
    app.run(debug=True)
