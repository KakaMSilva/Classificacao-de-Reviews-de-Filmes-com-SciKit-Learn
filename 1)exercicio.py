from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn import metrics
movie_reviews_data_folder = r"./data"
dataset = load_files(movie_reviews_data_folder, shuffle=False)
print("n_samples: %d" % len(dataset.data))
docs_train, docs_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, test_size=0.25, random_state=None)
tfidf_vectorizer = TfidfVectorizer(
    stop_words='english',
    max_df=0.5,
    min_df=2
)
svm_classifier = LinearSVC(
    C=1,
    loss='hinge',
    random_state=42
)
pipeline = Pipeline([
    ('tfidf', tfidf_vectorizer),
    ('svm', svm_classifier)
])
pipeline.fit(docs_train, y_train)
y_predicted = pipeline.predict(docs_test)
print(metrics.classification_report(y_test, y_predicted, target_names=dataset.target_names))
cm = metrics.confusion_matrix(y_test, y_predicted)
print(cm)
