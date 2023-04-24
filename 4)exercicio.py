from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn import metrics
movie_reviews_data_folder = r"./data"
dataset = load_files(movie_reviews_data_folder, shuffle=False)
print("n_samples: %d" % len(dataset.data))
docs_train, docs_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, test_size=0.25, random_state=None)
count_vectorizer = CountVectorizer(stop_words='english')
naive_bayes_classifier = MultinomialNB()
pipeline = Pipeline([
    ('count_vectorizer', count_vectorizer),
    ('naive_bayes', naive_bayes_classifier)
])
pipeline.fit(docs_train, y_train)
y_predicted = pipeline.predict(docs_test)
print(metrics.classification_report(y_test, y_predicted, target_names=dataset.target_names))
cm = metrics.confusion_matrix(y_test, y_predicted)
print(cm)
