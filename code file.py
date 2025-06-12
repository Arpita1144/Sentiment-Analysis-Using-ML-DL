from sklearn import tree
from sklearn.feature_extraction.text import CountVectorizer

positive_set = ["I love this sandwich","I like this restaurant","It is amazing"]
negative_set = ["I hate this sandwich","I do not like this restaurant","It is terrible"]
sample_set = ["I love this sandwich","I do not like this restaurant","It is terrible"]

# Fix: Create flat list instead of nested list
data_set = positive_set + negative_set
data_labels = ["Positive"] * len(positive_set) + ["Negative"] * len(negative_set)

print("Training data:\n", data_set)
print("Labels:\n", data_labels)

# Use CountVectorizer() to convert text to numerical features
vectorizer = CountVectorizer()

# Use fit() to learn the vocabulary of trianing data and for matrix representation of numerical data
vectorizer.fit(data_set)

# Use transform() to transform the training data into a matrix vectors
data_vectors =vectorizer.transform(data_set)

sample_vectors = vectorizer.transform(sample_set)

# use get_feature_names_out() to get the feature names
feature_names = vectorizer.get_feature_names_out()
print("Feature names:\n", feature_names)

# printing sample_vectors in array format
print("Data vectors:\n",data_vectors.toarray())
print("Sample vectors:\n",sample_vectors.toarray())

# Now use the tree module for classification
classifier = tree.DecisionTreeClassifier()

# Use fit() to train the classifier with data_vectors and data_labels
classifier.fit(data_vectors,data_labels)

# Use predict() to predict the labels of sample_vectors
predictions = classifier.predict(sample_vectors)
print(predictions)