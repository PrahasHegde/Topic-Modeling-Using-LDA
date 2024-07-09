#LDA -> group text by category


documents = [
    "Smartphones are an essential piece of technology in today's digital age.",
    "Professional athletes train rigorously to excel in their respective sports.",
    "Electric vehicles represent a significant advancement in automotive technology.",
    "Watching sports events live offers an exhilarating experience for fans.",
    "Virtual reality technology immerses users in simulated environments for gaming and entertainment.",
    "Sports equipment manufacturers continually innovate to improve performance and safety.",
    "Artificial intelligence is being integrated into various aspects of modern technology.",
    "Participating in sports promotes physical fitness and overall well-being."
]

#imports
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Convert text data into numerical feature vectors using CountVectorizer
vectorizer = CountVectorizer(stop_words='english', max_features=1000)
X = vectorizer.fit_transform(documents)

"""Our goal with this code is to classify the text in the documents into different topics. 
We can accomplish that using LatentDirichletAllocation"""
# Apply LDA
lda = LatentDirichletAllocation(n_components=2, random_state=42)  # Assuming there are 2 topics
lda.fit(X)


# Assign each document to the topic with the highest probability
topic_assignments = lda.transform(X).argmax(axis=1)

# Group documents by their assigned topics
topic_documents = {'Technology': [], 'Sports': []}
for i, topic_idx in enumerate(topic_assignments):
    topic = 'Technology' if topic_idx == 0 else 'Sports'
    topic_documents[topic].append(documents[i])

# Print documents grouped by topics
for topic, docs in topic_documents.items():
    print(f"{topic}:")
    for doc in docs:
        print("-", doc)
    print()