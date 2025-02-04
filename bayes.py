class NaiveBayesClassifier:
    def __init__(self):
        self.word_counts = {'spam': {}, 'not_spam': {}}
        self.class_counts = {'spam': 0, 'not_spam': 0}
        self.total_words = {'spam': 0, 'not_spam': 0}

    def tokenize(self, text):
        """Manually split text into words (tokens)"""
        text = text.lower()  # Convert to lowercase
        text = ''.join([c if c.isalnum() else ' ' for c in text])  # Remove punctuation
        return text.split()  # Split into words

    def train(self, dataset):
        """Train the classifier using a dataset"""
        for text, label in dataset:
            self.class_counts[label] += 1
            words = self.tokenize(text)

            for word in words:
                if word not in self.word_counts[label]:
                    self.word_counts[label][word] = 0
                self.word_counts[label][word] += 1
                self.total_words[label] += 1

    def calculate_probability(self, text, label):
        """Calculate the probability of the text being in a given class"""
        words = self.tokenize(text)
        total_class_count = self.class_counts[label]
        total_documents = sum(self.class_counts.values())

        # Calculate prior probability P(Class)
        prob = total_class_count / total_documents if total_documents > 0 else 0.5

        for word in words:
            word_frequency = self.word_counts[label].get(word, 0) + 1  # Laplace smoothing
            total_count = self.total_words[label] + len(self.word_counts[label])  # Normalize
            prob *= word_frequency / total_count

        return prob

    def predict(self, text):
        """Predict whether a given text is 'spam' or 'not_spam'"""
        spam_prob = self.calculate_probability(text, 'spam')
        not_spam_prob = self.calculate_probability(text, 'not_spam')

        return 'spam' if spam_prob > not_spam_prob else 'not_spam'


dataset = [
    ("Win a free lottery ticket now", "spam"),
    ("Limited time offer, claim your prize", "spam"),
    ("Hey, how are you doing?", "not_spam"),
    ("Meeting at 10 AM, don't be late", "not_spam"),
    ("Congratulations! You won a gift card", "spam"),
    ("Let's catch up over coffee", "not_spam"),
]

nb = NaiveBayesClassifier()
nb.train(dataset)

test_email = "Win a free prize now"
print(f"Prediction for '{test_email}':", nb.predict(test_email))  # Expected: 'spam'

test_email2 = "See you at the meeting"
print(f"Prediction for '{test_email2}':", nb.predict(test_email2))  # Expected: 'not_spam'

test_email3 = "Win a free lottery ticket now"
print(f"Prediction for '{test_email3}':", nb.predict(test_email3))
