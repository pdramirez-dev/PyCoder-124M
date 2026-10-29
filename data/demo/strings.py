def word_counts(text):
    counts = {}
    for word in text.lower().split():
        counts[word] = counts.get(word, 0) + 1
    return counts


def reverse_words(text):
    return " ".join(reversed(text.split()))
