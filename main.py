import sys

from Utils import *
from Helpers import *
from EM import EM


# Initialize the clusters according to the instructions.
def em_initialization(articles, num_of_articles):
    clustered_articles = list()
    for i, article in enumerate(articles):
        clustered_articles.append([0] * num_of_articles)
        clustered_articles[i][i % 9] = 1
    return clustered_articles


# Filter rare words.
def filter_rare_words(articles, words):
    frequencies = Counter()
    frequencies.update(words)
    filtered_articles = [Counter(word for word in article.elements() if frequencies[word] > 3) for article in articles]
    filtered_words = [word for word in words if frequencies[word] > 3]
    filtered_words_set = set()
    for article in filtered_articles:
        filtered_words_set.update(set(article))
    vocab_size = len(filtered_words_set)
    print "Vocabulary size:", vocab_size
    return filtered_articles, vocab_size, filtered_words

# Start the iterations for the em algorithm.
def EM_Algorithm(em, list_of_words, topics, article_topics):
    likelihoods = [em.calculate_likelihood()]
    perplexities = [calculate_perplexity(likelihoods[-1], list_of_words)]

    # EM algorithm.
    while True:
        print likelihoods[-1]
        em.update_parameters()
        likelihoods.append(em.calculate_likelihood())
        perplexities.append(calculate_perplexity(likelihoods[-1], list_of_words))
        if abs(likelihoods[-1] - likelihoods[-2]) < 1.0:
            break
    print "Accuracy: " + str(em.calculate_accuracy(topics, article_topics))
    return likelihoods, perplexities

# Create the confusion matrix.
def create_confusion_matrix(articles, article_topics, em):
    conf_mat = [[0] * 10 for i in range(0, 9)]
    clustered_articles = em.cluster_articles(articles)

    for i in range(len(clustered_articles)):
        article_cluster = clustered_articles[i].index(1)
        topics_ind = [j for j, x in enumerate(article_topics[i]) if x == 1]
        for ind in topics_ind:
            conf_mat[article_cluster][ind] += 1
        conf_mat[article_cluster][-1] += 1
    return conf_mat

def print_confusion_matrix(conf_mat):
    print "=========Confusion Matrix========="
    for i in range(len(conf_mat)):
        for j in range(len(conf_mat[i])):
            print conf_mat[i][j],
        print
    print "=========Confusion Matrix========="




if __name__ == "__main__":

    train_file = sys.argv[1]
    topics_file = sys.argv[2]
    num_of_topics = 9
    # Read the articles and get the histograms of words for each article.
    articles = read_file(train_file, parse_sep_articles, " ")
    list_of_words = read_file(train_file, parse_no_title, " ")
    topics = read_file(topics_file, parse_topics)
    article_topics = read_file(train_file, parse_titile, "\t", topics)
    # Filter rare words.
    articles, vocab_size, list_of_words = filter_rare_words(articles, list_of_words)
    # Cluster the articles according to the initialization instructions.
    clusters = em_initialization(articles, num_of_topics)
    em = EM(num_of_topics, articles, clusters, vocab_size)
    likelihoods, perplexities = EM_Algorithm(em, list_of_words, topics, article_topics)
    conf_mat = create_confusion_matrix(articles, article_topics, em)
    print_confusion_matrix(conf_mat)
    list_of_topics = sorted(topics, key=topics.get)


