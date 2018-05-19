import copy
from collections import defaultdict, Counter
import math

EPSILON = 0.0001
LAMBDA = 1.0
K = 10

# EM class that stores the
class EM(object):
    def __init__(self, num_of_topics, articles, article_clusters, vocab_size):
        self._alphas = list()
        self._P = list()
        self._one_divided_by_N = 1.0 / float(len(articles))
        self._ntk = articles
        self._nt = list()
        self._article_clusters = article_clusters
        self._vocab_size = vocab_size  # needed for smoothing
        self._initialize_nt(articles)
        self._create_parameters(num_of_topics, articles, article_clusters)
        self._initialize()

    # Initialize the parameters.
    def _initialize(self):
        w = list()
        for t in range(0, len(self._ntk)):
            w.append(copy.deepcopy(self._article_clusters[t]))
        self._update_alphas(w)
        self._update_P(w)
        self._last_wti = w

    # Initialize the nt according to each length of an article.
    def _initialize_nt(self, articles):
        for article in articles:
            self._nt.append(len(article))

    # Create the list for alpha parameter and the dictionaries for the P parameter.
    def _create_parameters(self, num_of_topics, articles, article_clusters):
        for i in range(0, num_of_topics):
            self._alphas.append(1.0 / float(num_of_topics))
            self._P.append(defaultdict(lambda: (LAMBDA / (LAMBDA * float(self._vocab_size)))))
        self._fill_cluster_words(articles, article_clusters)

    # Fill the words for every P dictionary.
    def _fill_cluster_words(self, articles, article_clusters):
        for article, one_hot_vec in zip(articles, article_clusters):
            index = one_hot_vec.index(1)
            for word in article:
                self._P[index][word] = 1.0

    # Calculate the z value for the underflow management.
    def _calculate_z(self, t):
        z = [0.0] * len(self._alphas)
        for i in range(0, len(self._alphas)):
            z[i] = math.log(self._alphas[i])
            for word in self._ntk[t]:
                z[i] += self._ntk[t][word] * math.log(self._P[i][word])
        return z

    # Calculate the wti with the underflow management.
    def _calculate_stable_wti(self, z, i, m):
        numerator = math.exp(z[i] - m)
        return numerator

    # Calculate the numerator of wti.
    def _calculate_wti_numerator(self, z, m, i):
        # z = self._calculate_z(t)
        # m = max(z)
        if z[i] - m < -K:
            wti = 0.0
        else:
            wti = self._calculate_stable_wti(z, i, m)
        return wti

    # Update the alpha values.
    def _update_alphas(self, w):
        for i in range(0, len(self._alphas)):
            # self._alphas[i] = self._one_divided_by_N
            temp_sum = 0.0
            for t in range(0, len(w)):
                temp_sum += w[t][i]
                # self._alphas[i] *= w[t][i] # should be sum?
            new_alpha_i = self._one_divided_by_N * temp_sum
            self._alphas[i] = new_alpha_i if new_alpha_i > EPSILON else EPSILON
        alpha_sum = sum(self._alphas)
        for i in range(0, len(self._alphas)):
            self._alphas[i] /= alpha_sum

    # Update the P values.
    def _update_P(self, w):
        for i in range(0, len(self._P)):
            numerators = defaultdict(lambda: LAMBDA)
            denominator = self._vocab_size * LAMBDA
            for t in range(0, len(w)):
                # Calculate the numerator for the word.
                for word in self._ntk[t]:
                    numerators[word] += w[t][i] * self._ntk[t][word]
                # Calculate the denominator for all the words.
                denominator += w[t][i] * self._nt[t]
            for word in self._P[i]:
                self._P[i][word] = numerators[word] / denominator


    # Calculate the likelihood.
    def calculate_likelihood(self):
        total_ln_l = 0.0
        for t in range(0, len(self._ntk)):
            # z = []
            z = self._calculate_z(t)
            m = max(z)

            e_sum = 0.0
            for j in range(0, len(z)):
                if z[j] - m >= -K:
                    e_sum += math.exp(z[j] - m)

            total_ln_l += m + math.log(e_sum)

        return total_ln_l

    # Calculate the accuracy of the algorithm.
    def calculate_accuracy(self, topics, article_topics):
        num_to_topic = {v: k for k, v in topics.iteritems()}
        # Cluster the articles according to the current parameters.
        articles_clusters = self.cluster_articles(self._ntk)
        cluster_topic_dict = self.create_cluster_topic_dict(articles_clusters, topics, article_topics)
        correct_predictions = 0.0

        # Calculate the accuracy for that clustering.
        for t in xrange(len(articles_clusters)):
            cluster = articles_clusters[t].index(1)
            topics_names = [num_to_topic[i] for i, x in enumerate(article_topics[t]) if x == 1]
            if cluster_topic_dict[cluster] in topics_names:
                correct_predictions += 1
        return correct_predictions / len(articles_clusters)

    # Create a conversion vector that will map each cluster to its dominant topic.
    def create_cluster_topic_dict(self, articles_clusters, topics, article_topics):
        cluster_topic_dict = {}
        topic_couters = []
        num_to_topic = {v: k for k, v in topics.iteritems()}
        for i in xrange(len(self._alphas)):
            c = Counter()
            for t in topics.keys():
                c[t] = 0
            topic_couters.append(c)
        for t in xrange(len(article_topics)):
            for index, topic_value in enumerate(article_topics[t]):
                topic_couters[articles_clusters[t].index(1)][num_to_topic[index]] += topic_value
        for index, counter in enumerate(topic_couters):
            max_value_topic = max(counter, key=counter.get)
            cluster_topic_dict[index] = max_value_topic
        return cluster_topic_dict


    # Update parameters.
    def update_parameters(self):
        w = list()
        for t in range(0, len(self._ntk)):
            z = self._calculate_z(t)
            m = max(z)
            w.append(list())
            for i in range(0, len(self._alphas)):
                wti = self._calculate_wti_numerator(z, m, i)
                w[t].append(wti)
        for t in range(0, len(w)):
            alpha_j_sum = sum(w[t])
            for i in range(0, len(self._alphas)):
                w[t][i] /= alpha_j_sum
        self._last_wti = w
        self._update_alphas(w)
        self._update_P(w)

    # Cluster the given articles according to the last wti.
    def cluster_articles(self, articles):
        article_clusters = list()
        for t, article in enumerate(articles):
            article_clusters.append([0] * 9)
            best_cluster = self._last_wti[t].index(max(self._last_wti[t]))
            article_clusters[t][best_cluster] = 1
        return article_clusters


