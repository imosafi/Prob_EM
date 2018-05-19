# Parse the data file.
from collections import Counter

# Parse the articles without the titles to a list of words.
def parse_no_title(lines, seperator, topics=None):
    words = list()
    for i in range(2, len(lines), 4):
        parsed_line = lines[i].split(seperator)
        parsed_line.remove("")
        for word in parsed_line:
            words.append(word)
    return words

# Parse the title of the articles.
def parse_titile(lines, seperator, topics):
    articles = list()
    for i in range(0, len(lines), 4):
        articles.append([0] * 9)
        parsed_line = lines[i][0:-1].split(seperator)
        for word in parsed_line[2:]:
            if word in topics:
                articles[i / 4][topics[word]] = 1
    return articles

# Parse the topics.
def parse_topics(lines, seperator, topics):
    topics = dict()
    for i in range(0, len(lines), 2):
        topics[lines[i]] = len(topics)
    return topics

# Parse the articles to a counter of words for each artile.
def parse_sep_articles(lines, seperator, topics=None):
    articles = list()
    for i in range(2, len(lines), 4):
        article = Counter()
        parsed_line = lines[i].split(seperator)
        parsed_line.remove("")
        article.update(parsed_line)
        articles.append(article)
    return articles

# Read the data file.
def read_file(file_name, parse_func, separator=None, topics=None):
    file = open(file_name, 'r')
    lines = file.read().splitlines()
    file.close()
    return parse_func(lines, separator, topics)


# Write the output file.
def write_file(file_name, content):
    file = open(file_name, 'w')
    file.write(content)
    file.close()

