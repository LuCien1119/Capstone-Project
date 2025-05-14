import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import networkx as nx

plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

df = pd.read_csv('data.csv', header=0)


def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'@\w+', '', text)  # 移除@提及
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\d{10,}', '', text)  # 移除长数字（如电话号码）
    text = re.sub(r'[^\w\s]', '', text)  # 移除非字母数字字符
    text = re.sub(r'&amp;', '', text)

    tokens = word_tokenize(text.lower())

    custom_stopwords = set(stopwords.words('english')).union({
        'said', 'would', 'could', 'one', 'like', 'get', 'also', 'us', 'new', 'time'
    })
    tokens = [word for word in tokens if len(word) >= 3 and word not in custom_stopwords]

    # stemmer = PorterStemmer()
    # tokens = [stemmer.stem(word) for word in tokens]

    return ' '.join(tokens)

# from spellchecker import SpellChecker
#
# spell = SpellChecker()
# def correct_text(text):
#     words = text.split()
#     corrected = [spell.correction(word) or word for word in words]
#     return ' '.join(corrected)


df['Cleaned_Text'] = df['fullText'].apply(clean_text)
# df['Cleaned_Text'] = df['Cleaned_Text'].apply(correct_text)


political_keywords = {'vote', 'voter', 'poll', 'swing', 'Ballot',
                      'kamala', 'harris', 'joe','biden', 'donald','trump',
                      'tax', 'waste','wasting', 'wasted', 'money', 'dollar', 'efficiency', 'efficient',
                      'party', 'democracy', 'democratic', 'republic', 'republics', 'republican',
                      'speech', 'freedom', 'media',
                      'lgbt', 'maga'
}


def is_political(text):
    if not isinstance(text, str):
        return False
    pattern = r'\b(' + '|'.join(re.escape(word) for word in political_keywords) + r')\b'
    return bool(re.search(pattern, text.lower()))


political_tweets = df[df['Cleaned_Text'].apply(is_political)].copy()
tweets = political_tweets[['fullText']]
tweets.to_csv('political_tweets.csv', index=False)
print(f"Num of political tweets: {len(political_tweets)}")

vectorizer = TfidfVectorizer(
    max_df=0.90,
    min_df=3,
    ngram_range=(1, 2),
    stop_words='english'
)
X = vectorizer.fit_transform(political_tweets['Cleaned_Text'])

optimal_topics = 5
final_lda = LatentDirichletAllocation(
    n_components=optimal_topics,
    learning_method='online',
    batch_size=256,
    max_iter=50,
    random_state=41
)
final_lda.fit(X)

feature_names = vectorizer.get_feature_names_out()
# plot_top_words(final_lda, feature_names)


def plot_topic_barcharts(model, feature_names, n_top_words=5, figsize=(10, 15), bar_height=0.5):
    plt.figure(figsize=figsize)

    n_topics = model.components_.shape[0]  # 获取实际主题数量

    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        plt.subplot(n_topics, 1, topic_idx + 1)

        bars = plt.barh(top_features, weights, height=bar_height,
                        color=sns.color_palette("husl", n_top_words))

        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                     f'{width:.3f}',
                     ha='left', va='center', fontsize=10)

        plt.title(f'Topic {topic_idx + 1}', fontsize=12, pad=10)
        plt.xlabel('TF-IDF Weight', fontsize=10)
        plt.xlim(0, max(weights) * 1.2)

        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', which='both', left=False)
        plt.grid(axis='x', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('topic_topwords.png', dpi=300, bbox_inches='tight')
    plt.show()

plot_topic_barcharts(final_lda, feature_names)

def plot_topic_cooccurrence_network(lda_model, X, threshold=0.1):
    doc_topic = lda_model.transform(X)
    cooccur = np.zeros((lda_model.n_components, lda_model.n_components))

    for doc in doc_topic:
        topics = np.where(doc > threshold)[0]

        for i in topics:
            for j in topics:
                if i != j:
                    cooccur[i, j] += 1

    G = nx.Graph()

    for i in range(lda_model.n_components):
        G.add_node(i, label=f"Topic {i + 1}")

    max_cooccur = cooccur.max()
    for i in range(lda_model.n_components):
        for j in range(i + 1, lda_model.n_components):
            if cooccur[i, j] > 0:
                normalized_weight = 0.1 + 4.9 * (cooccur[i, j] / max_cooccur)
                G.add_edge(i, j, weight=normalized_weight, count=cooccur[i, j])

    plt.figure(figsize=(12, 9))

    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

    node_colors = [plt.cm.tab10(i) for i in range(lda_model.n_components)]
    nx.draw_networkx_nodes(G, pos, node_size=2200, node_color=node_colors, alpha=0.9)

    edges = G.edges(data=True)
    edge_weights = [d['weight'] for (u, v, d) in edges]
    nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.6, edge_color='gray')

    node_labels = {n: G.nodes[n]['label'] for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_weight='bold')

    edge_labels = {(u, v): f"{int(d['count'])}" for (u, v, d) in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    # plt.title("LDA Topics Co-occurrence Network", fontsize=14)

    plt.axis('off')
    plt.tight_layout()
    plt.savefig('topic_cooccurrence_network.png', dpi=300, bbox_inches='tight')
    plt.show()

plot_topic_cooccurrence_network(final_lda, X, threshold=0.15)
