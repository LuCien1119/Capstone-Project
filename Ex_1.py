import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
from gensim.models import CoherenceModel, LdaModel
from gensim.corpora import Dictionary
import multiprocessing
from scipy.special import kl_div


# 解决多进程警告
def set_multiprocessing():
    multiprocessing.freeze_support()
    if __name__ == '__main__':
        multiprocessing.set_start_method('spawn', force=True)


set_multiprocessing()

plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")


def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\d{10,}', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'&amp;', ' ', text)

    tokens = word_tokenize(text.lower())

    custom_stopwords = set(stopwords.words('english')).union({
        'said', 'would', 'could', 'one', 'like', 'get', 'also', 'us', 'new', 'time'
    })
    tokens = [word for word in tokens if len(word) >= 3 and word not in custom_stopwords]

    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]

    return ' '.join(tokens)


def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath, header=0)
    df['Cleaned_Text'] = df['fullText'].apply(clean_text)
    return df


def filter_political_tweets(df, keywords):
    pattern = r'\b(' + '|'.join(re.escape(word) for word in keywords) + r')\b'
    political_tweets = df[df['Cleaned_Text'].apply(
        lambda x: bool(re.search(pattern, x.lower())) if isinstance(x, str) else False
    )].copy()
    return political_tweets


def calculate_coherence(model, texts, dictionary):
    coherence_model = CoherenceModel(
        model=model,
        texts=texts,
        dictionary=dictionary,
        coherence='c_v'
    )
    return coherence_model.get_coherence()


def find_optimal_topics(X, texts, dictionary, min_topics=5, max_topics=20):
    topic_range = range(min_topics, max_topics + 1)
    metrics = {
        'n_topics': [],
        'perplexity': [],
        'coherence': [],
        'kl_divergence': []
    }

    for n_topics in topic_range:
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            learning_method='online',
            batch_size=256,
            max_iter=50,
            random_state=42,
            n_jobs=1
        )
        lda.fit(X)

        corpus = [dictionary.doc2bow(text) for text in texts]
        lda_gensim = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=n_topics,
            passes=10,
            random_state=42
        )

        metrics['n_topics'].append(n_topics)
        metrics['perplexity'].append(lda.perplexity(X))
        metrics['coherence'].append(calculate_coherence(lda_gensim, texts, dictionary))

        # KL divergence
        topic_dist = lda.components_ / lda.components_.sum(axis=1)[:, np.newaxis]
        kl_matrix = np.zeros((n_topics, n_topics))
        for i in range(n_topics):
            for j in range(n_topics):
                if i != j:
                    kl_matrix[i, j] = np.sum(kl_div(topic_dist[i], topic_dist[j]))
        metrics['kl_divergence'].append(np.mean(kl_matrix))

        print(f"Topics: {n_topics}, Perplexity: {metrics['perplexity'][-1]:.2f}, "
              f"Coherence: {metrics['coherence'][-1]:.4f}, KL Divergence: {metrics['kl_divergence'][-1]:.4f}")

    return metrics


# 新增可视化函数
def visualize_topic_metrics(metrics):
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.plot(metrics['n_topics'], metrics['perplexity'], 'bo-')
    plt.xlabel("Number of Topics")
    plt.ylabel("Perplexity")
    plt.title("Perplexity vs. Number of Topics")
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(metrics['n_topics'], metrics['coherence'], 'go-')
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence Score")
    plt.title("Coherence vs. Number of Topics")
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(metrics['n_topics'], metrics['kl_divergence'], 'ro-')
    plt.xlabel("Number of Topics")
    plt.ylabel("KL Divergence")
    plt.title("Topic Distinctness (KL Divergence)")
    plt.grid(True)

    # comprehensive score
    plt.subplot(2, 2, 4)
    perplexities_norm = (metrics['perplexity'] - np.min(metrics['perplexity'])) / \
                        (np.max(metrics['perplexity']) - np.min(metrics['perplexity']))
    coherences_norm = (np.max(metrics['coherence']) - metrics['coherence']) / \
                      (np.max(metrics['coherence']) - np.min(metrics['coherence']))
    kl_norm = (metrics['kl_divergence'] - np.min(metrics['kl_divergence'])) / \
              (np.max(metrics['kl_divergence']) - np.min(metrics['kl_divergence']))

    combined_score = 0.5 * coherences_norm + 0.3 * perplexities_norm + 0.2 * kl_norm
    plt.plot(metrics['n_topics'], combined_score, 'mo-', label='Combined Score')
    plt.plot(metrics['n_topics'], coherences_norm, 'g:', alpha=0.5, label='Coherence (norm)')
    plt.plot(metrics['n_topics'], perplexities_norm, 'b:', alpha=0.5, label='Perplexity (norm)')
    plt.plot(metrics['n_topics'], kl_norm, 'r:', alpha=0.5, label='KL Divergence (norm)')

    optimal_idx = np.argmin(combined_score)
    plt.scatter(metrics['n_topics'][optimal_idx], combined_score[optimal_idx],
                c='red', s=100, label=f'Optimal: {metrics["n_topics"][optimal_idx]}')

    plt.xlabel("Number of Topics")
    plt.ylabel("Normalized Scores")
    plt.title("Combined Metric Score")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('topic_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    political_keywords = {'vote', 'voter', 'poll', 'swing', 'Ballot',
                          'kamala', 'harris', 'joe', 'biden', 'donald', 'trump',
                          'tax', 'waste', 'wasting', 'wasted', 'money', 'dollar', 'efficiency', 'efficient',
                          'party', 'democracy', 'democratic', 'republic', 'republics', 'republican',
                          'speech', 'freedom', 'media',
                          'lgbt', 'maga'
                          }

    df = load_and_preprocess_data('data.csv')
    political_tweets = filter_political_tweets(df, political_keywords)
    political_tweets[['fullText']].to_csv('political_tweets.csv', index=False)
    print(f"Number of political tweets: {len(political_tweets)}")

    texts = [doc.split() for doc in political_tweets['Cleaned_Text']]
    dictionary = Dictionary(texts)

    vectorizer = CountVectorizer(
        max_df=0.90,
        min_df=5,
        ngram_range=(1, 2),
        stop_words='english'
    )
    X = vectorizer.fit_transform(political_tweets['Cleaned_Text'])

    metrics = find_optimal_topics(X, texts, dictionary)

    visualize_topic_metrics(metrics)

    perplexities_norm = (metrics['perplexity'] - np.min(metrics['perplexity'])) / \
                        (np.max(metrics['perplexity']) - np.min(metrics['perplexity']))
    coherences_norm = (np.max(metrics['coherence']) - metrics['coherence']) / \
                      (np.max(metrics['coherence']) - np.min(metrics['coherence']))
    kl_norm = (metrics['kl_divergence'] - np.min(metrics['kl_divergence'])) / \
              (np.max(metrics['kl_divergence']) - np.min(metrics['kl_divergence']))

    combined_score = 0.5 * coherences_norm + 0.3 * perplexities_norm + 0.2 * kl_norm
    optimal_idx = np.argmin(combined_score)
    optimal_topics = metrics['n_topics'][optimal_idx]

    print(f"\nSelected optimal topics: {optimal_topics}")
    print(f"Perplexity: {metrics['perplexity'][optimal_idx]:.2f}")
    print(f"Coherence: {metrics['coherence'][optimal_idx]:.4f}")
    print(f"KL Divergence: {metrics['kl_divergence'][optimal_idx]:.4f}")

    # model training
    final_lda = LatentDirichletAllocation(
        n_components=optimal_topics,
        learning_method='online',
        batch_size=256,
        max_iter=50,
        random_state=43,
        n_jobs=1
    )
    final_lda.fit(X)

    def plot_top_words(model, feature_names, n_top_words=15, figsize=(12, 8)):
        fig, axes = plt.subplots(1, optimal_topics, figsize=figsize, sharex=True)
        axes = axes.flatten()

        for topic_idx, topic in enumerate(model.components_):
            top_features = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
            weights = topic[topic.argsort()[:-n_top_words - 1:-1]]

            wordcloud = WordCloud(
                width=400, height=300,
                background_color='white',
                colormap='viridis'
            ).generate_from_frequencies(dict(zip(top_features, weights)))

            axes[topic_idx].imshow(wordcloud, interpolation='bilinear')
            axes[topic_idx].set_title(f'Topic {topic_idx + 1}', fontsize=14)
            axes[topic_idx].axis('off')

        plt.tight_layout()
        plt.savefig('topic_wordclouds.png', dpi=300, bbox_inches='tight')
        plt.show()

    # visualization
    feature_names = vectorizer.get_feature_names_out()
    plot_top_words(final_lda, feature_names)

    def plot_topic_barcharts(model, feature_names, n_top_words=5, figsize=(15, 10), bar_height=0.3):
        plt.figure(figsize=figsize)

        for topic_idx, topic in enumerate(model.components_):
            # weight
            top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
            top_features = [feature_names[i] for i in top_features_ind]
            weights = topic[top_features_ind]

            # subplot
            plt.subplot(1, optimal_topics, topic_idx + 1)

            bars = plt.barh(top_features, weights, height=bar_height, color=sns.color_palette("husl", n_top_words))

            # labels
            for bar in bars:
                width = bar.get_width()
                plt.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                         f'{width:.3f}',
                         ha='left', va='center', fontsize=10)

            plt.title(f'Topic {topic_idx + 1}', fontsize=12, pad=10)
            plt.xlabel('TF-IDF Weight', fontsize=10)
            plt.xlim(0, max(weights) * 1.2)

            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['left'].set_visible(False)
            plt.tick_params(axis='y', which='both', left=False)
            plt.grid(axis='x', linestyle='--', alpha=1.0)

        plt.tight_layout()
        plt.savefig('topic_barcharts.png', dpi=300, bbox_inches='tight')
        plt.show()

    plot_topic_barcharts(final_lda, feature_names)

if __name__ == '__main__':
    main()