import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import spacy
import networkx as nx
import os
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from sklearn.cluster import KMeans, DBSCAN, AffinityPropagation
from sklearn.decomposition import LatentDirichletAllocation, PCA
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import hamming, cityblock, euclidean
from wordcloud import WordCloud
import re
import string
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

# Page configuration
st.set_page_config(
    page_title="Automotive Reviews Analysis",
    page_icon="🚗",
    layout="wide"
)

# Initialize nltk data directory
nltk_data_dir = "./nltk_data"
os.makedirs(nltk_data_dir, exist_ok=True)

# Download stopwords if not already downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_dir)

# Load the SpaCy Indonesian model
try:
    nlp = spacy.blank('id')
    nlp.add_pipe('sentencizer')
except:
    st.error("Error loading SpaCy model. Make sure you have installed spaCy with the Indonesian language model.")

# Initialize session state for storing models and data
if 'tfidf_matrix' not in st.session_state:
    st.session_state.tfidf_matrix = None
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None
if 'filtered_df' not in st.session_state:
    st.session_state.filtered_df = None
if 'similarity_matrix' not in st.session_state:
    st.session_state.similarity_matrix = None
if 'w2v_model' not in st.session_state:
    st.session_state.w2v_model = None

# Custom tokenizer function


def custom_tokenizer(text):
    return [token.text for token in nlp(text)]

# Enhanced tokenizer with stopword removal


def enhanced_tokenizer(text):
    from nltk.corpus import stopwords
    try:
        stop_words = set(stopwords.words('indonesian'))
    except:
        import nltk
        nltk.download('stopwords', download_dir=nltk_data_dir)
        stop_words = set(stopwords.words('indonesian'))

    tokens = custom_tokenizer(text)
    tokens = [
        token for token in tokens if token not in stop_words and len(token) > 2]
    return tokens

# TextRank summarizer


def textrank_summarizer(text, num_sentences=5):
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]

    vectorizer = TfidfVectorizer(tokenizer=enhanced_tokenizer, stop_words=None)
    tfidf_matrix = vectorizer.fit_transform(sentences)

    similarity_matrix = cosine_similarity(tfidf_matrix)

    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)

    ranked_sentences = sorted(
        ((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

    sentence_scores_df = pd.DataFrame(
        ranked_sentences, columns=["Score", "Sentence"])

    summary = " ".join([sent for _, sent in ranked_sentences[:num_sentences]])

    return summary, sentence_scores_df

# Function to display topic words


def display_topics(model, feature_names, num_words=10):
    topics = {}
    for topic_idx, topic in enumerate(model.components_):
        topics[f"Topic {topic_idx}"] = [feature_names[i]
                                        for i in topic.argsort()[:-num_words - 1:-1]]

    topics_df = pd.DataFrame(topics)
    return topics_df

# Function to calculate Levenshtein edit distance


def levenshtein_distance(s1, s2):
    rows = len(s1) + 1
    cols = len(s2) + 1
    distance_matrix = [[0 for _ in range(cols)] for _ in range(rows)]

    for i in range(rows):
        distance_matrix[i][0] = i
    for j in range(cols):
        distance_matrix[0][j] = j

    for i in range(1, rows):
        for j in range(1, cols):
            if s1[i-1] == s2[j-1]:
                cost = 0
            else:
                cost = 1

            distance_matrix[i][j] = min(
                distance_matrix[i-1][j] + 1,      # deletion
                distance_matrix[i][j-1] + 1,      # insertion
                distance_matrix[i-1][j-1] + cost  # substitution
            )

    return distance_matrix[len(s1)][len(s2)]

# Function to calculate Hellinger distance


def hellinger_distance(p, q):
    return np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))


# App title and introduction
st.title("🚗 Automotive Reviews Analysis")
st.markdown("""
This application analyzes automotive reviews with a focus on negative sentiments about fuel consumption.
It provides various text analysis techniques including summarization, similarity, clustering, and more.
""")

# Sidebar for navigation
analysis_option = st.sidebar.selectbox(
    "Choose Analysis Method",
    ["Data Explorer", "Text Summarization", "Text Similarity", "Text Clustering",
     "Word2Vec Analysis", "Distance Metrics Comparison"]
)

# Data loading function


@st.cache_data
def load_data():
    try:
        df = pd.read_csv('./data/train_preprocess.csv')
        return df
    except:
        st.error(
            "Could not find the data file. Please make sure 'data/train_preprocess.csv' exists.")
        return None


# Load data
df = load_data()

if df is not None:
    # Data Explorer
    if analysis_option == "Data Explorer":
        st.header("Data Explorer")

        # Basic dataset info
        st.subheader("Dataset Overview")
        st.write(f"Number of rows: {df.shape[0]}")
        st.write(f"Number of columns: {df.shape[1]}")

        # Display sample data
        st.subheader("Sample Data")
        st.dataframe(df.head())

        # Filter data
        st.subheader("Filter Data")
        sentiment = st.selectbox("Select sentiment type:", [
                                 "negative", "neutral", "positive"])
        column = st.selectbox(
            "Select column:", ["fuel", "machine", "part", "price", "service", "others"])

        filtered_df = df[df[column] == sentiment][['sentence']]
        st.session_state.filtered_df = filtered_df

        st.write(
            f"Filtered to {len(filtered_df)} sentences with {sentiment} sentiment about {column}")
        st.dataframe(filtered_df.head(10))

        # Show sentiment distribution
        st.subheader("Sentiment Distribution")
        col1, col2, col3 = st.columns(3)
        col11, col22, col33 = st.columns(3)

        with col1:
            st.write("Fuel Sentiment")
            fig, ax = plt.subplots()
            df['fuel'].value_counts().plot(kind='bar', ax=ax)
            st.pyplot(fig)

        with col2:
            st.write("Machine Sentiment")
            fig, ax = plt.subplots()
            df['machine'].value_counts().plot(kind='bar', ax=ax)
            st.pyplot(fig)

        with col3:
            st.write("Part Sentiment")
            fig, ax = plt.subplots()
            df['part'].value_counts().plot(kind='bar', ax=ax)
            st.pyplot(fig)
        with col11:
            st.write("Price Sentiment")
            fig, ax = plt.subplots()
            df['price'].value_counts().plot(kind='bar', ax=ax)
            st.pyplot(fig)
        with col22:
            st.write("Service Sentiment")
            fig, ax = plt.subplots()
            df['service'].value_counts().plot(kind='bar', ax=ax)
            st.pyplot(fig)
        with col33:
            st.write("Others Sentiment")
            fig, ax = plt.subplots()
            df['others'].value_counts().plot(kind='bar', ax=ax)
            st.pyplot(fig)

    # Text Summarization
    elif analysis_option == "Text Summarization":
        st.header("Text Summarization")

        # Filter data if not already filtered
        if st.session_state.filtered_df is None:
            st.session_state.filtered_df = df[df['fuel'] == 'negative'][[
                'sentence']]

        filtered_df = st.session_state.filtered_df

        st.write(
            f"Analyzing {len(filtered_df)} sentences with negative sentiment about fuel.")

        num_sentences = st.slider("Number of sentences in summary:", 1, 10, 5)

        if st.button("Generate Summary"):
            # Combine all sentences into one text
            text = " ".join(filtered_df['sentence'])

            # Apply TextRank summarization
            with st.spinner("Generating summary..."):
                summary, sentence_scores = textrank_summarizer(
                    text, num_sentences)

            # Display top ranked sentences
            st.subheader("Top Ranked Sentences")
            st.dataframe(sentence_scores.head(num_sentences))

            # Visualize sentence scores
            st.subheader("Sentence Score Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(range(len(sentence_scores.head(10))),
                   sentence_scores.head(10)['Score'])
            ax.set_xlabel('Sentence Index')
            ax.set_ylabel('Score')
            ax.set_title('Top 10 Sentence Scores')
            st.pyplot(fig)

    # Text Similarity
    elif analysis_option == "Text Similarity":
        st.header("Text Similarity Analysis")

        # Filter data if not already filtered
        if st.session_state.filtered_df is None:
            st.session_state.filtered_df = df[df['fuel'] == 'negative'][[
                'sentence']]

        filtered_df = st.session_state.filtered_df

        st.write(
            f"Analyzing {len(filtered_df)} sentences with negative sentiment about fuel.")

        # Limit to a subset for visualization purposes
        max_sentences = st.slider(
            "Maximum number of sentences to compare:", 5, 30, 10)
        subset_df = filtered_df.head(max_sentences)

        if st.button("Calculate Similarity Matrix"):
            with st.spinner("Calculating similarity matrix..."):
                # Convert text data to TF-IDF features
                vectorizer = TfidfVectorizer()
                tfidf_matrix = vectorizer.fit_transform(subset_df['sentence'])

                # Compute cosine similarity matrix
                similarity_matrix = cosine_similarity(tfidf_matrix)

                # Store in session state
                st.session_state.tfidf_matrix = tfidf_matrix
                st.session_state.vectorizer = vectorizer
                st.session_state.similarity_matrix = similarity_matrix

            # Display similarity matrix
            st.subheader("Cosine Similarity Matrix")

            # Create a DataFrame for better visualization
            sim_df = pd.DataFrame(
                similarity_matrix,
                index=subset_df['sentence'].str[:30] + '...',
                columns=subset_df['sentence'].str[:30] + '...'
            )

            # Plot heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(sim_df, annot=True, cmap='viridis', fmt=".2f", ax=ax)
            plt.title("Cosine Similarity Between Sentences")
            st.pyplot(fig)

            # Find most similar sentence pairs
            st.subheader("Most Similar Sentence Pairs")
            pairs = []
            for i in range(len(subset_df)):
                for j in range(i+1, len(subset_df)):
                    pairs.append((i, j, similarity_matrix[i, j]))

            top_pairs = sorted(pairs, key=lambda x: x[2], reverse=True)[:5]

            for i, j, score in top_pairs:
                st.write(f"**Similarity Score: {score:.4f}**")
                st.write(f"Sentence 1: {subset_df['sentence'].iloc[i]}")
                st.write(f"Sentence 2: {subset_df['sentence'].iloc[j]}")
                st.write("---")

    # Text Clustering
    elif analysis_option == "Text Clustering":
        st.header("Text Clustering")

        # Filter data if not already filtered
        if st.session_state.filtered_df is None:
            st.session_state.filtered_df = df[df['fuel'] == 'negative'][[
                'sentence']]

        filtered_df = st.session_state.filtered_df

        st.write(
            f"Analyzing {len(filtered_df)} sentences with negative sentiment about fuel.")

        # Choose clustering method
        clustering_method = st.selectbox(
            "Select Clustering Method:",
            ["K-Means",
                "Latent Dirichlet Allocation (LDA)", "Hierarchical Clustering"]
        )

        if clustering_method == "K-Means":
            num_clusters = st.slider("Number of clusters:", 2, 10, 3)

            if st.button("Apply K-Means Clustering"):
                with st.spinner("Applying K-Means clustering..."):
                    # Always recalculate TF-IDF matrix for the current filtered_df
                    # to ensure dimensions match
                    vectorizer = TfidfVectorizer()
                    tfidf_matrix = vectorizer.fit_transform(
                        filtered_df['sentence'])

                    # Store in session state
                    st.session_state.tfidf_matrix = tfidf_matrix
                    st.session_state.vectorizer = vectorizer

                    # Apply KMeans clustering
                    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
                    clusters = kmeans.fit_predict(tfidf_matrix)

                    # Now clusters will have same length as filtered_df
                    filtered_df['cluster'] = clusters

                # Display clustered sentences
                st.subheader("Clustered Sentences")
                for cluster in range(num_clusters):
                    st.write(f"**Cluster {cluster}:**")
                    cluster_sentences = filtered_df[filtered_df['cluster']
                                                    == cluster]['sentence'].values
                    for i, sentence in enumerate(cluster_sentences[:5]):
                        st.write(f"{i+1}. {sentence}")
                    if len(cluster_sentences) > 5:
                        st.write(f"... and {len(cluster_sentences) - 5} more")
                    st.write("---")

                # Visualize clusters (2D projection with PCA)
                st.subheader("Cluster Visualization")
                pca = PCA(n_components=2)
                coords = pca.fit_transform(tfidf_matrix.toarray())

                # Create a DataFrame for the plot
                plot_df = pd.DataFrame({
                    'x': coords[:, 0],
                    'y': coords[:, 1],
                    'cluster': clusters,
                    'sentence': filtered_df['sentence'].values
                })

                fig, ax = plt.subplots(figsize=(10, 8))
                scatter = ax.scatter(
                    plot_df['x'], plot_df['y'], c=plot_df['cluster'], cmap='viridis', alpha=0.7, s=100)

                # Add a legend
                legend = ax.legend(
                    *scatter.legend_elements(), title="Clusters")
                ax.add_artist(legend)

                # Add labels and title
                ax.set_xlabel('PCA Component 1')
                ax.set_ylabel('PCA Component 2')
                ax.set_title('K-Means Clustering Visualization')

                st.pyplot(fig)

        elif clustering_method == "Latent Dirichlet Allocation (LDA)":
            num_topics = st.slider("Number of topics:", 2, 10, 3)
            num_words = st.slider("Number of words per topic:", 5, 20, 10)

            if st.button("Apply LDA Topic Modeling"):
                with st.spinner("Applying LDA topic modeling..."):
                    # Convert text data to TF-IDF features if not already done
                    if st.session_state.tfidf_matrix is None or st.session_state.vectorizer is None:
                        vectorizer = TfidfVectorizer()
                        tfidf_matrix = vectorizer.fit_transform(
                            filtered_df['sentence'])
                        st.session_state.tfidf_matrix = tfidf_matrix
                        st.session_state.vectorizer = vectorizer
                    else:
                        tfidf_matrix = st.session_state.tfidf_matrix
                        vectorizer = st.session_state.vectorizer

                    # Apply LDA
                    lda = LatentDirichletAllocation(
                        n_components=num_topics, random_state=42)
                    lda_matrix = lda.fit_transform(tfidf_matrix)

                    # Assign the most probable topic to each sentence
                    filtered_df['topic'] = np.argmax(lda_matrix, axis=1)

                    # Get feature names from the vectorizer
                    feature_names = vectorizer.get_feature_names_out()

                    # Get top words for each topic
                    topics_df = display_topics(lda, feature_names, num_words)

                # Display topics
                st.subheader("Topics and Top Words")
                st.dataframe(topics_df)

                # Display sample sentences for each topic
                st.subheader("Sample Sentences by Topic")
                for topic in range(num_topics):
                    st.write(f"**Topic {topic}:**")
                    topic_sentences = filtered_df[filtered_df['topic']
                                                  == topic]['sentence'].values
                    for i, sentence in enumerate(topic_sentences[:3]):
                        st.write(f"{i+1}. {sentence}")
                    if len(topic_sentences) > 3:
                        st.write(f"... and {len(topic_sentences) - 3} more")
                    st.write("---")

                # Create word clouds for topics
                st.subheader("Word Clouds for Topics")
                fig, axes = plt.subplots(1, num_topics, figsize=(15, 5))
                for topic_idx, topic in enumerate(lda.components_):
                    if num_topics > 1:
                        ax = axes[topic_idx]
                    else:
                        ax = axes
                    word_freqs = {feature_names[i]: topic[i]
                                  for i in topic.argsort()[:-num_words - 1:-1]}
                    wordcloud = WordCloud(
                        background_color="white").generate_from_frequencies(word_freqs)
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis("off")
                    ax.set_title(f"Topic {topic_idx}")
                plt.tight_layout()
                st.pyplot(fig)

        elif clustering_method == "Hierarchical Clustering":
            if st.button("Apply Hierarchical Clustering"):
                with st.spinner("Applying hierarchical clustering..."):
                    # Convert text data to TF-IDF features if not already done
                    if st.session_state.tfidf_matrix is None or st.session_state.vectorizer is None:
                        vectorizer = TfidfVectorizer()
                        tfidf_matrix = vectorizer.fit_transform(
                            filtered_df['sentence'])
                        st.session_state.tfidf_matrix = tfidf_matrix
                        st.session_state.vectorizer = vectorizer
                    else:
                        tfidf_matrix = st.session_state.tfidf_matrix
                        vectorizer = st.session_state.vectorizer

                    # Limit to first 30 sentences for visualization
                    max_sentences = min(30, len(filtered_df))
                    subset_tfidf = tfidf_matrix[:max_sentences].toarray()
                    subset_sentences = filtered_df['sentence'].values[:max_sentences]

                    # Perform hierarchical clustering
                    import scipy.cluster.hierarchy as sch
                    linkage_matrix = sch.linkage(subset_tfidf, method='ward')

                    # Plot the dendrogram
                    fig, ax = plt.subplots(figsize=(16, 10))
                    sch.dendrogram(
                        linkage_matrix,
                        labels=[s[:30] + '...' for s in subset_sentences],
                        leaf_rotation=90,
                        leaf_font_size=8
                    )
                    plt.title("Hierarchical Clustering Dendrogram")
                    plt.xlabel("Sentences")
                    plt.ylabel("Distance")
                    plt.tight_layout()
                    st.pyplot(fig)

    # Word2Vec Analysis
    elif analysis_option == "Word2Vec Analysis":
        st.header("Word2Vec Analysis")

        # Filter data if not already filtered
        if st.session_state.filtered_df is None:
            st.session_state.filtered_df = df[df['fuel'] == 'negative'][[
                'sentence']]

        filtered_df = st.session_state.filtered_df

        st.write(
            f"Analyzing {len(filtered_df)} sentences with negative sentiment about fuel.")

        # Check if we already have a trained model
        if st.session_state.w2v_model is None:
            st.info("No Word2Vec model found. Please train a model first.")

            if st.button("Train Word2Vec Model"):
                with st.spinner("Training Word2Vec model... this may take a while"):
                    # Prepare all sentences for training
                    all_sentences = []
                    tokenized = [enhanced_tokenizer(
                        sent) for sent in filtered_df['sentence']]
                    all_sentences.extend(tokenized)

                    # Loss tracking callback
                    class EpochLogger(CallbackAny2Vec):
                        def __init__(self):
                            self.epoch = 0
                            self.losses = []

                        def on_epoch_end(self, model):
                            loss = model.get_latest_training_loss()
                            if self.epoch == 0:
                                self.loss_previous_step = loss
                            else:
                                current_loss = loss - self.loss_previous_step
                                self.losses.append(current_loss)
                                self.loss_previous_step = loss
                            self.epoch += 1

                    # Initialize callback
                    epoch_logger = EpochLogger()

                    # Train Word2Vec model
                    w2v_model = Word2Vec(
                        sentences=all_sentences,
                        vector_size=200,
                        window=6,
                        min_count=2,
                        sg=1,
                        hs=0,
                        negative=10,
                        ns_exponent=0.75,
                        seed=42,
                        workers=4,
                        epochs=10,  # Reduced from 20 for faster training
                        callbacks=[epoch_logger]
                    )

                    # Store model in session state
                    st.session_state.w2v_model = w2v_model

                st.success("Word2Vec model trained successfully!")

        else:
            st.success("Word2Vec model already trained.")

        # Once we have a model, show word similarity
        if st.session_state.w2v_model is not None:
            w2v_model = st.session_state.w2v_model

            st.subheader("Word Similarity Analysis")

            # Input word
            input_word = st.text_input(
                "Enter a word to find similar words:", "boros")

            if input_word:
                if input_word in w2v_model.wv:
                    similar_words = w2v_model.wv.most_similar(
                        input_word, topn=10)

                    st.write(f"Words similar to '{input_word}':")
                    for word, score in similar_words:
                        st.write(f"- {word}: {score:.4f}")

                    # Visualize word vectors using PCA
                    st.subheader("Word Vector Visualization")

                    # Get vectors for similar words
                    words_to_plot = [input_word] + \
                        [word for word, _ in similar_words]
                    word_vectors = [w2v_model.wv[word]
                                    for word in words_to_plot if word in w2v_model.wv]

                    if len(word_vectors) > 1:
                        # Apply PCA to reduce to 2 dimensions
                        pca = PCA(n_components=2)
                        reduced_vectors = pca.fit_transform(word_vectors)

                        # Create a DataFrame for plotting
                        plot_df = pd.DataFrame({
                            'x': reduced_vectors[:, 0],
                            'y': reduced_vectors[:, 1],
                            'word': [word for word in words_to_plot if word in w2v_model.wv]
                        })

                        # Plot
                        fig, ax = plt.subplots(figsize=(12, 8))
                        ax.scatter(plot_df['x'], plot_df['y'],
                                   c='blue', alpha=0.7)

                        # Add labels for each point
                        for _, row in plot_df.iterrows():
                            ax.annotate(
                                row['word'],
                                xy=(row['x'], row['y']),
                                xytext=(5, 2),
                                textcoords='offset points',
                                fontsize=12,
                                color='black'
                            )

                        ax.set_title('Word Embeddings Visualization')
                        st.pyplot(fig)
                    else:
                        st.warning("Not enough valid words for visualization.")
                else:
                    st.warning(
                        f"The word '{input_word}' is not in the vocabulary.")

            # Sentence similarity with Word2Vec
            st.subheader("Sentence Similarity with Word2Vec")

            col1, col2 = st.columns(2)

            with col1:
                sentence1 = st.text_area(
                    "Enter first sentence:", "avanza bahan bakar nya boros banget")

            with col2:
                sentence2 = st.text_area(
                    "Enter second sentence:", "kalau sudah di atas 120 km / jam boros banget avanza saya")

            if st.button("Calculate Similarity"):
                # Function to get sentence vector
                def get_sentence_vector(sentence):
                    tokens = enhanced_tokenizer(sentence)
                    valid_tokens = [
                        token for token in tokens if token in w2v_model.wv]
                    if not valid_tokens:
                        return None

                    vectors = [w2v_model.wv[token] for token in valid_tokens]
                    return np.mean(vectors, axis=0)

                vec1 = get_sentence_vector(sentence1)
                vec2 = get_sentence_vector(sentence2)

                if vec1 is not None and vec2 is not None:
                    # Calculate cosine similarity
                    similarity = cosine_similarity([vec1], [vec2])[0][0]

                    st.write(
                        f"Cosine similarity between sentences: {similarity:.4f}")

                    # Create a gauge chart to visualize similarity
                    import plotly.graph_objects as go

                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=similarity,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Similarity Score"},
                        gauge={
                            'axis': {'range': [0, 1]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 0.33], 'color': "lightgray"},
                                {'range': [0.33, 0.66], 'color': "gray"},
                                {'range': [0.66, 1], 'color': "darkgray"}
                            ]
                        }
                    ))

                    st.plotly_chart(fig)
                else:
                    st.error(
                        "Could not calculate vectors for one or both sentences. Make sure the words are in the vocabulary.")

    # Distance Metrics Comparison
    elif analysis_option == "Distance Metrics Comparison":
        st.header("Distance Metrics Comparison")

        # Filter data if not already filtered
        if st.session_state.filtered_df is None:
            st.session_state.filtered_df = df[df['fuel'] == 'negative'][[
                'sentence']]

        filtered_df = st.session_state.filtered_df

        st.write(
            f"Analyzing {len(filtered_df)} sentences with negative sentiment about fuel.")

        # Limit to a subset for visualization purposes
        max_sentences = st.slider("Number of sentences to compare:", 5, 15, 10)
        subset_df = filtered_df.head(max_sentences)

        # Choose distance metrics
        metrics = st.multiselect(
            "Select distance metrics to calculate:",
            ["Cosine", "Euclidean", "Manhattan", "Hamming", "Levenshtein"],
            default=["Cosine", "Euclidean"]
        )

        if 'distance_matrices' not in st.session_state:
            st.session_state.distance_matrices = None
        if 'comparison_subset_df' not in st.session_state:
            st.session_state.comparison_subset_df = None

        # Then modify your button and the code that follows like this:
        if st.button("Calculate Distance Matrices") or st.session_state.distance_matrices is not None:

            # Only calculate if we haven't already or if the subset has changed
            if (st.session_state.distance_matrices is None or
                st.session_state.comparison_subset_df is None or
                    not st.session_state.comparison_subset_df.equals(subset_df)):

                with st.spinner("Calculating distance matrices..."):
                    # Convert text data to TF-IDF features
                    vectorizer = TfidfVectorizer()
                    tfidf_matrix = vectorizer.fit_transform(
                        subset_df['sentence'])
                    tfidf_dense = tfidf_matrix.toarray()

                    # Calculate selected distance matrices
                    distance_matrices = {}

                    if "Cosine" in metrics:
                        distance_matrices["Cosine"] = cosine_distances(
                            tfidf_matrix)

                    if "Euclidean" in metrics:
                        euclidean_matrix = np.zeros(
                            (len(subset_df), len(subset_df)))
                        for i in range(len(subset_df)):
                            for j in range(len(subset_df)):
                                euclidean_matrix[i, j] = euclidean(
                                    tfidf_dense[i], tfidf_dense[j])
                        distance_matrices["Euclidean"] = euclidean_matrix

                    if "Manhattan" in metrics:
                        manhattan_matrix = np.zeros(
                            (len(subset_df), len(subset_df)))
                        for i in range(len(subset_df)):
                            for j in range(len(subset_df)):
                                manhattan_matrix[i, j] = cityblock(
                                    tfidf_dense[i], tfidf_dense[j])
                        distance_matrices["Manhattan"] = manhattan_matrix

                    if "Hamming" in metrics:
                        # Create binary vectors (word presence)
                        all_words = set()
                        tokenized_sentences = [enhanced_tokenizer(
                            sent) for sent in subset_df['sentence']]
                        for tokens in tokenized_sentences:
                            all_words.update(tokens)

                        binary_vectors = []
                        for tokens in tokenized_sentences:
                            tokens_set = set(tokens)
                            vector = [
                                1 if word in tokens_set else 0 for word in all_words]
                            binary_vectors.append(vector)

                        hamming_matrix = np.zeros(
                            (len(subset_df), len(subset_df)))
                        for i in range(len(subset_df)):
                            for j in range(len(subset_df)):
                                hamming_matrix[i, j] = hamming(
                                    binary_vectors[i], binary_vectors[j])
                        distance_matrices["Hamming"] = hamming_matrix

                    if "Levenshtein" in metrics:
                        levenshtein_matrix = np.zeros(
                            (len(subset_df), len(subset_df)))
                        for i in range(len(subset_df)):
                            for j in range(len(subset_df)):
                                levenshtein_matrix[i, j] = levenshtein_distance(
                                    subset_df['sentence'].iloc[i], subset_df['sentence'].iloc[j])

                        # Normalize by maximum value
                        max_val = np.max(levenshtein_matrix)
                        if max_val > 0:
                            levenshtein_matrix = levenshtein_matrix / max_val
                        distance_matrices["Levenshtein"] = levenshtein_matrix

                    # Store in session state
                    st.session_state.distance_matrices = distance_matrices
                    st.session_state.comparison_subset_df = subset_df.copy()
            else:
                # Use existing matrices from session state
                distance_matrices = st.session_state.distance_matrices

            # Display distance matrices
            st.subheader("Distance Matrices")

            # Only show tabs for metrics that we have matrices for
            available_metrics = list(distance_matrices.keys())
            tabs = st.tabs(available_metrics)

            for i, metric in enumerate(available_metrics):
                with tabs[i]:
                    # Create a DataFrame for better visualization
                    dist_df = pd.DataFrame(
                        distance_matrices[metric],
                        index=subset_df['sentence'].str[:30] + '...',
                        columns=subset_df['sentence'].str[:30] + '...'
                    )

                    # Plot heatmap
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(dist_df, annot=True,
                                cmap='viridis', fmt=".2f", ax=ax)
                    plt.title(f"{metric} Distance Matrix")
                    st.pyplot(fig)

            # Compare distances for a specific pair of sentences
            st.subheader("Distance Comparison for Specific Sentences")

            # Select sentences
            sentence_idx1 = st.selectbox("Select first sentence:", range(len(subset_df)),
                                         format_func=lambda x: subset_df['sentence'].iloc[x][:50] + "...")
            sentence_idx2 = st.selectbox("Select second sentence:", range(len(subset_df)),
                                         format_func=lambda x: subset_df['sentence'].iloc[x][:50] + "...")

            if sentence_idx1 is not None and sentence_idx2 is not None:
                st.write("**First sentence:**",
                         subset_df['sentence'].iloc[sentence_idx1])
                st.write("**Second sentence:**",
                         subset_df['sentence'].iloc[sentence_idx2])

                # Display distances
                st.write("**Distances:**")
                comparison_distances = []
                for metric in available_metrics:
                    distance = distance_matrices[metric][sentence_idx1,
                                                         sentence_idx2]
                    st.write(f"- {metric}: {distance:.4f}")
                    comparison_distances.append(distance)

                # Create a bar chart comparing distances
                comparison_df = pd.DataFrame({
                    'Metric': available_metrics,
                    'Distance': comparison_distances
                })

                fig, ax = plt.subplots(figsize=(10, 5))
                ax.bar(comparison_df['Metric'], comparison_df['Distance'])
                ax.set_xlabel('Distance Metric')
                ax.set_ylabel('Distance Value')
                ax.set_title(
                    'Comparison of Distance Metrics for Selected Sentences')
                st.pyplot(fig)
else:
    st.error("Failed to load data. Please check if the data file exists.")
