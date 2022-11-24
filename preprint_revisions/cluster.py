
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import argparse
import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from util import json_loader
from util.config import DATA_DIR

# preprocess class
class PreprocessText:
    def __init__(self, text):
        self.lemmatizer = WordNetLemmatizer()

    def __call__(self, text):
        text = self.to_lowercase(text)
        text = self.strip_punctuation(text)
        text = self.replace_numbers(text)
        text = self.remove_stopwords(text)
        text = self.lemmatize_words(text)
        return text

    def to_lowercase(self, text):
        """
        Convert text to lowercase
        """
        return text.lower()
    
    def strip_punctuation(self, text):
        """
        Remove punctuation from text
        """
        return re.sub(r'[^\w\s]', '', text)
    
    def replace_numbers(self, text):
        """
        Replace all numbers in the list of tokenized words with 'num'
        """
        return re.sub(r'\d+', 'num', text)

    def remove_stopwords(self, text):
        """
        Remove stopwords from text
        """
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(text)
        filtered_sentence = [w for w in word_tokens if not w in stop_words]
        return filtered_sentence

    def lemmatize_words(self, text):
        """
        Lemmatize words in text
        """
        lemmatised_text = [self.lemmatizer.lemmatize(word) for word in text]

        return ",".join(lemmatised_text)

    
def preprocess(text):
    """
    Preprocess the text by tokenizing, removing stopwords and lemmatizing
    args:
        text: string
    returns:
        tokens: list of strings
    """
    preprocessed_text = PreprocessText(text)
    return preprocessed_text(text)


def get_tfidf_matrix(texts):
    """
    Get tf-idf matrix
    args:
        texts: list of strings
    returns:
        tfidf_matrix: tf-idf matrix
        feature_names: numpy array of feature names
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = np.array(vectorizer.get_feature_names())
    return tfidf_matrix, feature_names

def cluster_kmeans(tfidf_matrix, feature_names, num_clusters):
    """
    Cluster using K-Means
    args:
        tfidf_matrix: tf-idf matrix
        feature_names: numpy array of feature names of tf-idf matrix
        num_clusters: number of clusters
    returns:
        labels: cluster labels
    """
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(tfidf_matrix)
    labels = kmeans.labels_

    for label in np.unique(labels):
        print(f"LABEL: {label}")
        subset = tfidf_matrix[np.where(labels==label)]
        subset_mean = np.squeeze(np.asarray(subset.mean(axis=0)))
        sorted_indices = np.argsort(subset_mean)[:-(10+1):-1]
        print(feature_names[sorted_indices])

    return labels

def plot_clusters(reduced_data, labels, filename):
    """
    Plot clusters
    args:
        reduced_data: reduced data
        labels: cluster labels
        filename: filename to save plot
    """

    labels_color_map = {
        0: '#20b2aa', 1: '#ff7373', 2: '#ffe4e1', 3: '#005073', 4: '#4d0404',
        5: '#ccc0ba', 6: '#4700f9', 7: '#f6f900', 8: '#00f91d', 9: '#da8c49'
    }

    fig, ax = plt.subplots()

    for index, _ in enumerate(reduced_data):
        # print instance, index, labels[index]
        pca_comp_1, pca_comp_2 = reduced_data[index]
        color = labels_color_map[labels[index]]
        ax.scatter(pca_comp_1, pca_comp_2, c=color)
        
    # save figure
    plt.savefig(filename)

def pca_transform(tfidf_matrix, num_components):
    """
    Perform PCA on tf-idf matrix
    args:
        tfidf_matrix: tf-idf matrix
        num_components: number of components
    returns:
        pca_matrix: PCA matrix
    """
    pca = PCA(n_components=num_components)
    pca_matrix = pca.fit_transform(tfidf_matrix.toarray())
    return pca_matrix


def save_text_clusters(text_list, labels, filename):
    cluster_dict = {}
    for i in range(len(text_list)):
        clust = str(labels[i])
        
        # if cluster is not in cluster_dict, add it
        if clust not in cluster_dict:
            cluster_dict[clust] = []
        
        else:
            cluster_dict[clust].append(text_list[i])
   
    json_loader.save_to_json(cluster_dict, filename)

def parse_args():
    """
    This function parses arguments
    """

    parser = argparse.ArgumentParser(description="Get sitemap tags from biorxiv and medrxiv")
    parser.add_argument("--server", type=str, help="preprint server to get sitemap tags for")
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    server = args.server
    revision_dict = json_loader.load_json(f'{DATA_DIR}/revision_dict_{server}.json')
    revision_texts = list(revision_dict.values())
   
    revision_texts = [preprocess(text) for text in revision_texts if text]
 
    if len(revision_texts)>10:
        print(revision_texts)
        tfidf_matrix, feature_names = get_tfidf_matrix(revision_texts)
        labels = cluster_kmeans(tfidf_matrix, feature_names, 10)
        pca_transformed = pca_transform(tfidf_matrix, 2)
        print(f'{DATA_DIR}/clusters_{server}.png')
        plot_clusters(pca_transformed, labels, f'{DATA_DIR}/clusters_{server}.png')
        save_text_clusters(revision_texts, labels,f'{DATA_DIR}/cluster_dict_{server}.json')
    
    else:
        print("Not enough texts to cluster")
        # make dummy image files
        plt.savefig(f'{DATA_DIR}/clusters_{server}.png')
        json_loader.save_to_json({"output": "Not enough texts to cluster"}, f'{DATA_DIR}/cluster_dict_{server}.json')
            
if __name__ == '__main__':
    main()


