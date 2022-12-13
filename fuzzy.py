
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing
from functools import partial
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin


class ExtractiveFuzzy(BaseEstimator, TransformerMixin):
    def __init__(self, n_top: int=1, threshold: float=0.8, n_jobs: int=0, batch_size: int=1000) -> None:
        self.n_top = n_top
        self.threshold = threshold
        self.n_jobs = n_jobs
        self.batch_size = batch_size
    
    
    def _check_format(self, a):
        if isinstance(a, (np.ndarray, list, np.array)):
            return pd.DataFrame(a)
        elif isinstance(a, pd.Series):
            return a.to_frame()
        else:
            return a
    
    
    def _word_2_ngrams(self, text, n=3):
        """
        Transform the text gave in parameter into part of n character
        text = "example" return --> ["exa", "xam", "amp", "mpl", "ple"]
        :param text: text to transform in ngrams
        :param n: size of grams
        :type text: str
        :type n: int
        :return: list
        """
        ngrams = [text[i: i+n] for i in range(len(text)-n+1)]
        return ngrams
    
    
    def _vectorizer(self, a, b):
        """
        Fit sklearn TfidfVectorizer on arr_1 and transform arr_1 and arr_2 in 2 documents*terms matrix
        
        :param a: query array of string to fit the TFIDF and to transform in documents*terms matrix
        :param b: population array of string to transform in documents*terms matrix
        
        :type a: array-like
        :type b: array-like
        
        :return: 2 array-like, shape(n_samples, n_ngrams)
        """
        vectorizer = TfidfVectorizer(analyzer=self._word_2_ngrams)
        a = vectorizer.fit_transform(a)
        b = vectorizer.transform(b)
        
        return a, b
    
    
    def _get_index_ntop_sim(self, csr_row):
        nnz = csr_row.getnnz()
        if nnz == 0:
            return [(-1 , -1)]
        elif nnz <= self.n_top:
            result = zip(csr_row.indices, csr_row.data)
        else:
            arg_idx = np.argpartition(csr_row.data, -self.n_top)[-self.n_top:]
            result = zip(csr_row.indices[arg_idx], csr_row.data[arg_idx])
        
        return sorted(result, key=lambda x: -x[1])
    
    
    def _get_cosine_sim(self, X, y):
        c = X.dot(y.transpose())
        return [self._get_index_ntop_sim(row) for row in c]
    
    
    def _batches(self, X, batch_size):
        """
        Create n batch from a of size batch_size
        :param a: Dataframe to partition
        :param batch_size: Size of the batches
        :type a: pd.DataFrame
        :type batch_size: int
        :return: list of indexes for batch i
        """
        x_lenght = len(X)
        for i in range(0, x_lenght, batch_size):
            yield X[i:i+batch_size]
    
    
    def _dictionary_index(self, old, new):
        """
        Map old values with correspondant new values
        :param old: List of old values
        :param new: List of new values
        :type old: list
        :type new: list
        :return: dictionnary of mapping
        """
        d = dict()
        for o, n in zip(old, new):
            d[n] = o
        return d
    
    
    def _map_dict(self, d, l):
        """
        Map a list l of values with correspondant values in the dictionnary d
        :param d: Dictionnary of value with correspondant old values
        :param l: List of new values to map with old values
        :type d: dict
        :type l: list
        :return: list of correspondant values
        """
        return [d.get(i, i) for i in l]
    
    
    def fit(self, X, y):
        pass
    
    
    def _compute_cosine(self, X, y):
        """Compute cosine similarity between query dataframe a and population dataframe b.
        Filter the population dataframe on n_column_merge in order to reduce the search.
        Create batches of query dataframe of size batch_size.
        Compute cosine similarity between each batch size and the filtered population dataframe b.
        
        Parameters
        ----------
            a: pd.DataFrame
                Query dataframe
            b: pd.DataFrame
                Population dataframe where to look for similar values
            n_column: str
                Name of the column where to look for similar values
            n_column_merge: str
                Name of the column to merge the 2 dataframe
            ntop: int
                Number of most similar value to find
            batch_size: int
                Number of element of dataframe a to compute at the same time
        
        Returns
        -------
        pd.DataFrame
            Dataframe a with its most similar value in b 
        """
        
        x_batches = self._batches(X.index.tolist(), self.batch_size)
        
        dict_c = {
            'index_batch': X.index.tolist(),
            'index_max': list(),
            'score': list()
        }
        
        for x_batch in tqdm(x_batches):
            # batch with columns and value
            x_temp = X.loc[x_batch]
            y_temp = y.copy()
            
            # Create dictionnary of reference between real indexes of b (old) and artificial indexes (new)
            old_indexes = y_temp.index.tolist()
            y_temp = y_temp.reset_index().drop("index", axis=1)
            new_indexes = y_temp.index.tolist()
            dict_indexes = self._dictionary_index(old_indexes, new_indexes)
            
            if y_temp.shape[0] > 0:
                x_array, y_array = self._vectorizer(x_temp.values.ravel(), y_temp.values.ravel())
                c = self._get_cosine_sim(x_array, y_array)
                
                for i in c:
                    dict_c['index_max'].append(self._map_dict(dict_indexes, [j[0] for j in i]))
                    dict_c['score'].append([j[1] for j in i])
        
        df_results = pd.DataFrame.from_dict(dict_c).set_index(['index_batch']).apply(pd.Series.explode).reset_index()
        X = X.merge(df_results, how='left', left_index=True, right_on='index_batch').drop(['index_batch'], axis=1)
        X = X.merge(y, how='left', right_index=True, left_on='index_max')
        return X
    
    
    def fit_transform(self, X, y, **fit_params):
        X = self._check_format(X)
        y = self._check_format(y)
        
        self.fit(X, y)
        
        if self.n_jobs == 0:
            return self._compute_cosine(X, y)
        n_cores = multiprocessing.cpu_count() if self.n_jobs == -1 else self.n_jobs
        X = np.array_split(X, n_cores)
        pool = multiprocessing.Pool(n_cores)
        return pd.concat(pool.map(partial(self._compute_cosine, y=y), X))
