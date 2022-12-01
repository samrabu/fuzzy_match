
import sys

import os
import time
import numpy as np
from fuzzywuzzy import fuzz
import pandas as pd
from . import utils_fuzzy as u
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from tqdm import tqdm
# import multiprocessing
from ray.util import multiprocessing
from functools import partial
import warnings

warnings.filterwarnings("ignore")
tqdm.pandas()


def create_matrix(arr_1, arr_2):
    """
    Fit sklearn TfidfVectorizer on arr_1 and transform arr_1 and arr_2 in 2 documents*terms matrix

    :param arr_1: query array of string to fit the TFIDF and to transform in documents*terms matrix
    :param arr_2: population array of string to transform in documents*terms matrix

    :type arr_1: array-like
    :type arr_2: array-like

    :return: 2 array-like, shape(n_samples, n_ngrams)
    """
    vectorizer = TfidfVectorizer(analyzer=u.word2ngrams)
    a = vectorizer.fit_transform(arr_1)
    b = vectorizer.transform(arr_2)
    
    return a, b


def set_threshold(a, threshold):
    a['threshold'] = np.where(a['Similarity'] >= threshold, 1, 0)


def output_cosine(a, b, merge_on, suffixes, threshold, common_values=None):
    a = a.merge(b, how=('left'), left_index=False, right_index=True, left_on=merge_on, suffixes=suffixes)
    col_to_drop =[col for col in a.columns if "_clean" in col.lower()]
    col_to_drop.extend([merge_on, 'col_fuzzy_match'+ suffixes[0], 'col_fuzzy_match'+suffixes[1], "index_batch"])
    a = a.drop(col_to_drop, axis=1)
    a["similarity"] = a["%_similarity"].map(lambda x: "Complète" if x > 0.99 else "Relative : {}".format(x) if x > threshold and x <= 0.99 else "Non")
    if common_values is not None:
        common_values = common_values[a.columns]
        a = pd.concat([a, common_values], axis=0)
    a = a.sort_index()
    return a


def ratio(str_a, str_b):
    if str_a == 'nan' or str_b == 'nan':
        return np.nan
    a_array, b_array = create_matrix([str_a], [str_b])
    c = a_array.dot(b_array.transpose()).data
    
    if len(c) == 0:
        return 0.0
    return c[0]


def ratio_fuzz(str_a: str, str_b: str):
    if str_a == 'nan' or str_b == 'nan':
        return np.nan
    return fuzz.ratio(str_a.lower().strip().replace(" ", ""), str_b.lower().strip().replace(" ", "")) / 100


def _get_csr_matrix_ntop_idx_data(csr_row, ntop):
    nnz = csr_row.getnnz()
    if nnz == 0:
        return [(-1 , -1)]
        # return None
    elif nnz <= ntop:
        result = zip(csr_row.indices, csr_row.data)
    else:
        arg_idx = np.argpartition(csr_row.data, -ntop)[-ntop:]
        result = zip(csr_row.indices[arg_idx], csr_row.data[arg_idx])
    
    return sorted(result, key=lambda x: -x[1])


def _cos_similarity_top(a, b, ntop, lower_bound=0):
    c = a.dot(b.transpose())
    return [_get_csr_matrix_ntop_idx_data(row, ntop) for row in c]


def _batches(a, batch_size):
    """
    Create n batch from a of size batch_size
    :param a: Dataframe to partition
    :param batch_size: Size of the batches
    :type a: pd.DataFrame
    :type batch_size: int
    :return: list of indexes for batch i
    """
    a_lenght = len(a)
    for i in range(0, a_lenght, batch_size):
        yield a[i:i+batch_size]


def _dictionary_index(old, new):
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


def _map_dict(d, l):
    """
    Map a list l of values with correspondant values in the dictionnary d
    :param d: Dictionnary of value with correspondant old values
    :param l: List of new values to map with old values
    :type d: dict
    :type l: list
    :return: list of correspondant values
    """
    return [d.get(i, i) for i in l]


def common(a: pd.DataFrame, b: pd.DataFrame, n_column: str) -> pd.DataFrame:
    a = a.reset_index()
    a["index_a"] = a["index"]
    a = a.drop("index", axis=1)
    
    common_values = a.merge(b, how='inner', on=n_column, suffixes=('_query', '_population'))
    a_index_common = common_values["index_a"].drop_duplicates()
    
    a = a.drop(a_index_common)
    a = a.drop("index_a", axis=1)
    
    common_values.index = common_values['index_a']
    common_values = common_values.drop("index_a", axis=1)
    common_values["%_similarity"] = 1
    common_values["similarity"] = "full"
    # b = b.drop(b_index_common)
    return a, common_values


def compute_cosine(a, b, n_column, n_column_merge, ntop=1, batch_size=1000, suffixes_multiple=""):
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
    
    # Filter b, keep common value between a & b on column n_column_merge
    if n_column_merge is not None:
        b = b.loc[b[n_column_merge].isin(a[n_column_merge])]
         
    a_batches = _batches(a.index.tolist(), batch_size)
    
    dict_c = {'index_batch': a.index.tolist(),
              'index_max': list(),
              '% - Similarity': list()}
    
    for a_batch in tqdm(a_batches):
        # batch with columns and value
        a_temp = a.loc[a_batch]
        
        # Filter b, keep common value between batch a & b on column n_column_merge
        if n_column_merge is not None:
            b_temp = b.loc[b[n_column_merge].isin(a_temp[n_column_merge])]
        else:
            b_temp = b.copy()
        
        # Create dictionnary of reference between real indexes of b (old) and artificial indexes (new)
        old_indexes = b_temp.index.tolist()
        b_temp = b_temp.reset_index().drop("index", axis=1)
        new_indexes = b_temp.index.tolist()
        dict_indexes = _dictionary_index(old_indexes, new_indexes)
        
        if b_temp.shape[0] > 0:
            a_array, b_array = create_matrix(a_temp[n_column], b_temp[n_column])
            c = _cos_similarity_top(a_array, b_array, ntop)
            
            for i in c:
                dict_c['index_max'].append(_map_dict(dict_indexes, [j[0] for j in i]))
                dict_c['%_similarity'].append([j[1] for j in i])
    
    df_results = pd.DataFrame.from_dict(dict_c).set_index(['index_batch']).apply(pd.Series.explode).reset_index()
    a = a.merge(df_results, how='left', left_index=True, right_on='index_batch')
    return a


def apply_cosine_multi_process(df_query, df_population, n_column, n_column_merge, ntop=1, batch_size=1000, threshold=0.8, suffixes_multiple=""):
    # df_query, common_values = common(df_query, df_population, n_column)
    out = compute_cosine(a=df_query, 
                         b=df_population, 
                         n_column=n_column, 
                         n_column_merge=n_column_merge, 
                         ntop=ntop, 
                         batch_size=batch_size)
    
    out = output_cosine(out, df_population, 
                        merge_on='index_max', 
                        suffixes=('_query', '_population'), 
                        threshold=threshold, 
                        common_values=None).sort_values(by=['%_similarity'], 
                                                                 ascending=False)
    return out


def main(df_query: str,
         df_population: str,
         query_columns_to_clean: str,
         population_columns_to_clean: str,
         n_column_merge_query: str = None,
         n_column_merge_population: str = None,
         n_top: int = 1,
         batch_size: int = 1000,
         threshold: float = 0.8,
         return_df_only: bool = False,
         path_file_out: str = "result.csv",
         sep: str = ";",
         n_jobs=1) -> pd.DataFrame:
    """
    Execute preprocessing and compute cosine similarity between 2 files
    
    Parameters
    ----------
        df_query: str or pd.DataFrame
            path to query file or query DataFrame
        df_population: or pd.DataFrame
            path to population file or population DataFrame
        query_columns_to_clean: list or str
            Column(s) to clean to concatenate and to use for fuzzy matching. If list then concatenation of all the columns and cleaning.
        population_columns_to_clean: list or str
            Column(s) to clean to concatenate and to use for fuzzy matching. If list then concatenation of all the columns and cleaning.
        n_column_merge_query: str 
            OPTIONAL. Name of the column to use in query dataframe for filter during cosine computation in order to reduce the computational time.
        n_column_merge_population: str 
            OPTIONAL. Name of the column to use in population dataframe for filter during cosine computation in order to reduce the computational time.
        n_top: int
            Number of most similar values to retrieve
        batch_size: int
            Number of element of dataframe a to compute at the same time
        threshold: float
            Minimal percentage of similarity to take into consideration  
        path_file_out: str
        
    Returns:
    -------
        pd.DataFrame
            Results
    """
    if (n_column_merge_population is None and n_column_merge_query is not None) or (n_column_merge_population is not None and n_column_merge_query is None):
        raise Exception("You must fill both n_column_merge arguments")
    
    print("Load data..")
    if isinstance(df_query, str):
        df_query = u.load_file(df_query)
    elif not isinstance(df_query, pd.DataFrame):
        raise Exception("Query input must be a string or a Pandas DataFrame")
    
    if isinstance(df_population, str):
        df_population = u.load_file(df_population).drop_duplicates()
    elif not isinstance(df_population, pd.DataFrame):
        raise Exception("Population input must be a string or a Pandas DataFrame")
    
    n_column = "col_fuzzy_match" # name of the column used for fuzzy match
    n_column_merge = None
    
    print("Preprocessing..")
    if (n_column_merge_query is not None and n_column_merge_query != '') and (n_column_merge_population is not None and n_column_merge_population != ''):
        
        n_column_merge ="n_column_merge_clean"
        df_population = u.preprocessing(df_population, n_column_merge_population, n_column_merge)
        df_query = u.preprocessing(df_query, n_column_merge_query, n_column_merge)
        
        if isinstance(population_columns_to_clean, list):
            population_columns_to_clean.extend([n_column_merge])
        else:
            population_columns_to_clean = [population_columns_to_clean, n_column_merge]
        
        if isinstance(query_columns_to_clean, list):
            query_columns_to_clean.extend([n_column_merge])
        else:
            query_columns_to_clean = [query_columns_to_clean, n_column_merge]
        
    df_population = u.preprocessing(df_population, population_columns_to_clean, n_column)
    df_query = u.preprocessing(df_query, query_columns_to_clean, n_column)
    
    if n_jobs == 1:
        print("Common values..")
        df_query, common_values = common(df_query, df_population, n_column)
        
        print("Compute similarity..")
        out = compute_cosine(a=df_query, 
                            b=df_population, 
                            n_column=n_column, 
                            n_column_merge=n_column_merge, 
                            ntop=n_top, 
                            batch_size=batch_size)
        
        print("Output..")
        out = output_cosine(out, df_population, 
                            merge_on='index_max', 
                            suffixes=('_query', '_population'), 
                            threshold=threshold, 
                            common_values=common_values)
    
    else:
        n_jobs = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
        print(n_jobs)
        df_query = np.array_split(df_query, n_jobs)
        pool = multiprocessing.Pool(n_jobs)
        out = pool.map(partial(apply_cosine_multi_process,
                                            df_population=df_population,
                                            n_column=n_column,
                                            n_column_merge=n_column_merge,
                                            ntop=n_top,
                                            batch_size=batch_size,
                                            threshold=threshold),
                                    df_query)
        out = pd.concat(out, axis=0)
    
    if return_df_only:
        return out
    else:
        out.to_csv(path_file_out, sep=';', encoding='utf-8-sig', index=False)
        print("Results exported.")
        return out
