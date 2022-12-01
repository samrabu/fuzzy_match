
import sys

import os
import re
import csv
import time
import unidecode
import numpy as np
import pandas as pd
from collections import Counter
from openpyxl.utils.cell import get_column_letter

import warnings
warnings.filterwarnings("ignore")

def load_sample_file(file: str, sep=";", frac=None, n=None) -> pd.DataFrame:
    """Import excel or csv file into pd.DataFrame
    
    Parameters
    ----------
        file : str
        sep : str
        frac : int, optional
        n : int, optional
        
    Returns:
    --------
        pd.DataFrame
    """
    _, file_extension = os.path.splitext(file)
    if file_extension == ".xlsx":
        df = pd.read_excel(file, dtype=object)
    elif file_extension in [".csv", ".txt"]:
        df = pd.read_csv(file, sep=sep, encoding='utf-16', dtype=object)
    else:
        raise ValueError("Only Excel, CSV and TXT file")
    
    if frac is not None and n is None:
        return df.sample(frac=frac)
    elif frac is None and n is not None:
        return df.sample(n=n)
    return df

class DefaultDialect(csv.Dialect):
    def __init__(self):
        self.delimiter = ";"
        self.quotechar = '"'
        self.escapechar = '\\'
        self.doublequote = False
        self.skipinitialspace = True
        self.quoting = csv.QUOTE_MINIMAL
#        self.lineterminator = '\r\n'

def t_detect_dialect(file_name):
        # delimiters detection
    dialect = DefaultDialect()
    encoding = detect_encoding(file_name)
    with open(file_name,'r', encoding = encoding) as csvfile:
        csvfile.seek(0)
        dial = False
        
        data = csvfile.readline()
        counter = Counter(data)
        
        nb1 = counter[';']/len(data)*100
        nb2 = counter[chr(10)]/len(data)*100
        nb3 = counter['|']/len(data)*100                      
        if nb1 > 5:
            dialect.delimiters = ";"
            return dialect
        else:
            if nb2 > 5:
                dialect.delimiters =chr(10)
                return dialect
            else:
                if nb3 > 5:
                    dialect.delimiters = '|'
                    return dialect
        
        
        csvfile.seek(0)
        for bb in [64,128,256,512,1024,2046,4092]:
            if dial == False:
                try:
                    dialect = csv.Sniffer().sniff(csvfile.read(bb))
                    dial = True
                except:
                    dial = False
                    pass
            else:
                break
        if dial == False:
            # print('default dialect')
            dialect = DefaultDialect()
    return dialect 


def load_file(file_path, sheet=0, chunk_size=None, usecols=None, nrows=None, sep=None):
    _, file_extension = os.path.splitext(file_path)
    encoding = detect_encoding(file_path)
    dialect = t_detect_dialect(file_path)
    if file_extension in ['.csv', '.txt']:
        df = pd.read_csv(file_path,
                         sep=sep,
                         dialect=dialect,
                         encoding=encoding,
                         chunksize=chunk_size,
                         infer_datetime_format=True,
                         usecols=usecols,
                         nrows=nrows,
                         error_bad_lines=False)
    elif file_extension in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path,
                           engine='openpyxl',
                           sheet_name=sheet)
    else:
        Exception("FORMAT NOT ACCEPTED")
        df = pd.DataFrame()
    return df


def detect_encoding(file_name, txt_key=''):
    '''
    detect encoding 
    
    TC
    
    '''
    
    encodings = [
        'cp1252',
        'latin_1',
        'ascii',
        'utf_8',
        'utf_8_sig',
        'utf-8',
        'big5',
        'big5hkscs',
        'cp037',
        'cp1006',
        'cp1026',
        'cp1140',
        'cp1250',
        'cp1251',
        'cp1252',
        'cp1253',
        'cp1254',
        'cp1255',
        'cp1256',
        'cp1257',
        'cp1258',
        'cp424',
        'cp437',
        'cp500',
        'cp737',
        'cp775',
        'cp850',
        'cp852',
        'cp855',
        'cp856',
        'cp857',
        'cp860',
        'cp861',
        'cp862',
        'cp863',
        'cp864',
        'cp865',
        'cp866',
        'cp869',
        'cp874',
        'cp875',
        'cp932',
        'cp949',
        'cp950',
        'euc_jis_2004',
        'euc_jisx0213',
        'euc_jp',
        'euc_kr',
        'gb18030',
        'gb2312',
        'gbk',
        'hz',
        'iso2022_jp',
        'iso2022_jp_1',
        'iso2022_jp_2',
        'iso2022_jp_2004',
        'iso2022_jp_3',
        'iso2022_jp_ext',
        'iso2022_kr',
        'iso8859_10',
        'iso8859_13',
        'iso8859_14',
        'iso8859_15',
        'iso8859_2',
        'iso8859_3',
        'iso8859_4',
        'iso8859_5',
        'iso8859_6',
        'iso8859_7',
        'iso8859_8',
        'iso8859_9',
        'johab',
        'koi8_r',
        'koi8_u',
        'mac_cyrillic',
        'mac_greek',
        'mac_iceland',
        'mac_latin2',
        'mac_roman',
        'mac_turkish',
        'ptcp154',
        'shift_jis',
        'shift_jis_2004',
        'shift_jisx0213',
        'utf_16',
        'utf_16_be',
        'utf_16_le',
        'utf_7']
    for e in encodings:
        try:
            fh = open(file_name, 'r', encoding=e)
            
            l = fh.readline()
            
            if len(l) > 0:
                if (ord(l[0]) < 128 or \
                    (ord(l[0]) >= 128 and ord(l[0]) < 239 and ord(l[1]) < 128)):
                    fh.close()
                    return e
                fh.close()
            else:
                raise Exception('Empty file')

        except UnicodeError:
            fh.close()
            pass
        else:
            fh.close()


def rules(text):
    text = text.lower()
    text = unidecode.unidecode(text)
    # text = re.sub(r'[^A-Za-z0-9]+', ' ', text)
    # text = re.sub(r'(monsieur\s*|madame\s*|mr\s+|mme\s+|m\s+|mlle\s+)', '', text)
    # text = re.sub(r'(nan\s+)', '', text)
    # text = re.sub(r'\s*', '', text)
    text = text.strip()
    # text = re.sub(r'^\s+$', '', text)
    return text

def preprocessing(df, l_columns, n_column):
    """
    Cleans up data by deleting punctuation, space and some words

    :param df: Dataframe
    :param n_column: Name of the column to clean or list of columns to clean
    :param l_columns: Optional, list of columns to combine

    :type df: pd.DataFrame
    :type n_column: str
    :type l_columns: None, str, list
    
    :return: pd.DataFrame 
    """
    print("Preprocessing : \n")
    if isinstance(l_columns, list):
        # print(l_columns)
        df[n_column] = df[l_columns].astype(str).apply(' '.join, axis=1)
    else:
        df[n_column] = df[l_columns].astype(str).replace(".0", '', regex=False)
        
    # apply cleaning on the new columns
    df[n_column] = df[n_column].astype(str).apply(rules)
    df = df.replace(r"^\s+$", 'nan', regex=True)
    # df = df.loc[df[n_column] != 'nan'].reset_index(drop=True)
    return df

def word2ngrams(text, n=3):
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
