# lab.py


import pandas as pd
import numpy as np
import os
import re


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def match_1(string):
    """
    DO NOT EDIT THE DOCSTRING!
    >>> match_1("abcde]")
    False
    >>> match_1("ab[cde")
    False
    >>> match_1("a[cd]")
    False
    >>> match_1("ab[cd]")
    True
    >>> match_1("1ab[cd]")
    False
    >>> match_1("ab[cd]ef")
    True
    >>> match_1("1b[#d] _")
    True
    """
    pattern = r'^..\[..\]'

    # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_2(string):
    """
    DO NOT EDIT THE DOCSTRING!
    >>> match_2("(123) 456-7890")
    False
    >>> match_2("858-456-7890")
    False
    >>> match_2("(858)45-7890")
    False
    >>> match_2("(858) 456-7890")
    True
    >>> match_2("(858)456-789")
    False
    >>> match_2("(858)456-7890")
    False
    >>> match_2("a(858) 456-7890")
    False
    >>> match_2("(858) 456-7890b")
    False
    """
    pattern = r'^\(858\) \d{3}-\d{4}$'

    # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_3(string):
    """
    DO NOT EDIT THE DOCSTRING!
    >>> match_3("qwertsd?")
    True
    >>> match_3("qw?ertsd?")
    True
    >>> match_3("ab c?")
    False
    >>> match_3("ab   c ?")
    True
    >>> match_3(" asdfqwes ?")
    False
    >>> match_3(" adfqwes ?")
    True
    >>> match_3(" adf!qes ?")
    False
    >>> match_3(" adf!qe? ")
    False
    """
    pattern = r'^[A-Za-z0-9\s?]{5,9}\?$'

    # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_4(string):
    """
    DO NOT EDIT THE DOCSTRING!
    >>> match_4("$$AaaaaBbbbc")
    True
    >>> match_4("$!@#$aABc")
    True
    >>> match_4("$a$aABc")
    False
    >>> match_4("$iiuABc")
    False
    >>> match_4("123$$$Abc")
    False
    >>> match_4("$$Abc")
    True
    >>> match_4("$qw345t$AAAc")
    False
    >>> match_4("$s$Bca")
    False
    >>> match_4("$!@$")
    False
    """
    pattern = r'^\$[^abc$]*\$(?:[aA]+)(?:[bB]+)(?:[cC]+)$'

    # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_5(string):
    """
    DO NOT EDIT THE DOCSTRING!
    >>> match_5("dsc80.py")
    True
    >>> match_5("dsc80py")
    False
    >>> match_5("dsc80..py")
    False
    >>> match_5("dsc80+.py")
    False
    """
    pattern = r'^[A-Za-z0-9_]+\.py$'

    # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_6(string):
    """
    DO NOT EDIT THE DOCSTRING!
    >>> match_6("aab_cbb_bc")
    False
    >>> match_6("aab_cbbbc")
    True
    >>> match_6("aab_Abbbc")
    False
    >>> match_6("abcdef")
    False
    >>> match_6("ABCDEF_ABCD")
    False
    """
    pattern = r'^[a-z]+_[a-z]+$'

    # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_7(string):
    """
    DO NOT EDIT THE DOCSTRING!
    >>> match_7("_abc_")
    True
    >>> match_7("abd")
    False
    >>> match_7("bcd")
    False
    >>> match_7("_ncde")
    False
    """
    pattern = r'^_.*_$'

    # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None



def match_8(string):
    """
    DO NOT EDIT THE DOCSTRING!
    >>> match_8("ASJDKLFK10ASDO")
    False
    >>> match_8("ASJDKLFK0ASDo!!!!!!! !!!!!!!!!")
    True
    >>> match_8("JKLSDNM01IDKSL")
    False
    >>> match_8("ASDKJLdsi0SKLl")
    False
    >>> match_8("ASDJKL9380JKAL")
    True
    """
    pattern = r'^[^Oi1]+$'

    # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None



def match_9(string):
    '''
    DO NOT EDIT THE DOCSTRING!
    >>> match_9('NY-32-NYC-1232')
    True
    >>> match_9('ca-23-SAN-1231')
    False
    >>> match_9('MA-36-BOS-5465')
    False
    >>> match_9('CA-56-LAX-7895')
    True
    >>> match_9('NY-32-LAX-0000') # If the state is NY, the city can be any 3 letter code, including LAX or SAN!
    True
    >>> match_9('TX-32-SAN-4491')
    False
    '''
    pattern = r'^(?:NY-\d{2}-[A-Z]{3}-\d{4}|CA-\d{2}-(?:SAN|LAX)-\d{4})$'

    # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_10(string):
    '''
    DO NOT EDIT THE DOCSTRING!
    >>> match_10('ABCdef')
    ['bcd']
    >>> match_10(' DEFaabc !g ')
    ['def', 'bcg']
    >>> match_10('Come ti chiami?')
    ['com', 'eti', 'chi']
    >>> match_10('and')
    []
    >>> match_10('Ab..DEF')
    ['bde']
    
    '''
    s = string.lower()
    s = re.sub(r'[^a-z0-9]|a', '', s)
    return re.findall(r'.{3}', s)


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def extract_personal(s: str):
    email_pat = re.compile(r'\b([A-Za-z0-9]+)@([A-Za-z0-9]+)\.([A-Za-z]{2,})\b')
    emails = ['@'.join([u, f'{d}.{t}']) for (u, d, t) in email_pat.findall(s)]

    # 2) SSNs
    ssn_pat = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
    ssn = ssn_pat.findall(s)

    btc_pat = re.compile(r'\b[A-Za-z0-9]{26,}\b')
    bitcoin = btc_pat.findall(s)

    street_types = r'(?:street|st\.?|avenue|ave\.?|road|rd\.?|boulevard|blvd\.?|drive|dr\.?|court|ct\.?|lane|ln\.?|way)'
    addr_pat = re.compile(
        rf'\b(\d{{1,5}})\s+([A-Za-z]+(?:\s+[A-Za-z]+)*)\s+({street_types})\b',
        flags=re.IGNORECASE
    )
    addresses = ['{} {} {}'.format(num, name, stype) for (num, name, stype) in addr_pat.findall(s)]

    return (emails, ssn, bitcoin, addresses)


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def tfidf_data(reviews_ser, review):
    tokens = re.findall(r'\b\w+\b', review.lower())
    total = len(tokens)

    counts = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    N = len(reviews_ser)
    idfs = {}
    for w in counts.keys():
        pat = re.compile(rf'\b{re.escape(w)}\b', flags=re.IGNORECASE)
        df = int(np.sum(reviews_ser.apply(lambda s: bool(pat.search(s)))))
        idfs[w] = np.log(N / df)
    out = pd.DataFrame({
        'cnt': pd.Series(counts),
    })
    out['tf'] = out['cnt'] / total
    out['idf'] = pd.Series(idfs)
    out['tfidf'] = out['tf'] * out['idf']
    return out


def relevant_word(out: pd.DataFrame):
    tfidf = pd.to_numeric(out['tfidf'], errors='coerce').dropna()
    return tfidf.idxmax()




# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def hashtag_list(tweet_text):
    return tweet_text.apply(
        lambda txt: re.findall(r'#[^\s#]+', str(txt)) if pd.notna(txt) else []
    ).apply(lambda lst: [h[1:] for h in lst])


def most_common_hashtag(tweet_lists):
    freqs = {}
    for tags in tweet_lists:
        for tag in tags:
            freqs[tag] = freqs.get(tag, 0) + 1

    def pick(tags):
        if len(tags) == 0:
            return np.nan
        if len(tags) == 1:
            return tags[0]
        return max(tags, key=lambda t: freqs[t])

    return tweet_lists.apply(pick)


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------




    


def create_features(ira: pd.DataFrame):
    s = ira['text'].astype(str)

    hashtag_pat = r'#[^\s#]+'
    tag_pat = r'@[A-Za-z0-9]+'
    link_pat = r'https?://\S+'
    rt_start = r'^\s*RT\b'

    num_hashtags = s.str.findall(hashtag_pat).apply(len)
    num_tags     = s.str.findall(tag_pat).apply(len)
    num_links    = s.str.findall(link_pat).apply(len)
    is_retweet   = s.str.match(rt_start)

    tags_series  = hashtag_list(s)
    mc_hashtags  = most_common_hashtag(tags_series)

    cleaned = s.str.replace(link_pat, ' ', regex=True) \
               .str.replace(hashtag_pat, ' ', regex=True) \
               .str.replace(tag_pat, ' ', regex=True) \
               .str.replace(rt_start, ' ', regex=True) \
               .str.replace(r'[^A-Za-z0-9 ]+', ' ', regex=True) \
               .str.lower() \
               .str.split().str.join(' ')
    out = pd.DataFrame({
        'text'         : cleaned,
        'num_hashtags' : num_hashtags,
        'mc_hashtags'  : mc_hashtags,
        'num_tags'     : num_tags,
        'num_links'    : num_links,
        'is_retweet'   : is_retweet
    }, index=ira.index)

    return out
