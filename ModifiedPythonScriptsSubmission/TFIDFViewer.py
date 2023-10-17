#3 Code completion subtask here: Computing tf-idf's and cosine similarity
"""
.. module:: TFIDFViewer

TFIDFViewer
******

:Description: TFIDFViewer

    Receives two paths of files to compare (the paths have to be the ones used when indexing the files)

:Authors:
    bejar

:Version: 

:Date:  05/07/2017
"""

from elasticsearch import Elasticsearch
from elasticsearch.exceptions import NotFoundError
from elasticsearch.client import CatClient
from elasticsearch_dsl import Search
from elasticsearch_dsl.query import Q

import argparse

import numpy as np

__author__ = 'bejar'

def search_file_by_path(client, index, path):
    """
    Search for a file using its path

    :param path:
    :return:
    """
    s = Search(using=client, index=index)
    q = Q('match', path=path)  # exact search in the path field
    s = s.query(q)
    result = s.execute()

    lfiles = [r for r in result]
    if len(lfiles) == 0:
        raise NameError(f'File [{path}] not found')
    else:
        return lfiles[0].meta.id


def document_term_vector(client, index, id):
    """
    Returns the term vector of a document and its statistics a two sorted list of pairs (word, count)
    The first one is the frequency of the term in the document, the second one is the number of documents
    that contain the term

    :param client:
    :param index:
    :param id:
    :return:
    """
    termvector = client.termvectors(index=index, id=id, fields=['text'],
                                    positions=False, term_statistics=True)

    file_td = {}
    file_df = {}

    if 'text' in termvector['term_vectors']:
        for t in termvector['term_vectors']['text']['terms']:
            file_td[t] = termvector['term_vectors']['text']['terms'][t]['term_freq']
            file_df[t] = termvector['term_vectors']['text']['terms'][t]['doc_freq']
    return sorted(file_td.items()), sorted(file_df.items())

############### Changes below ###############

def toTFIDF(client, index, file_id):
    """
    Returns the term weights of a document.

    Compute the TF-IDF weights for terms in a document. Start by obtaining the term verctor and the
    Document frequency for a specific document from an index in some database or search client.
    There after the max frequency is calculated from the term vector. This is used to normalize the
    Term frequency later on. The total number of documents is calcualted later and safed in dcount.
    This is used for the IDF computation. There after the TF-IDF weights for each term in the document
    are calculated and finally normalized.

    :param client: The client to communicate with the data source.
    :param index: The index where the documents are stored.
    :param file_id: The ID of the file for which we need the TF-IDF.
    :return: The normalized term weight vector.
    """

    file_tv, file_df = document_term_vector(client, index, file_id)

    max_freq = max([f for _, f in file_tv])

    dcount = doc_count(client, index)

    tfidfw = []

    for (term, weight), (_, df) in zip(file_tv, file_df):
        ############### Our code implementation ###############
        idf = np.log(dcount / df)
        tf = weight / max_freq
        tfidf = tf * idf
        tfidfw.append((term, tfidf))
        ########################################################

    return normalize(tfidfw)



def print_term_weigth_vector(twv):
    """
    Prints the term vector and the corresponding weights

    This is just a basic print function.

    :param twv: The list that is given with terms and weights.
    :return: Nothing
    """
    ############### Our code implementation ###############
    for term, weight in twv:
        print(f"{term}, {weight}")
    ########################################################
    pass


def normalize(tw):
    """
    Normalizes the weights in t so that they form a unit-length vector
    It is assumed that not all weights are 0. First we convert the input list into
    a numpy array for more efficient calculations. Then the relevant weights are
    extracted. These are then used for the normalization. This norm shall not be 0.
    Once the normalized weights have be calculated the list is recreated and returned.
    :param tw: This shall be the list of weight vectors to be normalited
    :return: The normalized list of weights
    """
    ############### Our code implementation ###############
    tw = np.array(tw)
    weights = np.array([weight for name, weight in tw], dtype=float)
    norm = np.linalg.norm(weights)
    if norm == 0:
        raise ValueError("Norm is zero, cannot divide!")
    normalized_weights = weights / norm
    result = [(name, weight) for (name, _), weight in zip(tw, normalized_weights)]

    return result
    ########################################################


def cosine_similarity(tw1, tw2):
    """
        Computes the cosine similarity between two weight vectors, terms are alphabetically ordered.
        This is close to as efficently as possible in python.
        First we initialize the relevant terms to 0. THrought the function the lists are iteratet throughout. If a term
        can be found in both lists their weights are used to update the dot product and norms. If a term is only in one
        of the lists we only update the norm of that one list. There after the remaining terms are processed if
        necessary. Lastly the norms are squared and finally the cosine similarity is computed as the ratio of the dot
        product to the product of the magnitudes of the two vectors. If either of the norms is zero and the
        corresponding vector is a zero vector then the cosine similiarity is defined as 0 to avoid division by zero.
        :param tw1: First weight vector, sorted by term.
        :param tw2: Second weight vector, sorted by term.
        :return: Cosine similarity between the two vectors.
    """
    ############### Our code implementation ###############
    #initalize to zero
    dotproduct, tw1norm, tw2norm, i, j = 0, 0, 0 ,0 ,0
    while i < len(tw1) and j < len(tw2):
        term1, weight1 = tw1[i]
        term2, weight2 = tw2[j]
        # If terms match, add to dot product and move both lists one step forward
        if term1 == term2:
            dotproduct += weight1 * weight2
            tw1norm += weight1 ** 2
            tw2norm += weight2 ** 2
            j += 1
            i += 1
        # If term1 comes before term2 alphabetically, only move list one forward
        elif term1 < term2:
            tw1norm += weight1 ** 2
            i += 1
        # Otherwise, only move list 2 forward
        else:
            tw2norm += weight2 ** 2
            j += 1
    # Process any remaining terms in the lists
    while i < len(tw1):
        _, weight1 = tw1[i]
        tw1norm += weight1 ** 2
        i += 1
    while j < len(tw2):
        _, weight2 = tw2[j]
        tw2norm += weight2 ** 2
        j += 1
    tw1norm = np.sqrt(tw1norm)
    tw2norm = np.sqrt(tw2norm)
    if tw1norm == 0 or tw2norm == 0:
        return 0

    return dotproduct / (tw1norm * tw2norm)
    ########################################################

#################### Changes above ####################

def doc_count(client, index):
    """
    Returns the number of documents in an index

    :param client:
    :param index:
    :return:
    """
    return int(CatClient(client).count(index=[index], format='json')[0]['count'])



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', default=None, required=True, help='Index to search')
    parser.add_argument('--files', default=None, required=True, nargs=2, help='Paths of the files to compare')
    parser.add_argument('--print', default=False, action='store_true', help='Print TFIDF vectors')

    args = parser.parse_args()


    index = args.index

    file1 = args.files[0]
    file2 = args.files[1]

    client = Elasticsearch(timeout=1000)

    try:

        # Get the files ids
        file1_id = search_file_by_path(client, index, file1)
        file2_id = search_file_by_path(client, index, file2)

        # Compute the TF-IDF vectors
        file1_tw = toTFIDF(client, index, file1_id)
        file2_tw = toTFIDF(client, index, file2_id)

        if args.print:
            print(f'TFIDF FILE {file1}')
            print_term_weigth_vector(file1_tw)
            print ('---------------------')
            print(f'TFIDF FILE {file2}')
            print_term_weigth_vector(file2_tw)
            print ('---------------------')

        print(f"Similarity = {cosine_similarity(file1_tw, file2_tw):3.5f}")

    except NotFoundError:
        print(f'Index {index} does not exists')

