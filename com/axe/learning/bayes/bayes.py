# -*- coding:utf-8 -*-
import random
from numpy import *
import feedparser


def loadDataSet():
    posting_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return posting_list, class_vec


def createVocabList(data_set):
    vocab_set = set([])
    for document in data_set:
        vocab_set = vocab_set | set(document)
    return list(vocab_set)


def setOfWord2Vec(vocab_list, input_set):
    ret_vec = [0]*len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            ret_vec[vocab_list.index(word)] = 1
        # else:
            # print 'the word: %s is not in my vocabulary!' % word
    return ret_vec


def bagOfWords2Vec(vocab_list, input_set):
    ret_vec = [0]*len(vocab_list)
    for word in input_set:
        ret_vec[vocab_list.index(word)] += 1
    return ret_vec


def trainNB0(train_matrix, train_category):
    train_docs_num = len(train_matrix)
    word_nums = len(train_matrix[0])
    p_abusive = sum(train_category)/float(train_docs_num)
    p0_num = ones(word_nums) # count for every word
    p1_num = ones(word_nums)
    p0_denom = 2.0 # total word num for class 0
    p1_denom = 2.0 # totoal word num for class 1
    for i in range(train_docs_num):
        if train_category[i] == 1:
            p1_num += train_matrix[i]
            p1_denom += sum(train_matrix[i])
        else:
            p0_num += train_matrix[i]
            p0_denom += sum(train_matrix[i])

    p1_vec = log(p1_num/p1_denom)
    p0_vec = log(p0_num/p0_denom)
    return p0_vec, p1_vec, p_abusive


def classifyNB(classify_vec, p0_vec, p1_vec, p_abusive):
    p0 = sum(classify_vec*p0_vec) + log(1.0 - p_abusive)
    p1 = sum(classify_vec*p1_vec) + log(p_abusive)
    if p1 > p0:
        return 1
    return 0


def testNB():
    posting_list, class_list = loadDataSet()
    vocab_list = createVocabList(posting_list)
    train_mat = []
    for posting in posting_list:
        train_mat.append(setOfWord2Vec(vocab_list, posting))
    p0_vec, p1_vec, p_abusive = trainNB0(train_mat, class_list)
    test_entry = ['love', 'my', 'dalmation']
    this_doc = array(setOfWord2Vec(vocab_list, test_entry))
    print test_entry, 'classified as ', classifyNB(this_doc, p0_vec, p1_vec,
                                                   p_abusive)
    test_entry = ['stupid', 'garbage']
    this_doc = array(setOfWord2Vec(vocab_list, test_entry))
    print test_entry, 'classified as ', classifyNB(this_doc, p0_vec, p1_vec,
                                                   p_abusive)


def textParse(big_str):
    import re
    token_list = re.split(r'\W*', big_str)
    return [token.lower() for token in token_list if len(token) > 2]


def spamTest():
    doc_list = []; class_list = []; full_text = []
    for i in range(1, 26):
        filename = 'email/spam/%d.txt' % i
        print filename
        word_list = textParse(open(filename).read())
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(1)
        filename = 'email/ham/%d.txt' % i
        print filename
        word_list = textParse(open(filename).read())
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(0)
    vocab_list = createVocabList(doc_list)
    training_set = range(50)
    test_set = []
    for i in range(10):
        rand_index = int(random.uniform(0, len(training_set)))
        test_set.append(training_set[rand_index])
        del(training_set[rand_index])
    traing_mat = []
    training_classes = []
    for doc_index in training_set:
        traing_mat.append(setOfWord2Vec(vocab_list, doc_list[doc_index]))
        training_classes.append(class_list[doc_index])
    p0_vec, p1_vec, p_spam = trainNB0(array(traing_mat), array(training_classes))
    error_count = 0
    for doc_index in test_set:
        word_vec = setOfWord2Vec(vocab_list, doc_list[doc_index])
        if classifyNB(array(word_vec), p0_vec, p1_vec, p_spam) != class_list[
            doc_index]:
            error_count += 1
    print 'the error rate is', float(error_count)/len(test_set)


def calcMostFreq(vocab_list, full_text):
    import operator
    freq_dict = {}
    print full_text
    print vocab_list
    for token in vocab_list:
        # print 'token is', token
        freq_dict[token] = full_text.count(token)
    sorted_freq = sorted(freq_dict.iteritems(), key=operator.itemgetter(1),
                         reverse=True)
    return sorted_freq[:30]

def localWords(feed1, feed0):
    doc_list = []
    class_list = []
    full_text = []
    min_len = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(min_len):
        word_list = textParse(feed1['entries'][i]['summary'])
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(1)
        word_list = textParse(feed0['entries'][i]['summary'])
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(0)

    vocab_list = createVocabList(doc_list)
    full_text = [text.encode('utf-8') for text in full_text]
    vocab_list = [vocab.encode('utf-8') for vocab in vocab_list]

    top30_words = calcMostFreq(vocab_list, full_text)
    for pair_word in top30_words:
        if  pair_word[0] in vocab_list:
            vocab_list.remove(pair_word[0])

    training_set = range(2 * min_len)
    test_set = []
    for i in range(20):
        rand_index = int(random.uniform(0, len(training_set)))
        test_set.append(training_set[rand_index])
        del(training_set[rand_index])
    traing_mat = []
    training_classes = []
    for doc_index in training_set:
        traing_mat.append(setOfWord2Vec(vocab_list, doc_list[doc_index]))
        training_classes.append(class_list[doc_index])
    p0_vec, p1_vec, p_spam = trainNB0(array(traing_mat), array(training_classes))
    error_count = 0
    for doc_index in test_set:
        word_vec = setOfWord2Vec(vocab_list, doc_list[doc_index])
        if classifyNB(array(word_vec), p0_vec, p1_vec, p_spam) != class_list[
            doc_index]:
            error_count += 1
    print 'the error rate is', float(error_count)/len(test_set)
    return vocab_list, p0_vec, p1_vec


def testFeed():
    ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
    sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
    localWords(ny, sf)


def getTopWords(ny, sf):
    vocab_list, p0_vec, p1_vec = localWords(ny, sf)
    top_ny = []
    top_sf = []
    for i in range(len(p0_vec)):
        if p0_vec[i] > -6.0:
            top_sf.append((vocab_list[i], p0_vec[i]))
        if p1_vec[i] > -6.0:
            top_ny.append((vocab_list[i], p1_vec[i]))
    sorted_sf = sorted(top_sf, key = lambda pair: pair[1], reverse=True)
    print 'sf**sf**sf'
    for item in sorted_sf:
        print item[0]

    sorted_ny = sorted(top_ny, key = lambda pair: pair[1], reverse=True)
    print 'ny**ny**ny'
    for item in sorted_ny:
        print item[0]

if __name__ == '__main__':
    ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
    sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
    getTopWords(ny, sf)

