import sys
from collections import defaultdict
import math
import random
import os
import os.path
"""
COMS W4705 - Natural Language Processing - Fall 2024 
Programming Homework 1 - Trigram Language Models
Daniel Bauer
"""

def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  



def get_ngrams(sequence, n):
    """
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of n >= 1 
    """
    ngrams = []

    if n < 1:
        raise ValueError("n must be greater than or equal to 1")
    
    if len(sequence) == 0:
        return []
    
    #for unigram
    if n == 1:
        sequence_padded = ['START'] + sequence + ['STOP']
    else: 
        #for n > 1 ngrams
        sequence_padded = ['START'] * (n - 1) + sequence + ['STOP']

    for i in range(len(sequence_padded) - n + 1):
        ngram = tuple(sequence_padded[i:i + n])
        ngrams.append(ngram)

    return ngrams


class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")

        self.sentence_num = 0 #sentence number count
    
        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)

        #total number of unigrams
        total_unigram = sum(self.unigramcounts.values())
        #total number of unigrams excluding count for 'START'
        self.total_unigram_denominator = total_unigram - self.unigramcounts.get(('START',), 0)
    




    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """
   
        self.unigramcounts = defaultdict(int)
        self.bigramcounts = defaultdict(int)
        self.trigramcounts = defaultdict(int)

        ##Your code here
        for sentence in corpus:
            self.sentence_num += 1 # count number of sentences
            unigram = get_ngrams(sentence, 1)
            bigram = get_ngrams(sentence, 2)
            trigram = get_ngrams(sentence, 3)

            #count occurances for each ngram
            for uni in unigram:
                self.unigramcounts[uni] += 1
            for bi in bigram:
                self.bigramcounts[bi] += 1
            for tri in trigram:
                self.trigramcounts[tri] += 1

        return

    def raw_trigram_probability(self,trigram):
        """
        Returns the raw (unsmoothed) trigram probability
        """
        #special case ('start' 'start', w)
        if trigram[:2] == ('START', 'START'):
            return self.trigramcounts.get(trigram, 0) / self.sentence_num

        trigram_prob_denominator = self.bigramcounts[(trigram[0], trigram[1])]
        if trigram_prob_denominator == 0:
            return 1 / len(self.lexicon) #uniform distribution 
        
        return self.trigramcounts.get(trigram, 0) / trigram_prob_denominator

    def raw_bigram_probability(self, bigram):
        """
        Returns the raw (unsmoothed) bigram probability
        """
        V = len(self.lexicon)
        if self.unigramcounts[(bigram[0],)] == 0: 
            return 1/V
    
        bigram_prob_denominator = self.unigramcounts[(bigram[0],)]
        
        return self.bigramcounts[bigram] / bigram_prob_denominator
    
    def raw_unigram_probability(self, unigram):
        """
        Returns the raw (unsmoothed) unigram probability.
        """
        
        #hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once, 
        # store in the TrigramModel instance, and then re-use it.  
        return  self.unigramcounts[unigram] / self.total_unigram_denominator

    def generate_sentence(self,t=20): 
        """
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        return result            

    def smoothed_trigram_probability(self, trigram):
        """
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0

        trigram_weighted = lambda1 * self.raw_trigram_probability(trigram)
        bigram_weighted = lambda2 * self.raw_bigram_probability(trigram[1:])
        unigram_weighted = lambda3 * self.raw_unigram_probability((trigram[2],))

        return trigram_weighted + bigram_weighted + unigram_weighted
        
    def sentence_logprob(self, sentence):
        """
        Returns the log probability of an entire sequence.
        """
        log_probability = 0.0
        trigrams = get_ngrams(sentence, 3)

        for tri in trigrams:
            smoothed_probability = self.smoothed_trigram_probability(tri)

            if smoothed_probability == 0:
                log_probability += 0
            else:
                log_probability += math.log2(smoothed_probability)

        return log_probability

    def perplexity(self, corpus):
        """
        Returns the log probability of an entire sequence.
        """
        sum_log_probability = 0.0
        total_tokens = 0.0

        for sentence in corpus:
            sum_log_probability += self.sentence_logprob(sentence)
            total_tokens += len(sentence) + 1 # + 1 adds the count for the STOP token

        l = (1/total_tokens) * sum_log_probability
        perplexity = 2**(-l)
        return perplexity


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0
        correct = 0       

        #high 
        for f in os.listdir(testdir1):
            pp1 = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            pp2 = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
            total += 1
            if pp1 < pp2:
                correct += 1


        #low
        for f in os.listdir(testdir2):
            pp2 = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            pp1 = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
            total += 1
            if pp2 < pp1:
                correct += 1
        
        accuracy = correct / total
        return accuracy

if __name__ == "__main__":

    model = TrigramModel(sys.argv[1]) 
    #print sentence count 
    print(f"Sentence Count: {model.sentence_num}")
    
    print(f"Number of unigrams: {len(model.unigramcounts)}")
    print(f"Number of bigrams: {len(model.bigramcounts)}")
    print(f"Number of trigrams: {len(model.trigramcounts)}")

    #perplexity train 
    train_corpus = corpus_reader(sys.argv[1], model.lexicon)
    train_pp = model.perplexity(train_corpus)
    print(f"perplexity train:{train_pp}")

    #test perplexity 
    dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    pp = model.perplexity(dev_corpus)
    print(f"perplexity brown test:{pp}")


    #print("Unigram Counts:", model.unigramcounts)
    #print("Bigram Counts:", model.bigramcounts)
    #print("Trigram Counts:", model.trigramcounts)

    # essay scoring experiment 
    train_high = sys.argv[3]
    train_low = sys.argv[4]
    test_high_dir = sys.argv[5]
    test_low_dir = sys.argv[6]

    accuracy = essay_scoring_experiment(train_high, train_low, test_high_dir, test_low_dir)
    print(f"essay accuracy: {accuracy: }")
                                   



    # put test code here...
    # or run the script from the command line with 
    # $ python -i trigram_model.py [corpus_file]
    # >>> 
    #
    # you can then call methods on the model instance in the interactive 
    # Python prompt. 

    
    # Testing perplexity: 
    # dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    # pp = model.perplexity(dev_corpus)
    # print(pp)


    # Essay scoring experiment: 
    # acc = essay_scoring_experiment('train_high.txt', 'train_low.txt", "test_high", "test_low")
    # print(acc)

