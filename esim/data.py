"""
Preprocessor and dataset definition for NLI.
"""
# Aurelien Coet, 2018.

import string
import torch
import numpy as np
import csv

from collections import Counter
from torch.utils.data import Dataset


class Preprocessor(object):
    """
    Preprocessor class for Natural Language Inference datasets.

    The class can be used to read NLI datasets, build worddicts for them
    and transform their premises, hypotheses and labels into lists of
    integer indices.
    """

    def __init__(self,
                 lowercase=False,
                 ignore_punctuation=False,
                 num_words=None,
                 stopwords=[],
                 labeldict={},
                 bos=None,
                 eos=None):
        """
        Args:
            lowercase: A boolean indicating whether the words in the datasets
                being preprocessed must be lowercased or not. Defaults to
                False.
            ignore_punctuation: A boolean indicating whether punctuation must
                be ignored or not in the datasets preprocessed by the object.
            num_words: An integer indicating the number of words to use in the
                worddict of the object. If set to None, all the words in the
                data are kept. Defaults to None.
            stopwords: A list of words that must be ignored when building the
                worddict for a dataset. Defaults to an empty list.
            bos: A string indicating the symbol to use for the 'beginning of
                sentence' token in the data. If set to None, the token isn't
                used. Defaults to None.
            eos: A string indicating the symbol to use for the 'end of
                sentence' token in the data. If set to None, the token isn't
                used. Defaults to None.
        """
        self.lowercase = lowercase
        self.ignore_punctuation = ignore_punctuation
        self.num_words = num_words
        self.stopwords = stopwords
        self.labeldict = labeldict
        self.bos = bos
        self.eos = eos

    def get_expl_dict(self, esnli_dir_path, train=False):
        """
        Returns a dict mapping a sample ID of a premise-hypothesis pair to the corresponding explanation
        """
        id_to_explenation_dict = {}
        mydict2 = {}
        
        if train:
            with open('../../data/dataset/esnli/esnli_train_1.csv', mode='r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)
                id_to_explenation_dict = {rows[0]:rows[4] for rows in reader}
            
            with open('../../data/dataset/esnli/esnli_train_2.csv', mode='r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)
                mydict2 = {rows[0]:rows[4] for rows in reader}
        
        else:
            with open(esnli_dir_path, mode='r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)
                id_to_explenation_dict = {rows[0]:rows[4] for rows in reader}
        
        print("Num of explanations in first file:", len(id_to_explenation_dict))
        if train:
            id_to_explenation_dict.update(mydict2)
            print("Num of explanations after merging:", len(id_to_explenation_dict))
        
        return id_to_explenation_dict
            
    
    def read_data(self, filepath, mode, snli=True):
        """
        Read the premises, hypotheses and labels from some NLI dataset's
        file and return them in a dictionary. The file should be in the same
        form as SNLI's .txt files.

        Args:
            filepath: The path to a file containing some premises, hypotheses
                and labels that must be read. The file should be formatted in
                the same way as the SNLI (and MultiNLI) dataset.

        Returns:
            A dictionary containing three lists, one for the premises, one for
            the hypotheses, and one for the labels in the input data.
        """
        
        assert "snli" in filepath or "multinli" in filepath
        if mode=="train":
            print("Doing read_data for training")
            if "snli" in filepath:
                filepath2 = "../../data/dataset/multinli_1.0/multinli_1.0_train.txt"
            else:           
                filepath2 = filepath
                filepath = "../../data/dataset/snli_1.0/snli_1.0_train.txt"
            
            expl_dict = self.get_expl_dict("../../data/dataset/esnli", train=True)
            
            with open(filepath, "r", encoding="utf8") as input_data, open(filepath2, "r", encoding="utf8") as input_data2:
                ids, premises, hypotheses, labels, explanations = [], [], [], [], []
                ids2, premises2, hypotheses2, labels2 = [], [], [], []
    
                # Translation tables to remove parentheses and punctuation from
                # strings.
                parentheses_table = str.maketrans({"(": None, ")": None})
                punct_table = str.maketrans({key: " "
                                             for key in string.punctuation})
    
                # Ignore the headers on the first line of the file.
                next(input_data)
    
                for line in input_data:
                    line = line.strip().split("\t")
    
                    # Ignore sentences that have no gold label.
                    if line[0] == "-":
                        continue
    
                    pair_id = line[8]
                    premise = line[1]
                    hypothesis = line[2]
    
                    # Remove '(' and ')' from the premises and hypotheses.
                    premise = premise.translate(parentheses_table)
                    hypothesis = hypothesis.translate(parentheses_table)
                    
                    
                    assert pair_id in expl_dict.keys()
                    explanation = expl_dict[pair_id]

    
                    if self.lowercase:
                        premise = premise.lower()
                        hypothesis = hypothesis.lower()
                        explanation = explanation.lower()
    
                    if self.ignore_punctuation:
                        premise = premise.translate(punct_table)
                        hypothesis = hypothesis.translate(punct_table)
                        explanation = explanation.translate(punct_table)
                    
                    cnt_long_explanations = 0
                    
                    # Each premise and hypothesis is split into a list of words.
                    premises.append([w for w in premise.rstrip().split()
                                     if w not in self.stopwords])
                    hypotheses.append([w for w in hypothesis.rstrip().split()
                                       if w not in self.stopwords])
                    explanations.append([w for w in explanation.rstrip().split()
                                        if w not in self.stopwords])
                    labels.append(line[0])
                    ids.append(pair_id)
                    
                    if len([w for w in explanation.rstrip().split()
                                        if w not in self.stopwords]) >= 50:
                        cnt_long_explanations += 1
                    
                # Ignore the headers on the first line of the file.
                next(input_data2)
    
                for line in input_data2:
                    line = line.strip().split("\t")
    
                    # Ignore sentences that have no gold label.
                    if line[0] == "-":
                        continue
    
                    pair_id = line[8]
                    premise = line[1]
                    hypothesis = line[2]
    
                    # Remove '(' and ')' from the premises and hypotheses.
                    premise = premise.translate(parentheses_table)
                    hypothesis = hypothesis.translate(parentheses_table)
    
                    if self.lowercase:
                        premise = premise.lower()
                        hypothesis = hypothesis.lower()
    
                    if self.ignore_punctuation:
                        premise = premise.translate(punct_table)
                        hypothesis = hypothesis.translate(punct_table)
    
                    # Each premise and hypothesis is split into a list of words.
                    premises2.append([w for w in premise.rstrip().split()
                                     if w not in self.stopwords])
                    hypotheses2.append([w for w in hypothesis.rstrip().split()
                                       if w not in self.stopwords])
                    labels2.append(line[0])
                    ids2.append(pair_id) 
                    
                print("Number of long explanations:", cnt_long_explanations)
    
                return {"ids": ids,
                        "ids2": ids2,
                        "premises": premises,
                        "premises2": premises2,
                        "hypotheses": hypotheses,
                        "hypotheses2": hypotheses2,
                        "labels": labels,
                        "labels2": labels2,
                        "explanations": explanations}
        else:
            if snli:
                if mode == "dev":        
                    expl_dict = self.get_expl_dict("../../data/dataset/esnli/esnli_dev.csv", train=False)
                else:
                    expl_dict = self.get_expl_dict("../../data/dataset/esnli/esnli_test.csv", train=False)
            
            print("Doing read_data for dev/test")
            with open(filepath, "r", encoding="utf8") as input_data:
                ids, premises, hypotheses, labels = [], [], [], []
                if snli:
                    explanations = []
                
                # Translation tables to remove parentheses and punctuation from
                # strings.
                parentheses_table = str.maketrans({"(": None, ")": None})
                punct_table = str.maketrans({key: " "
                                             for key in string.punctuation})
    
                # Ignore the headers on the first line of the file.
                next(input_data)
    
                for line in input_data:
                    line = line.strip().split("\t")
    
                    # Ignore sentences that have no gold label.
                    if line[0] == "-":
                        continue
    
                    pair_id = line[8]
                    premise = line[1]
                    hypothesis = line[2]
    
                    # Remove '(' and ')' from the premises and hypotheses.
                    premise = premise.translate(parentheses_table)
                    hypothesis = hypothesis.translate(parentheses_table)
                    
                    if snli:
                        assert pair_id in expl_dict.keys()
                        explanation = expl_dict[pair_id]
    
                    if self.lowercase:
                        premise = premise.lower()
                        hypothesis = hypothesis.lower()
                        if snli:
                            explanation = explanation.lower()
    
                    if self.ignore_punctuation:
                        premise = premise.translate(punct_table)
                        hypothesis = hypothesis.translate(punct_table)
                        if snli:
                            explanation = explanation.translate(punct_table)
    
                    # Each premise and hypothesis is split into a list of words.
                    premises.append([w for w in premise.rstrip().split()
                                     if w not in self.stopwords])
                    hypotheses.append([w for w in hypothesis.rstrip().split()
                                       if w not in self.stopwords])
                    if snli:
                        explanations.append([w for w in explanation.rstrip().split()
                                           if w not in self.stopwords])
                    labels.append(line[0])
                    ids.append(pair_id)   
                
                if snli:
                    return {"ids": ids,
                            "premises": premises,
                            "hypotheses": hypotheses,
                            "explanations": explanations,
                            "labels": labels}
                else:
                    return {"ids": ids,
                            "premises": premises,
                            "hypotheses": hypotheses,
                            "labels": labels}                    

    def build_worddict(self, data):
        """
        Build a dictionary associating words to unique integer indices for
        some dataset. The worddict can then be used to transform the words
        in datasets to their indices.

        Args:
            data: A dictionary containing the premises, hypotheses and
                labels of some NLI dataset, in the format returned by the
                'read_data' method of the Preprocessor class.
        """
        words = []
        [words.extend(sentence) for sentence in data["premises"]]
        [words.extend(sentence) for sentence in data["hypotheses"]]
        [words.extend(sentence) for sentence in data["explanations"]]
        [words.extend(sentence) for sentence in data["premises2"]]
        [words.extend(sentence) for sentence in data["hypotheses2"]]
        
        counts = Counter(words)
        num_words = self.num_words
        if self.num_words is None:
            num_words = len(counts)
        print("num_words =", num_words)
        num_words = 50000

        self.worddict = {}

        # Special indices are used for padding, out-of-vocabulary words,
        # beginning and end of sentence tokens, and label names
        self.worddict["_PAD_"] = 0
        self.worddict["_OOV_"] = 1

        offset = 2
        if self.bos:
            self.worddict["_BOS_"] = 2
            offset += 1
        if self.eos:
            self.worddict["_EOS_"] = 3
            offset += 1
        
        self.worddict["entailment"] = 4
        self.worddict["contradiction"] = 5
        self.worddict["neutral"] = 6
        offset += 3
        
        j = 0
        for i, word in enumerate(counts.most_common(num_words)):
            if word[0] not in ["entailment","contradiction","neutral"]:
                self.worddict[word[0]] = j + offset
                j += 1

        if self.labeldict == {}:
#            label_names = set(data["labels"])
#            self.labeldict = {label_name: i
#                              for i, label_name in enumerate(label_names)}
            # Entailment: 0, Contradiction: 1, Neutral: 2
            label_names = ["contradiction", "neutral", "entailment"]
            self.labeldict = {label_name: i
                              for i, label_name in enumerate(label_names)}    
        
        for i in range(50008):
            if i not in self.worddict.values():
                print("Index", i, "not found!!!")
        
        # Build index to word dict
        self.index_to_word_dict = {ind:word for word, ind in self.worddict.items()}

    def words_to_indices(self, sentence):
        """
        Transform the words in a sentence to their corresponding integer
        indices.

        Args:
            sentence: A list of words that must be transformed to indices.

        Returns:
            A list of indices.
        """
        indices = []
        # Include the beggining of sentence token at the start of the sentence
        # if one is defined.
        if self.bos:
            indices.append(self.worddict["_BOS_"])

        for word in sentence:
            if word in self.worddict:
                index = self.worddict[word]
            else:
                # Words absent from 'worddict' are treated as a special
                # out-of-vocabulary word (OOV).
                index = self.worddict["_OOV_"]
            indices.append(index)
        # Add the end of sentence token at the end of the sentence if one
        # is defined.
        if self.eos:
            indices.append(self.worddict["_EOS_"])

        return indices

    def indices_to_words(self, indices):
        """
        Transform the indices in a list to their corresponding words in
        the object's worddict.

        Args:
            indices: A list of integer indices corresponding to words in
                the Preprocessor's worddict.

        Returns:
            A list of words.
        """
        return [list(self.worddict.keys())[list(self.worddict.values())
                                           .index(i)]
                for i in indices]

    def transform_to_indices(self, data):
        """
        Transform the words in the premises and hypotheses of a dataset, as
        well as their associated labels, to integer indices.

        Args:
            data: A dictionary containing lists of premises, hypotheses
                and labels, in the format returned by the 'read_data'
                method of the Preprocessor class.

        Returns:
            A dictionary containing the transformed premises, hypotheses and
            labels.
        """
        if "explanations" in data.keys():
            transformed_data = {"ids": [],
                                "premises": [],
                                "hypotheses": [],
                                "explanations": [],
                                "labels": []}
        else:
            transformed_data = {"ids": [],
                                "premises": [],
                                "hypotheses": [],
                                "labels": []}            
        
        for i, premise in enumerate(data["premises"]):
            # Ignore sentences that have a label for which no index was
            # defined in 'labeldict'.
            label = data["labels"][i]
            if label not in self.labeldict and label != "hidden":
                continue

            transformed_data["ids"].append(data["ids"][i])

            if label == "hidden":
                transformed_data["labels"].append(-1)
            else:
                transformed_data["labels"].append(self.labeldict[label])

            indices = self.words_to_indices(premise)
            transformed_data["premises"].append(indices)

            indices = self.words_to_indices(data["hypotheses"][i])
            transformed_data["hypotheses"].append(indices)
            
            if "explanations" in data.keys():
                indices = self.words_to_indices(data["explanations"][i])
                # Clip long explanations such that maximal explanation length is at most 50
                if len(indices) > 50:
                    transformed_data["explanations"].append(indices[:50])
                else:
                    transformed_data["explanations"].append(indices)
                

        
#        if "explanations" in data.keys():
#            explanations_lengths = [len(exp) for exp in transformed_data["explanations"]]
#            cnt_long = 0
#            for l in explanations_lengths:
#                if l >= 50:
#                    cnt_long += 1
#            print("Number of transformed explanations of length at least 50:", cnt_long)
                

        return transformed_data
        

    def build_embedding_matrix(self, embeddings_file):
        """
        Build an embedding matrix with pretrained weights for object's
        worddict.

        Args:
            embeddings_file: A file containing pretrained word embeddings.

        Returns:
            A numpy matrix of size (num_words+n_special_tokens, embedding_dim)
            containing pretrained word embeddings (the +n_special_tokens is for
            the padding and out-of-vocabulary tokens, as well as BOS and EOS if
            they're used).
        """
        # Load the word embeddings in a dictionnary.
        embeddings = {}
        with open(embeddings_file, "r", encoding="utf8") as input_data:
            for line in input_data:
                line = line.split()

                try:
                    # Check that the second element on the line is the start
                    # of the embedding and not another word. Necessary to
                    # ignore multiple word lines.
                    float(line[1])
                    word = line[0]
                    if word in self.worddict:                      
                        embeddings[word] = line[1:]

                # Ignore lines corresponding to multiple words separated
                # by spaces.
                except ValueError:
                    continue

        num_words = len(self.worddict)
        embedding_dim = len(list(embeddings.values())[0])
        embedding_matrix = np.zeros((num_words, embedding_dim))

        # Actual building of the embedding matrix.
        missed = 0
        for word, i in self.worddict.items():
            if word in embeddings:
                if word == "entailment":
                    print("Found embedding of entailment! Index:", i)
                if word == "contradiction":
                    print("Found embedding of contradiction! Index:", i)
                if word == "neutral":
                    print("Found embedding of neutral! Index:", i)                    
                embedding_matrix[i] = np.array(embeddings[word], dtype=float)
            else:
                if word == "_PAD_":
                    continue
                missed += 1
                # Out of vocabulary words are initialised with random gaussian
                # samples.
                embedding_matrix[i] = np.random.normal(size=(embedding_dim))
        print("Missed words: ", missed)
        
        print("Total vocabulary size:", embedding_matrix.shape[0])

        return embedding_matrix


class NLIDataset(Dataset):
    """
    Dataset class for Natural Language Inference datasets.

    The class can be used to read preprocessed datasets where the premises,
    hypotheses and labels have been transformed to unique integer indices
    (this can be done with the 'preprocess_data' script in the 'scripts'
    folder of this repository).
    """

    def __init__(self,
                 data,
                 mode,
                 padding_idx=0,
                 max_premise_length=None,
                 max_hypothesis_length=None,
                 max_explanation_length=None):
        """
        Args:
            data: A dictionary containing the preprocessed premises,
                hypotheses and labels of some dataset.
            padding_idx: An integer indicating the index being used for the
                padding token in the preprocessed data. Defaults to 0.
            max_premise_length: An integer indicating the maximum length
                accepted for the sequences in the premises. If set to None,
                the length of the longest premise in 'data' is used.
                Defaults to None.
            max_hypothesis_length: An integer indicating the maximum length
                accepted for the sequences in the hypotheses. If set to None,
                the length of the longest hypothesis in 'data' is used.
                Defaults to None.
        """
        
        self.mode = mode
        
        self.premises_lengths = [len(seq) for seq in data["premises"]]
        self.max_premise_length = max_premise_length
        if self.max_premise_length is None:
            self.max_premise_length = max(self.premises_lengths)

        self.hypotheses_lengths = [len(seq) for seq in data["hypotheses"]]
        self.max_hypothesis_length = max_hypothesis_length
        if self.max_hypothesis_length is None:
            self.max_hypothesis_length = max(self.hypotheses_lengths)
            
        if mode in ["train", "dev"]:
            self.explanations_lengths = [len(seq) for seq in data["explanations"]]
            self.max_explanation_length = max_explanation_length
            if self.max_explanation_length is None:
                self.max_explanation_length = max(self.explanations_lengths)  
            print("Max explanation length:", self.max_explanation_length)

        self.num_sequences = len(data["premises"])
        
        if mode == "test":
            self.data = {"ids": [],
                         "premises": torch.ones((self.num_sequences,
                                                 self.max_premise_length),
                                                dtype=torch.long) * padding_idx,
                         "hypotheses": torch.ones((self.num_sequences,
                                                   self.max_hypothesis_length),
                                                  dtype=torch.long) * padding_idx,
                         "labels": torch.tensor(data["labels"], dtype=torch.long)}
        else:
            self.data = {"ids": [],
                         "premises": torch.ones((self.num_sequences,
                                                 self.max_premise_length),
                                                dtype=torch.long) * padding_idx,
                         "hypotheses": torch.ones((self.num_sequences,
                                                   self.max_hypothesis_length),
                                                  dtype=torch.long) * padding_idx,
                         "explanations": torch.ones((self.num_sequences,
                                                   self.max_explanation_length),
                                                  dtype=torch.long) * padding_idx,
                         "labels": torch.tensor(data["labels"], dtype=torch.long)}            

        for i, premise in enumerate(data["premises"]):
            self.data["ids"].append(data["ids"][i])
            end = min(len(premise), self.max_premise_length)
            self.data["premises"][i][:end] = torch.tensor(premise[:end])

            hypothesis = data["hypotheses"][i]
            end = min(len(hypothesis), self.max_hypothesis_length)
            self.data["hypotheses"][i][:end] = torch.tensor(hypothesis[:end])
            
            if mode in ["train", "dev"]:
                explanation = data["explanations"][i]
                end = min(len(explanation), self.max_explanation_length)
                self.data["explanations"][i][:end] = torch.tensor(explanation[:end])

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, index):
        if self.mode == "test":
            return {"id": self.data["ids"][index],
                    "premise": self.data["premises"][index],
                    "premise_length": min(self.premises_lengths[index],
                                          self.max_premise_length),
                    "hypothesis": self.data["hypotheses"][index],
                    "hypothesis_length": min(self.hypotheses_lengths[index],
                                             self.max_hypothesis_length),
                    "label": self.data["labels"][index]}
        else:
            return {"id": self.data["ids"][index],
                    "premise": self.data["premises"][index],
                    "premise_length": min(self.premises_lengths[index],
                                          self.max_premise_length),
                    "hypothesis": self.data["hypotheses"][index],
                    "hypothesis_length": min(self.hypotheses_lengths[index],
                                             self.max_hypothesis_length),
                    "explanation": self.data["explanations"][index],
                    "explanation_length": min(self.explanations_lengths[index],
                                              self.max_explanation_length),
                    "label": self.data["labels"][index]}            
