"""
This file contains a function to run the positive words sentence level attack. We simply select words that occur multiple times
in the training data in benign comments, but relatively fewer times in negative training comments and add these words to the start and end of a sentence
based on a percentage of the sentence length. This percentage is determined by the attack strength. Data needs to be preprocessed after running
this attack
"""
import os
import pickle
from collections import Counter
from itertools import chain
import pandas as pd
import os
import sys
import inspect
import random
import math


import preprocess_utils as pu


def attack_row(row, attack_words, attack_strength):
    """
    Attacks a single row of the dataframe to evade the toxicity detector by adding a fixed % of words by length at the beginning and
    end of the sentence. Returns the modified sentence
    """

    sentence = row["Sentence"]

    # only attack abusive points
    if row["Abusive"] == "Yes":
        len_sent = len(sentence.split(" "))

        NUM_INS = min(len(attack_words), math.ceil(attack_strength * len_sent))

        insert_first = random.sample(attack_words, NUM_INS)
        insert_last = random.sample(attack_words, NUM_INS)

        return " ".join(insert_first) + " " + sentence + " " + " ".join(insert_last)
    else:
        return sentence

def get_counts(data_df):
    """
    returns a Counter object with the counts of each word in the given dataframe
    :param data_df:
    :return:
    """
    corpus = list(data_df["Sentence"].values)
    corpus = list(chain.from_iterable([sentence.split() for sentence in corpus]))
    return Counter(corpus)

def get_attack_words(all_counts,pos_counts,neg_counts, top_n):
    """
    Takes in counters of words in all, just benign and just toxic sentences and returns words that will be inserted as part
    of the attack
    :param all_counts:
    :param pos_counts:
    :param neg_counts:
    :return:
    """
    count_df = pd.DataFrame(columns=["word", "pos_ratio", "neg_ratio"])
    count_df["word"] = list(all_counts.keys())
    count_df["pos_ratio"] = count_df["word"].apply(lambda word: pos_counts[word] / all_counts[word])

    count_df["neg_ratio"] = count_df["word"].apply(lambda word: neg_counts[word] / all_counts[word])
    count_df["count"] = count_df["word"].apply(lambda word: all_counts[word])

    count_df["dif"] = count_df["pos_ratio"] - count_df["neg_ratio"]
    count_df = count_df.sort_values(by=["dif", "count"], ascending=False)
    attack_words = count_df[0:top_n]["word"].tolist()
    return attack_words

def pos_words_sent_attack(data_folder, attack_strength, top_n=50):
    """

    :param data_folder: Folder in which we will find the data we want to attack as well as where we will save the
    data
    :param attack_strength: % wise additions before and after a sentence to be inserted based on length
    :param top_n: The number of top best attack words to use when attacking
    :return:
    """

    # name of attack (folder name we will save it in)
    attack_name = "adv_pos_words_sent"
    # path to save adversarial examples
    adv_folder = os.path.join(data_folder, f"{attack_name}_{str(attack_strength).replace('.', '_')}_{top_n}")

    path_to_training = os.path.join(data_folder, "train_sentence_df")
    # path to testing data we want to attack
    target_path = os.path.join(data_folder, "test_sentence_df")

    # create if it does not already exist
    if not os.path.isdir(adv_folder):
        os.mkdir(adv_folder)

    adv_path = os.path.join(adv_folder, f"raw")
    train_df = pd.read_pickle(path_to_training)

    #get counters for all, just benign and just abusitve comments
    all_counts =get_counts(train_df)
    pos_counts = get_counts(train_df[train_df['label']==0])
    neg_counts = get_counts(train_df[train_df['label']==1])

    attack_words = get_attack_words(all_counts,pos_counts,neg_counts,top_n=top_n)

    # load data we will be attacking

    target = pd.read_pickle(target_path)
    attacked_target = target
    target["Sentence"] = target.apply(lambda row: attack_row(row, attack_words,attack_strength), axis=1)

    # save the attacked data to test (only necessary columns)
    attacked_target = attacked_target[["Comment #", "Sentence", "Abusive"]]





    attacked_target.to_csv(adv_path,index=False)
    pu.preprocess_save_raw_df(adv_path,adv_folder)
    return attacked_target



