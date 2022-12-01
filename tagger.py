# The tagger.py starter code for CSC384 A4.
# Currently reads in the names of the training files, test file and output file,
# and calls the tagger (which you need to implement)
import os
import pprint
import sys
import numpy as np
import time

END_SENTENCE_SIG = [".", "?", "!"]
ALL_TAGS = ['AJ0', 'AJC', 'AJS', 'AT0', 'AV0', 'AVP', 'AVQ', 'CJC', 'CJS', 'CJT', 'CRD', 'DPS', 'DT0', 'DTQ', 'EX0', 'ITJ', 'NN0', 'NN1', 'NN2', 'NP0', 'ORD', 'PNI', 'PNP', 'PNQ', 'PNX', 'POS', 'PRF', 'PRP', 'PUL', 'PUN', 'PUQ', 'PUR', 'TO0', 'UNC', 'VBB', 'VBD', 'VBG', 'VBI', 'VBN', 'VBZ', 'VDB', 'VDD', 'VDG', 'VDI', 'VDN', 'VDZ', 'VHB', 'VHD', 'VHG', 'VHI', 'VHN', 'VHZ', 'VM0', 'VVB', 'VVD', 'VVG', 'VVI', 'VVN', 'VVZ', 'XX0', 'ZZ0', 'AJ0-AV0', 'AJ0-VVN', 'AJ0-VVD', 'AJ0-NN1', 'AJ0-VVG', 'AVP-PRP', 'AVQ-CJS', 'CJS-PRP', 'CJT-DT0', 'CRD-PNI', 'NN1-NP0', 'NN1-VVB', 'NN1-VVG', 'NN2-VVZ', 'VVD-VVN']
# print(len(ALL_TAGS))
AMBIGUITY_TAGS = ['AJ0-AV0', 'AJ0-VVN', 'AJ0-VVD', 'AJ0-NN1', 'AJ0-VVG', 'AVP-PRP', 'AVQ-CJS', 'CJS-PRP', 'CJT-DT0', 'CRD-PNI', 'NN1-NP0', 'NN1-VVB', 'NN1-VVG', 'NN2-VVZ', 'VVD-VVN']

pp = pprint.PrettyPrinter()


def read_file_to_list(filename):
    f = open(filename, 'r')
    return f.readlines()


def _ambiguity_check(tag):
    """Check the ambiguity tag, if it's not in the desired order, replace
    For example: 'AJ0-AV0' == 'AV0-AJ0'
        If we encountered 'AV0-AJ0', it's in the AMBIGUITY_TAGS, we just switch
            AV0-AJ0 to AJ0-AV0
    """
    if len(tag) > 3 and tag not in AMBIGUITY_TAGS:
        split = tag.split("-")
        # pp.pprint(split)
        tag = str(split[1] + "-" + split[0])
        return tag
    return tag


def _initial_probability(word_tag):
    """Compute the probability of a tag being the first in a sentence.
    Initial Matrix: A dictionary with tag as key and probalility of first tag
                        as val
    Transition Matrix: A dictionary ...
        key: query (tag after)
        val = dict...
                key = evidence (tag before)
                val = probability of (query | evidence)

    Emission Matrix: A dictionary ...
        key: tag
        val: dict...
                key = evidence (word)
                val = probability of (word|tag)
    >>> wt = []
    >>> _initial_probability(wt)
    """
    tag_first_dict = dict.fromkeys(ALL_TAGS, 0)
    tag_count_dict = dict.fromkeys(ALL_TAGS, 0)
    # init the transition matrix
    transition_dict = dict.fromkeys(ALL_TAGS, None)
    # init the emission probability Matrix
    emission_dict = dict.fromkeys(ALL_TAGS, None)
    emission_dict2 = dict.fromkeys(ALL_TAGS, None)

    for key in transition_dict:
        transition_dict[key] = dict.fromkeys(ALL_TAGS, 0)
        emission_dict[key] = dict()
        emission_dict2[key] = dict()

    word, tag = word_tag[0]
    tag_first_dict[tag] += 1
    tag_count_dict[tag] += 1
    emission_dict[tag][word] = 1
    for i in range(1, len(word_tag)):
        prev_word, prev_tag = word_tag[i - 1][0], word_tag[i - 1][1]
        word, tag = word_tag[i][0], word_tag[i][1]
        prev_tag = _ambiguity_check(prev_tag)
        tag = _ambiguity_check(tag)

        # transition_dict
        transition_dict[tag][prev_tag] += 1     # tag comes after prev_tag
        # emission_dict
        if word in emission_dict[tag]:
            emission_dict[tag][word] += 1
        else:
            emission_dict[tag][word] = 1
        # emission_dict 2
        if word[-2:].isalpha() and len(word) >= 2:
            last_two = word[-2:]
            if last_two in emission_dict2[tag]:
                emission_dict2[tag][last_two] += 1
            else:
                emission_dict2[tag][last_two] = 1
        # tag_first_dict
        if prev_word in END_SENTENCE_SIG:
            # tag is at the beginning of the sentence
            tag_first_dict[tag] += 1
        tag_count_dict[tag] += 1

    ## Initial Probability
    # Compute the probability of each tag starts at a sentence
    for k in tag_first_dict:
        if tag_count_dict[k] != 0:
            tag_first_dict[k] /= tag_count_dict[k]
    # with open("prior probability.txt", 'w') as f:
        # pp1 = pprint.PrettyPrinter(stream=f)
        # pp1.pprint("Initial Probability:")
        # pp1.pprint(tag_first_dict)
        # pp1.pprint("Tag Count:")
        # pp1.pprint(tag_count_dict)
        # pp1.pprint("Transition Count:")
        # pp1.pprint(transition_dict)
        # pp1.pprint("Emission Count:")
        # pp1.pprint(emission_dict)

    ## Compute the transition probability
    for cur_tag in transition_dict:
        for prev in transition_dict[cur_tag]:
            if tag_count_dict[prev] == 0:
                transition_dict[cur_tag][prev] = float(0)
                continue
            count = transition_dict[cur_tag][prev]
            transition_dict[cur_tag][prev] = count / tag_count_dict[prev]
    # pp.pprint(transition_dict)

    ## Compute the emission probability
    for cur_tag in emission_dict:
        for word in emission_dict[cur_tag]:
            if tag_count_dict[cur_tag] == 0:
                emission_dict[cur_tag][word] = float(0)
                continue
            count = emission_dict[cur_tag][word]
            emission_dict[cur_tag][word] = count / tag_count_dict[cur_tag]
        for last_two in emission_dict2[cur_tag]:
            if tag_count_dict[cur_tag] == 0:
                emission_dict2[cur_tag][last_two] = float(0)
                continue
            count = emission_dict2[cur_tag][last_two]
            emission_dict2[cur_tag][last_two] = count / tag_count_dict[cur_tag]
    # pp.pprint(emission_dict)
    print("Done Processing")

    return tag_first_dict, transition_dict, emission_dict, emission_dict2


def transform_emission_dict(word_emission, last2_emission):
    trained_words = []
    trained_last2 = []
    for k in word_emission:
        trained_words.extend(word_emission[k].keys())
        trained_last2.extend(last2_emission[k].keys())
    trained_words = np.unique(trained_words)
    trained_last2 = np.unique(trained_last2)
    # create word to index dictionaries
    word_index = dict()
    last2_index = dict()
    for i, word in enumerate(trained_words):
        word_index[word] = i
    for j, last in enumerate(trained_last2):
        last2_index[last] = j
    word_length = len(trained_words)
    tag_length = len(ALL_TAGS)
    last2_length = len(trained_last2)
    word_matrix = np.full((word_length + 1, tag_length), 1e-7)
    last2_matrix = np.full((last2_length + 1, tag_length), 1e-7)
    for i, tag in enumerate(ALL_TAGS):
        for word in word_emission[tag]:
            w_index = word_index[word]
            prob = word_emission[tag][word]
            word_matrix[w_index][i] = prob
        for last2 in last2_emission[tag]:
            l_index = last2_index[last2]
            prob = last2_emission[tag][last2]
            last2_matrix[l_index][i] = prob

    return word_matrix, last2_matrix, word_index, last2_index


def check_word_observed(word, to_index):
    try:
        return to_index[word]
    except KeyError:
        return -1


def viterbi(observe, initial, trans, emission, emission2):
    """
    observe: word observed.
        List[str]
    initial: initial probability.
        Dict[key: tag, val: prob]
    trans: transition probability
        Dict[key: after_tag, val:
                                Dict[key: prev_tag, val: prob]]
    emission: emission probability
        Dict[key: tag, val:
                            Dict[key: word, val: prob]]
    """
    word_length = len(observe)
    tag_length = len(ALL_TAGS)
    # prob[i,j] is the probability of being tag j at time i
    prob = np.zeros((word_length, tag_length))
    # prev[i,j] is the tag at time i
    prev = np.zeros((word_length, tag_length))

    # Turn initial dict into matrix
    initial_mat = np.array(list(initial.values())).T
    # pp.pprint(initial_mat)

    # Turn transition and emission dictionaries into matrices
    trans_mat = [list(trans[k].values()) for k in trans]
    trans_mat = np.array(trans_mat)
    # pp.pprint(trans_mat)

    emis_mat1, emis_mat2, word_index, last2_index = transform_emission_dict(emission, emission2)

    pp1 = pprint.PrettyPrinter(stream=open("prob_matrix.txt", 'w'))

    hyperparam = 0.455

    first_word = observe[0].lower()
    last2 = -1
    if len(first_word) >= 2 and first_word[-2:].isalpha():
        last2 = check_word_observed(first_word[-2:], last2_index)
    first_word = check_word_observed(first_word, word_index)
    prob[0] = initial_mat * emis_mat1[first_word] * emis_mat2[last2] ** hyperparam
    prev[0] = None

    # normalize
    norm_factor = prob[0].sum()
    prob[0] = prob[0] / norm_factor

    for t in range(1, word_length):
        cur_word = observe[t].lower()
        last2 = -1
        if len(cur_word) >= 2 and cur_word[-2:].isalpha():
            last2 = check_word_observed(cur_word[-2:], last2_index)
        cur_word_index = check_word_observed(cur_word, word_index)
        for i in range(tag_length):
            if last2 == -1:
                cur_prob = prob[t-1] * trans_mat[i] * emis_mat1[cur_word_index]
            else:
                cur_prob = prob[t-1] * trans_mat[i] * emis_mat1[cur_word_index] * emis_mat2[last2] ** hyperparam

            x = np.argmax(cur_prob)
            # Caution
            prob[t, i] = prob[t-1, x] * trans_mat[i, x] * emis_mat1[cur_word_index, x] * emis_mat2[last2, x] ** hyperparam
            prev[t, i] = x
        # normalize
        norm_factor = prob[t].sum()
        prob[t] = prob[t] / norm_factor


    # for i in range(tag_length):
    #     tag = ALL_TAGS[i]
    #     prob[0, i] = initial[tag] * emission[tag].get(observe[0], 1e-7) * emission2[tag].get(observe[0][-2:], 1e-7) ** 0.5
    #     prev[0, i] = None
    # # normalize
    # norm_factor = prob[0].sum()
    # # pp.pprint(norm_factor)
    # prob[0] = prob[0] / norm_factor
    # # pp1.pprint(prob[0])
    # # pp.pprint(prev)
    #
    # for t in range(1, word_length):
    #     for i in range(tag_length):
    #         # find the most likely tag at time t
    #         cur_tag = ALL_TAGS[i]
    #         cur_word = observe[t]
    #         most_likely_tag = 0
    #         max_prob = float(0)
    #         for k, tag in enumerate(ALL_TAGS):
    #             tran = trans[cur_tag].get(tag, 1e-7)
    #             emis = emission[cur_tag].get(cur_word, 1e-7)
    #             if len(cur_word) >= 2 and cur_word[-2:].isalpha():
    #                 emis2 = emission2[cur_tag].get(cur_word[-2:], 1e-7)
    #                 cur_prob = prob[t-1, k] * tran * emis * emis2 ** 0.5
    #             else:
    #                 cur_prob = prob[t-1, k] * tran * emis
    #             if cur_prob > max_prob:
    #                 max_prob = cur_prob
    #                 most_likely_tag = k
    #         prob[t, i] = max_prob
    #         prev[t, i] = most_likely_tag
    #     # normalize
    #     norm_factor = prob[t].sum()
    #     prob[t] = prob[t] / norm_factor
    # pp1.pprint(prob[300:310])
    # pp1.pprint(prev)
    last_tag_index = np.argmax(prob[-1])
    last_tag = ALL_TAGS[last_tag_index]
    pred_list = [last_tag]
    for t in range(word_length - 1, 0, -1):
        index = int(prev[t, last_tag_index])
        pred_list.append(ALL_TAGS[index])
        last_tag_index = index
    pred_list.reverse()
    pred_list.insert(0, pred_list.pop())
    return pred_list


def tag(training_list, test_file, output_file):
    # Tag the words from the untagged input file and write them into the output file.
    # Doesn't do much else beyond that yet.
    print("Tagging the file.")

    ## read each training file and put them into a list of word-tag pairs
    # and update the probabilities
    full_training_list = []
    for train_file in training_list:
        full_training_list.extend(read_file_to_list(train_file))
    print(f'Training size: {len(full_training_list)}')
    for i in range(len(full_training_list)):
        full_training_list[i] = full_training_list[i].strip().split(' : ')
        if len(full_training_list[i]) > 2 and full_training_list[i][-1] == "PUN":
            full_training_list[i] = [':', 'PUN']
        full_training_list[i][0] = full_training_list[i][0].lower()
    # print(full_training_list[0:13])

    ## read the test files and put them into a list of word
    test_list = read_file_to_list(test_file)
    lower_test_list = []
    for i in range(len(test_list)):
        test_list[i] = test_list[i].strip()
        lower_test_list.append(test_list[i].lower())
    # pp.pprint(test_list)

    ## Find the initial, transition, emission probability matrix
    Initial, Transition, Emission, Emission2 = _initial_probability(full_training_list)

    ## Viterbi algorithm
    result = viterbi(lower_test_list, Initial, Transition, Emission, Emission2)
    # pp.pprint(result)

    # write to output_file
    with open(output_file, 'w') as f:
        for i in range(len(test_list)):
            f.write(test_list[i] + " : " + result[i] + "\n")


if __name__ == '__main__':
    # Run the tagger function.
    print("Starting the tagging process.")

    # Tagger expects the input call: "python3 tagger.py -d <training files> -t <test file> -o <output file>"
    parameters = sys.argv
    training_list = parameters[parameters.index("-d")+1:parameters.index("-t")]
    test_file = parameters[parameters.index("-t")+1]
    output_file = parameters[parameters.index("-o")+1]
    # print("Training files: " + str(training_list))
    # print("Test file: " + test_file)
    # print("Output file: " + output_file)

    st = time.time()
    # Start the training and tagging operation.
    tag(training_list, test_file, output_file)

    solution_txt = 'data/training1.txt'
    results_txt = 'data/resultst5t1.txt'

    # need to change solution file
    with open(output_file, "r") as output_file, \
            open(solution_txt, "r") as solution_file, \
            open(results_txt, "w") as results_file:
        # Each word is on a separate line in each file.
        output = output_file.readlines()
        solution = solution_file.readlines()
        total_matches = 0

        # generate the report
        for index in range(len(output)):
            if output[index] != solution[index]:
                results_file.write(f"Line {index + 1}: "
                                   f"expected <{solution[index].strip()}> "
                                   f"but got <{output[index].strip()}>\n")
            else:
                total_matches = total_matches + 1

        # Add stats at the end of the results file.
        results_file.write(f"Total words seen: {len(output)}.\n")
        results_file.write(f"Total matches: {total_matches}.\n")
        results_file.write(f'Accuracy: {total_matches / len(output)}.\n')
        et = time.time()
        results_file.write(f'Elapsed time: {et - st} seconds')
