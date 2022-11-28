# The tagger.py starter code for CSC384 A4.
# Currently reads in the names of the training files, test file and output file,
# and calls the tagger (which you need to implement)
import os
import pprint
import sys
import numpy as np

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

    for key in transition_dict:
        transition_dict[key] = dict.fromkeys(ALL_TAGS, 0)
        emission_dict[key] = dict()

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
    with open("prior probability.txt", 'w') as f:
        # pp.pprint(tag_first_dict)
        # pp.pprint(tag_count_dict)
        # pp.pprint(transition_dict)
        # pp.pprint(emission_dict)

        pp1 = pprint.PrettyPrinter(stream=f)
        pp1.pprint("Initial Probability:")
        pp1.pprint(tag_first_dict)
        pp1.pprint("Tag Count:")
        pp1.pprint(tag_count_dict)
        pp1.pprint("Transition Count:")
        pp1.pprint(transition_dict)
        pp1.pprint("Emission Count:")
        pp1.pprint(emission_dict)

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
    # pp.pprint(emission_dict)

    return tag_first_dict, transition_dict, emission_dict


def viterbi(observe, initial, trans, emission):
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

    for i in range(tag_length):
        tag = ALL_TAGS[i]
        prob[0, i] = initial[tag] * emission[tag].get(observe[0], float(0))
        prev[0, i] = None
    # pp.pprint(prob)
    # pp.pprint(prev)

    for t in range(1, word_length):
        for i in range(tag_length):
            # find the most likely tag at time t
            cur_tag = ALL_TAGS[i]
            cur_word = observe[t]
            most_likely_tag = 0
            max_prob = float(0)
            for k, tag in enumerate(ALL_TAGS):
                tran = trans[cur_tag].get(tag, 0)
                emis = emission[cur_tag].get(cur_word, 0)
                cur_prob = prob[t-1, k] * tran * emis
                if cur_prob > max_prob:
                    max_prob = cur_prob
                    most_likely_tag = k
            prob[t, i] = max_prob
            prev[t, i] = most_likely_tag
    last_tag_index = np.argmax(prob[-1])
    last_tag = ALL_TAGS[last_tag_index]
    pred_list = [last_tag]
    for t in range(word_length - 1, 0, -1):
        index = int(prev[t, last_tag_index])
        pred_list.append(ALL_TAGS[index])
        last_tag_index = index
    pred_list.reverse()
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
    print(len(full_training_list))
    for i in range(len(full_training_list)):
        full_training_list[i] = full_training_list[i].replace(" ", "").strip().split(':')
        if len(full_training_list[i]) > 2 and full_training_list[i][-1] == "PUN":
            full_training_list[i] = [':', 'PUN']
    # print(full_training_list[0:13])

    ## read the test files and put them into a list of word
    test_list = read_file_to_list(test_file)
    for i in range(len(test_list)):
        test_list[i] = test_list[i].strip()

    # pp.pprint(test_list)

    ## Find the initial, transition, emission probability matrix
    Initial, Transition, Emission = _initial_probability(full_training_list)

    ## Viterbi algorithm
    result = viterbi(test_list, Initial, Transition, Emission)
    pp.pprint(result)

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

    # Start the training and tagging operation.
    tag(training_list, test_file, output_file)
