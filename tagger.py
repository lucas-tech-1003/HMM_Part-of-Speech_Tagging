# The tagger.py starter code for CSC384 A4.
# Currently reads in the names of the training files, test file and output file,
# and calls the tagger (which you need to implement)
import os
import sys

END_SENTENCE_SIG = [".", "?", "!"]
ALL_TAGS = ['AJ0', 'AJC', 'AJS', 'AT0', 'AV0', 'AVP', 'AVQ', 'CJC', 'CJS', 'CJT', 'CRD', 'DPS', 'DT0', 'DTQ', 'EX0', 'ITJ', 'NN0', 'NN1', 'NN2', 'NP0', 'ORD', 'PNI', 'PNP', 'PNQ', 'PNX', 'POS', 'PRF', 'PRP', 'PUL', 'PUN', 'PUQ', 'PUR', 'TO0', 'UNC', 'VBB', 'VBD', 'VBG', 'VBI', 'VBN', 'VBZ', 'VDB', 'VDD', 'VDG', 'VDI', 'VDN', 'VDZ', 'VHB', 'VHD', 'VHG', 'VHI', 'VHN', 'VHZ', 'VM0', 'VVB', 'VVD', 'VVG', 'VVI', 'VVN', 'VVZ', 'XX0', 'ZZ0', 'AJ0-AV0', 'AJ0-VVN', 'AJ0-VVD', 'AJ0-NN1', 'AJ0-VVG', 'AVP-PRP', 'AVQ-CJS', 'CJS-PRP', 'CJT-DT0', 'CRD-PNI', 'NN1-NP0', 'NN1-VVB', 'NN1-VVG', 'NN2-VVZ', 'VVD-VVN']
# print(len(ALL_TAGS))
AMBIGUITY_TAGS = ['AJ0-AV0', 'AJ0-VVN', 'AJ0-VVD', 'AJ0-NN1', 'AJ0-VVG', 'AVP-PRP', 'AVQ-CJS', 'CJS-PRP', 'CJT-DT0', 'CRD-PNI', 'NN1-NP0', 'NN1-VVB', 'NN1-VVG', 'NN2-VVZ', 'VVD-VVN']


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
    >>> wt = []
    >>> _initial_probability(wt)
    """
    tag_first_dict = dict.fromkeys(ALL_TAGS, 0)
    tag_count_dict = dict.fromkeys(ALL_TAGS, 0)
    # init the transition matrix
    transition_dict = dict.fromkeys(ALL_TAGS, dict.fromkeys(ALL_TAGS, 0))

    # print(tag_first_dict)
    word, tag = word_tag[0]
    tag_first_dict[tag] += 1
    tag_count_dict[tag] += 1
    for i in range(1, len(word_tag)):
        prev_word, prev_tag = word_tag[i - 1][0], word_tag[i - 1][1]
        word, tag = word_tag[i][0], word_tag[i][1]
        prev_tag = _ambiguity_check(prev_tag)
        tag = _ambiguity_check(tag)

        transition_dict[tag][prev_tag] += 1     # tag comes after prev_tag
        if prev_word in END_SENTENCE_SIG:
            # tag is at the beginning of the sentence
            tag_first_dict[tag] += 1
        tag_count_dict[tag] += 1

    ## Initial Probability
    # Compute the probability of each tag starts at a sentence
    for k in tag_first_dict:
        if tag_count_dict[k] != 0:
            tag_first_dict[k] /= tag_count_dict[k]
    print(tag_first_dict)

    ## Compute the transition probability



def tag(training_list, test_file, output_file):
    # Tag the words from the untagged input file and write them into the output file.
    # Doesn't do much else beyond that yet.
    print("Tagging the file.")

    # read each training file and put them into a list of word-tag pairs
    # and update the probabilities
    full_training_list = []
    for train_file in training_list:
        full_training_list.extend(read_file_to_list(train_file))
    print(len(full_training_list))
    for i in range(len(full_training_list)):
        full_training_list[i] = full_training_list[i].replace(" ", "").strip().split(':')
        if len(full_training_list[i]) > 2 and full_training_list[i][-1] == "PUN":
            full_training_list[i] = [':', 'PUN']
    print(full_training_list[0:13])
    # Find the initial probability matrix
    _initial_probability(full_training_list)


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
