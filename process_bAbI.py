import os
import glob
import pickle
import itertools
import numpy as np
from collections import defaultdict
from sortedcontainers import SortedDict, SortedSet

from misc import count_pred_true

def get_words_from_txt(path):
    words = []
    if path is not None:
        with open(path, "r") as f:
            for line in f:
                words += line.lower().strip().split(",")
    return set(words)


def get_babi_paths(folder, flag):
    txt_dict = SortedDict()
    for path in glob.glob(os.path.join(folder, "*_" + flag + ".txt")):
        key = int(os.path.basename(path).split("_")[0][2:])
        txt_dict[key] = path
    return txt_dict


def replace_multiple(val, item_dict):
    for k, v in item_dict.items():
        val = val.replace(k, v)
    return val


def get_babi_paths_dict(folder):
    return get_babi_paths(folder, "train"), get_babi_paths(folder, "test")


def parse_babi_line(line, useless_words, replace_dict):
    line = line.lower().strip().split("\t")
    line = (replace_multiple(item, replace_dict) for item in line)
    line = (word for item in line for word in item.strip().split(" ") if word not in useless_words)
    line = (int(word) if word.isnumeric() else word for word in line)

    return tuple(line)


def extract_data_from_line(q_type, index, line, is_question):

    if q_type in [1, 2]:
        if is_question:
            return (index, ) + line[::-1] + (None, ),
        else:
            return (index, ) + line,
    elif q_type == 3:
        if is_question:
            return (index, line[2], line[0], line[3]), (index, ) + (line[1], line[0], None)
        else:
            return (index, ) + line,
    elif q_type == 4:
        if is_question:
            if line[1] in ("north", "west", "south", "east"):
                return (index, line[1], "of", line[2]), (index, None) + line[0:2]
            else:
                return (index, line[1], line[0], line[2]), (index, line[2], "of", None)
        else:
            return (index, ) + line[0:3], (index, line[2], "of", line[3])
    elif q_type == 5:
        if is_question:
            if len(line) == 2:
                return (index, None) + line,
            else:
                return (index, ) + line, (index, ) + line[:2] + (None, )
        else:
            if len(line) == 3:
                return (index, ) + line,
            else:
                return (index, ) + line[0:3], (index, ) + line[0:2] + line[3:]
    elif q_type == 6:
        if is_question:
            return (index, line[1], line[0], line[2]), (index, None, line[0], line[2])
        else:
            return (index, ) + line,
    elif q_type == 7:
        if is_question:
            return (index,) + line[2:] + line[0:1], (index,) + (None, "many", "objects")
        else:
            if len(line) == 3:
                return (index,) + line,
            else:
                return (index,) + line[0:3], (index,) + line[0:2] + line[3:]
    elif q_type == 8:
        if is_question:
            return (index,) + line[1:] + (None, ),
        else:
            return (index, ) + line,
    elif q_type == 9:
        if is_question:
            return (index, line[1], line[0], line[2]), (index, None, line[0], line[2])
        else:
            if len(line) == 4:
                return (index, line[0], line[1], line[3]), (index, line[0], line[1], line[2])
            else:
                return (index,) + line,
    elif q_type == 10:
        if is_question:
            return (index, line[1], line[0], line[2]), (index, None, line[0], line[2])
        else:
            if len(line) == 3:
                return (index, ) + line,
            else:
                return (index, ) + line[0:2] + line[3:4], (index, ) + line[0:2] + line[5:]
    elif q_type == 11:
        if is_question:
            return (index, line[1], line[0], None),
        else:
            return (index,) + line,
    elif q_type == 12:
        if is_question:
            return (index, line[1], line[0], None),
        else:
            return (index, line[0]) + line[3:], (index, line[2]) + line[3:]
    elif q_type == 13:
        if is_question:
            return (index, line[1], line[0], None),
        else:
            if len(line) == 3:
                return (index,) + line,
            else:
                return (index, line[0]) + line[3:], (index, line[2]) + line[3:]
    elif q_type == 14:
        if is_question:
            return (index, line[2], line[0], line[3]), (index, line[1], line[0], None)
        else:
            if line[0] in ("afternoon", "evening", "yesterday", "before", "morning"):
                return (index, line[1], line[2], line[0]), (index, ) + line[1:]
            else:
                return (index, line[0], line[1], line[-1]), (index, ) + line[0:3]
    elif q_type == 15:
        if is_question:
            return (index, line[1], line[0], line[2]), (index, line[1], line[2], None)
        else:
            return (index, ) + line,
    elif q_type == 16:
        if is_question:
            return (index, line[2], line[1], line[0]), (index, ) + line[0:2] + (None, )
        else:
            return (index,) + line,
    elif q_type == 17:

        line = tuple(w for w in line if w != "is")
        first_half, second_half = line[0:2], line[-2:]
        position = [w for w in line if w in ('above', 'below', 'left', 'right')][0]

        if first_half[0] in ('blue', 'pink', 'yellow', 'red'):
            first_obj = first_half[1]
            first_half = ((index, first_half[1], "is", first_half[0]),)
        else:
            first_obj = first_half[0]
            first_half = ()

        if second_half[-2] in ('blue', 'pink', 'yellow', 'red'):
            second_obj = second_half[-1]
            second_half = ((index, second_half[1], "is", second_half[0]),)
        else:
            second_obj = second_half[-1]
            second_half = ()

        if is_question:
            return first_half + second_half + ((index, first_obj, position, second_obj), (index, None, 'is', position))
        else:
            return first_half + ((index, first_obj, position, second_obj), ) + second_half

    elif q_type == 18:
        if 'chocolates' in line:
            box_of_chocolates = ((index,) + ('box', 'of', 'chocolates'),)
            line = tuple(w for w in line if w != "chocolates")
        else:
            box_of_chocolates = ()

        if is_question:
            line = tuple(w for w in line if w != "is")
            return box_of_chocolates + ((index, ) + line, (index, None, 'is', line[2]))
        else:
            return box_of_chocolates + ((index, ) + line[:3], (index, line[0]) + line[2:4])
    elif q_type == 19:
        if is_question:
            return (index, line[2], line[0], line[3]), (index, "you", "go", None)
        else:
            return (index, line[0], line[2], line[3]),
    elif q_type == 20:
        if is_question:
            if len(line) == 2:
                return (index, ) + line + (None, ),
            else:
                return (index, ) + line, (index, "why", line[1], None)
        else:
            return (index, ) + line,


def get_story(f, q_type, useless_words, replace_dict, words, prev_line=None):

    cur_idx = 1 if prev_line is not None else 0
    prev_line = prev_line if prev_line is not None else parse_babi_line(next(f), useless_words, replace_dict)

    story = SortedDict()
    questions = []
    answers = []
    relevant = []

    lines = itertools.chain((prev_line, ), (parse_babi_line(line, useless_words, replace_dict) for line in f))

    finished = False

    while True:

        line = next(lines, None)

        if line is None:
            finished = True
            break

        line_idx = line[0]
        if line_idx < cur_idx:
            break

        cur_idx = line_idx

        if isinstance(line[-1], int):

            relevant_idx = len(line) - 2
            while isinstance(line[relevant_idx], int):
                relevant_idx -= 1

            question = line[:relevant_idx]
            answer = tuple(line[relevant_idx].split(","))
            relevant_data = line[relevant_idx + 1:]

            data = extract_data_from_line(q_type, question[0], question[1:], True)

            questions.append(data)
            answers.append(answer)
            relevant.append(sorted(relevant_data))

            words.update((w for d in data for w in d[1:] if w is not None))
            words.update(answer)

            for d in data:
                assert len(d) == 4, data

            if q_type == 3:
                print("QUESTION:", data)
                print("ANSWER:", answer)

        else:

            data = extract_data_from_line(q_type, line[0], line[1:], False)
            for d in data:
                assert len(d) == 4, data
            words.update((w for d in data for w in d[1:] if w is not None))
            story[data[0][0]] = data

            if q_type == 3:
                print("FACT:", data)

    return story, tuple(questions), tuple(answers), tuple(relevant), line, finished


def iterate_babi_file(path, q_type, useless_words, replace_dict, words):

    with open(path, "r") as f:
        prev_line = None
        stories = []
        finished = False
        while not finished:
            story, questions, answers, relevant, prev_line, finished = \
                get_story(f, q_type, useless_words, replace_dict, words, prev_line=prev_line)
            stories.append((story, questions, answers, relevant))
    return stories


def process_babi(input_folder, output_pkl_file, useless_words, replace_dict):

    train_paths, test_paths = get_babi_paths(input_folder, "train"), get_babi_paths(input_folder, "test")

    train_dict = SortedDict()
    test_dict = SortedDict()
    words = SortedSet()

    for q_type, path in train_paths.items():
        train_dict[q_type] = iterate_babi_file(path, q_type, useless_words, replace_dict, words)
    for q_type, path in test_paths.items():
        test_dict[q_type] = iterate_babi_file(path, q_type, useless_words, replace_dict, words)

    pickle.dump(train_dict, open(output_pkl_file + "train.pkl", "wb"))
    pickle.dump(test_dict, open(output_pkl_file + "test.pkl", "wb"))
    pickle.dump(words, open(output_pkl_file + "words.pkl", "wb"))


def word_variable_or_constant(variables, word, words):
    return variables[word] if variables is not None and word in variables else "w" + str(words.index(word))


def sentences_and_answer_to_rule(story, question, answer, relevant, words, rules):

    relevant_sentences = tuple(story[r] for r in relevant)
    sentence_and_question_words = set(w for sentence in relevant_sentences for triple in sentence for w in triple[1:])
    sentence_and_question_words.update((w for triple in question for w in triple[1:]))

    for ans in answer:

        counts = defaultdict(int)

        question_words = set(w for w in question[-1][1:] if w is not None)
        question_words.add(ans)

        for word in itertools.chain(*(triple[1:] for sentence in relevant_sentences for triple in sentence),
                                    *(triple[1:] for triple in question), (ans, )):
            if word is not None:
                counts[word] += 1

        variables = {}
        cur_variable = "a"

        for k, v in counts.items():
            if v > 1:
                variables[k] = cur_variable
                cur_variable = chr(ord(cur_variable) + 1)

        time = len(relevant_sentences) - 1
        antecedents = ""

        for sentence in relevant_sentences:
            for triple in sentence:

                antecedents += "I[t"+str(time)+","
                antecedents += word_variable_or_constant(variables, triple[1], words) + ","
                antecedents += word_variable_or_constant(variables if triple[2] in question_words else None, triple[2], words) + ","
                antecedents += word_variable_or_constant(variables, triple[3], words)
                antecedents += "]"

            time -= 1

        for triple in question[:-1]:

            antecedents += "Q["
            antecedents += word_variable_or_constant(variables, triple[1], words) + ","
            antecedents += word_variable_or_constant(variables if triple[2] in question_words else None, triple[2], words) + ","
            antecedents += word_variable_or_constant(variables, triple[3], words)
            antecedents += "]"

        head = "A["
        head += word_variable_or_constant(variables, question[-1][1] or ans, words) + ","
        head += word_variable_or_constant(variables if question[-1][2] in sentence_and_question_words else None, question[-1][2], words) + ","
        head += word_variable_or_constant(variables, question[-1][3] or ans, words)
        head += "]"

        rules.add("einsum " + head + ":-" + antecedents)


def sentences_and_answer_to_data(q_type, story, question, answer, relevant, words, data):

    relevant_sentences = tuple(story[r] for r in relevant)

    time = len(relevant_sentences) - 1

    story_data = []

    for sentence in relevant_sentences:
        for triple in sentence:
            s_data = np.array([time] + [words.index(word) for word in triple[1:]])
            assert s_data.shape == (4,), triple
            story_data.append(np.array([time] + [words.index(word) for word in triple[1:]]))
        time -= 1

    story_data = np.array(story_data).T

    question_data = []
    for triple in question[:-1]:
        question_data.append(np.array([words.index(word) for word in triple[1:]]))
    question_data = np.array(question_data).reshape((-1, 3)).T

    answer_data = np.array([words.index(word) for word in answer])

    query = question[-1]
    query_data = dict(q1=words.index(query[1]) if query[1] is not None else None,
                      q2=words.index(query[2]) if query[2] is not None else None,
                      q3=words.index(query[3]) if query[3] is not None else None)

    data.append((q_type, story_data, question_data, answer_data, query_data))


def triples_to_rules(pkl_file, program_path, data_path, weighted_rules):

    train_dict = SortedDict(pickle.load(open(pkl_file + "train.pkl", "rb")))
    test_dict = SortedDict(pickle.load(open(pkl_file + "test.pkl", "rb")))
    words = SortedSet(pickle.load(open(pkl_file + "words.pkl", "rb")))

    rules = set()

    for q_type, items in train_dict.items():
        for story, questions, answers, relevants in items:
            for question, answer, relevant in zip(questions, answers, relevants):
                sentences_and_answer_to_rule(story, question, answer, relevant, words, rules)

    new_rules = []
    if weighted_rules:
        for i, rule in enumerate(rules):
            lhs, rhs = rule.split(":-")
            new_rules.append(lhs + ":-W" + str(i) + "[]" + rhs)
    else:
        new_rules = rules

    new_rules = tuple(new_rules)

    with open(program_path, "w") as f:

        f.write("range int " + " ".join("w" + str(i) for i in range(len(words))) + "\n")
        f.write("range int " + " ".join("t" + str(i) for i in range(8)) + "\n")
        f.write("\n")

        f.write("range coords (iv.0, iv.1, iv.2, iv.3)\n")
        f.write("range coords (qv.0, qv.1, qv.2)\n")
        f.write("\n")

        f.write("tensor I shape=(8, 128, 128, 128)\n")
        f.write("tensor Q shape=(128, 128, 128)\n")
        f.write("tensor A shape=(128, 128, 128)\n")
        f.write("\n")

        f.write("input I[iv.0, iv.1, iv.2, iv.3] :- \n")
        f.write("input Q[qv.0, qv.1, qv.2] :- \n")
        f.write("\n")

        for rule in new_rules:
            f.write(rule + "\n")

        f.write("\n")

    data = []

    for q_type, items in test_dict.items():
        for story, questions, answers, relevants in items:
            for question, answer, relevant in zip(questions, answers, relevants):
                sentences_and_answer_to_data(q_type, story, question, answer, relevant, words, data)

    np.random.shuffle(data)
    pickle.dump(data, open(data_path + "_test.pkl", "wb"))


if __name__ == "__main__":

    babi_folder = "/home/ovmurad/Documents/Projects/TensorLog/tasks_1-20_v1-2/en"
    output_folder = "/home/ovmurad/Documents/Projects/TensorLog"
    output_pkl_file_ = "/home/ovmurad/Documents/Projects/TensorLog/triples_"
    program_path_ = "/home/ovmurad/PycharmProjects/tensorlog/bAbI_program_no_weights.txt"
    data_path_ = "/home/ovmurad/PycharmProjects/tensorlog/bAbI_data"

    useless_words_ = get_words_from_txt(os.path.join(output_folder, "ignore_words.txt"))
    composite_words_ = get_words_from_txt(os.path.join(output_folder, "composite_words.txt"))

    replace_dict_ = {w: "".join(w.split(" ")) for w in composite_words_}
    replace_dict_.update({"?": "", ".": ""})

    process_babi(babi_folder, output_pkl_file_, useless_words_, replace_dict_)
    # triples_to_rules(output_pkl_file_, program_path_, data_path_, weighted_rules=False)