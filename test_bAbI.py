import pickle

import torch
import numpy as np

from program import Program
from structs import QueryData


def bAbI():

    data = pickle.load(open("bAbI_data_test.pkl", "rb"))

    p = Program(program_path="bAbI_program_no_weights.txt")
    p.compile(constant_ranges=dict(**{"w" + str(i): i for i in range(0, 128)},
                                   **{"t" + str(i): i for i in range(0, 8)}))

    counts = np.array([0] * 20, dtype=np.float32)
    correct = np.array([0] * 20, dtype=np.float32)

    for i, (q_type, story_data, question_data, answer_data, query_data) in enumerate(data):

        input_ranges = dict(iv=story_data, qv=question_data)
        query_data = QueryData(tensor="A", domain_tuple=("q1", "q2", "q3"), domain_vals=query_data)

        answer = p.run(input_ranges=input_ranges,
                       input_tensors=dict(I=True, Q=True),
                       queries=[query_data])[0]

        if answer.ndim > 0:
            counts[q_type - 1] += 1
            if np.allclose(answer_data, torch.topk(answer, len(answer_data))[1]):
                correct[q_type - 1] += 1
            print(np.stack([counts, correct]))

bAbI()