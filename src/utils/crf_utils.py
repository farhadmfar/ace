import torch
import numpy as np
from typing import List

class Span:
    """
    A class of `Span` where we use it during evaluation.
    We construct spans for the convenience of evaluation.
    """
    def __init__(self, left: int, right: int, type: str):
        """
        A span compose of left, right (inclusive) and its entity label.
        :param left:
        :param right: inclusive.
        :param type:
        """
        self.left = left
        self.right = right
        self.type = type

    def __eq__(self, other):
        return self.left == other.left and self.right == other.right and self.type == other.type

    def __hash__(self):
        return hash((self.left, self.right, self.type))

def log_sum_exp_pytorch(vec: torch.Tensor) -> torch.Tensor:
    """
    Calculate the log_sum_exp trick for the tensor.
    :param vec: [batchSize * from_label * to_label].
    :return: [batchSize * to_label]
    """
    maxScores, idx = torch.max(vec, 1)
    maxScores[maxScores == -float("Inf")] = 0
    maxScoresExpanded = maxScores.view(vec.shape[0] ,1 , vec.shape[2]).expand(vec.shape[0], vec.shape[1], vec.shape[2])
    return maxScores + torch.log(torch.sum(torch.exp(vec - maxScoresExpanded), 1))



def evaluate_batch_insts(batch_pred_ids: torch.LongTensor,
                         batch_gold_ids: torch.LongTensor,
                         word_seq_lens: torch.LongTensor,
                         idx2label: List[str]) -> np.ndarray:
    """
    Evaluate a batch of instances and handling the padding positions.
    :param batch_insts:  a batched of instances.
    :param batch_pred_ids: Shape: (batch_size, max_length) prediction ids from the viterbi algorithm.
    :param batch_gold_ids: Shape: (batch_size, max_length) gold ids.
    :param word_seq_lens: Shape: (batch_size) the length for each instance.
    :param idx2label: The idx to label mapping.
    :return: numpy array containing (number of true positive, number of all positive, number of true positive + number of false negative)
             You can also refer as (number of correctly predicted entities, number of entities predicted, number of entities in the dataset)
    """
    idx2label = list(idx2label.values())
    p = 0
    total_entity = 0
    total_predict = 0
    word_seq_lens = word_seq_lens.tolist()
    predictions = []
    golds = []
    p_partial = 0
    total_entity_partial = 0
    total_predict_partial = 0
    for idx in range(len(batch_pred_ids)):
        length = word_seq_lens[idx]
        output = batch_gold_ids[idx][:length].tolist()
        prediction = batch_pred_ids[idx][:length].tolist()

        prediction = prediction[::-1]

        output = [idx2label[l] for l in output]


        prediction =[idx2label[l] for l in prediction]
        
        
        output_spans = set()
        start = -1
        for i in range(len(output)):
            if output[i].startswith("B-"):
                start = i
            if output[i].startswith("E-"):
                end = i
                output_spans.add(Span(start, end, output[i][2:]))
            if output[i].startswith("S-"):
                output_spans.add(Span(i, i, output[i][2:]))
        predict_spans = set()
        for i in range(len(prediction)):
            if prediction[i].startswith("B-"):
                start = i
            if prediction[i].startswith("E-"):
                end = i
                predict_spans.add(Span(start, end, prediction[i][2:]))
            if prediction[i].startswith("S-"):
                predict_spans.add(Span(i, i, prediction[i][2:]))

        
        for o_s in list(output_spans): total_entity_partial += o_s.right - o_s.left + 1
        for p_s in list(predict_spans): total_predict_partial += p_s.right - p_s.left + 1
        
        for o_s in list(output_spans):
            for p_s in list(predict_spans):
                if o_s.type == p_s.type:
                        l_o = set([i for i in range(o_s.left, o_s.right + 1)])
                        l_p = set([i for i in range(p_s.left, p_s.right + 1)])
                        p_partial += len(l_p.intersection(l_o))
                    
        
        total_entity += len(output_spans)
        total_predict += len(predict_spans)
        p += len(predict_spans.intersection(output_spans))

        # In case you need the following code for calculating the p/r/f in a batch.
        # (When your batch is the complete dataset)
        # precision = p * 1.0 / total_predict * 100 if total_predict != 0 else 0
        # recall = p * 1.0 / total_entity * 100 if total_entity != 0 else 0
        # fscore = 2.0 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0


    return np.asarray([p, total_predict, total_entity, p_partial, total_predict_partial, total_entity_partial], dtype=int)
        
