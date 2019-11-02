import torch as T
import numpy as np
from mlmo.utils.helpers.search import traverse_table


class BeamSearch(object):
    """Wrapper over ONMT beam search that works over batches."""

    def __init__(self, batch_size, beam_size, start_id, end_id, device='cpu',
                 min_lens=None, sample=False, len_norm=False, n_best=1,
                 excl_ids=None):
        """
        :param sample: whether to sample or take arg-max over word scores.
        :param min_lens: minimum lengths of sequences. One per batch unit.
        """
        if min_lens is not None and \
                not isinstance(min_lens, (list, np.ndarray)):
            raise ValueError("Please provide a list/array of minimum lengths!")

        if sample:
            raise NotImplementedError
        self.batch_size = batch_size
        self.beam_size = beam_size
        self.device = device
        if min_lens is None:
            min_lens = [0] * batch_size
        self._beams = [_BeamSearch(beam_size=beam_size, start_id=start_id,
                                   end_id=end_id, device=device,
                                   len_norm=len_norm, excl_ids=excl_ids,
                                   n_best=n_best, min_length=min_lens[i])
                       for i in range(batch_size)]

    def advance(self, word_log_probs):
        """
        :param word_log_probs: [batch_size, beam_size, vocab_size]
                               log probabilities over next words.
        """
        assert word_log_probs.size(0) == self.batch_size
        for i in range(self.batch_size):
            beam = self._beams[i]
            if not beam.done():
                beam.advance(word_log_probs[i])

    def get_current_state(self):
        """
        :return: [batch_size, beam_size]
                 current selected candidate (word) ids.
        """
        word_ids = T.zeros((self.batch_size, self.beam_size), dtype=T.int64,
                           device=self.device)
        for i in range(self.batch_size):
            word_ids[i] = self._beams[i].get_current_state()
        return word_ids

    def get_current_origin(self):
        """
        Returns current back-pointers to the previous time-step. Can be used
        to shuffle hidden states.
        :return: [batch_size, beam_size]
        """
        coll = T.zeros((self.batch_size, self.beam_size), dtype=T.int64,
                       device=self.device)
        for i in range(self.batch_size):
            coll[i] = self._beams[i].get_current_origin()
        return coll

    def get_finished_best(self, minimum=None, **kwargs):
        """
        Returns an array of the best completed hypothesis (word ids lists).
        Each starts with 'start_id', and ends with 'end_id'.
        Optionally, traverses additional parameters (**kwargs) that are passed
        using back-pointers. E.g., some artifacts produced by the decoder.

        :param kwargs: dict of arrays/tensors [steps, batch_size * beam_size, x]
        """
        best_seqs = np.empty(self.batch_size, dtype='object')
        new_kwargs = {k: np.empty(self.batch_size, dtype='object')
                      for k in kwargs}
        for i in range(self.batch_size):
            beam = self._beams[i]
            _, ts_and_ks = beam.sort_finished(minimum=minimum)

            # traversing both the internal word id table and additional params
            if len(ts_and_ks):
                t, k = ts_and_ks[0]
                best_seqs[i] = traverse_table(time_step=t, beam_indx=k,
                                              back_pointers=beam.back_pointers,
                                              elems_table=beam.selected_ids)

                for pname, pval in kwargs.items():
                    start_indx = self.beam_size * i
                    beam_params = pval[:, start_indx:
                                          (start_indx + self.beam_size)]
                    # it's t-1 because when the <END> token is generated
                    # the decoder stops.
                    best_params = traverse_table(time_step=t-1, beam_indx=k,
                                                 back_pointers=beam.back_pointers,
                                                 elems_table=beam_params)
                    new_kwargs[pname][i] = best_params
            else:
                best_seqs[i] = []
                for pname in kwargs:
                    new_kwargs[pname][i] = []

        if kwargs:
            return best_seqs, new_kwargs
        return best_seqs

    def done(self):
        all_beams_done = all([beam.done() for beam in self._beams])
        return all_beams_done


class _BeamSearch(object):
    def __init__(self, beam_size, start_id, end_id, n_best=1, device='cpu',
                 min_length=0, len_norm=False, excl_ids=None):
        self.beam_size = beam_size

        self.hyp_scores = T.zeros(beam_size, dtype=T.float32, device=device)

        # The back-pointers at each time-step
        self.back_pointers = []

        # The outputs at each time-step
        self.selected_ids = [T.full(size=(beam_size,), fill_value=start_id,
                                    device=device, dtype=T.int64)]

        # Has EOS topped the beam yet
        self._end_id = end_id
        self.end_id_top = False

        # Time and k pair for finished
        self.finished = []
        self.n_best = n_best

        # Minimum prediction length
        self.min_length = min_length

        self.len_norm = len_norm

        self.excl_ids = excl_ids

    def get_current_state(self):
        "Get the outputs for the current timestep."
        return self.selected_ids[-1]

    def get_current_origin(self):
        "Get the backpointers for the current timestep."
        return self.back_pointers[-1]

    def advance(self, word_log_probs):
        """
        Given log-prob over words for every last beam `wordLk` and attention
        :param word_log_probs: [K, vocab_size]
        :return: True if beam search is complete.
        """
        num_words = word_log_probs.size(1)

        # force the output to be longer than self.min_length
        cur_len = len(self.selected_ids)
        if cur_len < self.min_length:
            for k in range(len(word_log_probs)):
                word_log_probs[k][self._end_id] = -1e20

        # Sum the previous scores.
        if len(self.back_pointers):

            # excluding words from being considered by setting their scores to
            # a very small value such that they would not be selected
            if self.excl_ids:
                word_log_probs[:, self.excl_ids] = -1e20

            # here we're summing log-probs of histories and next words
            beam_scores = word_log_probs + self.hyp_scores.unsqueeze(1)

            # Don't let EOS have children.
            for i in range(self.selected_ids[-1].size(0)):
                if self.selected_ids[-1][i] == self._end_id:
                    beam_scores[i] = -1e20
        else:
            beam_scores = word_log_probs[0]
        flat_beam_scores = beam_scores.view(-1)

        best_scores, best_scores_id = flat_beam_scores.topk(self.beam_size,
                                                            0, True, True)

        self.hyp_scores = best_scores

        # best_scores_id is flattened beam x word array, so calculate which
        # word and beam score came from
        prev_k = best_scores_id / num_words
        self.back_pointers.append(prev_k)
        self.selected_ids.append((best_scores_id - prev_k * num_words))

        for i in range(self.selected_ids[-1].size(0)):
            if self.selected_ids[-1][i] == self._end_id:
                s = self.hyp_scores[i]
                self.finished.append((s, len(self.selected_ids) - 1, i))

        # End condition is when top-of-beam is EOS and no global score
        # only applicable when lengths are not normalized
        if not self.len_norm and self.selected_ids[-1][0] == self._end_id:
            self.end_id_top = True

    def done(self):
        if self.len_norm:
            return len(self.finished) >= self.n_best
        else:
            return self.end_id_top and len(self.finished) >= self.n_best

    def sort_finished(self, minimum=None):
        """
        :param minimum: the minimum number of hypotheses to add to the beam,
                        even if they are incomplete.
        """
        if minimum is not None:
            i = 0
            # Add from beam until we have minimum outputs.
            while len(self.finished) < minimum:
                s = self.hyp_scores[i]
                self.finished.append((s, len(self.selected_ids) - 1, i))
                i += 1

        if self.len_norm:
            self.finished = [(sc/float(ln), ln, i) for sc, ln, i
                             in self.finished]

        self.finished.sort(key=lambda a: -a[0])

        scores = [sc for sc, _, _ in self.finished]
        time_step_and_k = [(t, k) for _, t, k in self.finished]
        return scores, time_step_and_k