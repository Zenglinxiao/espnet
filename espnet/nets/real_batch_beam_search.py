import math
import torch
from typing import List, Tuple, Dict, IntNumber
import logging
from .batch_beam_search import BatchHypothesis, BatchBeamSearch
from .beam_search import BeamSearch, Hypothesis

from espnet.nets.e2e_asr_common import end_detect


class BatchBeamHypothesis(BatchHypothesis):
    """Batchfied/Vectorized hypothesis data type."""

    ids: torch.Tensor = torch.tensor([])  # (batch*beam,)
    x: torch.Tensor = torch.tensor([])  # (batch, beam, T, D)
    yseq: torch.Tensor = torch.tensor([])  # (batch, beam, maxlen)
    score: torch.Tensor = torch.tensor([])  # (batch, beam)
    length: torch.Tensor = torch.tensor([])  # (batch, beam)
    scores: Dict[str, torch.Tensor] = dict()  # values: (batch, beam)
    states: Dict[str, List[List]] = dict()

    def __len__(self) -> int:
        """Return number of hypothesis."""
        return self.length.numel()

    @classmethod
    def from_batch_hyps(
        cls, batch_hyps: BatchHypothesis, x: torch.Tensor, beam_size: IntNumber
    ) -> "BatchBeamHypothesis":
        """Build BatchBeamHypothesis from BatchHypothesis with ids.

        Repeat batch_hyps with beam_size times and enrich it with original
        batch index and encoded feature x as additional information to form
        a BatchBeamHypothesis instance.

        Args:
            batch_hyps (BatchHypothesis): (batch, *)
            x (Tensor): (batch, T, D)
            beam_size (int)

        Returns:
            BatchBeamHypothesis
                .ids: (batch * beam, *)
                .x: (batch, beam, T, D)
                .yseq: (batch, beam, seq_len)
                .score, .length: (batch, beam)
                .scores: Dict[str, Tensor(batch, beam)]
                .states: Dict[str, List[List](batch, beam)]

        """
        bsz, T, D = x.size()
        assert len(batch_hyps) == bsz, "batch_hyps mismatch x!"
        ids = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)\
                   .to(x.device).long()  # (bsz*beam_size)
        running_x = x.index_select(0, ids).view(bsz, beam_size, T, D)
        # yseq: (batch, seq_len) -> (batch, beam, seq_len)
        yseq = batch_hyps.yseq.index_select(0, ids).view(bsz, beam_size, -1)
        # score, length: (batch) -> (batch, beam)
        score = batch_hyps.score.index_select(0, ids).view(bsz, beam_size)
        length = batch_hyps.length.index_select(0, ids).view(bsz, beam_size)
        # scores: Dict[str, Tensor(batch)] -> Dict[str, Tensor(batch, beam)]
        scores = {
            k: v.index_select(0, ids).view(bsz, beam_size)
            for k, v in batch_hyps.scores.items()
        }
        # states: Dict[str, List[Tensor](batch)] -> Dict[str, List[List](batch, beam)]
        states = {
            k: [[hyp_state for _ in range(beam_size)] for hyp_state in v]
            for k, v in batch_hyps.states.items()
        }
        return cls(
            ids=ids,
            x=running_x,
            yseq=yseq,
            score=score,
            length=length,
            scores=scores,
            states=states
        )

    def index_reorder(self, index: torch.Tensor) -> "BatchBeamHypothesis":
        """Select hypothese path indicated by 1-D tensor index."""
        assert len(index.size()) == 1, "index should be 1-D tensor."
        assert index.size().item() == len(self), "number of path not match."
        _new_ids = self.ids.index_select(0, index)  # (bsz*beam)
        bsz, beam, T, D = self.x.size()
        # (batch, beam, T, D)  -> (batch*beam, T, D) -> (batch, beam, T, D)
        _new_x = self.x.view(-1, T, D).index_select(0, index).view(-1, beam, T, D)
        # yseq: (batch, beam, seq_len) -> (batch*beam, seq_len) -> (batch, beam, seq_len)
        bsz_y, beam_y, seq_len = self.yseq.size()
        assert bsz == bsz_y and beam == bsz_y, "Error: size mismatch!"
        _new_yseq = self.yseq.view(-1, seq_len).index_select(0, index).view(-1, beam, seq_len)
        # score, length: (batch, beam) -> (batch*beam) -> (batch, beam)
        _new_score = self.score.view(-1).index_select(0, index).view(-1, beam)
        _new_length = self.length.view(-1).index_select(0, index).view(-1, beam)
        # scores: Dict[str, Tensor(batch, beam) -> (batch*beam) -> (batch, beam)]
        _new_scores = {
            k: v.view(-1).index_select(0, ids).view(bsz, beam)
            for k, v in self.scores.items()
        }
        # states: Dict[str, List[List](batch, beam) -> (batch, beam)]
        _new_states = {
            k: [[v[path_id // beam][path_id % beam] for path_id in batch_ids]
                for batch_ids in index]
            for k, v in self.states.items()
        }
        return BatchBeamHypothesis(
            ids=_new_ids,
            x=_new_x,
            yseq=_new_yseq,
            score=_new_score,
            length=_new_length,
            scores=_new_scores,
            states=_new_states
        )

    # def append_tokens(self, token_ids: torch.Tensor) -> None:
    #     """Append token_ids as new predicted token.

    #     Args:
    #         token_ids (torch.Tensor): (batch, beam) candidate token index for each path
    #             of the batched beam.

    #     """
    #     bsz, beam = token_ids.size()
    #     bsz_y, beam_y, seq_len = self.yseq.size()
    #     assert bsz == bsz_y and beam == beam_y, "size not match."
    #     self.yseq = torch.cat((self.yseq, token_ids.unsqueeze(-1)), dim=-1)
    #     self.length += 1

    def is_finished(self, eos_id):
        """Return index of finished path."""
        return self.yseq[:, :, -1].eq(eos_id)  # (batch, beam)

    def get_hypothesis_at(self, path_id: IntNumber):
        """Return the path_id Hypothesis.

        Args:
            path_id (int): hypothesis path index, ranging in [0, batch*beam)
        """
        bsz, beam = self.length.size()
        batch_id = path_id // beam
        beam_id = path_id % beam
        return Hypothesis(
            yseq=self.yseq[batch_id][beam_id],
            score=self.score[batch_id][beam_id],
            scores={k: v[batch_id][beam_id] for k, v in self.scores.items()},
            states={k: v[batch_id][beam_id] for k, v in self.states.items()}
        )

    def get_alive_batch(self, alive_batch_id: List[int]):
        """Choose the subset of hypothesis that will continue.

        Args:
            alive_batch_id (List[int]): (n_alived_sent,)

        """
        bsz, beam = self.length.size()
        _new_scores = {
            k: v[alive_batch_id]
            for k, v in self.scores.items()
        }
        _new_states = {
            k: [v[batch_id] for batch_id in alive_batch_id]
            for k, v in self.states.items()
        }
        return BatchBeamHypothesis(
            ids=self.ids.view(bsz, beam)[alive_batch_id].view(-1),
            x=self.x[alive_batch_id],
            yseq=self.yseq[alive_batch_id],
            score=self.score[alive_batch_id],
            length=self.length[alive_batch_id],
            scores=_new_scores,
            states=_new_states
        )


class RealBatchBeamSearch(BatchBeamSearch):
    """Batch beam search to decode B example at once."""

    def init_hyp(self, x: torch.Tensor) -> BatchBeamHypothesis:
        """Get an initial hypothesis data.

        Args:
            x (torch.Tensor): The encoder output feature (B, T, D)

        Returns:
            BatchBeamHypothesis: The initial hypothesis of the batch.

        """
        # NOTE: this may not compatible for CTCPrefixScorer
        # As they get different CTCPrefixScore/CTCPrefixScoreTH
        batch_hyp = []
        for ex_x in x:
            # ex_x (T, D) of each example
            init_states = dict()
            init_scores = dict()
            for k, d in self.scorers.items():
                init_states[k] = d.init_state(ex_x)
                init_scores[k] = 0.0
            # each example have one begin hypothesis
            batch_hyp.append(
                Hypothesis(
                    score=0.0,
                    scores=init_scores,
                    states=init_states,
                    yseq=torch.tensor([self.sos], device=x.device),
                )
            )
        batch_hyp = self.batchfy(batch_hyp)
        return BatchBeamHypothesis.from_batch_hyps(
            batch_hyp, x, self.beam_size
        )

    def initialize(
        self, x: torch.Tensor
    ) -> Tuple[BatchBeamHypothesis, List[bool], List[List[Hypothesis]]]:
        """Initialize for parallel beam search.

        Args:
            x (torch.Tensor): Encoded speech feature (bsz, T, D)

        Returns:
            running_hyps (): hypothsis of the batch (bsz, beam_size, *)
            finished (BoolTensor): ended batch hypothsis (bsz,), True when all
                beam of sentence i ended
            finalized (List[List[Hypothese]]): list of hypothese that finalized.

        """
        bsz, T, D = x.size()
        running_hyps = self.init_hyp(x)  # (batch, *)
        finished = [False for _ in range(bsz)]
        finalized = [[] for _ in range(bsz)]
        return running_hyps, finished, finalized

    # def score_full(
    #     self, hyp: BatchHypothesis, x: torch.Tensor
    # ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    #     """Score new hypothesis by `self.full_scorers`.

    #     Args:
    #         hyp (Hypothesis): Hypothesis with prefix tokens to score
    #         x (torch.Tensor): Corresponding input feature (B, T, D)

    #     Returns:
    #         Tuple[Dict[str, torch.Tensor], Dict[str, Any]]: Tuple of
    #             score dict of `hyp` that has string keys of `self.full_scorers`
    #             and tensor score values of shape: `(self.n_vocab,)`,
    #             and state dict that has string keys
    #             and state values of `self.full_scorers`

    #     """
    #     scores = dict()
    #     states = dict()
    #     for k, d in self.full_scorers.items():
    #         scores[k], states[k] = d.batch_score(hyp.yseq, hyp.states[k], x)
    #     return scores, states

    # def score_partial(
    #     self, hyp: BatchHypothesis, ids: torch.Tensor, x: torch.Tensor
    # ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    #     """Score new hypothesis by `self.full_scorers`.

    #     Args:
    #         hyp (BatchHypothesis): Hypothesis with prefix tokens to score
    #         ids (torch.Tensor): 2D tensor of new partial tokens to score
    #             (B, pre_beam_size)
    #         x (torch.Tensor): Corresponding input feature (B, T, D)

    #     Returns:
    #         Tuple[Dict[str, torch.Tensor], Dict[str, Any]]: Tuple of
    #             score dict of `hyp` that has string keys of `self.full_scorers`
    #             and tensor score values of shape: `(self.n_vocab,)`,
    #             and state dict that has string keys
    #             and state values of `self.full_scorers`

    #     """
    #     scores = dict()
    #     states = dict()
    #     for k, d in self.part_scorers.items():
    #         # FIXME: x in this function is 2D but we have 3D: for loop ?
    #         # NOTE: x is actually not used in ctx & no other stuff need this
    #         scores[k], states[k] = d.batch_score_partial(
    #             hyp.yseq, ids, hyp.states[k], x
    #         )
    #     return scores, states

    def batch_beam(
        self, weighted_scores: torch.Tensor, ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Batch-compute topk full token ids and partial token ids.

        Args:
            weighted_scores (torch.Tensor): The weighted sum scores for each tokens.
                Its shape is `(batch, beam, self.vocab_size)`.
            ids (torch.Tensor): The partial token ids to compute topk.
                Its shape is `(batch, beam, self.pre_beam_size)`.

        # Returns: FIXME partial part
        #     Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        #         The topk full (prev_hyp, new_token) ids
        #         and partial (prev_hyp, new_token) ids.
        #         Their shapes are all `(self.beam_size,)`

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                The topk hypothese path ids and its new token ids.
                Their shapes are `(batch, self.beam_size,)`

        """
        # top_ids = weighted_scores.topk(self.beam_size, dim=-1)[1]  # (B, beam_size)
        bsz, beam, _ = weighted_scores.size()
        top_scores, top_ids = weighted_scores.view(bsz, -1).topk(self.beam_size, dim=-1)
        batch_beam_ids = top_ids // self.n_vocab  # (batch, beam) top beam path
        new_token_ids = top_ids % self.n_vocab  # (batch, beam)
        return top_scores, batch_beam_ids, new_token_ids
        # # Because of the flatten above, `top_ids` is organized as:
        # # [hyp1 * V + token1, hyp2 * V + token2, ..., hypK * V + tokenK],
        # # where V is `self.n_vocab` and K is `self.beam_size`
        # prev_hyp_ids = top_ids // self.n_vocab
        # new_token_ids = top_ids % self.n_vocab
        # return prev_hyp_ids, new_token_ids, prev_hyp_ids, new_token_ids

    def search(
        self, running_hyps: BatchBeamHypothesis, x: torch.Tensor
    ) -> BatchBeamHypothesis:
        """Search new tokens for running hypotheses and encoded speech x.

        Args:
            running_hyps (BatchBeamHypothesis): Running hypotheses on beam
            x (torch.Tensor): Encoded speech feature (batch, beam, T, D)

        Returns:
            BatchHypothesis: Best sorted hypotheses (B*beam_size, Ty+1)

        """
        batch_size = running_hyps.x.size(0)
        n_path = len(running_hyps)
        part_ids = None  # no pre-beam
        # NOTE 1. Calculating Score
        # batch scoring: (batch*beam, V)
        weighted_scores = torch.zeros(
            n_path, self.n_vocab, dtype=x.dtype, device=x.device
        )
        # (batch*beam, V)
        scores, states = self.score_full(running_hyps, x)  # FIXME
        for k in self.full_scorers:
            weighted_scores += self.weights[k] * scores[k]
        # partial scoring
        if self.do_pre_beam:
            pre_beam_scores = (
                weighted_scores
                if self.pre_beam_score_key == "full"
                else scores[self.pre_beam_score_key]
            )
            # part_ids: (B, pre_beam_size)
            part_ids = torch.topk(pre_beam_scores, self.pre_beam_size, dim=-1)[1]
            # NOTE: part_ids.view(batch_size, -1, self.pre_beam_size)
        # NOTE(takaaki-hori): Unlike BeamSearch, we assume that score_partial returns
        # full-size score matrices, which has non-zero scores for part_ids and zeros
        # for others.
        part_scores, part_states = self.score_partial(running_hyps, part_ids, x)  # FIXME
        for k in self.part_scorers:
            weighted_scores += self.weights[k] * part_scores[k]
        # add previous hyp scores
        weighted_scores += running_hyps.score.view(-1).unsqueeze(1)

        # # TODO(karita): do not use list. use batch instead
        # # see also https://github.com/espnet/espnet/pull/1402#discussion_r354561029
        # # update hyps
        # best_hyps = []
        # prev_hyps = self.unbatchfy(running_hyps)
        # for (
        #     full_prev_hyp_id,
        #     full_new_token_id,
        #     part_prev_hyp_id,
        #     part_new_token_id,
        # ) in zip(*self.batch_beam(weighted_scores, part_ids)):
        #     prev_hyp = prev_hyps[full_prev_hyp_id]
        #     best_hyps.append(
        #         Hypothesis(
        #             score=weighted_scores[full_prev_hyp_id, full_new_token_id],
        #             yseq=self.append_token(prev_hyp.yseq, full_new_token_id),
        #             scores=self.merge_scores(
        #                 prev_hyp.scores,
        #                 {k: v[full_prev_hyp_id] for k, v in scores.items()},
        #                 full_new_token_id,
        #                 {k: v[part_prev_hyp_id] for k, v in part_scores.items()},
        #                 part_new_token_id,
        #             ),
        #             states=self.merge_states(
        #                 {
        #                     k: self.full_scorers[k].select_state(v, full_prev_hyp_id)
        #                     for k, v in states.items()
        #                 },
        #                 {
        #                     k: self.part_scorers[k].select_state(
        #                         v, part_prev_hyp_id, part_new_token_id
        #                     )
        #                     for k, v in part_states.items()
        #                 },
        #                 part_new_token_id,
        #             ),
        #         )
        #     )

        # Replacement: batch implementation
        # NOTE 2. Rank Score at batch level: each sentence got beam candidate
        # (batch, beam), (batch, beam), (batch, beam)
        top_scores, batch_beam_ids, token_ids = self.batch_beam(
            weighted_scores.view(batch_size, -1, self.n_vocab),
            part_ids
        )
        batch_beam_offset = torch.arange(
            0, batch_size * self.beam_size, step=self.beam_size).to(batch_beam_ids)  # (batch)
        hyp_ids = batch_beam_ids + batch_beam_offset.unsqueeze(1)  # (batch, beam) + (batch, 1)

        # NOTE 3. update running_hyps
        # reorder by hyp_ids: choose beam selected path
        reordered_hyps = running_hyps.index_reorder(hyp_ids.view(-1))
        # append last prediction: (batch, beam, seq_len) + (batch, beam, 1)
        reordered_hyps.yseq = torch.cat(
            (reordered_hyps.yseq, token_ids.unsqueeze(-1)), dim=-1)
        reordered_hyps.length += 1
        # update score as chosen by beam
        reordered_hyps.score = top_scores
        # FIXME running_hyps.scores/states updated by reorder but need change
        # if we need also partial to merge XX to reordered_hyp
        return reordered_hyps

    def forward(
        self, x: torch.Tensor, maxlenratio: float = 0.0, minlenratio: float = 0.0
    ) -> List[List[Hypothesis]]:
        """Perform real batch beam search.

        Args:
            x (torch.Tensor): Encoded speech feature (B, T, D)
            maxlenratio (float): Input length ratio to obtain max output length.
                If maxlenratio=0.0 (default), it uses a end-detect function
                to automatically find maximum hypothesis lengths
            minlenratio (float): Input length ratio to obtain min output length.

        Returns:
            List[List[Hypothesis]]: N-best decoding results for batch_size sentences

        """
        # set length bounds:
        # FIXME this may need to be changed as ensure_min/max_length
        if maxlenratio == 0:
            maxlen = x.shape[1]  # NOTE: or -2
        else:
            maxlen = max(1, int(maxlenratio * x.size(1)))
        minlen = int(minlenratio * x.size(1))
        logging.info("decoder input length: " + str(x.shape[1]))
        logging.info("max output length: " + str(maxlen))
        logging.info("min output length: " + str(minlen))

        # init running states
        running_hyps, finished, finalized = self.initialize(x)

        # main loop of prefix search
        for i in range(maxlen):
            logging.debug("position " + str(i))
            best = self.search(running_hyps, running_hyps.x)  # (B0, *), (B0, T, D) -> (B0, *)
            # post process of one iteration: remove ended hypothsis, get remain running hyps
            # update ended_hyps, maintain running hyps, etc...
            running_hyps = self.post_process(
                i, maxlen, maxlenratio, best, finalized, finished)
            # end detection
            if all(finished):
                assert len(running_hyps) == 0, "all finish but running_hyps not empty."
                logging.info(f"all batch ended")
                break
            # FIXME: not sure understand this end_detect logic
            # if maxlenratio == 0.0 and end_detect([h.asdict() for h in finalized], i):
            #     logging.info(f"end detected at {i}")
            #     break
            if len(running_hyps) == 0:
                logging.info("no hypothesis. Finish decoding.")
                break
            else:
                logging.debug(f"remained hypotheses: {len(running_hyps)}")

        # sort finalized hypothesis by score (descending)
        best_finalized = [
            sorted(hyps, key=lambda x: x.score, reverse=True) for hyps in finalized
        ]
        assert all([len(hyps) >= self.beam_size for hyps in best_finalized]), \
            "Not enough hypothesis generated."
        # prune to contain only {beam_size} hypothesis for each sentence
        if any([len(hyps) > self.beam_size for hyps in best_finalized]):
            best_finalized = [hyps[:self.beam_size] for hyps in best_finalized]
        return best_finalized

    def post_process(self, i, maxlen, maxlenratio, running_hyps, ended_hyps, finished):
        """Perform post-processing of beam search iterations.

        Args:
            i (int): The length of hypothesis tokens.
            maxlen (int): The maximum length of tokens in beam search.
            maxlenratio (int): The maximum length ratio in beam search.
            running_hyps (BatchBeamHypothesis): The running hypotheses in beam search.
            ended_hyps (List[List[Hypothesis]]): The ended hypotheses in beam search.
            finished (List[bool]): flags indicate if one sentence is finished.

        Returns:
            BatchHypothesis: The new running hypotheses.

        """
        batch_size = running_hyps.x.size(0)
        n_hyps = len(running_hyps)  # =batch*beam
        logging.debug("the number of running: hypotheses {} / {} sentence.".format(
            n_hyps, batch_size))
        # if self.token_list is not None:
        #     logging.debug(
        #         "best hypo: "
        #         + "".join([self.token_list[x] for x in running_hyps[0].yseq[1:]])
        #     )
        # add eos in the final loop to avoid that there are no ended hyps
        if i == maxlen - 1:
            logging.info("adding <eos> in the last position in the loop")
            # eos: (bsz, beam, 1), running_hyps: (bsz, beam, seq_len)
            eos_ids = torch.full(running_hyps.yseq.size()[:-1], self.eos).unsqueeze(-1)
            # running_hyps.append_eos(eos_ids)
            running_hyps.yseq = torch.cat((running_hyps.yseq, eos_ids), dim=-1)

        # add ended hypotheses to a final list, and removed them from current hypotheses
        # (this will be a problem, number of hyps < beam)
        # path_newly_finished = running_hyps.is_finished(eos=self.eos)  # (batch, beam)
        path_newly_finished_mask = (
            running_hyps.yseq[:, :, -1].eq(self.eos_id)
            & running_hyps.score.ne(-math.inf)
        )

        # NOTE add to result when path finish, reduce runing batch when batch finish
        # only reduce batch when one sentence have enough candidate in result.
        if path_newly_finished_mask.any():
            # 1. add newly finished path to ended_hyps
            # any path newly finished
            path_newly_finished_id = torch.mask_select(
                torch.arange(0, n_hyps).to(running_hyps.ids),
                mask=path_newly_finished_mask.view(-1)
            )
            # map path id to original sentence id
            path_sent_id = running_hyps.ids.index_select(
                0, index=path_newly_finished_id
            )
            assert path_newly_finished_id.size() == path_sent_id.size(), \
                "path_newly_finished_id, path_sent_id size mismatch."
            # add finished hypothesis to ended_hyps
            finished_batch_id = set()  # [0, batch_size)
            for path_id, sent_id in zip(
                    path_newly_finished_id.tolist(), path_sent_id.tolist()):
                finished_hyp = running_hyps.get_hypothesis_at(path_id)
                # TODO finalize hyp with partial's score?
                ended_hyps[sent_id].append(finished_hyp)
                # sentence finish when it has beam_size finished hypothesis
                if len(ended_hyps[sent_id]) >= self.beam_size:
                    finished_batch_id.update(path_id // self.beam_size)
                    finished[sent_id] = True  # set finish flag to True
            # NOTE mask score of finished path to -inf to avoid be selected
            # again by beam
            running_hyps = running_hyps.score.masked_fill(
                mask=path_newly_finished_mask, value=-math.inf)
            if len(finished_batch_id) > 0:
                # 2. remove finished batch from runing hypothesis
                batch_id_alive = [
                    i for i in range(batch_size) if i not in finished_batch_id]
                if len(batch_id_alive) == 0:
                    logging.debug(f"All batch finished.")
                    return BatchBeamHypothesis()
                running_hyps = running_hyps.get_alive_batch(batch_id_alive)
                new_bsz = running_hyps.length.size(0)
                logging.debug(f"Update finished batch: {batch_size}->{new_bsz}.")
            if running_hyps.score.eq(-math.inf).all(dim=-1).any():
                # (batch, beam) -> (batch,) -> 1
                raise RuntimeError("sentence are finished but not removed from beam!")
        return running_hyps
        # remained_hyps = []
        # for hyp in running_hyps:
        #     if hyp.yseq[-1] == self.eos:
        #         # e.g., Word LM needs to add final <eos> score
        #         for k, d in chain(self.full_scorers.items(), self.part_scorers.items()):
        #             s = d.final_score(hyp.states[k])
        #             hyp.scores[k] += s
        #             hyp = hyp._replace(score=hyp.score + self.weights[k] * s)
        #         ended_hyps.append(hyp)
        #     else:
        #         remained_hyps.append(hyp)
        # return remained_hyps
