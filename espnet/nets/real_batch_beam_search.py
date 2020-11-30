import math
import torch
from typing import List, Tuple, Any, Dict, Optional, NamedTuple
import logging
from .batch_beam_search import BatchBeamSearch
from .beam_search import BeamSearch, Hypothesis

from espnet.nets.e2e_asr_common import end_detect


class BatchBeamHypothesis(NamedTuple):
    """Batchfied/Vectorized hypothesis data type."""

    ids: torch.Tensor = torch.tensor([])  # (batch*beam,)
    x: torch.Tensor = torch.tensor([])  # (batch, beam, T, D)
    yseq: torch.Tensor = torch.tensor([])  # (batch, beam, maxlen)
    score: torch.Tensor = torch.tensor([])  # (batch, beam)
    length: torch.Tensor = torch.tensor([])  # (batch, beam)
    scores: Dict[str, torch.Tensor] = dict()  # values: (batch*beam)
    states: Dict[str, List[List]] = dict()  # values: (batch*beam)

    def __len__(self) -> int:
        """Return number of hypothesis."""
        return self.length.numel()

    # @classmethod
    # def from_batch_hyps(
    #     cls, batch_hyps: BatchHypothesis, x: torch.Tensor, beam_size: int
    # ) -> "BatchBeamHypothesis":
    #     """Build BatchBeamHypothesis from BatchHypothesis with ids.

    #     Repeat batch_hyps with beam_size times and enrich it with original
    #     batch index and encoded feature x as additional information to form
    #     a BatchBeamHypothesis instance.

    #     Args:
    #         batch_hyps (BatchHypothesis): (batch, *)
    #         x (Tensor): (batch, T, D)
    #         beam_size (int)

    #     Returns:
    #         BatchBeamHypothesis
    #             .ids: (batch * beam, *)
    #             .x: (batch, beam, T, D)
    #             .yseq: (batch, beam, seq_len)
    #             .score, .length: (batch, beam)
    #             .scores: Dict[str, Tensor(batch, beam)]
    #             .states: Dict[str, List[List](batch, beam)]

    #     """
    #     bsz, T, D = x.size()
    #     assert len(batch_hyps) == bsz, "batch_hyps mismatch x!"
    #     ids = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)\
    #                .to(x.device).long()  # (bsz*beam_size)
    #     running_x = x.index_select(0, ids).view(bsz, beam_size, T, D)
    #     # yseq: (batch, seq_len) -> (batch, beam, seq_len)
    #     yseq = batch_hyps.yseq.index_select(0, ids).view(bsz, beam_size, -1)
    #     # score, length: (batch) -> (batch, beam)
    #     score = batch_hyps.score.index_select(0, ids).view(bsz, beam_size)
    #     length = batch_hyps.length.index_select(0, ids).view(bsz, beam_size)
    #     # scores: Dict[str, Tensor(batch)] -> Dict[str, Tensor(batch, beam)]
    #     scores = {
    #         k: v.index_select(0, ids).view(bsz, beam_size)
    #         for k, v in batch_hyps.scores.items()
    #     }
    #     # states: Dict[str, List[Tensor](batch)] -> Dict[str, List[List](batch, beam)]
    #     states = {
    #         k: [[hyp_state for _ in range(beam_size)] for hyp_state in v]
    #         for k, v in batch_hyps.states.items()
    #     }
    #     return cls(
    #         ids=ids,
    #         x=running_x,
    #         yseq=yseq,
    #         score=score,
    #         length=length,
    #         scores=scores,
    #         states=states
    #     )

    # def index_reorder(self, index: torch.Tensor) -> "BatchBeamHypothesis":
    #     """Select hypothese path indicated by 1-D tensor index."""
    #     assert len(index.size()) == 1, "index should be 1-D tensor."
    #     # assert index.size().item() == len(self), "number of path not match."
    #     _new_ids = self.ids.index_select(0, index)  # (bsz*beam)
    #     bsz, beam, T, D = self.x.size()
    #     # (batch, beam, T, D)  -> (batch*beam, T, D) -> (batch, beam, T, D)
    #     _new_x = self.x.view(-1, T, D).index_select(0, index).view(-1, beam, T, D)
    #     # yseq: (batch, beam, seq_len) -> (batch*beam, seq_len) -> (batch, beam, seq_len)
    #     bsz_y, beam_y, seq_len = self.yseq.size()
    #     assert bsz == bsz_y and beam == bsz_y, "Error: size mismatch!"
    #     _new_yseq = self.yseq.view(-1, seq_len).index_select(0, index).view(-1, beam, seq_len)
    #     # score, length: (batch, beam) -> (batch*beam) -> (batch, beam)
    #     _new_score = self.score.view(-1).index_select(0, index).view(-1, beam)
    #     _new_length = self.length.view(-1).index_select(0, index).view(-1, beam)
    #     # scores: Dict[str, Tensor(batch, beam) -> (batch*beam) -> (batch, beam)]
    #     _new_scores = {
    #         k: v.view(-1).index_select(0, index).view(bsz, beam)
    #         for k, v in self.scores.items()
    #     }
    #     # states: Dict[str, List[List](batch, beam) -> (batch, beam)]
    #     _new_states = {
    #         k: [[v[path_id // beam][path_id % beam] for path_id in batch_ids]
    #             for batch_ids in index]
    #         for k, v in self.states.items()
    #     }
    #     return BatchBeamHypothesis(
    #         ids=_new_ids,
    #         x=_new_x,
    #         yseq=_new_yseq,
    #         score=_new_score,
    #         length=_new_length,
    #         scores=_new_scores,
    #         states=_new_states
    #     )

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

    # def is_finished(self, eos_id):
    #     """Return index of finished path."""
    #     return self.yseq[:, :, -1].eq(eos_id)  # (batch, beam)

    # def get_hypothesis_at(self, path_id: int):
    #     """Return the path_id Hypothesis.

    #     Args:
    #         path_id (int): hypothesis path index, ranging in [0, batch*beam)
    #     """
    #     bsz, beam = self.length.size()
    #     batch_id = path_id // beam
    #     beam_id = path_id % beam
    #     return Hypothesis(
    #         yseq=self.yseq[batch_id][beam_id],
    #         score=self.score[batch_id][beam_id],
    #         scores={k: v[batch_id][beam_id] for k, v in self.scores.items()},
    #         states={k: v[batch_id][beam_id] for k, v in self.states.items()}
    #     )

    # def get_alive_batch(self, alive_batch_id: List[int]):
    #     """Choose the subset of hypothesis that will continue.

    #     Args:
    #         alive_batch_id (List[int]): (n_alived_sent,)

    #     """
    #     bsz, beam = self.length.size()
    #     _new_scores = {
    #         k: v[alive_batch_id]
    #         for k, v in self.scores.items()
    #     }
    #     _new_states = {  # FIXME
    #         k: [v[batch_id] for batch_id in alive_batch_id]
    #         for k, v in self.states.items()
    #     }
    #     return BatchBeamHypothesis(
    #         ids=self.ids.view(bsz, beam)[alive_batch_id].view(-1),
    #         x=self.x[alive_batch_id],
    #         yseq=self.yseq[alive_batch_id],
    #         score=self.score[alive_batch_id],
    #         length=self.length[alive_batch_id],
    #         scores=_new_scores,
    #         states=_new_states
    #     )

    def _path_view(self):
        """Return a new instance with path view (batch*beam, *)."""
        bsz, beam, T, D = self.x.size()
        bsz_y, beam_y, seq_len = self.yseq.size()
        assert bsz == bsz_y and beam == beam_y, "Error: size mismatch!"
        # _new_scores = {  # scores: Dict[str, Tensor(batch, beam)
        #     k: v.view(-1) for k, v in self.scores.items()
        # }
        # _new_states = {  # states: Dict[str, List[List](batch, beam)]
        #     k: [path_state for batch_state in v for path_state in batch_state]
        #     for k, v in self.states.items()
        # }
        return BatchBeamHypothesis(
            ids=self.ids,
            x=self.x.view(-1, T, D),
            yseq=self.yseq.view(-1, seq_len),
            score=self.score.view(-1),
            length=self.length.view(-1),
            scores=self.scores,
            states=self.states  # _new_states
        )

    def _standard_view(self, batch_size):
        """Return to standard batch beam view."""
        n_path, T, D = self.x.size()
        n_path_y, seq_len = self.yseq.size()
        assert n_path == n_path_y, "Error: size mismatch!"
        beam = n_path // batch_size
        assert n_path % batch_size == 0, "n_path should be dividable by batch_size"
        # _new_scores = {  # scores: Dict[str, Tensor(batch, beam)
        #     k: v.view(batch_size, beam) for k, v in self.scores.items()
        # }
        # _new_states = {  # states: Dict[str, List[List](batch, beam)]
        #     k: [
        #         [v[batch_id * batch_size + beam_id] for beam_id in range(beam)]
        #         for batch_id in range(batch_size)
        #     ]
        #     for k, v in self.states.items()
        # }
        return BatchBeamHypothesis(
            ids=self.ids,
            x=self.x.view(batch_size, beam, T, D),
            yseq=self.yseq.view(batch_size, beam, seq_len),
            score=self.score.view(batch_size, beam),
            length=self.length.view(batch_size, beam),
            scores=self.scores,
            states=self.states
        )


class RealBatchBeamSearch(BatchBeamSearch):
    """Batch beam search to decode B example at once."""

    def _select(self, hyps: BatchBeamHypothesis, path_id: int) -> Hypothesis:
        """Return a single Hypothesis from hyps in path index i."""
        path_hyps = hyps._path_view()
        return Hypothesis(
            yseq=path_hyps.yseq[path_id, : path_hyps.length[path_id]],
            score=path_hyps.score[path_id],
            scores={k: v[path_id] for k, v in path_hyps.scores.items()},
            states={
                k: self.scorers[k].select_state(v, path_id)
                for k, v in path_hyps.states.items()
            },
        )

    def _batch_select(self, hyps: BatchBeamHypothesis, batch_ids: List[int]) -> BatchBeamHypothesis:
        batch_size, beam_size = hyps.length.size()
        new_batch_size = len(batch_ids)
        path_ids = [
            i for rg in (
                range(bid * beam_size, (bid + 1) * beam_size) for bid in batch_ids)
            for i in rg]
        assert len(path_ids) == new_batch_size * beam_size, "_batch_select error."
        return BatchBeamHypothesis(
            ids=hyps.ids[path_ids],
            x=hyps.x[batch_ids],
            yseq=hyps.yseq[batch_ids],
            score=hyps.score[batch_ids],
            length=hyps.length[batch_ids],
            scores={k: v[path_ids] for k, v in hyps.scores.items()},
            states={
                k: [self.scorers[k].select_state(v, i) for i in path_ids]
                for k, v in hyps.states.items()
            },
        )

    def init_hyp(self, x: torch.Tensor) -> BatchBeamHypothesis:
        """Get an initial hypothesis data.

        Args:
            x (torch.Tensor): The encoder output feature (B, T, D)

        Returns:
            BatchBeamHypothesis: The initial hypothesis of the batch.

        """
        # NOTE: this is compatible for CTCPrefixScorer
        # As they get different CTCPrefixScore/CTCPrefixScoreTH
        # FIXME: CTCPrefixScorer need to initialize CTC module in the score
        # therefore can only be called once for init_state,
        # we need to switch this to one single batch_init_state & feed it with
        # whole x (B, T, D), each step, select right path for ctc is needed
        # but this is not supported in current CTC module!!!
        # NOTE: the change is tricy for CTC, but for others like transformer decoder
        # this seems easy, as batch_init_state return None just as init_state
        # TODO candidate replace:
        # ...
        # init_states = dict()
        # init_scores = dict()
        # for k, d in self.scorers.items():
        #     init_states[k] = d.batch_init_state(x)
        #     init_scores[k] = [0.0 for _ in range(len(x))]
        # ...
        batch_hyp = []
        for ex_x in x:
            # ex_x (T, D) of each example
            init_states = dict()
            init_scores = dict()
            for k, d in self.scorers.items():
                init_states[k] = d.batch_init_state(ex_x)
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
        bsz, T, D = x.size()
        return BatchBeamHypothesis(
            ids=torch.arange(bsz).to(x.device).long(),  # (bsz*beam)
            x=x.unsqueeze(1),  # (bsz, T, D) -> (bsz, beam, T, D)
            yseq=batch_hyp.yseq.unsqueeze(1),  # (batch, S) -> # (batch, beam, S)
            score=batch_hyp.score.unsqueeze(1).to(x.device),  # (batch,) -> # (batch, beam)
            length=batch_hyp.length.unsqueeze(1).to(x.device),  # (batch,) -> # (batch, beam)
            scores=batch_hyp.scores,  # (bsz*beam,)
            states=batch_hyp.states,  # (bsz*beam,)
        )
        # return BatchBeamHypothesis.from_batch_hyps(
        #     batch_hyp, x, self.beam_size
        # )

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

    def score_full(
        self, hyp: BatchBeamHypothesis, x: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Score new hypothesis by `self.full_scorers`.

        Args:
            hyp (BatchBeamHypothesis): BatchBeamHypothesis with prefix tokens to score
            x (torch.Tensor): Corresponding input feature (batch, beam, T, D)

        Returns:
            Tuple[Dict[str, torch.Tensor], Dict[str, Any]]: Tuple of
                score dict of `hyp` that has string keys of `self.full_scorers`
                and tensor score values of shape: `(batch*beam, self.n_vocab,)`,
                and state dict that has string keys
                and state values of `self.full_scorers`

        """
        bsz, beam = hyp.length.size()
        bsz_x, beam_x, T, D = x.size()
        assert bsz == bsz_x, "mismatch batch size"
        assert beam == beam_x, "mismatch beam size"
        n_path = bsz * beam
        # 1. flatten hyp from (batch, beam, *) -> (batch*beam, *)
        yseq = hyp.yseq.view(n_path, -1)
        # currunt_states = {
        #     k: [hyp_state for batch_state in v for hyp_state in batch_state]
        #     for k, v in hyp.states.items()
        # }
        flatten_x = x.view(n_path, T, D)
        # 2. got scores/states in shape (batch*beam, *)
        scores = dict()
        states = dict()
        for k, d in self.full_scorers.items():
            scores[k], states[k] = d.batch_score(yseq, hyp.states[k], flatten_x)
        return scores, states

    # def score_partial(  FIXME
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

    def pre_batch_beam(self, scores: torch.Tensor):
        """Returns pre_beam part_ids."""
        batch_size, beam_size, n_vocab = scores.size()
        top_ids = torch.topk(scores.view(batch_size, -1), self.pre_beam_size, dim=-1)[1]
        batch_beam_ids = top_ids // self.n_vocab  # (batch, pre_beam_size) top beam path
        new_token_ids = top_ids % self.n_vocab  # (batch, pre_beam_size)
        return batch_beam_ids, new_token_ids

    # def merge_batch_scores(
    #     self,
    #     prev_scores: Dict[str, float],
    #     next_full_scores: Dict[str, torch.Tensor],
    #     # next_part_scores: Optional[Dict[str, torch.Tensor]] = None
    # ) -> torch.Tensor:
    #     """Merge scores of current step into running scores."""
    #     new_scores = dict()
    #     for k, v in next_full_scores.items():
    #         new_scores[k] = prev_scores[k] + v
    #     # for k, v in next_part_scores.items():
    #     #     new_scores[k] = prev_scores[k] + v
    #     return new_scores

    # def _path_select(
    #     self, running_hyps: BatchBeamHypothesis, index: torch.Tensor
    # ) -> BatchBeamHypothesis:
    #     """Select hypothese path indicated by 1-D tensor index."""
    #     assert len(index.size()) == 1, "index should be 1-D tensor."
    #     assert index.size().item() == len(running_hyps), "number of path not match."
    #     # 1. flatten batch, beam by path
    #     bsz, beam, T, D = running_hyps.x.size()
    #     running_hyps = running_hyps._path_view()
    #     # 2. select by path index
    #     _new_ids = running_hyps.ids.index_select(0, index)  # (bsz*beam)
    #     _new_x = running_hyps.x.index_select(0, index)
    #     _new_yseq = running_hyps.yseq.index_select(0, index)
    #     _new_score = running_hyps.score.index_select(0, index)
    #     _new_length = running_hyps.length.index_select(0, index)
    #     _new_scores = {
    #         k: v.view(-1).index_select(0, index).view(bsz, beam)
    #         for k, v in running_hyps.scores.items()
    #     }
    #     _new_states = {
    #         k: [path_state for path_state in v]
    #         for k, v in running_hyps.states.items()
    #     }
    #     running_hyps = BatchBeamHypothesis(
    #         ids=_new_ids,
    #         x=_new_x,
    #         yseq=_new_yseq,
    #         score=_new_score,
    #         length=_new_length,
    #         scores=_new_scores,
    #         states=_new_states
    #     )
    #     # 3. reshape to (batch, beam) view
    #     _new_ids = running_hyps.ids.index_select(0, index)  # (bsz*beam)
    #     bsz, beam, T, D = running_hyps.x.size()
    #     # (batch, beam, T, D)  -> (batch*beam, T, D) -> (batch, beam, T, D)
    #     _new_x = running_hyps.x.view(-1, T, D).index_select(0, index).view(bsz, -1, T, D)
    #     # yseq: (batch, beam, seq_len) -> (batch*beam, seq_len) -> (batch, beam, seq_len)
    #     bsz_y, beam_y, seq_len = running_hyps.yseq.size()
    #     assert bsz == bsz_y and beam == bsz_y, "Error: size mismatch!"
    #     _new_yseq = running_hyps.yseq.view(-1, seq_len).index_select(0, index).view(-1, beam, seq_len)
    #     # score, length: (batch, beam) -> (batch*beam) -> (batch, beam)
    #     _new_score = running_hyps.score.view(-1).index_select(0, index).view(-1, beam)
    #     _new_length = running_hyps.length.view(-1).index_select(0, index).view(-1, beam)
    #     # scores: Dict[str, Tensor(batch, beam) -> (batch*beam) -> (batch, beam)]
    #     _new_scores = {
    #         k: v.view(-1).index_select(0, index).view(bsz, beam)
    #         for k, v in running_hyps.scores.items()
    #     }
    #     # states: Dict[str, List[List](batch, beam) -> (batch, beam)]
    #     _new_states = {
    #         k: [[v[path_id // beam][path_id % beam] for path_id in batch_ids]
    #             for batch_ids in index]
    #         for k, v in running_hyps.states.items()
    #     }
    #     return BatchBeamHypothesis(
    #         ids=_new_ids,
    #         x=_new_x,
    #         yseq=_new_yseq,
    #         score=_new_score,
    #         length=_new_length,
    #         scores=_new_scores,
    #         states=_new_states
    #     )

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
        batch_size, beam, T, D = running_hyps.x.size()
        n_path = len(running_hyps)
        # part_ids = None  # no pre-beam

        # NOTE 1. Calculating Score
        # batch scoring: (batch*beam, V)
        weighted_scores = torch.zeros(
            n_path, self.n_vocab, dtype=x.dtype, device=x.device
        )
        # (batch*beam, V)
        scores, states = self.score_full(running_hyps, x)
        for k in self.full_scorers:
            weighted_scores += self.weights[k] * scores[k]
        # partial scoring:
        # assert not self.do_pre_beam, "PartialScorer is not supported yet."
        # if self.do_pre_beam:
        #     pre_beam_scores = (
        #         weighted_scores
        #         if self.pre_beam_score_key == "full"
        #         else scores[self.pre_beam_score_key]
        #     )
        #     # NOTE: reshape to (batch_size, -1) -> topk as batch_beam
        #     part_batch_beam_ids, part_new_token_ids = self.pre_batch_beam(
        #         pre_beam_scores.view(batch_size, self.beam_size, self.n_vocab))
        #     # batch_beam_id -> path_id
        #     part_batch_beam_offset = torch.arange(
        #         0, batch_size * self.beam_size, step=self.beam_size
        #     ).to(part_batch_beam_ids)  # (batch)
        #     # (batch, pre_beam_size) + (batch, 1)
        #     part_hyp_ids = part_batch_beam_ids + part_batch_beam_offset.unsqueeze(1)
        #     # DELETE THESE
        #     # part_ids: (B, pre_beam_size)
        #     # part_ids = torch.topk(pre_beam_scores, self.pre_beam_size, dim=-1)[1]
        #     # NOTE: part_ids.view(batch_size, -1, self.pre_beam_size)
        # else:
        #     part_hyp_ids, part_new_token_ids = None, None
        # # NOTE(takaaki-hori): Unlike BeamSearch, we assume that score_partial returns
        # # full-size score matrices, which has non-zero scores for part_ids and zeros
        # # for others.
        # part_scores, part_states = self.score_partial(
        #     running_hyps, part_hyp_ids, part_new_token_ids, x)  # FIXME
        # for k in self.part_scorers:
        #     weighted_scores += self.weights[k] * part_scores[k]

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

        # NOTE 2. Rank Score at batch level: each sentence got self.beam_size candidate
        # (batch, self.beam_size), (batch, self.beam_size), (batch, self.beam_size)
        top_scores, batch_beam_ids, token_ids = self.batch_beam(
            weighted_scores.view(batch_size, -1, self.n_vocab),
            ids=None  # not use in definition
        )
        batch_beam_offset = torch.arange(
            0, batch_size * beam, step=beam).to(batch_beam_ids)  # (batch)
        hyp_ids = batch_beam_ids + batch_beam_offset.unsqueeze(1)  # (batch, beam) + (batch, 1)
        # partial reuse full as in batch_beam_search.py: see its batch_beam method
        # part_hyp_ids, part_token_ids = hyp_ids, token_ids

        # Replacement: batch implementation
        # Loop to merge score & states using merge_scores/merge_states
        _hyps_scores, _hyps_states = [], []  # List[Dict]
        for hyp_id, new_token_id in zip(hyp_ids.view(-1), token_ids.view(-1)):
            hyp_scores = self.merge_scores(
                {k: v[hyp_id] for k, v in running_hyps.scores.items()},  # (batch*beam,) -> (1)
                {k: v[hyp_id] for k, v in scores.items()},  # (batch*beam, V) -> (V)
                new_token_id,
                {},
                new_token_id,
            )
            hyp_states = self.merge_states(
                {
                    k: self.full_scorers[k].select_state(v, hyp_id)
                    for k, v in states.items()
                },  # (batch*beam)
                {},
                new_token_id,
            )
            _hyps_scores.append(hyp_scores)
            _hyps_states.append(hyp_states)
        batch_scores = {k: torch.tensor([_scores[k] for _scores in _hyps_scores]) for k in self.scorers}
        batch_states = {k: [_states[k] for _states in _hyps_states] for k in self.scorers}

        # NOTE 3. update running_hyps
        # # reorder by hyp_ids: choose beam selected path
        # reordered_hyps = running_hyps.index_reorder(hyp_ids.view(-1))
        # # append last prediction: (batch, beam, seq_len) + (batch, beam, 1)
        # reordered_hyps.yseq = torch.cat(
        #     (reordered_hyps.yseq, token_ids.unsqueeze(-1)), dim=-1)
        # reordered_hyps.length += 1
        # # # update score as chosen by beam
        # reordered_hyps.score = top_scores
        # reordered_hyps.scores = batch_scores
        # reordered_hyps.states = batch_states
        # return reordered_hyps

        # get batch hypothesis info by hyp_ids
        batch_ids = running_hyps.ids.index_select(0, hyp_ids.view(-1))   # (bsz*beam)
        batch_x = running_hyps.x.view(batch_size * beam, T, D).index_select(0, hyp_ids.view(-1))
        batch_yseq = running_hyps.yseq.view(batch_size * beam, -1).index_select(0, hyp_ids.view(-1))
        batch_length = running_hyps.length.view(batch_size * beam).index_select(0, hyp_ids.view(-1))
        new_bsz, new_beam = hyp_ids.size()
        assert new_bsz == batch_size, "batch size should match!"
        assert new_beam == self.beam_size, "each batch should have beam_size path after search."
        # update prediction
        new_yseq = torch.cat(
            (batch_yseq.view(new_bsz, new_beam, -1), token_ids.unsqueeze(-1)), dim=-1)
        new_length = batch_length.view(new_bsz, new_beam) + 1
        return BatchBeamHypothesis(
            ids=batch_ids,
            x=batch_x.view(new_bsz, new_beam, T, D),
            yseq=new_yseq,
            score=top_scores,
            length=new_length,
            scores=batch_scores,
            states=batch_states
        )

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
            # FIXME: not sure understand this end_detect logic (relate to CTC)
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
        assert all((len(hyps) >= self.beam_size for hyps in best_finalized)), \
            "Not enough hypothesis generated."
        # prune to contain only {beam_size} hypothesis for each sentence
        if any((len(hyps) > self.beam_size for hyps in best_finalized)):
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
            running_hyps.yseq[:, :, -1].eq(self.eos)
            & running_hyps.score.ne(-math.inf)
        )

        # NOTE add to result when path finish, reduce runing batch when batch finish
        # only reduce batch when one sentence have enough candidate in result.
        if path_newly_finished_mask.any():
            # 1. add newly finished path to ended_hyps
            # any path newly finished
            path_newly_finished_id = torch.masked_select(
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
                finished_hyp = self._select(running_hyps, path_id)
                # TODO finalize hyp with partial's score?
                ended_hyps[sent_id].append(finished_hyp)
                # sentence finish when it has beam_size finished hypothesis
                if len(ended_hyps[sent_id]) >= self.beam_size:
                    finished_batch_id.add(path_id // self.beam_size)
                    finished[sent_id] = True  # set finish flag to True
            # NOTE mask score of finished path to -inf to avoid be selected
            # again by beam
            running_hyps.score.masked_fill_(
                mask=path_newly_finished_mask, value=-math.inf)
            if len(finished_batch_id) > 0:
                # 2. remove finished batch from runing hypothesis
                batch_id_alive = [
                    i for i in range(batch_size) if i not in finished_batch_id]
                if len(batch_id_alive) == 0:
                    logging.debug(f"All batch finished.")
                    return BatchBeamHypothesis()
                running_hyps = self._batch_select(running_hyps, batch_id_alive)  #.get_alive_batch(batch_id_alive)
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
