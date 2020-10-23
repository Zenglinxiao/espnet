#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Decoder definition."""

import logging

from typing import Any
from typing import List
from typing import Tuple

import torch

from espnet.nets.pytorch_backend.nets_utils import rename_state_dict
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.decoder_layer import DecoderLayer
from espnet.nets.pytorch_backend.transformer.dynamic_conv import DynamicConvolution
from espnet.nets.pytorch_backend.transformer.dynamic_conv2d import DynamicConvolution2D
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.lightconv import LightweightConvolution
from espnet.nets.pytorch_backend.transformer.lightconv2d import LightweightConvolution2D
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.scorer_interface import BatchScorerInterface

from espnet.nets.pytorch_backend.nets_utils import mask_by_length
from espnet.nets.pytorch_backend.nets_utils import to_device
import numpy as np

def _pre_hook(
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
):
    # https://github.com/espnet/espnet/commit/3d422f6de8d4f03673b89e1caef698745ec749ea#diff-bffb1396f038b317b2b64dd96e6d3563
    rename_state_dict(prefix + "output_norm.", prefix + "after_norm.", state_dict)


class Decoder(BatchScorerInterface, torch.nn.Module):
    """Transfomer decoder module.

    Args:
        odim (int): Output diminsion.
        self_attention_layer_type (str): Self-attention layer type.
        attention_dim (int): Dimention of attention.
        attention_heads (int): The number of heads of multi head attention.
        conv_wshare (int): The number of kernel of convolution. Only used in
            self_attention_layer_type == "lightconv*" or "dynamiconv*".
        conv_kernel_length (Union[int, str]): Kernel size str of convolution
            (e.g. 71_71_71_71_71_71). Only used in self_attention_layer_type
            == "lightconv*" or "dynamiconv*".
        conv_usebias (bool): Whether to use bias in convolution. Only used in
            self_attention_layer_type == "lightconv*" or "dynamiconv*".
        linear_units (int): The number of units of position-wise feed forward.
        num_blocks (int): The number of decoder blocks.
        dropout_rate (float): Dropout rate.
        positional_dropout_rate (float): Dropout rate after adding positional encoding.
        self_attention_dropout_rate (float): Dropout rate in self-attention.
        src_attention_dropout_rate (float): Dropout rate in source-attention.
        input_layer (Union[str, torch.nn.Module]): Input layer type.
        use_output_layer (bool): Whether to use output layer.
        pos_enc_class (torch.nn.Module): Positional encoding module class.
            `PositionalEncoding `or `ScaledPositionalEncoding`
        normalize_before (bool): Whether to use layer_norm before the first block.
        concat_after (bool): Whether to concat attention layer's input and output.
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied. i.e. x -> x + att(x)

    """

    def __init__(
        self,
        odim,
        selfattention_layer_type="selfattn",
        attention_dim=256,
        attention_heads=4,
        conv_wshare=4,
        conv_kernel_length=11,
        conv_usebias=False,    
        linear_units=2048,     
        num_blocks=6,          
        dropout_rate=0.1,
        positional_dropout_rate=0.1,
        self_attention_dropout_rate=0.0,
        src_attention_dropout_rate=0.0,
        input_layer="embed",
        use_output_layer=True,
        pos_enc_class=PositionalEncoding,
        normalize_before=True,
        concat_after=False,
    ):
        """Construct an Decoder object."""
        torch.nn.Module.__init__(self)
        self._register_load_state_dict_pre_hook(_pre_hook)
        if input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(odim, attention_dim),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(odim, attention_dim),
                torch.nn.LayerNorm(attention_dim),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif isinstance(input_layer, torch.nn.Module):
            self.embed = torch.nn.Sequential(
                input_layer, pos_enc_class(attention_dim, positional_dropout_rate)
            )
        else:
            raise NotImplementedError("only `embed` or torch.nn.Module is supported.")
        self.normalize_before = normalize_before
        if selfattention_layer_type == "selfattn":
            logging.info("decoder self-attention layer type = self-attention")
            self.decoders = repeat(
                num_blocks,
                lambda lnum: DecoderLayer(
                    attention_dim,
                    MultiHeadedAttention(
                        attention_heads, attention_dim, self_attention_dropout_rate
                    ),
                    MultiHeadedAttention(
                        attention_heads, attention_dim, src_attention_dropout_rate
                    ),
                    PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                    dropout_rate,
                    normalize_before,
                    concat_after,
                ),
            )
        elif selfattention_layer_type == "lightconv":
            logging.info("decoder self-attention layer type = lightweight convolution")
            self.decoders = repeat(
                num_blocks,
                lambda lnum: DecoderLayer(
                    attention_dim,
                    LightweightConvolution(
                        conv_wshare,
                        attention_dim,
                        self_attention_dropout_rate,
                        conv_kernel_length,
                        lnum,
                        use_kernel_mask=True,
                        use_bias=conv_usebias,
                    ),
                    MultiHeadedAttention(
                        attention_heads, attention_dim, src_attention_dropout_rate
                    ),
                    PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                    dropout_rate,
                    normalize_before,
                    concat_after,
                ),
            )
        elif selfattention_layer_type == "lightconv2d":
            logging.info(
                "decoder self-attention layer "
                "type = lightweight convolution 2-dimentional"
            )
            self.decoders = repeat(
                num_blocks,
                lambda lnum: DecoderLayer(
                    attention_dim,
                    LightweightConvolution2D(
                        conv_wshare,
                        attention_dim,
                        self_attention_dropout_rate,
                        conv_kernel_length,
                        lnum,
                        use_kernel_mask=True,
                        use_bias=conv_usebias,
                    ),
                    MultiHeadedAttention(
                        attention_heads, attention_dim, src_attention_dropout_rate
                    ),
                    PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                    dropout_rate,
                    normalize_before,
                    concat_after,
                ),
            )
        elif selfattention_layer_type == "dynamicconv":
            logging.info("decoder self-attention layer type = dynamic convolution")
            self.decoders = repeat(
                num_blocks,
                lambda lnum: DecoderLayer(
                    attention_dim,
                    DynamicConvolution(
                        conv_wshare,
                        attention_dim,
                        self_attention_dropout_rate,
                        conv_kernel_length,
                        lnum,
                        use_kernel_mask=True,
                        use_bias=conv_usebias,
                    ),
                    MultiHeadedAttention(
                        attention_heads, attention_dim, src_attention_dropout_rate
                    ),
                    PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                    dropout_rate,
                    normalize_before,
                    concat_after,
                ),
            )
        elif selfattention_layer_type == "dynamicconv2d":
            logging.info(
                "decoder self-attention layer type = dynamic convolution 2-dimentional"
            )
            self.decoders = repeat(
                num_blocks,
                lambda lnum: DecoderLayer(
                    attention_dim,
                    DynamicConvolution2D(
                        conv_wshare,
                        attention_dim,
                        self_attention_dropout_rate,
                        conv_kernel_length,
                        lnum,
                        use_kernel_mask=True,
                        use_bias=conv_usebias,
                    ),
                    MultiHeadedAttention(
                        attention_heads, attention_dim, src_attention_dropout_rate
                    ),
                    PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                    dropout_rate,
                    normalize_before,
                    concat_after,
                ),
            )
        self.selfattention_layer_type = selfattention_layer_type
        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)
        if use_output_layer:
            self.output_layer = torch.nn.Linear(attention_dim, odim)
        else:
            self.output_layer = None

    def forward(self, tgt, tgt_mask, memory, memory_mask):
        """Forward decoder.

        Args:
            tgt (torch.Tensor): Input token ids, int64 (#batch, maxlen_out) if
                input_layer == "embed". In the other case, input tensor
                (#batch, maxlen_out, odim).
            tgt_mask (torch.Tensor): Input token mask (#batch, maxlen_out).
                dtype=torch.uint8 in PyTorch 1.2- and dtype=torch.bool in PyTorch 1.2+
                (include 1.2).
            memory (torch.Tensor): Encoded memory, float32 (#batch, maxlen_in, feat).
            memory_mask (torch.Tensor): Encoded memory mask (#batch, maxlen_in).
                dtype=torch.uint8 in PyTorch 1.2- and dtype=torch.bool in PyTorch 1.2+
                (include 1.2).

        Returns:
            torch.Tensor: Decoded token score before softmax (#batch, maxlen_out, odim)
                   if use_output_layer is True. In the other case,final block outputs
                   (#batch, maxlen_out, attention_dim).
            torch.Tensor: Score mask before softmax (#batch, maxlen_out).

        """
        x = self.embed(tgt)
        x, tgt_mask, memory, memory_mask = self.decoders(
            x, tgt_mask, memory, memory_mask
        )
        if self.normalize_before:
            x = self.after_norm(x)
        if self.output_layer is not None:
            x = self.output_layer(x)
        return x, tgt_mask

    def forward_one_step(self, tgt, tgt_mask, memory, cache=None):
        """Forward one step.

        Args:
            tgt (torch.Tensor): Input token ids, int64 (#batch, maxlen_out).
            tgt_mask (torch.Tensor): Input token mask (#batch, maxlen_out).
                dtype=torch.uint8 in PyTorch 1.2- and dtype=torch.bool in PyTorch 1.2+
                (include 1.2).
            memory (torch.Tensor): Encoded memory, float32 (#batch, maxlen_in, feat).
            cache (List[torch.Tensor]): List of cached tensors.
                Each tensor shape should be (#batch, maxlen_out - 1, size).

        Returns:
            torch.Tensor: Output tensor (batch, maxlen_out, odim).
            List[torch.Tensor]: List of cache tensors of each decoder layer.

        """
        x = self.embed(tgt)
        if cache is None:
            cache = [None] * len(self.decoders)
        new_cache = []
        for c, decoder in zip(cache, self.decoders):
            x, tgt_mask, memory, memory_mask = decoder(
                x, tgt_mask, memory, None, cache=c
            )
            new_cache.append(x)

        if self.normalize_before:
            y = self.after_norm(x[:, -1])
        else:
            y = x[:, -1]
        if self.output_layer is not None:
            y = torch.log_softmax(self.output_layer(y), dim=-1)

        return y, new_cache

    # beam search API (see ScorerInterface)
    def score(self, ys, state, x):
        """Score."""
        ys_mask = subsequent_mask(len(ys), device=x.device).unsqueeze(0)
        if self.selfattention_layer_type != "selfattn":
            # TODO(karita): implement cache
            logging.warning(
                f"{self.selfattention_layer_type} does not support cached decoding."
            )
            state = None
        logp, state = self.forward_one_step(
            ys.unsqueeze(0), ys_mask, x.unsqueeze(0), cache=state
        )
        return logp.squeeze(0), state

    # batch beam search API (see BatchScorerInterface)
    def batch_score(
        self, ys: torch.Tensor, states: List[Any], xs: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Any]]:
        """Score new token batch (required).

        Args:
            ys (torch.Tensor): torch.int64 prefix tokens (n_batch, ylen).
            states (List[Any]): Scorer states for prefix tokens.
            xs (torch.Tensor):
                The encoder feature that generates ys (n_batch, xlen, n_feat).

        Returns:
            tuple[torch.Tensor, List[Any]]: Tuple of
                batchfied scores for next token with shape of `(n_batch, n_vocab)`
                and next state list for ys.

        """
        # merge states
        n_batch = len(ys)
        n_layers = len(self.decoders)
        if states[0] is None:
            batch_state = None
        else:
            # transpose state of [batch, layer] into [layer, batch]
            batch_state = [
                torch.stack([states[b][i] for b in range(n_batch)])
                for i in range(n_layers)
            ]

        # batch decoding
        ys_mask = subsequent_mask(ys.size(-1), device=xs.device).unsqueeze(0)
        logp, states = self.forward_one_step(ys, ys_mask, xs, cache=batch_state)

        # transpose state of [layer, batch] into [batch, layer]
        state_list = [[states[i][b] for i in range(n_layers)] for b in range(n_batch)]
        return logp, state_list

    def recognize_beam_batch(
        self,
        h,
        hlens,
        lpz,
        recog_args,
        char_list,
        rnnlm=None,
        normalize_score=True,
        strm_idx=0,
        lang_ids=None,
    ):

        num_encs = 1
        # to support mutiple encoder asr mode, in single encoder mode,
        # convert torch.Tensor to List of torch.Tensor
        if num_encs == 1:
            h = [h]
            hlens = [hlens]
            lpz = [lpz]
        if num_encs > 1 and lpz is None:
            lpz = [lpz] * num_encs

        #att_idx = min(strm_idx, len(self.att) - 1)
        att_idx = min(strm_idx, 0)
        for idx in range(num_encs):
            logging.info(
                "Number of Encoder:{}; enc{}: input lengths: {}.".format(
                    num_encs, idx + 1, h[idx].size(1)
                )
            )
            h[idx] = mask_by_length(h[idx], hlens[idx], 0.0)

        # search params
        batch = len(hlens[0])
        beam = recog_args.beam_size
        penalty = recog_args.penalty
        ctc_weight = getattr(recog_args, "ctc_weight", 0)  # for NMT
        att_weight = 1.0 - ctc_weight
        ctc_margin = getattr(
            recog_args, "ctc_window_margin", 0
        )  # use getattr to keep compatibility
        # weights-ctc,
        # e.g. ctc_loss = w_1*ctc_1_loss + w_2 * ctc_2_loss + w_N * ctc_N_loss
        if lpz[0] is not None and num_encs > 1:
            weights_ctc_dec = recog_args.weights_ctc_dec / np.sum(
                recog_args.weights_ctc_dec
            )  # normalize
            logging.info(
                "ctc weights (decoding): " + " ".join([str(x) for x in weights_ctc_dec])
            )
        else:
            weights_ctc_dec = [1.0]

        n_bb = batch * beam
        pad_b = to_device(self, torch.arange(batch) * beam).view(-1, 1)

        max_hlen = np.amin([max(hlens[idx]) for idx in range(num_encs)])
        if recog_args.maxlenratio == 0:
            maxlen = max_hlen
        else:
            maxlen = max(1, int(recog_args.maxlenratio * max_hlen))
        minlen = int(recog_args.minlenratio * max_hlen)
        logging.info("max output length: " + str(maxlen))
        logging.info("min output length: " + str(minlen))

        # initialization
        dunits = self.decoders[0].feed_forward.w_1.out_features
        c_prev = [
            to_device(self, torch.zeros(n_bb, dunits)) for _ in range(len(self.decoders))
        ]
        z_prev = [
            to_device(self, torch.zeros(n_bb, dunits)) for _ in range(len(self.decoders))
        ]
        c_list = [
            to_device(self, torch.zeros(n_bb, dunits)) for _ in range(len(self.decoders))
        ]
        z_list = [
            to_device(self, torch.zeros(n_bb, dunits)) for _ in range(len(self.decoders))
        ]
        vscores = to_device(self, torch.zeros(batch, beam))

        rnnlm_state = None
        if num_encs == 1:
            a_prev = [None]
            att_w_list, ctc_scorer, ctc_state = [None], [None], [None]
            #self.att[att_idx].reset()  # reset pre-computation of h
        else:
            a_prev = [None] * (num_encs + 1)  # atts + han
            att_w_list = [None] * (num_encs + 1)  # atts + han
            att_c_list = [None] * (num_encs)  # atts
            ctc_scorer, ctc_state = [None] * (num_encs), [None] * (num_encs)
            for idx in range(num_encs + 1):
                self.att[idx].reset()  # reset pre-computation of h in atts and han

        if self.replace_sos and recog_args.tgt_lang:
            logging.info("<sos> index: " + str(char_list.index(recog_args.tgt_lang)))
            logging.info("<sos> mark: " + recog_args.tgt_lang)
            yseq = [
                [char_list.index(recog_args.tgt_lang)] for _ in six.moves.range(n_bb)
            ]
        elif lang_ids is not None:
            # NOTE: used for evaluation during training
            yseq = [
                [lang_ids[b // recog_args.beam_size]] for b in six.moves.range(n_bb)
            ]
        else:
            logging.info("<sos> index: " + str(self.sos))
            logging.info("<sos> mark: " + char_list[self.sos])
            yseq = [[self.sos] for _ in six.moves.range(n_bb)]

        accum_odim_ids = [self.sos for _ in six.moves.range(n_bb)]
        stop_search = [False for _ in six.moves.range(batch)]
        nbest_hyps = [[] for _ in six.moves.range(batch)]
        ended_hyps = [[] for _ in range(batch)]

        exp_hlens = [
            hlens[idx].repeat(beam).view(beam, batch).transpose(0, 1).contiguous()
            for idx in range(num_encs)
        ]
        exp_hlens = [exp_hlens[idx].view(-1).tolist() for idx in range(num_encs)]
        exp_h = [
            h[idx].unsqueeze(1).repeat(1, beam, 1, 1).contiguous()
            for idx in range(num_encs)
        ]
        exp_h = [
            exp_h[idx].view(n_bb, h[idx].size()[1], h[idx].size()[2])
            for idx in range(num_encs)
        ]

        if lpz[0] is not None:
            scoring_num = min(
                int(beam * CTC_SCORING_RATIO)
                if att_weight > 0.0 and not lpz[0].is_cuda
                else 0,
                lpz[0].size(-1),
            )
            ctc_scorer = [
                CTCPrefixScoreTH(lpz[idx], hlens[idx], 0, self.eos, margin=ctc_margin,)
                for idx in range(num_encs)
            ]

        for i in six.moves.range(maxlen):
            logging.debug("position " + str(i))

            vy = to_device(self, torch.LongTensor(self._get_last_yseq(yseq)))
            ey = self.dropout_emb(self.embed(vy))
            if num_encs == 1:
                att_c, att_w = self.att[att_idx](
                    exp_h[0], exp_hlens[0], self.dropout_dec[0](z_prev[0]), a_prev[0]
                )
                att_w_list = [att_w]
            else:
                for idx in range(num_encs):
                    att_c_list[idx], att_w_list[idx] = self.att[idx](
                        exp_h[idx],
                        exp_hlens[idx],
                        self.dropout_dec[0](z_prev[0]),
                        a_prev[idx],
                    )
                exp_h_han = torch.stack(att_c_list, dim=1)
                att_c, att_w_list[num_encs] = self.att[num_encs](
                    exp_h_han,
                    [num_encs] * n_bb,
                    self.dropout_dec[0](z_prev[0]),
                    a_prev[num_encs],
                )
            ey = torch.cat((ey, att_c), dim=1)

            # attention decoder
            z_list, c_list = self.rnn_forward(ey, z_list, c_list, z_prev, c_prev)
            if self.context_residual:
                logits = self.output(
                    torch.cat((self.dropout_dec[-1](z_list[-1]), att_c), dim=-1)
                )
            else:
                logits = self.output(self.dropout_dec[-1](z_list[-1]))
            local_scores = att_weight * F.log_softmax(logits, dim=1)

            # rnnlm
            if rnnlm:
                rnnlm_state, local_lm_scores = rnnlm.buff_predict(rnnlm_state, vy, n_bb)
                local_scores = local_scores + recog_args.lm_weight * local_lm_scores

            # ctc
            if ctc_scorer[0]:
                local_scores[:, 0] = self.logzero  # avoid choosing blank
                part_ids = (
                    torch.topk(local_scores, scoring_num, dim=-1)[1]
                    if scoring_num > 0
                    else None
                )
                for idx in range(num_encs):
                    att_w = att_w_list[idx]
                    att_w_ = att_w if isinstance(att_w, torch.Tensor) else att_w[0]
                    local_ctc_scores, ctc_state[idx] = ctc_scorer[idx](
                        yseq, ctc_state[idx], part_ids, att_w_
                    )
                    local_scores = (
                        local_scores
                        + ctc_weight * weights_ctc_dec[idx] * local_ctc_scores
                    )

            local_scores = local_scores.view(batch, beam, self.odim)
            if i == 0:
                local_scores[:, 1:, :] = self.logzero

            # accumulate scores
            eos_vscores = local_scores[:, :, self.eos] + vscores
            vscores = vscores.view(batch, beam, 1).repeat(1, 1, self.odim)
            vscores[:, :, self.eos] = self.logzero
            vscores = (vscores + local_scores).view(batch, -1)

            # global pruning
            accum_best_scores, accum_best_ids = torch.topk(vscores, beam, 1)
            accum_odim_ids = (
                torch.fmod(accum_best_ids, self.odim).view(-1).data.cpu().tolist()
            )
            accum_padded_beam_ids = (
                (accum_best_ids // self.odim + pad_b).view(-1).data.cpu().tolist()
            )

            y_prev = yseq[:][:]
            yseq = self._index_select_list(yseq, accum_padded_beam_ids)
            yseq = self._append_ids(yseq, accum_odim_ids)
            vscores = accum_best_scores
            vidx = to_device(self, torch.LongTensor(accum_padded_beam_ids))

            a_prev = []
            num_atts = num_encs if num_encs == 1 else num_encs + 1
            for idx in range(num_atts):
                if isinstance(att_w_list[idx], torch.Tensor):
                    _a_prev = torch.index_select(
                        att_w_list[idx].view(n_bb, *att_w_list[idx].shape[1:]), 0, vidx
                    )
                elif isinstance(att_w_list[idx], list):
                    # handle the case of multi-head attention
                    _a_prev = [
                        torch.index_select(att_w_one.view(n_bb, -1), 0, vidx)
                        for att_w_one in att_w_list[idx]
                    ]
                else:
                    # handle the case of location_recurrent when return is a tuple
                    _a_prev_ = torch.index_select(
                        att_w_list[idx][0].view(n_bb, -1), 0, vidx
                    )
                    _h_prev_ = torch.index_select(
                        att_w_list[idx][1][0].view(n_bb, -1), 0, vidx
                    )
                    _c_prev_ = torch.index_select(
                        att_w_list[idx][1][1].view(n_bb, -1), 0, vidx
                    )
                    _a_prev = (_a_prev_, (_h_prev_, _c_prev_))
                a_prev.append(_a_prev)
            z_prev = [
                torch.index_select(z_list[li].view(n_bb, -1), 0, vidx)
                for li in range(self.dlayers)
            ]
            c_prev = [
                torch.index_select(c_list[li].view(n_bb, -1), 0, vidx)
                for li in range(self.dlayers)
            ]

            # pick ended hyps
            if i >= minlen:
                k = 0
                penalty_i = (i + 1) * penalty
                thr = accum_best_scores[:, -1]
                for samp_i in six.moves.range(batch):
                    if stop_search[samp_i]:
                        k = k + beam
                        continue
                    for beam_j in six.moves.range(beam):
                        _vscore = None
                        if eos_vscores[samp_i, beam_j] > thr[samp_i]:
                            yk = y_prev[k][:]
                            if len(yk) <= min(
                                hlens[idx][samp_i] for idx in range(num_encs)
                            ):
                                _vscore = eos_vscores[samp_i][beam_j] + penalty_i
                        elif i == maxlen - 1:
                            yk = yseq[k][:]
                            _vscore = vscores[samp_i][beam_j] + penalty_i
                        if _vscore:
                            yk.append(self.eos)
                            if rnnlm:
                                _vscore += recog_args.lm_weight * rnnlm.final(
                                    rnnlm_state, index=k
                                )
                            _score = _vscore.data.cpu().numpy()
                            ended_hyps[samp_i].append(
                                {"yseq": yk, "vscore": _vscore, "score": _score}
                            )
                        k = k + 1

            # end detection
            stop_search = [
                stop_search[samp_i] or end_detect(ended_hyps[samp_i], i)
                for samp_i in six.moves.range(batch)
            ]
            stop_search_summary = list(set(stop_search))
            if len(stop_search_summary) == 1 and stop_search_summary[0]:
                break

            if rnnlm:
                rnnlm_state = self._index_select_lm_state(rnnlm_state, 0, vidx)
            if ctc_scorer[0]:
                for idx in range(num_encs):
                    ctc_state[idx] = ctc_scorer[idx].index_select_state(
                        ctc_state[idx], accum_best_ids
                    )

        torch.cuda.empty_cache()

        dummy_hyps = [
            {"yseq": [self.sos, self.eos], "score": np.array([-float("inf")])}
        ]
        ended_hyps = [
            ended_hyps[samp_i] if len(ended_hyps[samp_i]) != 0 else dummy_hyps
            for samp_i in six.moves.range(batch)
        ]
        if normalize_score:
            for samp_i in six.moves.range(batch):
                for x in ended_hyps[samp_i]:
                    x["score"] /= len(x["yseq"])

        nbest_hyps = [
            sorted(ended_hyps[samp_i], key=lambda x: x["score"], reverse=True)[
                : min(len(ended_hyps[samp_i]), recog_args.nbest)
            ]
            for samp_i in six.moves.range(batch)
        ]

        return nbest_hyps
