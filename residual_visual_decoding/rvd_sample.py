import copy
import inspect
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn

from transformers.generation.logits_process import (
    LogitsProcessorList,
)
from transformers.generation.stopping_criteria import (
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)
import transformers
from transformers.generation.utils import SampleOutput



def sample(
    self,
    input_ids: torch.LongTensor,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    logits_warper: Optional[LogitsProcessorList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    **model_kwargs,
) -> Union[SampleOutput, torch.LongTensor]:
    # init values
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id


    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
    output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
    output_attentions = (
        output_attentions if output_attentions is not None else self.generation_config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
    )

    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # keep track of which sequences are already finished
    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

    this_peer_finished = False  # used by synced_gpus only
    
    rvd_flag = model_kwargs.get("rvd")
    adb_flag = model_kwargs.get("adb")
    
    if rvd_flag:
        rvd_input_ids = model_kwargs.pop("rvd_input_ids")
        model_kwargs_rvd = copy.deepcopy(model_kwargs)
        if adb_flag:
            adb_input_ids = model_kwargs.pop("adb_input_ids")
            model_kwargs_blind = copy.deepcopy(model_kwargs)
            model_kwargs_blind.pop("images")
    # auto-regressive generation
    while True:
        if synced_gpus:
            # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
            # The following logic allows an early break if all peers finished generating their sequence
            this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
            # send 0.0 if we finished, 1.0 otherwise
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            # did all peers finish? the reduced sum will be 0.0 then
            if this_peer_finished_flag.item() == 0.0:
                break

        # prepare model inputs
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        # forward pass to get next token
        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        if synced_gpus and this_peer_finished:
            continue  # don't waste resources running the code we don't need

        next_token_logits = outputs.logits[:, -1, :]
        

        output_attentions_wo_img = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states_wo_img = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        
        # prepare inputs for RVD and ADB
        if rvd_flag:
            
            # prepare for residual visual distribution
            inputs_tensor, model_input_name, model_kwargs_rvd = self._prepare_model_inputs(
                rvd_input_ids, self.generation_config.bos_token_id, model_kwargs_rvd
            )
            
            model_kwargs_rvd["attention_mask"] = self._prepare_attention_mask_for_generation(
                inputs_tensor, self.generation_config.pad_token_id, self.generation_config.eos_token_id
            )
            
            
            model_inputs_rvd = self.prepare_inputs_for_generation(rvd_input_ids, **model_kwargs_rvd)
            outputs_rvd = self(
                **model_inputs_rvd,
                return_dict=True,
                output_attentions=output_attentions_wo_img,
                output_hidden_states=output_hidden_states_wo_img,
            )
            next_token_logits_rvd = outputs_rvd.logits[:, -1, :]
            
            
            if adb_flag:
                # prepare for blind distribution
                blind_inputs_tensor, model_input_name, model_kwargs_blind = self._prepare_model_inputs(
                    adb_input_ids, self.generation_config.bos_token_id, model_kwargs_blind
                )
                
                model_kwargs_blind["attention_mask"] = self._prepare_attention_mask_for_generation(
                    blind_inputs_tensor, self.generation_config.pad_token_id, self.generation_config.eos_token_id
                )
                
                model_inputs_adb = self.prepare_inputs_for_generation(adb_input_ids, **model_kwargs_blind)
                outputs_adb = self(
                    **model_inputs_adb,
                    return_dict=True,
                    output_attentions=output_attentions_wo_img,
                    output_hidden_states=output_hidden_states_wo_img,
                )
                next_token_logits_adb = outputs_adb.logits[:, -1, :]
            
            
            rvd_alpha = model_kwargs.get("rvd_alpha")
            rvd_beta = model_kwargs.get("rvd_beta")
            
            # Adaptive Distribution Blending
            if adb_flag:
                # calculate the KL divergence
                rvd_probs =  torch.nn.functional.softmax(next_token_logits_rvd, dim=-1)
                adb_probs = torch.nn.functional.softmax(next_token_logits_adb, dim=-1)
                total_probs = 0.5 * (rvd_probs + adb_probs)
                total_log_probs = torch.log(total_probs)
                rvd_log_probs = torch.nn.functional.log_softmax(next_token_logits_rvd, dim=-1)
                adb_log_probs = torch.nn.functional.log_softmax(next_token_logits_adb, dim=-1)
                
                # calculating JS divergence 
                kl_score_1 = torch.nn.functional.kl_div(total_log_probs, adb_log_probs, log_target=True, reduction="batchmean")
                kl_score_2 = torch.nn.functional.kl_div(total_log_probs, rvd_log_probs, log_target=True, reduction="batchmean")
                kl_score = 0.5 * (kl_score_1 + kl_score_2)
                    
                scale = min((kl_score * rvd_beta), 1.0)
                diffs = (1-scale)*next_token_logits + scale * next_token_logits_rvd
                rvd_logits = diffs

            # Disable ADB, run with parameter alpha 
            else:
                diffs = (1-rvd_alpha) * next_token_logits + rvd_alpha * next_token_logits_rvd
                rvd_logits = diffs


            next_token_scores = logits_processor(input_ids, rvd_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )


        # finished sentences should have their next token be a padding token
        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)


        # updating input ids and model_kwargs for RVD
        if rvd_flag:
            model_kwargs_rvd = self._update_model_kwargs_for_generation(
                outputs_rvd, model_kwargs_rvd, is_encoder_decoder=self.config.is_encoder_decoder
            )
            rvd_next_tokens = copy.deepcopy(next_tokens)
            rvd_input_ids = torch.cat([rvd_input_ids, rvd_next_tokens[:, None]], dim=-1)
            if adb_flag:
                model_kwargs_blind = self._update_model_kwargs_for_generation(
                    outputs_adb, model_kwargs_blind, is_encoder_decoder=self.config.is_encoder_decoder
                )
                adb_next_tokens = copy.deepcopy(next_tokens)
                adb_input_ids = torch.cat([adb_input_ids, adb_next_tokens[:, None]], dim=-1)
                
            

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )

        # if eos_token was found in one sentence, set sentence to finished
        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )

            # stop when each sentence is finished
            if unfinished_sequences.max() == 0:
                this_peer_finished = True
        # possible wrong
        # stop if we exceed the maximum length
        if stopping_criteria(input_ids, scores):
            this_peer_finished = True

        if this_peer_finished and not synced_gpus:
            break

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return SampleEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
            )
        else:
            return SampleDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
            )
    else:
        return input_ids

def rvd_sampling():
    transformers.generation.utils.GenerationMixin.sample = sample