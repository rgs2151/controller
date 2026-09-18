import torch as th
from transformers import AutoTokenizer, AutoModelForCausalLM
import steer.lqr_utils as lqr
from functools import partial
from enum import Enum
import time
import torch.nn.functional as F
from sklearn.decomposition import PCA
from contextlib import contextmanager
import functools
from typing import Callable, List, Tuple
import torch.nn as nn
# import numpy as np

class Mode(Enum):
    COLLECTING = 0
    TRACKING = 1
    STEERING = 2
    SETPOINT = 3
    MULTI = 4

class LQRSteering:
    '''
    Contrastive method currently assuming precomputed:
        - jacobians (A)
        - contrastive vectors
    '''


    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        q: float = 10,
        r: float = 10,
        qf: float = 1,
        A: th.Tensor = None,    
        contrastive_vecs: th.Tensor = None,
        preserve_mem: bool = True,
    ):
        self.model = model
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        self.tokenizer = tokenizer
        self.A = A
        self.E = contrastive_vecs
        self.contrastive = False

        self.T = len(model.model.layers)
        self.n = model.model.embed_tokens.embedding_dim
        self.m = self.n

        print(f"model device: {self.model.device}")
        self.Q = th.eye(self.n).unsqueeze(0).repeat(self.T, 1, 1).to(self.device) * q
        self.R = th.eye(self.n).unsqueeze(0).repeat(self.T, 1, 1).to(self.device) * r
        self.Qf = th.eye(self.n).to(self.device) * qf
        
        if preserve_mem:
            self.K = lqr.time_varying_lqr_noB_mem_efficient(self.A, self.Q, self.R, self.Qf) if A is not None else None
            del self.A
            del self.Q
            del self.R
            del self.Qf
        else:
            self.B = th.eye(self.n).repeat(self.T, 1, 1).to(self.device) 
            self.K = lqr.time_varying_lqr(self.A, self.B, self.Q, self.R, self.Qf) if A is not None else None
            # print(f"self.K[0]: {self.K[0]}")
        th.cuda.empty_cache()
        if (self.model.device == 'cpu'):
            self.model=self.model.to(self.device)
        self.X = None # to allocate at runtime
        self.U = th.zeros((self.T, self.n), device=self.device)

        self.X_cl = None

        self.betas = None
        self.E_unit = None
        self.setpoint_type = "linear"
        self.basis2 = None
        self.target_degree = None

        self.hooks = []
        self.mode = None
        self.ALL_TOKENS = False
        

        self.setpoint_signals = []
        self.iter = 0

        self.SIGNAL_COLLECT = False


    def hook_steering(self, layer_idx, module, input, output):
        u_t = self.K[layer_idx]@(self.E[layer_idx]) # can be computed offline
        self.U[layer_idx] = u_t
        self.X[layer_idx] = input[0][0,-1,:]

        output[0][...,-1,:] = output[0][...,-1,:] + u_t # new

        if (layer_idx == self.T-1):
            self.X[self.T] = output[0][...,-1,:] + u_t
        return output
        

    def register_steering_hooks(self):
        """Register the hooks."""

        for layer_idx, layer in enumerate(self.model.model.layers):
            def hook_wrapper(layer_idx):
                def hook(module, input, output):
                    return self.hook_steering(layer_idx, module, input, output)

                return hook

            self.hooks.append(
                layer.register_forward_hook(
                    hook_wrapper(layer_idx)
                )
            )

    def hook_collector(self, layer_idx, module, input, output):
        if self.iter == 0:
            self.X[self.iter][layer_idx] = input[0]
            if layer_idx == self.T-1:
                self.X[self.iter][self.T] = output[0]
                self.iter = self.iter + 1

        else: # for everything other than the first layer, only collect last token position 
            self.X[self.iter][layer_idx] = input[0][0,-1,:]
            if layer_idx == self.T-1:
                self.X[self.iter][self.T] = output[0][...,-1,:]
                self.iter = self.iter + 1

        return output
    
    def register_collection_hooks(self):
        """Register the hooks."""

        for layer_idx, layer in enumerate(self.model.model.layers):
            def hook_wrapper(layer_idx):
                def hook(module, input, output):
                    return self.hook_collector(layer_idx, module, input, output)

                return hook

            self.hooks.append(
                layer.register_forward_hook(
                    hook_wrapper(layer_idx)
                )
            )


    def hook_tracking(self, layer_idx, module, input, output):
        x_t = input[0][0,-1,:]

        if layer_idx == 0:
            self.X_cl[self.iter][layer_idx] = x_t

        diff = x_t - self.X[self.iter][layer_idx,-1,:]
        u_t = -self.K[layer_idx]@(diff)
        # u_t = -(diff)
        self.U[layer_idx] = u_t

        if isinstance(output,tuple):
            output[0][...,-1,:] = output[0][...,-1,:] + u_t
            if layer_idx == self.T-1:
                self.X_cl[self.iter][layer_idx+1] = output[0][...,-1,:]

        else: 
            output[...,-1,:] = output[...,-1,:] + u_t
            if layer_idx == self.T-1:
                self.X_cl[self.iter][layer_idx+1] = output[...,-1,:]

        if (layer_idx == self.T-1):
            # self.X[self.iter][self.T] = output[0][...,-1,:] + u_t
            self.iter = self.iter + 1
        return output
    
    def register_tracking_hooks(self):
        """Register the hooks."""

        for layer_idx, layer in enumerate(self.model.model.layers):
            def hook_wrapper(layer_idx):
                def hook(module, input, output):
                    return self.hook_tracking(layer_idx, module, input, output)

                return hook

            self.hooks.append(
                layer.register_forward_hook(
                    hook_wrapper(layer_idx)
                )
            )

    def hook_setpoint_tracking(self, layer_idx, module, input, output):
        # assume E_normed is unit vector in direction of contrastive feature

        if self.ALL_TOKENS:
            x = input[0]
            self.X[layer_idx] = x[-1,-1,:]
            if self.setpoint_type == "linear":
                v = self.E_unit[layer_idx]
                b_mat = self.betas[layer_idx] * th.ones([x.shape[0], x.shape[1]], device=self.device)
                probe_mat = x @ v.T
                alpha = b_mat - probe_mat
                v_mat = v.expand(x.shape[0], x.shape[1], -1)
                e = alpha.unsqueeze(-1) * v_mat
            elif self.setpoint_type == "angular":
                e = self.get_angular_sp(x, layer_idx) - x
            else:
                raise ValueError("Unsupported setpoint type")

            u_t = e @ self.K[layer_idx].T
            
            self.U[layer_idx] = u_t[-1,-1]

            if isinstance(output,tuple):
                output[0][...] = output[0] + u_t
            else: 
                output = output + u_t
            return output

        else:
            x = input[0][:,-1,:]
            self.X[layer_idx] = x[-1,:]

            if self.setpoint_type == "linear":
                v = self.E_unit[layer_idx]
                alpha = th.tensor([self.betas[layer_idx] for i in range(x.shape[0])], device=self.device) - th.bmm(v.unsqueeze(0).unsqueeze(0), th.transpose(x.unsqueeze(0),-2,-1))
                e = alpha.squeeze(0).T @ v.unsqueeze(0)
            elif self.setpoint_type == "angular":
                e = self.get_angular_sp(x, layer_idx) - x
            else:
                raise ValueError("Unsupported setpoint type")
            u_t = th.bmm(self.K[layer_idx].unsqueeze(0), th.transpose(e.unsqueeze(0),-2,-1)).squeeze(0).T
            self.U[layer_idx] = u_t[-1]

            if isinstance(output,tuple):
                output[0][...,-1,:] = output[0][...,-1,:] + u_t
            else: 
                output[...,-1,:] = output[...,-1,:] + u_t

            # print(f"output: {output}")
            return output

    def register_setpoint_tracking_hooks(self):
        """Register the hooks."""

        for layer_idx, layer in enumerate(self.model.model.layers):
            def hook_wrapper(layer_idx):
                def hook(module, input, output):
                    return self.hook_setpoint_tracking(layer_idx, module, input, output)

                return hook

            self.hooks.append(
                layer.register_forward_hook(
                    hook_wrapper(layer_idx)
                )
            )

    def hook_multisteer_tracking(self, layer_idx, module, input, output):
        x = input[0][:,-1,:]
        self.X[layer_idx] = x[-1,:]

        v = self.E_unit[layer_idx]
        v_alt = self.E_alt_unit[layer_idx]
        alpha = th.tensor([self.betas[layer_idx] for i in range(x.shape[0])], device=self.device) - th.bmm(v.unsqueeze(0).unsqueeze(0), th.transpose(x.unsqueeze(0),-2,-1))
        alpha_alt = th.tensor([self.betas_alt[layer_idx] for i in range(x.shape[0])], device=self.device) - th.bmm(v_alt.unsqueeze(0).unsqueeze(0), th.transpose(x.unsqueeze(0),-2,-1))
        
        e = alpha.squeeze(0).T @ v.unsqueeze(0)
        e_alt = alpha_alt.squeeze(0).T @ v_alt.unsqueeze(0)
        u_t = th.bmm(self.K[layer_idx].unsqueeze(0), th.transpose(e.unsqueeze(0),-2,-1)).squeeze(0).T
        u_t_alt = th.bmm(self.K[layer_idx].unsqueeze(0), th.transpose(e_alt.unsqueeze(0),-2,-1)).squeeze(0).T
        self.U[layer_idx] = u_t[-1]

        if isinstance(output,tuple):
            output[0][...,-1,:] = output[0][...,-1,:] + u_t + u_t_alt
        else: 
            output[...,-1,:] = output[...,-1,:] + u_t + u_t_alt

        # print(f"output: {output}")
        return output

    def register_multisteer_tracking_hooks(self):
        """Register the hooks."""

        for layer_idx, layer in enumerate(self.model.model.layers):
            def hook_wrapper(layer_idx):
                def hook(module, input, output):
                    return self.hook_multisteer_tracking(layer_idx, module, input, output)

                return hook

            self.hooks.append(
                layer.register_forward_hook(
                    hook_wrapper(layer_idx)
                )
            )

    def hook_get_sp_signal(self, layer_idx, module, input, output):
        x = input[0][:,-1,:]
        v = self.E_unit[layer_idx]
        raw_signal = th.bmm(v.unsqueeze(0).unsqueeze(0), th.transpose(x.unsqueeze(0),-2,-1))
        nm = th.norm(self.E[layer_idx])
        signal = raw_signal
        self.setpoint_signals.append(th.mean(signal).item())

        if layer_idx == self.T-1:
            if isinstance(output,tuple):
                x = output[0][...,-1,:]
            else: 
                x = output[...,-1,:]
            if self.mode != None:
                alpha = th.tensor([self.betas[layer_idx] for i in range(x.shape[0])], device=self.device) - th.bmm(v.unsqueeze(0).unsqueeze(0), th.transpose(x.unsqueeze(0),-2,-1))
                e = alpha.squeeze(0).T @ v.unsqueeze(0)
                u_t = th.bmm(self.K[layer_idx].unsqueeze(0), th.transpose(e.unsqueeze(0),-2,-1)).squeeze(0).T
                x = x + u_t
                
            v = self.E_unit[0]
            raw_signal = th.bmm(v.unsqueeze(0).unsqueeze(0), th.transpose(x.unsqueeze(0),-2,-1))
            nm = th.norm(self.E[layer_idx+1])
            signal = raw_signal / nm
            self.setpoint_signals.append(th.mean(signal).item())
        return output

    def register_setpoint_signal_hooks(self):
        """Register the hooks."""

        for layer_idx, layer in enumerate(self.model.model.layers):
            def hook_wrapper(layer_idx):
                def hook(module, input, output):
                    return self.hook_get_sp_signal(layer_idx, module, input, output)

                return hook

            self.hooks.append(
                layer.register_forward_hook(
                    hook_wrapper(layer_idx)
                )
            )

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def __enter__(self):
        if self.mode == Mode.COLLECTING:
            self.register_collection_hooks()
        elif self.mode == Mode.STEERING:
            self.register_steering_hooks()
        elif self.mode == Mode.TRACKING: 
            self.register_tracking_hooks()
        elif self.mode == Mode.SETPOINT:
            self.register_setpoint_tracking_hooks()
        elif self.mode == Mode.MULTI:
            self.register_multisteer_tracking_hooks()
        else:
            print("generating with no steering applied")

        if self.SIGNAL_COLLECT:
            self.register_setpoint_signal_hooks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()

    

    def evaluate(self, prompt, max_new_tokens, do_sample=False, temp=0.7):
        '''
        Steers with no setpoint, always 'tracking' the full contrastive vector.
        Likley not the desired behavior, not considered in the manuscript.
        '''
        self.mode = Mode.STEERING
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        self.X = th.zeros((self.T+1, self.n)).to(self.device)

        with self: # I think just an elegant way to trigger __enter__ and __exit__ to manage hooks
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                do_sample=do_sample,
                temperature=temp,
                use_cache=False,
                pad_token_id=self.tokenizer.eos_token_id,
                # **model_generation_kwargs, #
            )

        output_str = self.tokenizer.decode(output.sequences[0], skip_special_tokens=True)
        return output_str

    def track_setpoint(self, prompt, max_new_tokens, lmbda=1, do_sample=False, temp=1, all_tokens=False, return_tokens=False):
        '''
        A-LQR implementation: tracks the LFS setpoint with an LQR controller at every layer.

        args:
            prompt: list of text inputs
            max_new_tokens: maximum tokens to generate
            lmbda: setpoint target (typically 1-2.5)
            do_sample: greedy decoding if False, set to True in all scripts except refusal
            temp: sampling temperature, N/A if do_sample = False
            return_tokens: return generated tokens before decoding
            all_tokens: steer on all token positions (typically false)

        '''
        
        self.mode = Mode.SETPOINT
        self.setpoint_type = "linear"
        self.SIGNAL_COLLECT = True
        self.setpoint_signals = []
        self.ALL_TOKENS = all_tokens
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
        ).to(self.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        self.X = th.zeros((self.T+1, self.n)).to(self.device)

        self.E_unit = th.zeros_like(self.E)
        self.betas = [0 for i in range(self.T+1)]
        for i, e in enumerate(self.E):
            nrm = th.linalg.norm(e)
            if nrm == 0:
                self.E_unit[i] = self.E_unit[i]*0
            else:
                self.E_unit[i] = e / nrm
                self.betas[i] = lmbda * nrm

        with self:
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                do_sample=do_sample,
                top_p=0.3,
                repetition_penalty=1.2 if do_sample else None,
                temperature=temp,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
                # **model_generation_kwargs, #
            )

        if return_tokens:
            return output.sequences
        output_str = self.tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
        self.ALL_TOKENS = False
        return output_str
        

    def multisteer(self, prompt, max_new_tokens, alt_contr, lmbda=1, alt_lmbda=1, do_sample=False, temp=1, all_tokens=False, return_tokens=False):
        '''
        Track multiple (2) setpoints wth A-LQR.

        args:
            prompt: list of text inputs
            max_new_tokens: maximum tokens to generate
            alt_contr: contrastive vectors corresponding to secondary concept
            lmbda: setpoint target (typically 1-3.5)
            alt_lmbda: alternate setpoint target (typically 1-3.5)
            do_sample: greedy decoding if False, set to True in all scripts except refusal
            temp: sampling temperature, N/A if do_sample = False
            return_tokens: return generated tokens before decoding
            all_tokens: steer on all token positions (typically False - only final token position)

        '''
            
        self.mode = Mode.MULTI
        self.setpoint_type = "linear"
        self.SIGNAL_COLLECT = True
        self.setpoint_signals = []
        self.ALL_TOKENS = all_tokens
        # print("python is garbage")
        # inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
        ).to(self.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        self.X = th.zeros((self.T+1, self.n)).to(self.device)

        self.E_unit = th.zeros_like(self.E)
        self.betas = [0 for i in range(self.T+1)]
        for i, e in enumerate(self.E):
            # print(f"e in setpoint: {e}")
            nrm = th.linalg.norm(e)
            # print(f"nrm in setpoint: {nrm}")
            if nrm == 0:
                self.E_unit[i] = self.E_unit[i]*0
            else:
                self.E_unit[i] = e / nrm
                self.betas[i] = lmbda * nrm

        self.E_alt_unit = th.zeros_like(alt_contr)
        self.betas_alt = [0 for i in range(self.T+1)]
        for i, e in enumerate(alt_contr):
            # print(f"e in setpoint: {e}")
            nrm = th.linalg.norm(e)
            # print(f"nrm in setpoint: {nrm}")
            if nrm == 0:
                self.E_alt_unit[i] = self.E_unit[i]*0
            else:
                self.E_alt_unit[i] = e / nrm
                self.betas_alt[i] = alt_lmbda * nrm

        with self:
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                do_sample=do_sample,
                top_p=0.3,
                repetition_penalty=1.2 if do_sample else None,
                temperature=temp,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
                # **model_generation_kwargs, #
            )

        if return_tokens:
            return output.sequences
        output_str = self.tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
        self.ALL_TOKENS = False
        return output_str


    def track_tokens(self, nom_prompt, prompt, k=1):
        '''
        Tracks a latent 'trajectory' defined by nom_prompt

        args:
            nom_prompt: nominal text input to track (str)
            prompt: text input (str)
            k: number of tokens to track (k > 1 not supported)

        '''
        self.mode = Mode.COLLECTING

        start_time = time.perf_counter()

        nom_inputs = self.tokenizer(nom_prompt, return_tensors="pt").to(self.device)
        nom_input_ids = nom_inputs["input_ids"]
        nom_attention_mask = nom_inputs["attention_mask"].float()

        embedding_layer = self.model.get_input_embeddings()
        hidden_states = embedding_layer(nom_input_ids)
        self.X = [th.zeros_like(hidden_states).repeat(self.T+1, 1, 1).to(self.device)]
        
        sublist = [th.zeros_like(hidden_states[...,-1,:]).repeat(self.T+1, 1, 1).to(self.device) for i in range(k-1)]
        self.X = self.X + sublist
        self.iter = 0

        # print(f"len X: {len(self.X)}")
        # print(f"X[0] shape: {self.X[0].shape}")
        # self.X = th.zeros((self.T+1, self.n)).to(self.device)
        with self:
            with th.no_grad():
                nom_output = self.model.generate(
                    input_ids=nom_input_ids,
                    attention_mask=nom_attention_mask,
                    max_new_tokens=k,
                    return_dict_in_generate=True,
                    do_sample=False,
                    use_cache=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    # **model_generation_kwargs, #
                )

        end_nom_time = time.perf_counter()

        print(f"Nom rollout time: {end_nom_time - start_time}")

        nom_output_str = self.tokenizer.decode(nom_output.sequences[0], skip_special_tokens=True)
        print(f"nom_output: {nom_output_str}<END>")
        
        

        if self.A is None:
            batch_size, seq_len = nom_input_ids.shape
            position_ids = th.arange(seq_len, dtype=th.long, device=self.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len).to(self.device)

            position_embeddings = self.model.model.rotary_emb(hidden_states, position_ids)
            wrapped_tfs_temp = [partial(lqr.tf_block_wrapper, tf, nom_attention_mask, position_ids, position_embeddings) for tf in self.model.model.layers]
            tfs_with_control_temp = [partial(lqr.transformerBlockControl, tf) for tf in wrapped_tfs_temp]
            self.A = lqr.linearize(tfs_with_control_temp,self.T,self.m,self.X[0]) # linearizing about first subtrajectory

        lin_time = time.perf_counter()
        print(f"Linearize time: {lin_time - end_nom_time}")

        print(self.A.device)
        self.K = lqr.time_varying_lqr(self.A, self.B, self.Q, self.R, self.Qf)

        self.mode = Mode.TRACKING
        self.X_cl = [th.zeros_like(hidden_states).repeat(self.T+1, 1, 1).to(self.device)]
        
        sublist = [th.zeros_like(hidden_states[...,-1,:]).repeat(self.T+1, 1, 1).to(self.device) for i in range(k-1)]
        self.X_cl = self.X_cl + sublist

        self.iter = 0

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        with self: # I think just an elegant way to trigger __enter__ and __exit__ to manage hooks
            with th.no_grad():
                output = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=k,
                    return_dict_in_generate=True,
                    do_sample=False,
                    use_cache=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    # **model_generation_kwargs, #
                )

        output_str = self.tokenizer.decode(output.sequences[0], skip_special_tokens=True)
        # print(f"steered output: {output_str}")
        
        end_time = time.perf_counter()

        print(f"Tracking time: {end_time - lin_time}")
        print(f"Total time: {end_time - start_time}")
        return output_str


    def track_traj(self, X_nom, prompt, k=1, do_sample=False, temp=0.7):
        '''
        Tracks a latent 'trajectory' defined by X_nom

        args:
            X_nom: nominal activation trajectory
            prompt: text input (str)
            k: number of tokens to track (k > 1 not supported)
        '''

        self.mode = Mode.COLLECTING

        start_time = time.perf_counter()
        self.X = [X_nom for i in range(k)]        


        
        self.mode = Mode.TRACKING
        self.iter = 0

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        if self.A is None:
            embedding_layer = self.model.get_input_embeddings()
            hidden_states = embedding_layer(input_ids)
            batch_size, seq_len = input_ids.shape
            position_ids = th.arange(seq_len, dtype=th.long, device=self.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len).to(self.device)

            position_embeddings = self.model.model.rotary_emb(hidden_states, position_ids)
            wrapped_tfs_temp = [partial(lqr.tf_block_wrapper, tf, attention_mask, position_ids, position_embeddings) for tf in self.model.model.layers]
            tfs_with_control_temp = [partial(lqr.transformerBlockControl, tf) for tf in wrapped_tfs_temp]
            self.A, _ = lqr.linearize(tfs_with_control_temp,self.T,self.m,self.X[0]) # linearizing about first subtrajectory
            self.K = lqr.time_varying_lqr(self.A, self.B, self.Q, self.R, self.Qf)


        with self: # I think just an elegant way to trigger __enter__ and __exit__ to manage hooks
            with th.no_grad():
                output = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=k,
                    return_dict_in_generate=True,
                    do_sample=do_sample,
                    temperature=temp if do_sample else None,
                    use_cache=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    # **model_generation_kwargs, #
                )

        output_str = self.tokenizer.decode(output.sequences[0], skip_special_tokens=True)
        
        end_time = time.perf_counter()

        return output_str

    
    def complete_rollout(self, prompt, k=1):
        self.mode = Mode.COLLECTING
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"].float()
        embedding_layer = self.model.get_input_embeddings()
        hidden_states = embedding_layer(input_ids)
        self.X = [th.zeros_like(hidden_states).repeat(self.T+1, 1, 1).to(self.device)]
        
        sublist = [th.zeros_like(hidden_states[...,-1,:]).repeat(self.T+1, 1, 1).to(self.device) for i in range(k-1)]
        self.X = self.X + sublist

        with self: # I think just an elegant way to trigger __enter__ and __exit__ to manage hooks
            with th.no_grad():
                output = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=k,
                    return_dict_in_generate=True,
                    do_sample=False,
                    use_cache=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    # **model_generation_kwargs, #
                )

        batch_size, seq_len = input_ids.shape
        position_ids = th.arange(seq_len, dtype=th.long, device=self.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len).to(self.device)

        position_embeddings = self.model.model.rotary_emb(hidden_states, position_ids)
        wrapped_tfs_temp = [partial(lqr.tf_block_wrapper, tf, attention_mask, position_ids, position_embeddings) for tf in self.model.model.layers]
        tfs_with_control_temp = [partial(lqr.transformerBlockControl, tf) for tf in wrapped_tfs_temp]
        
        self.A = [th.eye(self.n).unsqueeze(0).repeat(self.T, 1, 1).to(self.device) for i in range (k)]
        for i in range(k):
            self.A[i], _ = lqr.linearize(tfs_with_control_temp,self.T,self.m,self.X[i])
        
        return self.X, self.A, output

    