import torch as th
import numpy as np
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import steer.lqr_utils as lqr
from functools import partial
from datasets import load_dataset
import random
import pickle
from steer.steering import Mode
from contextlib import contextmanager
import functools
from typing import Callable, List, Tuple
import torch.nn as nn
from timeit import default_timer as timer
import gc
from pathlib import Path
from steer.config import config
# PICKLE_JAR = config["environment"]["pickle_jar"]
PICKLE_JAR = Path(__file__).resolve().parent / 'toxicity' / config["environment"]["pickle_jar"]


class ContrastiveBuilder:
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        dataset_name: str = None,
    ):
        self.model = model
        self.device = self.model.device
        print(f"model device: {self.device}")
        self.tokenizer = tokenizer
        self.dataset = load_dataset(dataset_name) if dataset_name is not None else None

        self.T = len(self.model.model.layers)
        self.n = self.model.model.embed_tokens.embedding_dim
        self.m = self.n
        print(f"Latent dim: {self.n}")
        self.A_sum = None
        self.X_sum = None
        self.X_mean = None

        self.X = None # to allocate at runtime -- dependent on input length

        self.e_prev = None
        self.e_sum = None

        self.targets = None

        self.hooks = []
        self.mode = Mode.COLLECTING

        self.Kp = None
        self.Ki = None
        self.Kd = None

    def hook_collector(self, layer_idx, module, input, output):
        self.X[layer_idx] = input[0]
        if layer_idx == self.T-1:
            self.X[self.T] = output[0]
        return output
    
    def register_hooks(self):
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

    def hook_PID(self, layer_idx, module, input, output):
        x = input[0][:,-1,:]

        # if layer_idx == 6:
        e = self.targets[layer_idx] - x
        self.e_sum += e.squeeze(0)
        # print(f"alpha: {alpha/th.norm(self.E[layer_idx])}")

        u_t = self.Kp*e + self.Ki*self.e_sum + self.Kd*(e - self.e_prev)
        # print(f"x shape: {x.shape}")
        self.e_prev = e
        self.X[layer_idx] = x[-1,:]
        self.U[layer_idx] = u_t[-1]

        if isinstance(output,tuple):
            output[0][...,-1,:] = output[0][...,-1,:] + u_t
        else: 
            output[...,-1,:] = output[...,-1,:] + u_t
        return output

    def register_PID_hooks(self):
        """Register the hooks."""

        for layer_idx, layer in enumerate(self.model.model.layers):
            def hook_wrapper(layer_idx):
                def hook(module, input, output):
                    return self.hook_PID(layer_idx, module, input, output)

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
            self.register_hooks()
        elif self.mode == Mode.STEERING:
            self.register_PID_hooks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()

    def collect_data_test(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        # print(f"inputs: {inputs}")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"].float()
        embedding_layer = self.model.get_input_embeddings()
        hidden_states = embedding_layer(input_ids)
        self.X = th.zeros_like(hidden_states).repeat(self.T+1, 1, 1).to(self.device)

        with self:
            self.model.generate(input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=1,
                    return_dict_in_generate=True,
                    do_sample=False,
                    use_cache=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    )
        

        batch_size, seq_len = input_ids.shape
        position_ids = th.arange(seq_len, dtype=th.long, device=self.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len).to(self.device)

        position_embeddings = self.model.model.rotary_emb(hidden_states, position_ids)

        wrapped_tfs_temp = [partial(lqr.tf_block_wrapper, tf, attention_mask, position_ids, position_embeddings) for tf in self.model.model.layers]
        tfs_with_control_temp = [partial(lqr.transformerBlockControl, tf) for tf in wrapped_tfs_temp]
        # print(f"Xshape: {self.X.shape}")
        A, _ = lqr.linearize(tfs_with_control_temp,self.T,self.m,self.X)


        self.A_sum = self.A_sum + A

    def collect_data(self, num_samples, num_tokens, trait, filename, lb=0, ub=0.1, split="train", collect_A = False):#, num_A = 1):
        self.mode = Mode.COLLECTING
        data = self.dataset[split]
        filtered_data = [
            item["text"]
            for item in data["prompt"]
            if item[trait] is not None and item[trait] <= ub and item[trait] >= lb
        ]

        # A_iter = num_A
        sample = random.sample(filtered_data, num_samples)
        for prompt in sample:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            # print(f"inputs: {inputs}")
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"].float()
            embedding_layer = self.model.get_input_embeddings()
            hidden_states = embedding_layer(input_ids)
            self.X = th.zeros_like(hidden_states).repeat(self.T+1, 1, 1).to(self.device)

            with self:
                self.model.generate(input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=num_tokens,
                        return_dict_in_generate=True,
                        do_sample=False,
                        use_cache=False,
                        pad_token_id=self.tokenizer.eos_token_id,
                        )
            
            self.X_sum = self.X_sum + self.X[:,-1,:]


            if collect_A:# and A_iter > 0:
                batch_size, seq_len = input_ids.shape
                position_ids = th.arange(seq_len, dtype=th.long, device=self.device)
                position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len).to(self.device)

                position_embeddings = self.model.model.rotary_emb(hidden_states, position_ids)

                wrapped_tfs_temp = [partial(lqr.tf_block_wrapper, tf, attention_mask, position_ids, position_embeddings) for tf in self.model.model.layers]
                tfs_with_control_temp = [partial(lqr.transformerBlockControl, tf) for tf in wrapped_tfs_temp]
                A, _ = lqr.linearize(tfs_with_control_temp,self.T,self.m,self.X)
                self.A_sum = self.A_sum + A
                # A_iter -= 1


        total = num_samples*num_tokens
        print(f"total: {total}")
        if collect_A:
            tensor_dict = {
                "X": self.X_sum / total,
                "A": self.A_sum / total,
            } 
        else:
            tensor_dict = {
                "X": self.X_sum / total,
            } 

        with open((PICKLE_JAR / filename).with_suffix(".pkl"), "wb") as f:
            pickle.dump(tensor_dict, f)

    
    def collect_data_batch(self, prompts, num_samples, filename, num_tokens=1, batch_size=50):
        self.mode = Mode.COLLECTING
        X_sum = th.zeros((self.T+1, self.n,)).to(self.device)

        samples = random.sample(prompts, num_samples)
        for i in range(0,len(samples), batch_size):
            sample = samples[i:i+batch_size]
            inputs = self.tokenizer(
                sample, 
                return_tensors="pt", 
                padding=True,
                truncation=True,
            ).to(self.device)
            # print(f"inputs: {inputs}")
            input_ids = inputs["input_ids"]
            B,L = input_ids.shape
            # print(f"B,L: {B,L}")
            attention_mask = inputs["attention_mask"].float()
            embedding_layer = self.model.get_input_embeddings()
            hidden_states = embedding_layer(input_ids)
            self.X = th.zeros(self.T+1, B, L, hidden_states.size(-1), device=self.device)

            with self:
                self.model.generate(input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=num_tokens,
                        return_dict_in_generate=True,
                        do_sample=False,
                        use_cache=False,
                        pad_token_id=self.tokenizer.eos_token_id,
                        )
                
            X_sum += th.sum(self.X[:,:,-1,:], dim = 1)
            # X_mean = th.mean(self.X[:,:,-1,:], dim = 1)
        X_mean = X_sum / len(samples)

        total = num_samples*num_tokens
        print(f"total: {total}")

        tensor_dict = {
            "X": X_mean,
        } 

        with open((PICKLE_JAR / filename).with_suffix(".pkl"), "wb") as f:
            pickle.dump(tensor_dict, f)
        
        del self.X
        self.X = None

    def collect_activations(self, prompts, num_samples, filename=None, num_tokens=1, batch_size=50):
        self.mode = Mode.COLLECTING

        acts = th.zeros((num_samples, self.T+1, self.n))

        samples = random.sample(prompts, num_samples)
        for i in range(0,len(samples), batch_size):
            sample = samples[i:i+batch_size]
            inputs = self.tokenizer(
                sample, 
                return_tensors="pt", 
                padding=True,
                truncation=True,
            ).to(self.device)
            # print(f"inputs: {inputs}")
            input_ids = inputs["input_ids"]
            B,L = input_ids.shape
            # print(f"B,L: {B,L}")
            attention_mask = inputs["attention_mask"].float()
            embedding_layer = self.model.get_input_embeddings()
            hidden_states = embedding_layer(input_ids)
            self.X = th.zeros(self.T+1, B, L, hidden_states.size(-1), device=self.device)

            with self:
                self.model.generate(input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=num_tokens,
                        return_dict_in_generate=True,
                        do_sample=False,
                        use_cache=False,
                        pad_token_id=self.tokenizer.eos_token_id,
                        )

            acts[i:i+batch_size] = th.transpose(self.X[:,:,-1,:],0,1).detach().cpu()
            # X_mean = th.mean(self.X[:,:,-1,:], dim = 1)
        return acts

    def collect_acts_and_jacs(self, prompts, num_samples, filename, num_tokens=1, max_ctx=512): # 24 works for llama 8-9b
        self.mode = Mode.COLLECTING
        jacs = th.zeros((num_samples, self.T, self.n, self.n,))
        acts = th.zeros((num_samples, self.T+1, self.n))

        sample = random.sample(prompts, num_samples)
        print(f"sample: {sample}")
        iter = 1
        for i, prompt in enumerate(sample):
            print(f"iter: {iter}, prompt: {prompt}")
            iter += 1
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_ctx).to(self.device)

            # print(f"inputs: {inputs}")
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"].float()
            embedding_layer = self.model.get_input_embeddings()
            hidden_states = embedding_layer(input_ids)
            self.X = th.zeros_like(hidden_states).repeat(self.T+1, 1, 1).to(self.device)

            with th.no_grad():
                with self:
                    self.model.generate(input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=num_tokens,
                            return_dict_in_generate=True,
                            do_sample=False,
                            use_cache=False,
                            pad_token_id=self.tokenizer.eos_token_id,
                            )
            
            # self.X_sum = self.X_sum + self.X[:,-1,:]


            # and A_iter > 0:
            batch_size, seq_len = input_ids.shape
            position_ids = th.arange(seq_len, dtype=th.long, device=self.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len).to(self.device)

            position_embeddings = self.model.model.rotary_emb(hidden_states, position_ids)

            wrapped_tfs_temp = [partial(lqr.tf_block_wrapper, tf, attention_mask, position_ids, position_embeddings) for tf in self.model.model.layers]
            tfs_with_control_temp = [partial(lqr.transformerBlockControl, tf) for tf in wrapped_tfs_temp]
            A = lqr.linearize(tfs_with_control_temp, self.T,self.m,self.X)
            jacs[i] = A.detach().cpu()
            acts[i] = self.X[:,-1,:].detach().cpu()
            del A
            A = None
            del self.X
            self.X = None

        return acts,jacs


    def collect_jacobians(self, prompts, num_samples, filename, num_tokens=1, max_ctx=512): # 24 works for llama 8-9b
        self.mode = Mode.COLLECTING
        self.A_sum = th.zeros((self.T, self.n, self.n,)).to(self.device)

        sample = random.sample(prompts, num_samples)

        collection_times = []

        iter = 1
        for prompt in sample:
            print(f"iter: {iter}")
            start = timer()
            iter += 1
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_ctx).to(self.device)

            # print(f"inputs: {inputs}")
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"].float()
            embedding_layer = self.model.get_input_embeddings()
            hidden_states = embedding_layer(input_ids)
            self.X = th.zeros_like(hidden_states).repeat(self.T+1, 1, 1).to(self.device)

            with th.no_grad():
                with self:
                    self.model.generate(input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=num_tokens,
                            return_dict_in_generate=True,
                            do_sample=False,
                            use_cache=False,
                            pad_token_id=self.tokenizer.eos_token_id,
                            )
            
            # self.X_sum = self.X_sum + self.X[:,-1,:]


            # and A_iter > 0:
            batch_size, seq_len = input_ids.shape
            position_ids = th.arange(seq_len, dtype=th.long, device=self.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len).to(self.device)

            position_embeddings = self.model.model.rotary_emb(hidden_states, position_ids)

            wrapped_tfs_temp = [partial(lqr.tf_block_wrapper, tf, attention_mask, position_ids, position_embeddings) for tf in self.model.model.layers]
            tfs_with_control_temp = [partial(lqr.transformerBlockControl, tf) for tf in wrapped_tfs_temp]
            # A, _ = lqr.linearize(tfs_with_control_temp,self.T,self.m,self.X)
            A = lqr.linearize(tfs_with_control_temp,self.T,self.m,self.X)
            self.A_sum = self.A_sum + A
                # A_iter -= 1
            del A
            A = None
            del self.X
            self.X = None

            th.cuda.synchronize()
            end = timer()

            rt = end - start
            
            print(f"loop runtime: {rt}")
            collection_times.append(rt)
            gc.collect()
            if th.cuda.is_available():
                device_id = th.cuda.current_device()

                # Print allocated memory (currently used by tensors)
                print(f"th.cuda.memory_allocated: {th.cuda.memory_allocated(device_id)/1024**3:.3f}GB")
                
                # Print reserved memory (allocated by PyTorch's internal memory manager, including cached free blocks)
                print(f"th.cuda.memory_reserved: {th.cuda.memory_reserved(device_id)/1024**3:.3f}GB")
                
                # Print peak memory usage during the current process lifetime
                print(f"th.cuda.max_memory_reserved: {th.cuda.max_memory_reserved(device_id)/1024**3:.3f}GB")

                # Optional: Clear the memory cache (can make `nvidia-smi` report lower usage, but doesn't affect PyTorch's ability to allocate new tensors)
                th.cuda.empty_cache() 
            else:
                print("CUDA not available")


        total = num_samples*num_tokens
        print(f"total: {total}")
        tensor_dict = {
            "A": self.A_sum / total,
        } 

        # print("========================= Collection Times =========================")
        # times_th = th.tensor(collection_times)
        # print(f"th mean: {th.mean(times_th)}, std: {th.std(times_th)}")
        # print("===============================================================")



        with open((PICKLE_JAR / filename).with_suffix(".pkl"), "wb") as f:
            pickle.dump(tensor_dict, f)

        del self.A_sum
        self.A_sum = None
        

    def collect_jacobians_vram(self, prompts, num_samples, filename, num_tokens=1, max_ctx=512):
        self.mode = Mode.COLLECTING
        self.A_sum = th.zeros((self.T, self.n, self.n,)).to(self.device)

        sample = random.sample(prompts, num_samples)

        collection_times = []

        lqr.print_curr_mem("beginning of jac collection")

        iter = 1
        for prompt in sample:
            print(f"iter: {iter}")
            start = timer()
            iter += 1
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_ctx).to(self.device)

            # print(f"inputs: {inputs}")
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"].float()
            embedding_layer = self.model.get_input_embeddings()
            hidden_states = embedding_layer(input_ids)
            self.X = th.zeros_like(hidden_states).repeat(self.T+1, 1, 1).to(self.device)
            print(f"x nom shape: {self.X.shape}")

            with th.no_grad():
                with self:
                    self.model.generate(input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=num_tokens,
                            return_dict_in_generate=True,
                            do_sample=False,
                            use_cache=False,
                            pad_token_id=self.tokenizer.eos_token_id,
                            )
            
            # self.X_sum = self.X_sum + self.X[:,-1,:]

            lqr.print_curr_mem("before linearization in loop")

        

            # and A_iter > 0:
            batch_size, seq_len = input_ids.shape
            position_ids = th.arange(seq_len, dtype=th.long, device=self.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len).to(self.device)

            position_embeddings = self.model.model.rotary_emb(hidden_states, position_ids)

            wrapped_tfs_temp = [partial(lqr.tf_block_wrapper, tf, attention_mask, position_ids, position_embeddings) for tf in self.model.model.layers]
            tfs_with_control_temp = [partial(lqr.transformerBlockControl, tf) for tf in wrapped_tfs_temp]
            
            lqr.print_curr_mem("right before linearization in loop")
            
            with th.backends.cuda.sdp_kernel(
                enable_flash=False,
                enable_mem_efficient=False,
                enable_math=True
            ):
                print("RUNNING THE STREAMED JVP")
                lqr.linearize_jvp_streamed_gpu(tfs_with_control_temp,self.T,self.m,self.X,self.A_sum)
            # self.A_sum = self.A_sum + A
                # A_iter -= 1
            # del A
            # A = None
            del self.X
            self.X = None

            th.cuda.synchronize()
            end = timer()

            rt = end - start
            
            print(f"loop runtime: {rt}")
            lqr.print_curr_mem(f"after linearization in loop {iter}")
        total = num_samples*num_tokens
        print(f"total: {total}")
        tensor_dict = {
            "A": self.A_sum.detach().cpu() / total,
        } 

        print("========================= Collection Times =========================")
        times_th = th.tensor(collection_times)
        print(f"th mean: {th.mean(times_th)}, std: {th.std(times_th)}")
        print("===============================================================")



        with open((PICKLE_JAR / filename).with_suffix(".pkl"), "wb") as f:
            pickle.dump(tensor_dict, f)

        del self.A_sum
        self.A_sum = None
        

        


    def collect_sequentialPID(self, prompts, num_samples, filename, target_acts, kp=0.5, ki=0.01, kd=0.01, num_tokens=1):
        self.mode = Mode.STEERING
        
        self.Kp = kp
        self.Ki = ki
        self.Kd = kd

        self.targets = target_acts
        sample = random.sample(prompts, num_samples)
        

        inputs = self.tokenizer(
            sample, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
        ).to(self.device)

        input_ids = inputs["input_ids"]
        B,L = input_ids.shape
        self.e_sum = th.zeros((input_ids.shape[0], target_acts[0].shape[0]), device=self.device)
        self.e_prev = th.zeros((input_ids.shape[0], target_acts[0].shape[0]), device=self.device)

        attention_mask = inputs["attention_mask"].float()
        embedding_layer = self.model.get_input_embeddings()
        hidden_states = embedding_layer(input_ids)
        self.X = th.zeros(self.T+1, B, L, hidden_states.size(-1), device=self.device)

        with th.no_grad():
            with self:
                self.model.generate(input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=num_tokens,
                        return_dict_in_generate=True,
                        do_sample=False,
                        use_cache=False,
                        pad_token_id=self.tokenizer.eos_token_id,
                        )
            
            self.X_mean = th.mean(self.X[:,:,-1,:], dim = 1)

        total = num_samples*num_tokens
        print(f"total: {total}")

        tensor_dict = {
            "X_contr": self.X_mean,
        } 

        with open("../../scratch/" + filename + ".pkl", "wb") as f:
            pickle.dump(tensor_dict, f)


    @contextmanager
    def add_hooks(
        self,
        module_forward_pre_hooks: List[Tuple[nn.Module, Callable]] = None,
        module_forward_hooks: List[Tuple[nn.Module, Callable]] = None,
        **kwargs,
    ):
        """Context manager for temporarily adding forward hooks.

        Args:
            module_forward_pre_hooks: List of (module, hook_fn) tuples for pre-hooks
            module_forward_hooks: List of (module, hook_fn) tuples for forward hooks
            **kwargs: Additional keyword arguments passed to hook functions

        Yields:
            None. Hooks are active within the context, removed on exit.
        """
        module_forward_pre_hooks = module_forward_pre_hooks or []
        module_forward_hooks = module_forward_hooks or []
        handles = []
        try:
            for module, hook in module_forward_pre_hooks:
                partial_hook = functools.partial(hook, **kwargs)
                handles.append(module.register_forward_pre_hook(partial_hook))
            for module, hook in module_forward_hooks:
                partial_hook = functools.partial(hook, **kwargs)
                handles.append(module.register_forward_hook(partial_hook))
            yield
        finally:
            for h in handles:
                h.remove()


    def get_angular_steering_output_hook_replicate(
        self,
        steering_config: dict,
        target_degree: float,
        adaptive_mode: int = 1,
    ):
        """Create a hook that applies angular steering to layer outputs.

        Args:
            steering_config: Dict with 'first_direction' and 'second_direction' numpy arrays
            target_degree: Rotation angle in degrees (0-360)
            adaptive_mode: Steering application mode:
                        0 = always steer all activations
                        1 = only steer when activation is aligned with first_direction (conditional)

        Returns:
            Hook function that applies angular steering transformation to module outputs
        """
        first_direction = th.from_numpy(steering_config["first_direction"])
        second_direction = th.from_numpy(steering_config["second_direction"])

        # Compute rotation
        device = first_direction.device
        theta_rad = th.tensor(target_degree * th.pi / 180.0, device=device)

        # Orthonormalize directions
        b1 = first_direction / first_direction.norm()
        b2 = second_direction - (second_direction @ b1) * b1
        b2 = b2 / b2.norm()

        # Projection matrix
        proj_matrix = th.outer(b1, b1) + th.outer(b2, b2)

        # Rotation matrix
        cos_theta = th.cos(theta_rad)
        sin_theta = th.sin(theta_rad)
        rotation_matrix = th.stack(
            [th.stack([cos_theta, -sin_theta]), th.stack([sin_theta, cos_theta])]
        )

        # Steering vector
        unit_vector = th.tensor([1.0, 0.0], device=device)
        rotated_2d = rotation_matrix @ unit_vector
        steering_vector = rotated_2d[0] * b1 + rotated_2d[1] * b2

        _cache = {}

        def steering_hook(_module, _input, output):
            device = output.device
            dtype = output.dtype
            cache_key = (device, dtype)

            if cache_key not in _cache:
                _cache[cache_key] = (
                    proj_matrix.to(device=device, dtype=dtype),
                    steering_vector.to(device=device, dtype=dtype),
                    first_direction.to(device=device, dtype=dtype),
                )

            proj, steer, first_dir = _cache[cache_key]

            projected = output @ proj
            scale = projected.norm(dim=-1, keepdim=True)

            if adaptive_mode == 0:
                steered = output - projected + scale * steer
                return steered
            elif adaptive_mode == 1:
                proj_to_first = output @ first_dir
                mask = (proj_to_first > 0).unsqueeze(-1)
                steered = output - projected + scale * steer
                return th.where(mask, steered, output)
            else:
                raise ValueError(f"Unknown adaptive_mode: {adaptive_mode}")

        return steering_hook

    def angular_steer_collection(self, prompts, num_samples, filename, config, target_degree=180, max_new_tokens=1, lmbda=1, do_sample=False, temp=0.7, batch_size=50):
        self.mode = Mode.COLLECTING
        self.target_degree = th.tensor(target_degree)

        # inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        X_sum = th.zeros((self.T+1, self.n,)).to(self.device)

        samples = random.sample(prompts, num_samples)
        for i in range(0,len(samples), batch_size):
            sample = samples[i:i+batch_size]
            inputs = self.tokenizer(
                sample, 
                return_tensors="pt", 
                padding=True,
                truncation=True,
            ).to(self.device)
            # print(f"inputs: {inputs}")
            input_ids = inputs["input_ids"]
            B,L = input_ids.shape
            # print(f"B,L: {B,L}")
            attention_mask = inputs["attention_mask"].float()
            embedding_layer = self.model.get_input_embeddings()
            hidden_states = embedding_layer(input_ids)
            self.X = th.zeros(self.T+1, B, L, hidden_states.size(-1), device=self.device)

            module_dict = dict(self.model.named_modules())

            print(f"module dict:\n{module_dict}")


            output_hooks = [
                    (
                        module_dict[module_name],
                        self.get_angular_steering_output_hook_replicate(
                            steering_config=steering_config,
                            target_degree=target_degree,
                            adaptive_mode=1,
                        ),
                    )
                for module_name, steering_config in config.items() if "post_attention_layernorm" in module_name 
                ]

            with self.add_hooks(
                module_forward_hooks=output_hooks,
            ):
                with self:
                    self.model.generate(
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
            X_sum += th.sum(self.X[:,:,-1,:], dim = 1)
            # X_mean = th.mean(self.X[:,:,-1,:], dim = 1)
        X_mean = X_sum / len(samples)

        total = num_samples*max_new_tokens
        print(f"total: {total}")

        tensor_dict = {
            "X": X_mean,
        } 

        with open((PICKLE_JAR / filename).with_suffix(".pkl"), "wb") as f:
            pickle.dump(tensor_dict, f)
        
        del self.X
        self.X = None
