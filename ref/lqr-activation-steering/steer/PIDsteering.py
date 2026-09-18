import torch as th
from transformers import AutoTokenizer, AutoModelForCausalLM
import steer.lqr_utils as lqr
from functools import partial
from enum import Enum
import time
from tqdm import tqdm

class Mode(Enum):
    STEERING = 0
    SETPOINT = 1
    ACTADD = 2
th.autograd.set_detect_anomaly(True)


class PIDSteering:
    '''
    Contrastive method currently assuming precomputed:
        - contrastive vectors
    '''

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        kp: float = 10,
        ki: float = 10,
        kd: float = 1,
        A: th.Tensor = None,    
        contrastive_vecs: th.Tensor = None,
    ):
        self.model = model
        self.device = model.device
        self.tokenizer = tokenizer
        self.A = A
        self.E = contrastive_vecs
        self.contrastive = False

        self.T = len(model.model.layers)
        self.n = model.model.embed_tokens.embedding_dim
        self.m = self.n


        self.Kp = kp
        self.Ki = ki
        self.Kd = kd
        
        
        
        self.X = None # to allocate at runtime
        self.e_prev = None
        self.U = th.zeros((self.T, self.n), device=self.device)
        self.e_sum = th.zeros_like(self.E[0])

        self.betas = None
        self.E_unit = th.zeros_like(self.E)

        self.hooks = []
        self.mode = Mode.SETPOINT
        

        self.iter = 0

        self.layer_inds = []


    def hook_steering(self, layer_idx, module, input, output):
        # u_t = self.K[layer_idx]@(self.E[layer_idx]) # can be computed offline
        e = self.E[layer_idx]
        self.e_sum += e
        u_t = self.Kp*e + self.Ki*self.e_sum + self.Kd*(e - self.e_prev)
        self.e_prev = e

        self.U[layer_idx] = u_t[-1,:]
        self.X[layer_idx] = input[0][0,-1,:]

        # output[0][:,-1,:] = output[0][:,-1,:] + u_t # 4.40
        # output[0][...,-1,:] = output[0][...,-1,:] + u_t # new

        if isinstance(output,tuple):
            output[0][...,-1,:] = output[0][...,-1,:] + u_t
        else: 
            output[...,-1,:] = output[...,-1,:] + u_t
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

        else:
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

        diff = x_t - self.X[self.iter][layer_idx,-1,:]
        u_t = -self.K[layer_idx]@(diff)
        # u_t = -(diff)
        self.U[layer_idx] = u_t

        output[0][...,-1,:] = output[0][...,-1,:] + u_t # new

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
        x = input[0][:,-1,:]

        v = self.E_unit[layer_idx].to(x.dtype)
        alpha = th.tensor([self.betas[layer_idx] for i in range(x.shape[0])], device=self.device, dtype=x.dtype) - th.bmm(v.unsqueeze(0).unsqueeze(0), th.transpose(x.unsqueeze(0),-2,-1))
        e = alpha.squeeze(0).T @ v.unsqueeze(0)
        self.e_sum += e

        if layer_idx % 10 == 0:
            self.e_sum = self.e_sum * 0        

        u_t = self.Kp*e + self.Ki*self.e_sum + self.Kd*(e - self.e_prev)
        # print(f"x shape: {x.shape}")
        self.e_prev = e
        self.X[layer_idx] = x[-1,:]
        self.U[layer_idx] = u_t[-1]


        if not th.isfinite(u_t).all():
            print(f"layer index: {layer_idx}")
            print(f"e: {e}")
            print(f"e_prev: {self.e_prev}")
            print(f"e_prev: {self.e_sum}")
            raise RuntimeError("u_t contains NaN or Inf")
        if isinstance(output,tuple):
            output[0][...,-1,:] = output[0][...,-1,:] + u_t
        else: 
            output[...,-1,:] = output[...,-1,:] + u_t
            

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

    def register_actadd_setpoint_tracking_hooks(self):
        """Register the hooks."""

        for layer_idx in self.layer_inds:
            layer = self.model.model.layers[layer_idx]
            def hook_wrapper(layer_idx):
                def hook(module, input, output):
                    return self.hook_setpoint_tracking(layer_idx, module, input, output)

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
        if self.mode == Mode.STEERING:
            self.register_steering_hooks()
        elif self.mode == Mode.SETPOINT:
            self.register_setpoint_tracking_hooks()
        elif self.mode == Mode.ACTADD:
            self.register_actadd_setpoint_tracking_hooks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()

    def evaluate(self, prompt, max_new_tokens, do_sample=False, temp=0.7):
        '''
        Steers with no setpoint, always 'tracking' the full contrastive vector.
        Likley not the desired behavior, not considered in the manuscript.
        '''
        self.mode = Mode.STEERING
        
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True,
            truncation=True,).to(self.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        self.X = th.zeros((self.T+1, self.n)).to(self.device)

        self.e_sum = th.zeros((input_ids.shape[0], self.E[0].shape[0]), device=self.device)
        self.e_prev = th.zeros((input_ids.shape[0], self.E[0].shape[0]), device=self.device)

        with th.no_grad():
            with self:
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

        output_str = self.tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
        return output_str



    def track_setpoint_actadd(self, prompt, max_new_tokens, lmbda=1, layer_inds=[5], do_sample=False, temp=1, return_tokens=False):
        '''
        Run ActAddLFS, which tracks the LFS setpoint with a simple P controller.

        Only intervenes at layers specified by layer_inds
        
        '''
        
        self.mode = Mode.ACTADD
        self.Kd = 0
        self.Ki = 0
        self.layer_inds = layer_inds

        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
        ).to(self.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        self.X = th.zeros((self.T+1, self.n)).to(self.device)
        self.e_sum = th.zeros((input_ids.shape[0], self.E[0].shape[0]), device=self.device)
        self.e_prev = th.zeros((input_ids.shape[0], self.E[0].shape[0]), device=self.device)

        self.betas = [0 for i in range(self.T+1)]
        for i, e in enumerate(self.E):
            nrm = th.linalg.norm(e)
            if nrm < 1e-6:
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

        # output_str = self.tokenizer.decode(output.sequences[0], skip_special_tokens=True)
        if return_tokens:
            return output.sequences

        output_str = self.tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
        return output_str


    def track_setpoint(self, prompt, max_new_tokens, lmbda=1, do_sample=False, temp=1, return_tokens=False):
        '''
        S-PID implementation: tracks the LFS setpoint with a PID controller at every layer.

        args:
            prompt: list of text inputs
            max_new_tokens: maximum tokens to generate
            lmbda: setpoint target (typically 1-2.5)
            do_sample: greedy decoding if False, set to True in all scripts except refusal
            temp: sampling temperature, N/A if do_sample = False
            return_tokens: return generated tokens before decoding

        '''
        
        
        self.mode = Mode.SETPOINT

        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
        ).to(self.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        self.X = th.zeros((self.T+1, self.n)).to(self.device)
        # print(f"E shape: {self.E[0].shape}")
        self.e_sum = th.zeros((input_ids.shape[0], self.E[0].shape[0]), device=self.device)
        self.e_prev = th.zeros((input_ids.shape[0], self.E[0].shape[0]), device=self.device)


        self.betas = [0 for i in range(self.T+1)]
        for i, e in enumerate(self.E):
            # print(f"e: {e}")
            nrm = th.linalg.norm(e)
            if nrm < 1e-6:
                self.E_unit[i] = self.E_unit[i]*0
                # raise ValueError("norm is 0")
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
        return output_str
