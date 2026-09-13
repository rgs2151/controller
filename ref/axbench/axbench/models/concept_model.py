from .model import Model
import torch, einops
import random
from tqdm.auto import tqdm
import os
import pandas as pd
from pyvene import (
    IntervenableConfig,
    IntervenableModel
)
from .interventions import (
    SubspaceIntervention,
    AdditionIntervention,
    ConceptVectorIntervention
)
from ..utils.constants import EXAMPLE_TAG
from torch.utils.data import DataLoader
from ..utils.model_utils import (
    set_decoder_norm_to_unit_norm,
    remove_gradient_parallel_to_decoder_directions,
    gather_residual_activations, 
    get_lr,
    calculate_l1_losses
)
from transformers import get_scheduler
from transformers import set_seed
from .preference_model import *


class ConceptModel(Model):
    def __str__(self):
        return 'ConceptModel'

    def make_model(self, **kwargs):
        pass

    def make_preference_dataloader(self, examples, **kwargs):
        data_module = make_preference_data_module(self.tokenizer, examples, **kwargs)
        g = torch.Generator()
        g.manual_seed(self.seed)
        train_dataloader = DataLoader(
            data_module["train_dataset"], shuffle=True, # we shuffle for examples.
            batch_size=self.training_args.batch_size, 
            collate_fn=data_module["data_collator"],
            generator=g)
        return train_dataloader

    def train(self, examples, **kwargs):
        if self.use_wandb:
            import wandb
            logging_metadata = kwargs["logging_metadata"]
            run_name = f"{logging_metadata['model_name']}_{logging_metadata['layer']}_{logging_metadata['concept_id']}"
            wandb_proj = kwargs.get("wandb_project", None)
            wandb_name = kwargs.get("wandb_name", None)
            run = wandb.init(
                project=f"{wandb_proj}", 
                entity=wandb_name,
                name=run_name,
                dir="wandb",
            )

        train_dataloader = self.make_preference_dataloader(
            examples, **kwargs)
        torch.cuda.empty_cache()

        # Optimizer and lr
        optimizer = torch.optim.AdamW(
            self.ax_model.parameters(), 
            lr=self.training_args.lr, weight_decay=self.training_args.weight_decay)
        num_training_steps = self.training_args.n_epochs * (len(train_dataloader) // self.training_args.gradient_accumulation_steps)
        lr_scheduler = get_scheduler(
            "linear", optimizer=optimizer,
            num_warmup_steps=0, num_training_steps=num_training_steps)
        # Main training loop.
        rank = torch.distributed.get_rank()
        progress_bar, curr_step, logging_step = tqdm(range(num_training_steps), position=rank, leave=True), 0, 0
        
        for epoch in range(self.training_args.n_epochs):
            for step, batch in enumerate(train_dataloader):
                # Implement minibatching to prevent OOM
                # Calculate the actual batch size after expansion
                expanded_batch_size = self.training_args.batch_size * len(self.preference_pairs)
                # Determine minibatch size (use original training args batch size)
                minibatch_size = self.training_args.batch_size
                # Number of minibatches needed
                num_minibatches = (expanded_batch_size + minibatch_size - 1) // minibatch_size
                
                # Prepare all inputs first
                winning_inputs = {
                    "input_ids": [],
                    "attention_mask": [],
                    "labels": [],
                    "intervention_locations": [],
                    "steering_factors": [],
                }
                # losing_inputs = {
                #     "input_ids": [],
                #     "attention_mask": [],
                #     "labels": [],
                #     "intervention_locations": [],
                #     "steering_factors": [],
                # }
                # winning and losing are separate minibatches
                # so that we can compute pair-wise losses
                for i in range(self.training_args.batch_size):
                    for pair in self.preference_pairs:
                        # winning
                        winning_inputs["input_ids"].append(batch[f"{pair}_winning_input_ids"][i])
                        winning_inputs["attention_mask"].append(batch[f"{pair}_winning_attention_mask"][i])
                        winning_inputs["labels"].append(batch[f"{pair}_winning_labels"][i])
                        winning_inputs["intervention_locations"].append(batch[f"{pair}_winning_intervention_locations"][i])
                        # # losing
                        # losing_inputs["input_ids"].append(batch[f"{pair}_losing_input_ids"][i])
                        # losing_inputs["attention_mask"].append(batch[f"{pair}_losing_attention_mask"][i])
                        # losing_inputs["labels"].append(batch[f"{pair}_losing_labels"][i])
                        # losing_inputs["intervention_locations"].append(batch[f"{pair}_losing_intervention_locations"][i])
                        winning_inputs["steering_factors"].append(torch.tensor(random.choice(self.training_args.steering_factors)))
                        # if "_add" in pair:
                        #     # if it is x, and winning = steered, losing = not steered
                        #     winning_inputs["steering_factors"].append(torch.tensor(random.choice(self.training_args.steering_factors)))
                        #     if self.training_args.substraction_type == "null_it_out":
                        #         losing_inputs["steering_factors"].append(torch.tensor(0.0))
                        #     else:
                        #         losing_inputs["steering_factors"].append(torch.tensor(random.choice(self.training_args.steering_factors)))
                        # else:
                        #     # if it is x, and winning = not steered, losing = steered
                        #     losing_inputs["steering_factors"].append(torch.tensor(random.choice(self.training_args.steering_factors)))
                        #     if self.training_args.substraction_type == "null_it_out":
                        #         winning_inputs["steering_factors"].append(torch.tensor(0.0))
                        #     else:
                        #         winning_inputs["steering_factors"].append(torch.tensor(random.choice(self.training_args.steering_factors)))
                
                # Initialize metrics accumulation for this batch
                batch_metrics = {}
                loss_sum = 0
                
                # Process each minibatch
                for mb in range(num_minibatches):
                    start_idx = mb * minibatch_size
                    end_idx = min((mb + 1) * minibatch_size, expanded_batch_size)
                    
                    if start_idx >= expanded_batch_size:
                        break
                    
                    # Create minibatch inputs
                    # winning is always before losing!
                    # minibatch_inputs = {
                    #     k: torch.stack(winning_inputs[k][start_idx:end_idx]+losing_inputs[k][start_idx:end_idx], dim=0).to(self.device) 
                    #     for k, _ in winning_inputs.items()
                    # }
                    minibatch_inputs = {
                        k: torch.stack(winning_inputs[k][start_idx:end_idx], dim=0).to(self.device) 
                        for k, _ in winning_inputs.items()
                    }
                    if isinstance(self.ax, list):
                        unit_locations = {"sources->base": (
                            None,
                            # repeat along first dimension
                            minibatch_inputs["intervention_locations"].permute(1, 0, 2).tolist() * len(self.ax)
                        )}
                    else:
                        unit_locations = {"sources->base": (
                            None,
                            minibatch_inputs["intervention_locations"].permute(1, 0, 2).tolist()
                        )}

                    # run ref, and policy with intv
                    subspaces = [{
                        "k": self.training_args.topk,
                        "steering_factor": minibatch_inputs["steering_factors"], 
                    }]
                    subspace_repeat = 1 if not isinstance(self.ax, list) else len(self.ax)
                    subspaces = subspaces * subspace_repeat
                    # forward
                    _, cf_outputs = self.ax_model(
                        base={
                            "input_ids": minibatch_inputs["input_ids"],
                            "attention_mask": minibatch_inputs["attention_mask"]
                        }, unit_locations=unit_locations, labels=minibatch_inputs["labels"],
                        subspaces=subspaces, use_cache=False)
                    
                    # loss
                    steer_loss = cf_outputs.loss
                    minibatch_loss = steer_loss
                    
                    # Normalize loss by total number of minibatches for this step
                    # (instead of dividing by gradient_accumulation_steps)
                    minibatch_loss = minibatch_loss / (num_minibatches * self.training_args.gradient_accumulation_steps)
                    
                    # Backward pass for this minibatch
                    minibatch_loss.backward()
                    
                    # Track total loss for logging
                    loss_sum += steer_loss.detach() * (end_idx - start_idx)
                
                loss = loss_sum / expanded_batch_size

                # Perform optimization step every gradient_accumulation_steps
                if (step + 1) % self.training_args.gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                    torch.nn.utils.clip_grad_norm_(self.ax_model.parameters(), 1.0)
                    curr_lr = get_lr(optimizer)
                    # optim
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    progress_bar.set_description(
                        "lr %.6f || loss %.6f" % (
                            curr_lr, loss))
                    curr_step += 1

        progress_bar.close()
        if self.use_wandb:
            run.finish()

    @torch.no_grad()
    def predict_latent(self, examples, **kwargs):
        self.ax.eval()
        batch_size = kwargs.get('batch_size', 32)
        return_max_act_only = kwargs.get("return_max_act_only", False)
        is_chat_model = kwargs.get("is_chat_model", False)
        eager_prepare_df = kwargs.get("eager_prepare_df", False)
        overwrite_concept_id = kwargs.get("overwrite_concept_id", None)

        all_acts = []
        all_max_act = []
        all_max_act_idx = []
        all_max_token = []
        all_tokens = []
        # Process in batches
        progress_bar = tqdm(range(0, len(examples), batch_size), desc="Processing batches")
        for i in progress_bar:
            batch = examples.iloc[i:i + batch_size]
            if eager_prepare_df:
                batch = prepare_df(batch, self.tokenizer, is_chat_model)

            # Batch encode all inputs
            inputs = self.tokenizer(
                batch["input"].tolist(), return_tensors="pt", 
                add_special_tokens=True, padding=True, truncation=True).to(self.device)
            
            gather_acts = gather_residual_activations(
                self.model, self.layer, inputs)
            outputs = self.ax(
                gather_acts[:, kwargs["prefix_length"]:],  # no bos token
                subspaces={
                    "subspaces": torch.tensor([overwrite_concept_id]*len(batch["input"])).to(self.device) \
                    if overwrite_concept_id is not None else torch.tensor(batch["concept_id"].tolist()).to(self.device),
                    "k": 1
                })
            ax_acts = outputs.latent[0].float().detach().cpu()

            seq_lens = inputs["attention_mask"].sum(dim=1) - kwargs["prefix_length"] # no bos token
            # Process each sequence in the batch
            for seq_idx, ax_seq in enumerate(ax_acts):
                acts = ax_seq[:seq_lens[seq_idx]].flatten().data.numpy().tolist()
                acts = [round(x, 3) for x in acts]
                max_act = max(acts)
                all_max_act.append(max_act)
            # clear memory and cache
            del ax_acts
            del gather_acts
            torch.cuda.empty_cache()

        return {
            "max_act": all_max_act
        }
        
    @torch.no_grad()
    def predict_latents(self, examples, **kwargs):
        self.ax.eval()
        batch_size = kwargs.get('batch_size', 32)
        all_acts = []
        all_max_act = []
        all_max_act_idx = []
        all_max_token = []
        all_tokens = []
        # Process in batches
        for i in range(0, len(examples), batch_size):
            batch = examples.iloc[i:i + batch_size]
            # Batch encode all inputs
            inputs = self.tokenizer(
                batch["input"].tolist(), return_tensors="pt", 
                add_special_tokens=True, padding=True, truncation=True).to(self.device)
            
            gather_acts = gather_residual_activations(
                self.model, self.layer, inputs)
            
            ax_acts_batch = torch.relu(torch.matmul(
                gather_acts[:, kwargs["prefix_length"]:], # bs, s, h
                self.ax.proj.weight.permute(1, 0) # h, d
            )).float().cpu().numpy()
            
            # Process each sequence in the batch
            seq_lens = inputs["attention_mask"].sum(dim=1) - kwargs["prefix_length"] # no bos token
            for seq_idx, row in enumerate(batch.itertuples()):
                # select acts with attention mask
                acts_batch = ax_acts_batch[
                    seq_idx, :seq_lens[seq_idx]]
                
                concept_acts = []
                concept_max_act = []
                concept_max_act_idx = []
                concept_max_token = []
                concept_tokens = []
                for row_idx in range(ax_acts_batch.shape[-1]):
                    # row_idx here is the concept id
                    acts = acts_batch[:, row_idx].flatten().tolist()
                    acts = [round(x, 3) for x in acts]
                    max_act = max(acts)
                    max_act_indices = [i for i, x in enumerate(acts) if x == max_act]
                    max_act_idx = max_act_indices[0]
                    # Get tokens for this specific sequence
                    tokens = self.tokenizer.tokenize(row.input)[kwargs["prefix_length"]-1:] # -1 is because it does not prepend BOS token
                    max_token = tokens[max_act_idx]
                    concept_acts.append(acts)
                    concept_max_act.append(max_act)
                    concept_max_act_idx.append(max_act_idx)
                    concept_max_token.append(max_token)
                    concept_tokens.append(tokens)
                all_acts.append(concept_acts)
                all_max_act.append(concept_max_act)
                all_max_act_idx.append(concept_max_act_idx)
                all_max_token.append(concept_max_token)
                all_tokens.append(concept_tokens)
        return {
            # "acts": all_acts,
            "max_act": all_max_act,
            # "max_act_idx": all_max_act_idx,
            # "max_token": all_max_token,
            # "tokens": all_tokens
        }
    
    def pre_compute_mean_activations(self, dump_dir, **kwargs):
        self.max_activations = {}
        return self.max_activations