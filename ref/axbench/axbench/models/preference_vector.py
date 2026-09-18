from .preference_model import *

class PreferenceVector(PreferenceModel):
    # the base class for all preference models
    preference_pairs = ["orig_add"] # "orig_add", "orig_sub", "steered_add", "steered_sub"
    def __str__(self):
        return 'PreferenceVector'

    def make_model(self, **kwargs):
        mode = kwargs.get("mode", "latent")
        overwrite_component = kwargs.get("overwrite_component", None)
        print("**Getting embed dim from the following model config**")
        if mode == "steering":
            intervention_type = kwargs.get("intervention_type", "addition")
            if intervention_type == "addition":
                ax = AdditionIntervention(
                    embed_dim=kwargs.get("embed_dim", self.model.config.hidden_size), 
                    low_rank_dimension=kwargs.get("low_rank_dimension", 1),
                )
            elif intervention_type == "addition_suppression":
                ax = AdditionSuppressionIntervention(
                    embed_dim=kwargs.get("embed_dim", self.model.config.hidden_size), 
                    low_rank_dimension=kwargs.get("low_rank_dimension", 1),
                )
            else:
                raise ValueError(f"Intervention type {intervention_type} not supported")
        else:
            intervention_type = kwargs.get("intervention_type", "addition")
            if intervention_type == "addition":
                ax = PreferenceVectorIntervention(
                    embed_dim=kwargs.get("embed_dim", self.model.config.hidden_size), 
                    low_rank_dimension=kwargs.get("low_rank_dimension", 1),
                    dropout=kwargs.get("dropout", 0.0),
                    intervention_positions_dropout=kwargs.get("intervention_positions_dropout", 0.0)
                )
        self.intervention_type = intervention_type
        layers = self.steering_layers if self.steering_layers else [self.layer]
        self.ax = ax.to(self.device)
        self.ax.train()
        ax_config = IntervenableConfig(representations=[{
            "layer": l,
            "component": f"model.layers[{l}].output" if overwrite_component is None else overwrite_component,
            "low_rank_dimension": kwargs.get("low_rank_dimension", 1),
            "intervention": self.ax} for l in layers])
        ax_model = IntervenableModel(ax_config, self.model)
        ax_model.set_device(self.device)
        self.ax_model = ax_model
        self.preference_pairs = kwargs.get("preference_pairs", ["orig_add"])

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