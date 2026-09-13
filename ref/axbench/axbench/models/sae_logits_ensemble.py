from .model import Model
import torch
import pandas as pd
import logging
from pathlib import Path
from dotenv import load_dotenv
from pyvene import IntervenableConfig, IntervenableModel
from .interventions import AdditionIntervention
from ..utils.model_utils import set_decoder_norm_to_unit_norm, gather_residual_activations
from torch.utils.data import DataLoader
from .probe import DataCollator, make_data_module
from sae_lens import SAE
import numpy as np
import re
import ast
import pickle
from openai import OpenAI
import json


load_dotenv()

logging.basicConfig(
    format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.WARN
)
logger = logging.getLogger(__name__)

import os
os.environ["OPENAI_API_KEY"] = os.getenv('openai_key')
os.environ["NP_API_KEY"] = os.getenv('np_key')


def rank_sae_features(
    features: pd.DataFrame,
    labels: pd.Series,
    target_label: int = 1,
    alpha: float = 0.3,
    min_active: int = 5,
    require_target_higher: bool = True,
    contrastive_labels=None,
) -> pd.DataFrame:
    """
    Rank SAE features by target vs. contrastive label activation patterns.
    Matches logic from latent_ranking.ipynb.
    """
    # === Split label groups ===
    if contrastive_labels is not None:
        contrastive_mask = labels.isin(contrastive_labels)
        contrastive_mask &= (labels != target_label)
    else:
        contrastive_mask = labels != target_label
    target_mask = labels == target_label

    if target_mask.sum() == 0:
        raise ValueError("No samples for target_label.")
    if contrastive_mask.sum() == 0:
        raise ValueError("No contrastive samples.")

    F_target = features[target_mask]
    F_contr = features[contrastive_mask]

    # === Activation counts & frequencies ===
    active_target = (F_target > 0).sum()
    active_contr = (F_contr > 0).sum()
    freq_target = active_target / len(F_target)
    freq_contr = active_contr / len(F_contr)
    freq_diff = (freq_target - freq_contr).clip(lower=0)
    
    # === Conditional means (only over active samples) ===
    sum_target_active = F_target.where(F_target > 0, 0.0).sum()
    sum_contr_active = F_contr.where(F_contr > 0, 0.0).sum()
    cond_mean_target = (sum_target_active / active_target.replace(0, np.nan)).fillna(0.0)
    cond_mean_constr = (sum_contr_active / active_contr.replace(0, np.nan)).fillna(0.0)

    # === Normal means (baseline for diff_mean) ===
    mean_target = F_target.mean()
    mean_contr = F_contr.mean()
    diff_mean = mean_target - mean_contr

    # === Calculation for power score (from sae_logits.py) ===
    denom = mean_target.replace(0, np.nan)
    percent_gap = (diff_mean / denom).fillna(0.0)

    # === Relative conditional difference ===
    abs_diff_cond = cond_mean_target - cond_mean_constr
    ratio_cond = abs_diff_cond / (cond_mean_constr + 1e-9)

    # === Filtering ===
    if require_target_higher:
        mask = (cond_mean_target > cond_mean_constr) & (cond_mean_target > 0)
    else:
        mask = cond_mean_target > 0
    keep = (active_target[mask] >= min_active)

    # === Combined scores ===
    # score_strength_freq_pow = ratio_cond[mask] * (freq_diff[mask] ** alpha)
    score_strength_freq_pow = percent_gap[mask] * (freq_target[mask] ** alpha)
    score_strength_freq_add = 0.5 * ratio_cond[mask] + 0.5 * freq_diff[mask]
    score_strength_freq_hmean = 2 * (ratio_cond[mask] * freq_diff[mask]) / (
        ratio_cond[mask] + freq_diff[mask] + 1e-9
    )

    # === Build result DataFrame ===
    res = pd.DataFrame({
        "cond_mean_target": cond_mean_target[mask][keep],
        "cond_mean_contrastive": cond_mean_constr[mask][keep],
        "ratio_cond": ratio_cond[mask][keep],
        "freq_diff": freq_diff[mask][keep],
        "active_target": active_target[mask][keep],
        # Scores
        "score_strength_freq_pow": score_strength_freq_pow[keep],
        "score_strength_freq_add": score_strength_freq_add[keep],
        "score_strength_freq_hmean": score_strength_freq_hmean[keep],
        "diff_mean": diff_mean[mask][keep],
    }).sort_values("score_strength_freq_pow", ascending=False)

    return res

def sae_latent_logit_lens(
    feature_id: int,
    sae_model,
    model,
    tokenizer,
    topk: int = 5,
    device: str = 'cpu'
):
    """Compute top logit tokens for an SAE latent using LogitLens for Gemma 2."""
    dtype = torch.bfloat16
    
    # Get language model head weights
    lm_w = model.lm_head.weight.to(device).to(dtype)
    
    # Get SAE decoder vector for this feature
    w_dec = sae_model.W_dec[feature_id].to(device).to(dtype)
    
    # Apply RMSNorm (Gemma 2 specific)
    norm_w = model.model.norm.weight.to(device).to(dtype)
    eps = model.model.norm.eps  # Gemma 2 uses 'eps' instead of 'variance_epsilon'
    
    # Normalize the decoder vector
    variance = w_dec.pow(2).mean()
    w_dec_normed = w_dec * torch.rsqrt(variance + eps) * norm_w
    
    # Project to vocabulary space
    delta = w_dec_normed @ lm_w.T
    
    # Get top-k tokens
    inc_vals, inc_ids = torch.topk(delta, topk)
    
    # Convert token IDs to strings
    inc_tokens = tokenizer.convert_ids_to_tokens(inc_ids.tolist())
    
    return inc_tokens, inc_vals.tolist()


class SAELogitsEnsemble(Model):
    """SAE-based steering using logit lens feature ranking."""
    
    def __str__(self):
        return "SAELogitsEnsemble"
    
    def make_model(self, **kwargs):
        """
        Initialize the model and intervention.
        This follows the pattern from mean.py and lsreft.py.
        """
        mode = kwargs.get("mode", "train")
        self.eval_data_path = kwargs.get("eval_data_path")
        self.alpha = kwargs.get("alpha", 0.2)
        self.min_active = kwargs.get("min_active", 3)
        self.topk_logits = kwargs.get("topk_logits", 5)
        self.llm_model = kwargs.get("llm_model", "gpt-4o-mini")
        self.max_concepts = kwargs.get("max_concepts", None)
        self.concept_path = kwargs.get("concept_path", "./axbench/concept500/prod_9b_l31_v1/generate/metadata.jsonl")
        self.concepts = pd.read_json(self.concept_path, lines=True)
        self.analysis_data_path = kwargs.get("analysis_data_path", "")
        self.analysis_data_path = Path(self.analysis_data_path)

        self.concepts_without_lm_latents = []

        # Alphas for ensemble sweep
        self.alphas_sweep = kwargs.get("alphas_sweep", [0.05, 20.])

        intervention_type = kwargs.get("intervention_type", "addition")
        

        if mode == "steering":
            # Setup for inference/steering mode
            if intervention_type == "addition":
                ax = AdditionIntervention(
                    embed_dim=self.model.config.hidden_size, 
                    low_rank_dimension=kwargs.get("low_rank_dimension", 1),
                )
            else:
                raise NotImplementedError(f"Intervention type {intervention_type} not supported for SAELogits.")
            
            self.ax = ax
            self.ax.train()
            
            # Create IntervenableModel for steering
            ax_config = IntervenableConfig(representations=[{
                "layer": self.layer,
                "component": f"model.layers[{self.layer}].output",
                "low_rank_dimension": kwargs.get("low_rank_dimension", 1),
                "intervention": self.ax
            }])
            ax_model = IntervenableModel(ax_config, self.model)
            ax_model.set_device(self.device)
            self.ax_model = ax_model
            
            logger.info(f"SAELogits IntervenableModel created for layer {self.layer}")
        else:
            # Setup for training mode
            if intervention_type == "addition":
                ax = AdditionIntervention(
                    embed_dim=self.model.config.hidden_size, 
                    low_rank_dimension=kwargs.get("low_rank_dimension", 1),
                )
            else:
                raise NotImplementedError(f"Intervention type {intervention_type} not supported for SAELogits.")
            
            self.ax = ax.to(self.device)
            logger.info(f"SAELogits initialized for training on layer {self.layer}")

    def make_dataloader(self, examples, **kwargs):
        """Creates a DataLoader from the examples DataFrame."""
        data_module = make_data_module(self.tokenizer, self.model, examples)
        train_dataloader = DataLoader(
            data_module["train_dataset"], 
            shuffle=False,
            batch_size=self.training_args.batch_size,
            collate_fn=data_module["data_collator"]
        )
        return train_dataloader

    def train(self, examples, **kwargs):
        """
        Create steering vectors for concepts using your SAE-Logits method.
        This is a non-gradient based "training" step.
        """
        logger.info("Starting steering vector generation for SAELogits...")
        self.concepts = pd.read_json(self.concept_path, lines=True)
        concept_id = kwargs.get('logging_metadata')['concept_id']

        # Extract concept info
        concept_row = self.concepts.iloc[concept_id]
        target_concept = concept_row['concept']
        
        # Extract concept type (genre)
        concept_type = "unknown"
        genres_map = concept_row.get('concept_genres_map')
        if isinstance(genres_map, dict):
            types_list = genres_map.get(target_concept)
            if isinstance(types_list, list) and len(types_list) > 0:
                concept_type = types_list[0]

        # 1. Create a dataloader for the examples
        train_dataloader = self.make_dataloader(examples, **kwargs)
        torch.cuda.empty_cache()

        # 2. Gather positive and negative activations
        positive_activations = []
        negative_activations = []
        logger.info("Gathering activations from the dataset...")
        for batch in train_dataloader:
            # Prepare input and move to device
            inputs = {k: v.to(self.device) for k, v in batch.items()}
            
            # Get activations from the specified layer
            activations = gather_residual_activations(
                self.model, self.layer, 
                {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}
            ).detach()
            
            # Align activations with labels, removing the prompt/prefix part
            prefix_length = kwargs.get("prefix_length", 0)
            if prefix_length >= activations.shape[1]:
                continue # Skip if the sequence is shorter than the prefix

            nonbos_mask = inputs["attention_mask"][:, prefix_length:]
            activations = activations[:, prefix_length:][nonbos_mask.bool()]
            
            # The labels are repeated for each token in the sequence.
            # We need to align them with the non-masked activations.
            labels = inputs["labels"].unsqueeze(1).repeat(
                1, inputs["input_ids"].shape[1] - prefix_length)
            
            # Filter activations based on labels
            positive_activations.append(activations[labels[nonbos_mask.bool()] == 1])
            negative_activations.append(activations[labels[nonbos_mask.bool()] != 1])

        logger.info(f"Gathered {len(positive_activations)} batches of activations.")

        # Concatenate all activations from the batches
        all_positive_activations = torch.cat(positive_activations, dim=0)
        all_negative_activations = torch.cat(negative_activations, dim=0)

        release = "gemma-scope-9b-it-res-canonical"
        sae_id = "layer_31/width_131k/canonical"
        sae = SAE.from_pretrained(release, sae_id).to(self.device)
        
        # 3. Get SAE feature activations
        logger.info("Encoding activations with SAE...")
        
        # Process in batches to avoid OOM
        encode_batch_size = kwargs.get("encode_batch_size", 512)
        
        def encode_in_batches(activations, batch_size):
            """Encode activations in batches to avoid OOM."""
            features_list = []
            num_samples = activations.shape[0]
            
            for i in range(0, num_samples, batch_size):
                batch = activations[i:i + batch_size]
                with torch.no_grad():
                    batch_feats = sae.encode(batch)
                features_list.append(batch_feats.cpu())
                
                # Log progress
                if (i // batch_size) % 10 == 0:
                    logger.info(f"Encoded {i + batch.shape[0]}/{num_samples} samples...")
            
            return torch.cat(features_list, dim=0)
        
        logger.info(f"Encoding {all_positive_activations.shape[0]} positive activations in batches of {encode_batch_size}...")
        pos_feats = encode_in_batches(all_positive_activations, encode_batch_size)
        
        logger.info(f"Encoding {all_negative_activations.shape[0]} negative activations in batches of {encode_batch_size}...")
        neg_feats = encode_in_batches(all_negative_activations, encode_batch_size)
        
        all_feats_df = pd.DataFrame(torch.cat([pos_feats, neg_feats], dim=0).numpy())
        all_labels = pd.Series([1] * len(pos_feats) + [0] * len(neg_feats))

        # 4. Rank features (Ensemble Strategy)
        logger.info("Ranking SAE features with Ensemble Strategy...")
        
        ranking_results = {}
        top_50_indices = {}
        
        # A. Power Score Sweep
        for alpha in self.alphas_sweep:
            res = rank_sae_features(
                features=all_feats_df,
                labels=all_labels,
                target_label=1,
                alpha=alpha,
                min_active=self.min_active
            )
            key = f"power_alpha_{alpha}"
            ranking_results[key] = res.index.tolist()[:50]
            top_50_indices[key] = res.index[:50].tolist()
            
        # Use the result from the first alpha to get other scores (they don't depend on alpha)
        base_res = rank_sae_features(all_feats_df, all_labels, 1, alpha=self.alphas_sweep[0], min_active=self.min_active)
        
        # 5. Create Union & Logit Lens
        union_candidates = set()
        for indices in top_50_indices.values():
            union_candidates.update(indices)
            
        logger.info(f"Union set size: {len(union_candidates)} candidates from {len(top_50_indices)} methods.")

        # 6. LLM-based selection using logit lens
        logger.info("Computing logit lens scores for union candidates...")
        candidates_with_tokens = {}
        
        for idx in union_candidates:
            inc_tokens, inc_vals = sae_latent_logit_lens(
                feature_id=idx,
                sae_model=sae,
                model=self.model,
                tokenizer=self.tokenizer,
                topk=self.topk_logits,
                device=self.device
            )
            
            clean_tokens = [tok.replace(' ', '').strip() for tok in inc_tokens if tok.replace(' ', '').strip()]
            
            # Get weight from base_res (it contains all valid indices)
            if idx in base_res.index:
                sae_weight = base_res.loc[idx, 'cond_mean_target']
            else:
                sae_weight = 0.0
                
            weighted_score = inc_vals[0] * sae_weight if inc_vals else 0.0
            
            # Calculate average logit score for analysis (SAE Mean * Logit Avg)
            logit_avg = np.mean(inc_vals) if inc_vals else 0.0
            avg_logit_score = logit_avg * sae_weight
            
            candidates_with_tokens[idx] = {
                'tokens': ', '.join(clean_tokens[:self.topk_logits]),
                'score': weighted_score,
                'raw_logit': inc_vals[0] if inc_vals else 0.0,
                'sae_weight': sae_weight,
                'logit_avg': logit_avg,       # Store for individual inspection
                'avg_logit_score': avg_logit_score # Store for individual inspection
            }
        
        # Sort candidates by weighted score (descending) for the prompt
        sorted_candidates = sorted(
            candidates_with_tokens.items(), 
            key=lambda x: x[1]['logit_avg'], 
            reverse=True
        )
        
        candidates_str = '\n'.join([
            f"{idx}: {info['tokens']}"
            for idx, info in sorted_candidates
        ])
        
        prompt = f"""You are identifying the foundational SAE neurons that best represent the concept: '{target_concept}'.

**Goal**: Construct a precise steering vector by selecting only the neurons that capture the essence of the concept, filtering out diluted or circumstantial associations.

**Selection Guidelines**:
1. **Prioritize Exact Matches**: Retain neurons where the predicted tokens are direct synonyms, exact references, or the concept itself.
2. **Exclude Contextual & Compositional Features**: Do not include neurons that represent:
   - *Contextual associations* (e.g., for "Apple", exclude "pie", "tree", "eating").
   - *Grammatical or Structural patterns* (e.g., typical adjectives, prepositions, or sentence structures).
   - *Related but distinct* concepts.
3. **Maintain Purity**: Omit neurons that appear vague or polysemantic.
4. **Avoid Redundancy**: Before selecting a neuron to keep:
   - Track which token sets you've already included.
   - Check if any previously selected neuron already has a very similar or nearly identical token set. If a duplicate is found, **EXCLUDE** it to avoid diluting the steering vector with redundant information.

**Output Instructions**:
1. **Re-Rank by Relevance**: The input list is sorted statistically. Please output a new list re-ordered by **semantic relevance**, placing the most accurate representation first.
2. **Stop at Saturation**: Do not aim for a specific number of features. If a small set of latents (even just 1 or 2) fully embodies the concept, **STOP**. Including lower-quality or tangential latents may degrade the steering vector.

**Candidates**:
{candidates_str}

Output Format:
Thinking: [For each latent: brief reasoning on the selection and re-ordering, noting any redundancies detected]
FINAL_LIST: [best_id, second_best_id, ...]
"""
        
        logger.warning(f"Querying model for final feature selection...")

        client = OpenAI()
        response = client.responses.create(
            model="gpt-5.2",
            input=prompt,
            reasoning={ "effort": "high"},
            text={ "verbosity": "medium" },
        )
        response_text = response.output_text.strip()

        logger.warning(f"\n--- LLM Response ---\n{response_text}\n--- End Response ---\n")

        if "FINAL_LIST:" in response_text:
            after_label = response_text.split("FINAL_LIST:", 1)[1]
            list_match = re.search(r'\[[\d,\s]+\]', after_label)
            if list_match:
                selected_indices = ast.literal_eval(list_match.group(0))
            else:
                logger.warning("Found 'FINAL_LIST:' but no valid list after it. Using top 5 candidates.")
                selected_indices = [idx for idx, _ in sorted_candidates[:5]]
                concept_id = kwargs.get('logging_metadata')['concept_id']
                self.concepts_without_lm_latents.append(concept_id)

        else:
            list_match = re.search(r'\[[\d,\s]+\]', response_text)
            if list_match:
                selected_indices = ast.literal_eval(list_match.group(0))
            else:
                logger.warning("Could not find list in LLM response. Using top 5 candidates.")
                selected_indices = [idx for idx, _ in sorted_candidates[:5]]
                concept_id = kwargs.get('logging_metadata')['concept_id']
                self.concepts_without_lm_latents.append(concept_id)

        logger.info(f"LLM selected {len(selected_indices)} features for steering.")

        # Calculate average score for the concept based on selected steering features
        concept_avg_score = 0.0

        if selected_indices:
            concept_avg_score = np.mean([candidates_with_tokens[idx]['avg_logit_score'] for idx in selected_indices])

        # 7. Save Analysis Data
        analysis_data = {
            "concept_id": concept_id,
            "concept": target_concept,
            "concept_type": concept_type,
            "concept_avg_sae_mean_logit_score": concept_avg_score,
            "rankings": ranking_results, 
            "top_50_indices": top_50_indices,
            "union_candidates": union_candidates,
            "candidates_info": candidates_with_tokens,
            "llm_response_text": response_text,
            "relevant_indices": selected_indices,
            "filtered_indices": selected_indices,
            "alphas_used": self.alphas_sweep
        }
        
        # Save to disk
        save_dir = Path(self.analysis_data_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = self.analysis_data_path / f"concept_{concept_id}_analysis.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(analysis_data, f)
        logger.info(f"Saved analysis data to {save_path}")

        # 8. Create steering vector from SELECTED features
        steering_vector = torch.zeros(sae.cfg.d_in, device=self.device, dtype=self.model.dtype)

        # Filter indices to ensure we have data for them
        valid_indices = [idx for idx in selected_indices if idx in candidates_with_tokens]

        if valid_indices:
            # Rank-Monotonicity Enforcement:

            metrics = []
            for idx in valid_indices:
                info = candidates_with_tokens[idx]
                
                metrics.append({
                    'current_weight': info['sae_weight'],
                    'score': info['score'],
                    'raw_logit': info['raw_logit']
                })
            
            running_max_score = -float('inf')
            final_weights = []

            for m in reversed(metrics):
                target_score = max(m['score'], running_max_score)
                running_max_score = target_score
                
                if abs(m['raw_logit']) > 1e-4:
                    new_weight = target_score / m['raw_logit']
                else:
                    new_weight = m['current_weight']
                
                final_weights.append(new_weight)
            
            # Reverse back to original order
            final_weights.reverse()
            
            # Construct vector
            for i, idx in enumerate(valid_indices):
                steering_vector += sae.W_dec[idx] * final_weights[i]
        
        # 9. Set the steering vector in the intervention object
        self.ax.proj.weight.data = steering_vector.unsqueeze(0)
        set_decoder_norm_to_unit_norm(self.ax)

    def intervene(self, intervention_type="addition", **kwargs):
        """
        This method is not strictly needed if you follow the standard pattern,
        as the framework will handle creating the IntervenableModel for inference.
        """
        logger.info("SAELogits intervene method called (usually for inference setup).")
        return super().intervene(intervention_type, **kwargs)