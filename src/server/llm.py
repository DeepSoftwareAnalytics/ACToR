import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Tuple, List, Dict, Any
import numpy as np
from loguru import logger


class LLM:
    def __init__(self, model_path: str, max_new_tokens: int | None = 512, temperature: float | None = 0.7, n: int | None = 1):
        """
        Initialize LLM class (using transformers library)
        
        Args:
            model_path: Model path, can be local path or HuggingFace model name
            max_new_tokens: Maximum number of tokens to generate
            temperature: Temperature parameter
            n: Number of generated samples
        """
        try:
            logger.info(f"model_path: {model_path}")
            self.model_path = model_path
            self.temperature = temperature
            self.max_new_tokens = max_new_tokens
            self.n = n

            self.tokenizer = AutoTokenizer.from_pretrained(model_path, revision="main", local_files_only=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                revision="main",
                local_files_only=True,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="eager"
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Set pad_token (if it doesn't exist)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info(f"Successfully created transformers LLM instance, model path: {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to create transformers LLM instance: {str(e)}")
            raise

    def generate_text(
        self,
        prompt: str,
    ) -> List[str]:
        """
        Generate text using transformers LLM
        
        Args:
            prompt: User input prompt
            
        Returns:
            List[str]: List of generated texts
        """
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Set generation parameters based on temperature
            if self.temperature == 0:
                # Greedy decoding
                logger.info(f"Using greedy decoding with temperature {self.temperature}")
                generation_params = {
                    "max_new_tokens": self.max_new_tokens,
                    "num_return_sequences": self.n,
                    "do_sample": False,
                    "pad_token_id": self.tokenizer.eos_token_id,
                    "return_dict_in_generate": True,
                    "output_scores": True
                }
            else:
                # Temperature sampling
                logger.info(f"Using temperature sampling with temperature {self.temperature}")
                generation_params = {
                    "max_new_tokens": self.max_new_tokens,
                    "temperature": self.temperature,
                    "num_return_sequences": self.n,
                    "do_sample": True,
                    "pad_token_id": self.tokenizer.eos_token_id,
                    "return_dict_in_generate": True,
                    "output_scores": True
                }
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **generation_params
                )
            
            generated_texts = []
            prompt_len = len(prompt)
            for i in range(self.n):
                generated_tokens = outputs.sequences[i]
                generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                generated_text = generated_text[prompt_len:]
                generated_texts.append(generated_text)
            
            return generated_texts
            
        except Exception as e:
            logger.error(f"Text generation failed: {str(e)}")
            raise

    def generate_token(
        self,
        prompt: str,
    ) -> Tuple[torch.Tensor, str, int, Tuple[Tuple[torch.Tensor, ...], ...], List[Tuple[str, int, float]]]:
        """
        Generate a single token using transformers LLM (first generation, returns KV cache)
        
        Args:
            prompt: User input prompt
            
        Returns:
            Tuple[torch.Tensor, str, int, Tuple[Tuple[torch.Tensor, ...], ...]]: 
                - hidden_states: Hidden layer states, converted to tensor
                - generated_token: Generated token text
                - generated_token_id: Generated token id
                - past_key_values: KV cache
        """
        try:
            # First generation, input complete prompt
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1,
                    do_sample=True if self.temperature > 0 else False,
                    temperature=self.temperature if self.temperature > 0 else None,
                    pad_token_id=self.tokenizer.eos_token_id,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                    output_scores=True,
                    use_cache=True
                )
            
            # Get logits
            logits = outputs.scores[0][0]  # (vocab_size,)
            
            # Get generated token id (maintain original logic)
            generated_token_id = outputs.sequences[0][-1].item()
            
            # Decode generated token
            generated_token = self.tokenizer.decode([generated_token_id], skip_special_tokens=True)

            last_layer = outputs.hidden_states[0][-1]
            last_token_hidden_state = last_layer[:,-1,:]
            
            # Ensure hidden_state is 2D tensor, as classifier expects input to be [batch_size, features]
            if last_token_hidden_state.dim() == 1:
                last_token_hidden_state = last_token_hidden_state.unsqueeze(0)  # Add batch dimension
            
            # Get KV cache
            past_key_values = outputs.past_key_values
                             
            return last_token_hidden_state, generated_token, generated_token_id, past_key_values
            
        except Exception as e:
            logger.error(f"Token generation failed: {str(e)}")
            raise
    
    def generate_token_with_cache(
        self,
        token_id: int,
        past_key_values: Tuple[Tuple[torch.Tensor, ...], ...],
    ) -> Tuple[torch.Tensor, str, int, Tuple[Tuple[torch.Tensor, ...], ...]]:
        """
        Generate a single token using transformers LLM (continuation mode, using KV cache)
        
        Args:
            token_id: Input token id
            past_key_values: KV cache for accelerating generation
            
        Returns:
            Tuple[torch.Tensor, str, int, Tuple[Tuple[torch.Tensor, ...], ...]]: 
                - hidden_states: Hidden layer states, converted to tensor
                - generated_token: Generated token text
                - generated_token_id: Generated token id
                - new_past_key_values: New KV cache
        """
        try:
            # Continuation mode, only need to input token id
            assert past_key_values is not None
            input_ids = torch.tensor([[token_id]], device=self.model.device)
            
            # Calculate attention mask shape
            # past_key_values[0][0] shape is (batch, num_heads, past_seq_len, head_dim)
            # We need total length of past_seq_len + current_seq_len(1)
            past_seq_len = past_key_values[0][0].shape[2]  # Get past sequence length
            current_seq_len = input_ids.shape[1]  # Current input length (1)
            total_seq_len = past_seq_len + current_seq_len
            attention_mask = torch.ones(1, total_seq_len, device=self.model.device)
            
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    past_key_values=past_key_values,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=True,
                    return_dict=True
                )
            
            # Get logits and generate next token
            # logits shape: (batch, 1, vocab_size), take the last position
            logits = outputs.logits[:, -1, :]  # (batch, vocab_size)
            
            # Set generation strategy based on temperature
            if self.temperature == 0:
                # Greedy decoding
                generated_token_id = torch.argmax(logits, dim=-1).item()
            else:
                # Temperature sampling
                probs = torch.softmax(logits / self.temperature, dim=-1)
                generated_token_id = torch.multinomial(probs, 1).item()
            
            # Decode generated token
            generated_token = self.tokenizer.decode([generated_token_id], skip_special_tokens=True)
            
            # Get hidden states - last layer last position
            last_hidden_state = outputs.hidden_states[-1][:, -1, :]  # (batch, hidden_size)
            
            # Get new past_key_values
            new_past_key_values = outputs.past_key_values
            
            return last_hidden_state, generated_token, generated_token_id, new_past_key_values
            
        except Exception as e:
            logger.error(f"Token generation failed: {str(e)}")
            raise
    
    def generate_attention_score_list(self, prompt: str, ground_truth: str) -> List[List[float]]:
        """
        Generate attention score list.
        Returns: A list of length equal to ground_truth tokens. The i-th element is a list of attention scores
        from "the token itself (including self-attention) to the end of sequence" query positions to that token
        (in chronological order, from that token to the last token).
        """
        try:
            # --- 1) Tokenize (without special tokens, maintain original order) ---
            enc_prompt = self.tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False)
            enc_gt = self.tokenizer.encode(ground_truth, return_tensors="pt", add_special_tokens=False)

            # Move to model device
            device = self.model.device
            prompt_ids = enc_prompt.to(device)
            ground_truth_ids = enc_gt.to(device)

            # --- 2) Concatenate inputs (assuming single sample), and construct attention mask ---
            full_input_ids = torch.cat([prompt_ids, ground_truth_ids], dim=1)  # (1, seq_len)
            # If model has pad_token_id, construct mask based on pad, otherwise all 1
            pad_id = getattr(self.tokenizer, "pad_token_id", None)
            if pad_id is not None:
                attention_mask = (full_input_ids != pad_id).long()
            else:
                attention_mask = torch.ones_like(full_input_ids).long()

            prompt_len = prompt_ids.shape[1]
            gt_len = ground_truth_ids.shape[1]
            seq_len = full_input_ids.shape[1]

            # --- 3) Use forward to get attentions ---
            with torch.no_grad():
                outputs = self.model(
                    input_ids=full_input_ids,
                    attention_mask=attention_mask,
                    output_attentions=True,
                    use_cache=False,  # Disable KV cache to get attention weights
                    return_dict=True
                )
            
            try:
                last_layer_attn = outputs.attentions[-1]  # (batch, heads, seq_len, seq_len)
                avg_attention = last_layer_attn.mean(dim=(0, 1))  # (seq_len, seq_len)
            except Exception as e:
                logger.error(f"Failed to calculate attention average: {e}")
                raise

            # --- 4) Calculate attention score list for each ground-truth token being attended by "itself and subsequent tokens" ---
            attention_score_list: List[List[float]] = []
            for i in range(gt_len):
                # Absolute index (in full sequence)
                token_abs_idx = prompt_len + i
                # We want attention values from queries at rows = token_abs_idx ... seq_len-1 to token_abs_idx (as key)
                # avg_attention[row, col] -> row = query position, col = key position
                rows_start = token_abs_idx
                rows_end = seq_len  # exclusive
                col = token_abs_idx

                # Extract and convert to Python list
                # Note: avg_attention dtype is torch.float
                token_attention_scores = avg_attention[rows_start:rows_end, col].cpu().numpy().tolist()
                attention_score_list.append(token_attention_scores)
            if len(attention_score_list) != gt_len:
                logger.error(f"Attention score list length incorrect: {len(attention_score_list)} != {gt_len}")
                raise ValueError(f"Attention score list length incorrect: {len(attention_score_list)} != {gt_len}")
            return attention_score_list

        except Exception as e:
            logger.error(f"Failed to generate attention score list: {str(e)}")
            raise
        
    def get_training_data(self, prompt: str, ground_truth: str) -> List[Dict[str, Any]]:
        """
        Generate training data (teacher-forcing approach, using optimized cache management).
        Returns: For each generation step (corresponding to a token in ground truth), generates a dict containing:
        - step: Position index (starting from 0, relative to ground_truth)
        - pred_token_id: Model predicted token id
        - pred_token: Model predicted token text
        - pred_logprob: Model's log probability for the predicted token (consistent with sampling distribution when using self.temperature)
        - gt_token_id: Ground truth token id
        - gt_token: Ground truth token text
        - gt_logprob: Model's log probability for ground truth token (same distribution as above)
        - entropy: Entropy of model's output distribution at this step
        - hidden_state: Hidden representation of last layer last position (list of floats)
        - attention_scores: Corresponding attention scores from generate_attention_score_list (may be empty)
        """
        try:
            device = self.model.device
            prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(device)
            gt_ids = self.tokenizer.encode(ground_truth, return_tensors="pt", add_special_tokens=False).to(device)
            gt_len = gt_ids.shape[1]

            attention_score_list = self.generate_attention_score_list(prompt, ground_truth)

            records: List[Dict[str, Any]] = []
            mismatch_count = 0
            past_key_values = None
            
            # For each position i, use context = prompt + ground_truth[:i] to get next-token distribution
            for i in range(gt_len):
                # Build current context (teacher forcing)
                if i == 0:
                    context_ids = prompt_ids
                else:
                    context_ids = torch.cat([prompt_ids, gt_ids[:, :i]], dim=1)

                attention_mask = torch.ones_like(context_ids).to(device)

                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids=context_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=1,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id,
                        output_hidden_states=True,
                        return_dict_in_generate=True,
                        output_scores=True,
                        use_cache=True,
                        past_key_values=past_key_values
                    )

                past_key_values = outputs.past_key_values

                logits = outputs.scores[0]

                # Training uses greedy decoding
                scaled_logits = logits
                log_probs = F.log_softmax(scaled_logits, dim=-1)  # log p(·)
                probs = torch.exp(log_probs)
                pred_id = torch.argmax(scaled_logits, dim=-1).item()

                gt_id = gt_ids[0, i].item()

                if pred_id != gt_id:
                    mismatch_count += 1

                pred_logprob = log_probs[0, pred_id].item()
                gt_logprob = log_probs[0, gt_id].item()

                entropy = -torch.sum(probs * log_probs).item()

                # hidden state: get the last position vector of the last layer
                # outputs.hidden_states is a tuple (layer0, layer1, ..., last_layer), each (batch, seq_len, hidden)
                last_h = outputs.hidden_states[0][-1]  # (batch, seq_len, hidden)
                hidden_vec = last_h[0, -1].cpu().numpy().tolist()

                pred_token = self.tokenizer.decode([pred_id], skip_special_tokens=True)
                gt_token = self.tokenizer.decode([gt_id], skip_special_tokens=True)

                attn_scores = attention_score_list[i] if i < len(attention_score_list) else []

                record = {
                    "step": i,
                    "pred_token_id": pred_id,
                    "pred_token": pred_token,
                    "pred_logprob": pred_logprob,
                    "gt_token_id": gt_id,
                    "gt_token": gt_token,
                    "gt_logprob": gt_logprob,
                    "entropy": entropy,
                    "hidden_state": hidden_vec,
                    "attention_scores": attn_scores,
                }

                records.append(record)

            mismatch_ratio = mismatch_count / gt_len if gt_len > 0 else 0
            logger.info(f"Token mismatch statistics: {mismatch_count}/{gt_len} = {mismatch_ratio:.2%}")

            return records

        except Exception as e:
            logger.error(f"Failed to generate training data: {str(e)}")
            raise
