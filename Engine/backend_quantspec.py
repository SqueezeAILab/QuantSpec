import torch
from QuantSpec.Engine.model_quantspec import Transformer
from QuantSpec.Engine.utils import load_model_quantspec

class LMBackend:
    def __init__(self, dtype = torch.bfloat16, device: str = "cuda:0", dec_list: list = [1], draft_dec_list: list = [1]) -> None:
        self.dtype = dtype
        self.device = device
        self.model_forward = {}
        self.draft_forward = {}
        for dec_len in dec_list:
            if dec_len == 0: continue
            self.model_forward[dec_len] = lambda model, x, input_pos, cache_seqlens, qcache_seqlens: model(x, input_pos, cache_seqlens, qcache_seqlens)
        for dec_len in draft_dec_list:
            if dec_len == 0: continue
            self.draft_forward[dec_len] = lambda model, x, input_pos, cache_seqlens, qcache_seqlens: model.draft_forward(x, input_pos, cache_seqlens, qcache_seqlens)
        self.prefill = lambda model, x, input_pos, cache_seqlens, qcache_seqlens: model.prefill(x, input_pos, cache_seqlens, qcache_seqlens)
        self.cachelens = None
        self.q_cachelens = None
        self.residual_len = None

    def load_model(self, checkpoints: str, use_tp: bool, rank_group=None, group = None, quantize: bool = False, marlin_checkpoint: str = None):
        if quantize:
            assert marlin_checkpoint is not None, "Marlin checkpoint is required for quantization"
        self.model: Transformer = load_model_quantspec(checkpoint_path=checkpoints, device=self.device, precision=self.dtype, use_tp= use_tp, rank_group=rank_group, group = group, quantize = quantize, marlin_checkpoint = marlin_checkpoint)

    @torch.inference_mode()
    def setup_caches(self, max_batch_size: int = 1, max_seq_length: int = 2048, **cache_kwargs):
        self.max_length = max_seq_length
        self.batch_size = max_batch_size
        self.cachelens = torch.zeros(max_batch_size, dtype=torch.int32, device=self.device)
        self.q_cachelens = torch.zeros(max_batch_size, dtype=torch.int32, device=self.device)
        self.residual_len = cache_kwargs.get("residual_len", 128)
        self.group_size = cache_kwargs.get("group_size", 128)
        with torch.device(self.device):
            self.model.setup_caches(max_batch_size=max_batch_size, max_seq_length=max_seq_length, **cache_kwargs)

    def compile(self, encode=False):
        import torch._dynamo.config
        import torch._inductor.config
        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.triton.unique_kernel_names = True
        torch._inductor.config.fx_graph_cache = True # Experimental feature to reduce compilation times, will be on by default in future
        torch._functorch.config.enable_autograd_cache = True
        for key in self.model_forward.keys():
            self.model_forward[key] = torch.compile(self.model_forward[key], mode="max-autotune", fullgraph=True)
        for key in self.draft_forward.keys():
            self.draft_forward[key] = torch.compile(self.draft_forward[key], mode="max-autotune", fullgraph=True)
        if encode:
             self.prefill = torch.compile(self.prefill, mode="max-autotune", fullgraph=True)
             
    @torch.inference_mode()
    def inference(self, input_ids: torch.LongTensor, benchmark = False):
            dec_len = input_ids.shape[1]
            position_ids = self.cachelens.view(-1,1) + torch.arange(dec_len, device=self.device).unsqueeze(0).repeat(self.batch_size,1)
            if dec_len not in self.model_forward.keys():
                raise ValueError(f"Decoding length {dec_len} not supported")
            logits = self.model_forward[dec_len](
                model=self.model,  
                x=input_ids.clone(),
                input_pos=position_ids.clone(), 
                cache_seqlens=self.cachelens.clone(),
                qcache_seqlens=self.q_cachelens.clone()) if dec_len in self.model_forward.keys() else self.model.forward(input_ids.clone(), position_ids.clone(), self.cachelens.clone(), self.q_cachelens.clone())
            if not benchmark:
                self.cachelens += dec_len
            return logits
    
    @torch.inference_mode()
    def draft_inference(self, input_ids: torch.LongTensor, benchmark = False, cachelen_update = None):
            dec_len = input_ids.shape[1]
            position_ids = self.cachelens.view(-1,1) + torch.arange(dec_len, device=self.device).unsqueeze(0).repeat(self.batch_size,1)
            logits = self.draft_forward[dec_len](
                model=self.model, 
                x=input_ids,
                input_pos=position_ids, 
                cache_seqlens=self.cachelens,
                qcache_seqlens=self.q_cachelens) if dec_len in self.draft_forward.keys() else self.model.draft_forward(input_ids.clone(), position_ids.clone(), self.cachelens.clone(), self.q_cachelens.clone())
            if not benchmark:
                if cachelen_update == None:
                    self.cachelens += dec_len
                else:
                    self.cachelens += cachelen_update
            return logits
    
    @torch.inference_mode()
    def encode(self, input_ids: torch.LongTensor):
        self.cachelens.zero_()
        self.q_cachelens.zero_()
        self.clear_kv()
        logits = None
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0).repeat(self.batch_size,1)
        # TODO: implement for division
        division = False #seq_len > 1000
        if division:
            chunk_size = 32
            num_chunks = (seq_len + chunk_size - 1) // chunk_size  # Ceil division
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, seq_len)
                
                chunk_input_ids = input_ids[:, start_idx:end_idx]
                chunk_position_ids = position_ids[:, start_idx:end_idx]
                chunk_cache_seqlens = self.cachelens + start_idx

                logits = self.prefill(
                    model=self.model,
                    x=chunk_input_ids,
                    input_pos=chunk_position_ids,
                    cache_seqlens=chunk_cache_seqlens
                )

                if end_idx > self.streaming_budget:
                    chunk_position_ids = torch.arange(self.streaming_budget - chunk_input_ids.shape[1], self.streaming_budget, device = self.device).unsqueeze(0).repeat(input_ids.shape[0],1).long()
                self.draft_prefill(
                    model=self.model,
                    x=chunk_input_ids,
                    input_pos=chunk_position_ids,
                    cache_seqlens=chunk_cache_seqlens,
                    is_last = i == num_chunks-1
                )
        else:
            # raise NotImplementedError("Not implemented for seq_len < 1000")
            logits = self.prefill(
                model=self.model,
                x=input_ids,
                input_pos=position_ids,
                cache_seqlens=self.cachelens,
                qcache_seqlens=self.q_cachelens
            )
            self.cachelens += seq_len
            residual_cachelen = self.cachelens-self.q_cachelens
            mask = residual_cachelen > 2*self.residual_len
            to_quant_len = torch.where(mask, 
                residual_cachelen-self.residual_len-(residual_cachelen%self.group_size),
                torch.zeros_like(residual_cachelen))
            self.q_cachelens += to_quant_len
            
        return logits
          
    
    @torch.inference_mode()
    def clear_kv(self):
        for b in self.model.layers:
            b.attention.kv_cache.reset()

    
