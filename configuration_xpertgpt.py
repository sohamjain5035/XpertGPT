from transformers import PretrainedConfig

class XpertGPTConfig(PretrainedConfig):
    model_type = "xpertgpt"

    def __init__(
        self,
        vocab_size: int = 16384,
        block_size: int = 512,
        d_model: int = 256,
        d_thin: int = 384,
        num_layers: int = 6,
        num_blocks: int = 4,
        capacity_factor: float = 2.0,
        dropout: float = 0.1,
        **kwargs
    ):
        kwargs.setdefault("is_decoder", True)
        kwargs.setdefault("bos_token_id", 2)  # [CLS]
        kwargs.setdefault("eos_token_id", 3)  # [SEP]
        kwargs.setdefault("pad_token_id", 1)  # [PAD]

        self.vocab_size = vocab_size
        self.block_size = block_size
        self.d_model = d_model
        self.d_thin = d_thin
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.capacity_factor = capacity_factor
        self.dropout = dropout
        
        # Attribute parity for classification heads
        self.hidden_size = d_model
        self.num_hidden_layers = num_layers

        super().__init__(**kwargs)
