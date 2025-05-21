from dataclasses import dataclass, field


@dataclass
class SafeLoRAConfig:
    """
    This is the configuration class to store the configuration of a safeLoRA.
    """

    base_model_path: str = field(
        default=None,
        metadata={"help": "The path of the base model for obtaining the aligned matrix"},
    )

    aligned_model_path: str = field(
        default=None,
        metadata={"help": "The path of the aligned model for obtaining the aligned matrix"},
    )


    select_layers_type: str = field(
        default="number",
        metadata={"help": "How to select projection layers? options: [threshold, number]"},
    )

    threshold: float = field(
        default=0.5,
        metadata={"help": "The threshold of cosine similarity."},
    )

    num_proj_layers: int = field(
        default=10,
        metadata={"help": "The number of projected layers."},
    )

    devices: str = field(
        default="cuda",
        metadata = {"help": "Devices are used in SafeLoRA. (gpu or cpu)"}

    )

    def __post_init__(self):
        if self.base_model_path is None:
            raise ValueError("base_model_path cannot be None.")
        if self.aligned_model_path is None:
            raise ValueError("aligned_model_path cannot be None.")


# # Initialize configuration example
# config = SafeLoRAConfig(
#     base_model_path="meta-llama/Llama-2-7b-hf",
#     aligned_model_path="meta-llama/Llama-2-7b-chat-hf",
#     select_layers_type="number",
#     threshold=0.5,
#     num_proj_layers=10,
#     devices="cuda"  # Use "cpu" if no GPU
# )
