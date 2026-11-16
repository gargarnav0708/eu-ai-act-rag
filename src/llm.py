"""Load the configured LLM in 4-bit mode on a compatible CUDA machine."""

from .config import LLM_MODEL


def load_llm():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
        from langchain_huggingface import HuggingFacePipeline
    except ImportError as exc:
        raise ImportError(
            "LLM dependencies are missing. On a supported GPU machine, install the CUDA requirements."
        ) from exc

    # This starter intentionally targets the most dependable bitsandbytes path: NVIDIA CUDA.
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU was detected, so this project's 4-bit bitsandbytes Llama loader is not enabled.\n"
            "Keep working on ingestion/retrieval locally, then run the LLM stage on a supported NVIDIA GPU\n"
            "(for example, a CUDA-enabled cloud notebook or Linux workstation). Apple Silicon/MPS needs a\n"
            "different loading path; do not assume load_in_4bit works unchanged on a Mac."
        )

    model_id = LLM_MODEL
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    print(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading 4-bit model; first run downloads several GB of model weights...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map="auto",
        dtype=compute_dtype,
        low_cpu_mem_usage=True,
    )
    model.eval()

    text_generation_pipeline = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.05,
        return_full_text=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    return HuggingFacePipeline(pipeline=text_generation_pipeline)
