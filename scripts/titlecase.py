"""Smart title-casing for ML paper titles.

Applies context-aware casing that preserves acronyms (LSTM, CNN),
known ML terms (ImageNet, ResNet), and intentional internal casing
(DeepSeek, iPhone), while lowercasing articles/prepositions per
Chicago Manual of Style conventions.
"""

import re

# Words that stay lowercase in title case (unless first word or after colon).
_TITLE_SMALL_WORDS = frozenset({
    # articles
    "a", "an", "the",
    # coordinating conjunctions
    "and", "but", "or", "nor", "for", "yet", "so",
    # short prepositions
    "as", "at", "by", "en", "if", "in", "of", "on", "to", "up",
    "vs", "via", "per",
    "from", "into", "with", "over", "upon", "onto", "near",
    "like", "till", "past", "amid", "atop", "sans", "than",
})

# Known ML/AI terms with canonical casing.  Lookup is case-insensitive;
# the value is the correct display form.  Add new terms as needed.
_ML_TERMS: dict[str, str] = {}
for _term in [
    # Architectures & models
    "ImageNet", "ResNet", "ResNeXt", "VGG", "AlexNet", "GoogLeNet",
    "LeNet", "DenseNet", "EfficientNet", "EfficientDet", "MobileNet",
    "MobileNets", "ShuffleNet", "SqueezeNet", "InceptionNet",
    "BERT", "RoBERTa", "XLNet", "ALBERT", "DistilBERT", "ELECTRA",
    "DeBERTa", "BERTScore", "GPT", "ChatGPT", "InstructGPT",
    "DALL-E", "CLIP", "BLIP", "ViT", "DeiT", "DINO", "DINOv2",
    "MAE", "SAM", "LLaMA", "Llama", "Mistral", "Mixtral", "Gemma",
    "Gemini", "DeepSeek", "Qwen", "PaLM", "LaMDA", "Alpaca",
    "T5", "mT5", "FLAN", "UL2",
    "WaveNet", "WaveGlow", "Tacotron",
    "StyleGAN", "ProGAN", "BigGAN", "CycleGAN", "StarGAN",
    "pix2pix", "Pix2Pix",
    "NeRF", "DreamBooth", "ControlNet", "LoRA", "QLoRA", "DoRA",
    "AdaLoRA", "PEFT",
    "MoE", "UNet", "U-Net", "YOLO", "YOLOv5", "YOLOv8",
    "DETR", "Mask R-CNN", "Faster R-CNN", "Fast R-CNN", "R-CNN",
    "SSD", "FPN", "RPN", "ROI", "RoI",
    "SimCLR", "MoCo", "MoCoV2", "BYOL", "SwAV", "BarlowTwins",
    "AlphaGo", "AlphaFold", "AlphaZero", "MuZero",
    # Techniques & concepts
    "LSTM", "GRU", "GAN", "GANs", "VAE", "VAEs", "ViTs",
    "CNN", "CNNs", "RNN", "RNNs", "MLP", "MLPs", "LLM", "LLMs",
    "NLP", "NLU", "NLG", "NER", "POS",
    "RLHF", "DPO", "PPO", "TRPO", "SAC", "DDPG", "TD3", "A3C",
    "SGD", "Adam", "AdamW", "AdaGrad", "RMSProp", "LARS", "LAMB",
    "ROUGE", "BLEU", "METEOR", "CIDEr",
    "SLAM", "SfM", "MVS", "ORB",
    "GNN", "GNNs", "GCN", "GAT", "GraphSAGE",
    "SVM", "SVMs", "kNN", "k-NN", "PCA", "ICA", "LDA", "HMM",
    "MCMC", "EM", "MLE",
    "IoU", "mIoU", "mAP", "AP", "AUC", "ROC", "FID",
    "SSIM", "PSNR", "LPIPS",
    "BatchNorm", "LayerNorm", "GroupNorm", "RMSNorm",
    "SoftMax", "ReLU", "GELU", "SiLU", "Mish", "ELU", "PReLU",
    "LeakyReLU",
    "RAG", "CoT", "ICL", "SFT", "GRPO",
    "FlashAttention", "PagedAttention", "KV",
    # Frameworks & tools
    "TensorFlow", "PyTorch", "JAX", "NumPy", "SciPy",
    "HuggingFace", "LangChain", "OpenAI",
    "DeepSpeed", "Megatron", "ColossalAI", "vLLM", "Triton",
    "CUDA", "cuDNN", "TensorRT", "ONNX",
    "OpenCV", "PCL", "ROS",
    # Datasets & benchmarks
    "CIFAR", "MNIST", "COCO", "VOC", "LVIS", "ADE20K",
    "SQuAD", "GLUE", "SuperGLUE", "MMLU", "HellaSwag",
    "WMT", "LAMBADA", "WikiText", "BookCorpus",
    "Kinetics", "AVA", "ActivityNet", "UCF101", "HMDB51",
    "ShapeNet", "ModelNet", "ScanNet", "SUN RGB-D",
    "nuScenes", "KITTI", "Waymo", "Argoverse",
    # Common abbreviations in paper titles
    "GPU", "GPUs", "TPU", "TPUs", "CPU", "CPUs", "FPGA", "FPGAs",
    "RGB", "RGB-D", "LiDAR", "IMU",
    "API", "APIs", "SDK", "REST", "HTTP",
    "2D", "3D", "4D", "6DoF",
    "i.i.d.", "w.r.t.",
    "YouTube", "GitHub", "arXiv", "LaTeX", "BibTeX",
]:
    _ML_TERMS[_term.lower()] = _term


def _smart_title_case_word(word: str) -> str:
    """Apply title case to a single word, preserving special patterns."""
    if not word:
        return word

    # Strip trailing punctuation for lookups, reattach at the end.
    stripped = word.rstrip(",:;.!?)")
    suffix = word[len(stripped):]

    # Preserve anything that looks like an acronym (2+ uppercase letters,
    # possibly with digits): "LSTM", "CNN", "3D", "GPT-4", "RGB-D"
    alpha = "".join(c for c in stripped if c.isalpha())
    if len(alpha) >= 2 and alpha.isupper():
        return word

    # Check ML terms dictionary (case-insensitive lookup)
    lookup = stripped.lower()
    if lookup in _ML_TERMS:
        return _ML_TERMS[lookup] + suffix

    # Check for hyphenated compound: title-case each part, but keep
    # inner small words lowercase (e.g. "Image-to-Image", "State-of-the-Art")
    if "-" in stripped and not stripped.startswith("-"):
        parts = stripped.split("-")
        cased_parts = [_smart_title_case_word(parts[0])]
        for p in parts[1:]:
            if p.lower() in _TITLE_SMALL_WORDS:
                cased_parts.append(p.lower())
            else:
                cased_parts.append(_smart_title_case_word(p))
        return "-".join(cased_parts) + suffix

    # Preserve words with intentional internal casing (e.g. "Uni4D",
    # "iPhone", "DeepSeek").
    if alpha and not alpha.isupper() and not alpha.islower():
        return word

    # Default: capitalize first letter, lowercase the rest
    if len(stripped) > 1:
        return stripped[0].upper() + stripped[1:].lower() + suffix
    return stripped.upper() + suffix


def smart_title_case(title: str) -> str:
    """Apply smart title case to a paper title for display.

    Rules:
    - First word is always capitalized.
    - Articles, short prepositions, and coordinating conjunctions are
      lowercased unless they are the first word or follow a colon/dash.
    - Acronyms (2+ uppercase letters) are preserved as-is.
    - Known ML terms (ImageNet, ResNet, LSTM, etc.) use canonical casing.
    - Hyphenated compounds have each part title-cased.
    """
    if not title:
        return title

    # Split while preserving whitespace structure
    tokens = re.split(r"(\s+)", title)
    result = []
    capitalize_next = True  # first word is always capitalized

    for token in tokens:
        # Whitespace tokens pass through
        if not token or token.isspace():
            result.append(token)
            continue

        # Strip trailing punctuation for small-word check
        stripped = token.rstrip(",:;.!?)")
        lower = stripped.lower()

        # After colon, em-dash, or en-dash: capitalize like a new title
        if capitalize_next:
            cased = _smart_title_case_word(token)
            result.append(cased)
            capitalize_next = False
        elif lower in _TITLE_SMALL_WORDS:
            suffix = token[len(stripped):]
            result.append(stripped.lower() + suffix)
        else:
            result.append(_smart_title_case_word(token))

        # Set flag if this token ends with colon or is an em/en-dash
        if token.endswith(":") or token in ("\u2014", "\u2013", "-"):
            capitalize_next = True

    return "".join(result)
