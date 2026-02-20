"""Tests for title case normalization and LaTeX sanitization."""

import sys
from pathlib import Path

# Ensure project root is on sys.path for imports
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.common import normalize_title_case
from scripts.page_builders import _fix_latex_braces
from scripts.titlecase import smart_title_case


# ---------------------------------------------------------------------------
# Data layer: normalize_title_case (ALL CAPS / all lowercase → title case)
# ---------------------------------------------------------------------------

class TestNormalizeTitleCase:
    def test_all_caps(self):
        assert normalize_title_case("FACE RECOGNITION UNDER VARYING POSE") == (
            "Face Recognition Under Varying Pose"
        )

    def test_all_lowercase(self):
        assert normalize_title_case("face recognition under varying pose") == (
            "Face Recognition Under Varying Pose"
        )

    def test_mixed_case_untouched(self):
        title = "Face Recognition Under Varying Pose"
        assert normalize_title_case(title) == title

    def test_sentence_case_untouched(self):
        title = "Face recognition under varying pose"
        assert normalize_title_case(title) == title

    def test_empty(self):
        assert normalize_title_case("") == ""

    def test_no_alpha(self):
        assert normalize_title_case("123 - 456") == "123 - 456"

    def test_all_caps_with_numbers(self):
        assert normalize_title_case("3D OBJECT RECOGNITION") == "3D Object Recognition"


# ---------------------------------------------------------------------------
# Display layer: smart_title_case
# ---------------------------------------------------------------------------

class TestSmartTitleCase:
    # -- Basic title casing --

    def test_sentence_case_to_title(self):
        assert smart_title_case("Face recognition under varying pose") == (
            "Face Recognition Under Varying Pose"
        )

    def test_already_title_case(self):
        assert smart_title_case(
            "Deterministic Image-to-Image Translation via Denoising Models"
        ) == "Deterministic Image-to-Image Translation via Denoising Models"

    def test_empty(self):
        assert smart_title_case("") == ""

    # -- Small words --

    def test_small_words_lowercased(self):
        result = smart_title_case("attention is all you need")
        # "is" is not in the small words list (it's a verb), so it gets capitalized
        # "all" is not in the list either
        assert result == "Attention Is All You Need"

    def test_articles_lowercased(self):
        result = smart_title_case("the cat in the hat")
        assert result == "The Cat in the Hat"

    def test_first_word_always_capitalized(self):
        result = smart_title_case("a survey of deep learning")
        assert result == "A Survey of Deep Learning"

    def test_prepositions(self):
        result = smart_title_case("learning to see in the dark")
        assert result == "Learning to See in the Dark"

    # -- Acronyms --

    def test_acronym_preserved(self):
        result = smart_title_case("training LSTM networks for sequence prediction")
        assert "LSTM" in result

    def test_multiple_acronyms(self):
        result = smart_title_case("CNN and RNN for NLP tasks")
        assert "CNN" in result
        assert "RNN" in result
        assert "NLP" in result

    def test_short_acronym_2_chars(self):
        # 2-char all-caps should be preserved
        result = smart_title_case("an AI system for 3D reconstruction")
        assert "AI" in result
        assert "3D" in result

    # -- ML terms --

    def test_imagenet(self):
        result = smart_title_case("training on imagenet with resnet")
        assert "ImageNet" in result
        assert "ResNet" in result

    def test_bert(self):
        result = smart_title_case("fine-tuning bert for text classification")
        assert "BERT" in result

    def test_pytorch(self):
        result = smart_title_case("a pytorch implementation of transformers")
        assert "PyTorch" in result

    def test_lora(self):
        result = smart_title_case("parameter-efficient fine-tuning with lora")
        assert "LoRA" in result

    def test_nerf(self):
        result = smart_title_case("neural radiance fields: a nerf survey")
        assert "NeRF" in result

    def test_gan(self):
        result = smart_title_case("image synthesis with gan")
        assert "GAN" in result

    def test_vit(self):
        result = smart_title_case("an image is worth 16x16 words: vit at scale")
        assert "ViT" in result

    # -- Colon handling --

    def test_capitalize_after_colon(self):
        result = smart_title_case("neural radiance fields: a comprehensive survey")
        assert result == "Neural Radiance Fields: A Comprehensive Survey"

    def test_small_word_after_colon_capitalized(self):
        result = smart_title_case("transformers: the key to understanding language")
        assert result.startswith("Transformers: The")

    # -- Hyphenated compounds --

    def test_hyphenated_compound(self):
        result = smart_title_case("self-supervised learning for vision")
        assert "Self-Supervised" in result

    def test_hyphenated_with_small_word(self):
        result = smart_title_case("image-to-image translation")
        assert "Image-to-Image" in result or "Image-To-Image" in result

    # -- Real paper titles from the dataset --

    def test_cvpr1994_sentence_case(self):
        result = smart_title_case(
            "2D matching of 3D moving objects in color outdoor scenes"
        )
        assert result == "2D Matching of 3D Moving Objects in Color Outdoor Scenes"

    def test_cvpr1994_error_propagation(self):
        result = smart_title_case(
            "Error propagation in full 3D-from-2D object recognition"
        )
        # "in" stays lowercase, "3D-from-2D" preserves 3D/2D
        assert "in" in result.split()
        assert "3D" in result
        assert "2D" in result

    def test_already_correct_complex(self):
        title = "Towards Source-Free Machine Unlearning"
        assert smart_title_case(title) == title

    def test_real_iclr_title(self):
        result = smart_title_case(
            "scaling laws for neural language models"
        )
        assert result == "Scaling Laws for Neural Language Models"

    # -- Idempotency --

    def test_idempotent_on_title_case(self):
        title = "Attention Is All You Need"
        assert smart_title_case(title) == title

    def test_idempotent_on_acronym_heavy(self):
        title = "BERT: Pre-Training of Deep Bidirectional Transformers"
        result = smart_title_case(title)
        assert "BERT" in result
        assert "Pre-Training" in result

    # -- Edge cases --

    def test_single_word(self):
        assert smart_title_case("transformers") == "Transformers"

    def test_preserves_whitespace(self):
        result = smart_title_case("word  word")
        assert "  " in result

    def test_all_caps_title_with_acronyms(self):
        """ALL CAPS title should have acronyms preserved after title() in data layer."""
        # Data layer converts ALL CAPS → Title Case first,
        # then display layer applies smart rules
        data_normalized = normalize_title_case("CNN BASED IMAGE RECOGNITION")
        # After data layer: "Cnn Based Image Recognition" (title() doesn't know acronyms)
        # Display layer fixes it:
        result = smart_title_case(data_normalized)
        assert "CNN" in result
        assert "Image Recognition" in result

    # -- Trailing punctuation --

    def test_ml_term_with_colon(self):
        result = smart_title_case("LoRA: low-rank adaptation of large language models")
        assert result.startswith("LoRA:")

    def test_ml_term_with_comma(self):
        result = smart_title_case("on imagenet, we train resnet")
        assert "ImageNet," in result
        assert "ResNet" in result

    # -- Internal mixed case preserved --

    def test_mixed_case_model_name(self):
        result = smart_title_case("Uni4D: unifying visual foundation models")
        assert "Uni4D:" in result

    def test_mixed_case_preserved_mid_title(self):
        result = smart_title_case("training DeepSeek for code generation")
        assert "DeepSeek" in result

    # -- Prepositions with/from lowercased --

    def test_with_lowercased(self):
        result = smart_title_case("models with dual approximators")
        assert "with" in result.split()

    def test_from_lowercased(self):
        result = smart_title_case("learning from noisy labels")
        assert "from" in result.split()

    # -- Hyphenated preposition inside compound --

    def test_3d_from_2d_hyphenated(self):
        result = smart_title_case("Error propagation in full 3D-from-2D object recognition")
        assert "3D-from-2D" in result

    def test_state_of_the_art(self):
        result = smart_title_case("state-of-the-art results on CIFAR-10")
        assert "State-of-the-Art" in result


# ---------------------------------------------------------------------------
# LaTeX brace fixing
# ---------------------------------------------------------------------------

class TestFixLatexBraces:
    def test_mathcal_missing_braces(self):
        assert _fix_latex_braces(r"$\mathcalNP$-hard") == r"$\mathcal{NP}$-hard"

    def test_texttt_missing_braces(self):
        assert _fix_latex_braces(r"$\textttMoE-RBench$") == r"$\texttt{MoE-RBench}$"

    def test_mathtt_missing_braces(self):
        assert _fix_latex_braces(r"$\mathttVITS$") == r"$\mathtt{VITS}$"

    def test_textrm_missing_braces(self):
        assert _fix_latex_braces(r"$\textrmFlow$") == r"$\textrm{Flow}$"

    def test_sqrt_missing_braces(self):
        assert _fix_latex_braces(r"$\sqrtT$") == r"$\sqrt{T}$"

    def test_mathcal_O(self):
        assert _fix_latex_braces(r"$\mathcalO(L)$") == r"$\mathcal{O}(L)$"

    def test_already_braced_unchanged(self):
        s = r"$\mathcal{O}(\sqrt{T})$"
        assert _fix_latex_braces(s) == s

    def test_emph_well_formed_unchanged(self):
        s = r"\emph{De Novo} Mass Spectrometry"
        assert _fix_latex_braces(s) == s

    def test_no_latex_unchanged(self):
        s = "A Simple Title with No LaTeX"
        assert _fix_latex_braces(s) == s

    def test_real_phi_flow(self):
        s = r"$\bfΦ_\textrmFlow$: Differentiable Simulations"
        result = _fix_latex_braces(s)
        assert r"\textrm{Flow}" in result
        # \bf before Greek Φ is not captured (non-ASCII) — that's fine,
        # KaTeX handles \bf as a math-mode font switch for the next token

    def test_bf_missing_braces(self):
        result = _fix_latex_braces(r"\bfHello")
        assert result == r"\bf{Hello}"

    def test_rm_in_math(self):
        result = _fix_latex_braces(r"$\rmE(3)$")
        assert r"\rm{E}" in result
