"""
Tests for inference pipeline.
Tests use HuggingFace Hub model fallback when local model is not available.
"""
import pytest
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Singleton instance for reuse across tests (avoid reloading model)
_test_summarizer = None


def get_test_summarizer():
    """Get or create a shared summarizer instance for tests."""
    global _test_summarizer
    if _test_summarizer is None:
        from src.text_summarizer.pipeline.inference import Summarizer
        # Reset singleton to ensure fresh load
        Summarizer._instance = None
        Summarizer._model = None
        Summarizer._tokenizer = None
        _test_summarizer = Summarizer()
    return _test_summarizer


class TestInferencePipeline:
    """Test suite for inference pipeline."""
    
    def test_summarizer_import(self):
        """Test that Summarizer can be imported."""
        from src.text_summarizer.pipeline.inference import Summarizer
        assert Summarizer is not None
    
    def test_summarizer_initialization(self):
        """Test Summarizer initialization (uses HuggingFace Hub fallback if no local model)."""
        summarizer = get_test_summarizer()
        assert summarizer.model is not None
        assert summarizer.tokenizer is not None
        assert summarizer.tokenizer.vocab_size > 0
    
    def test_summarize_text(self):
        """Test summarization of sample text."""
        summarizer = get_test_summarizer()
        
        sample_dialogue = """
        John: Hey, are you coming to the party tonight?
        Sarah: I'm not sure, I have a lot of work to do.
        John: Come on, it'll be fun! Everyone's going to be there.
        Sarah: Okay, I'll try to finish early and come by around 8.
        John: Great! See you then!
        """
        
        summary = summarizer.summarize(sample_dialogue)
        
        assert summary is not None
        assert len(summary) > 0
        assert len(summary) < len(sample_dialogue)
    
    def test_summarize_empty_input(self):
        """Test that empty input returns empty string."""
        summarizer = get_test_summarizer()
        
        summary = summarizer.summarize("")
        assert summary == ""
        
        summary = summarizer.summarize("   ")
        assert summary == ""
    
    def test_summarize_short_input(self):
        """Test summarization of very short input."""
        summarizer = get_test_summarizer()
        
        short_dialogue = "John: Hi!\nSarah: Hello!"
        summary = summarizer.summarize(short_dialogue)
        
        assert summary is not None
        assert len(summary) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
