"""Tests for empathy scoring model."""

import pytest


class TestEmpathyScorer:
    """Tests for the EmpathyScorer class."""
    
    def test_empathic_phrases_detection(self):
        """Test that empathic phrases are detected."""
        from site_empathy_analysis.models.empathy_model import EmpathyScorer
        
        scorer = EmpathyScorer()
        
        # Test with empathic text
        result = scorer.score("We understand how difficult this journey must be for you.")
        
        assert "we understand" in result.empathic_phrases or "your journey" in result.empathic_phrases
        assert result.empathic_phrase_count > 0
    
    def test_non_empathic_detection(self):
        """Test that non-empathic indicators are detected."""
        from site_empathy_analysis.models.empathy_model import EmpathyScorer
        
        scorer = EmpathyScorer()
        
        # Test with transactional text
        result = scorer.score("Click here to submit the required fields per our policy.")
        
        assert len(result.non_empathic_indicators) > 0
    
    def test_score_range(self):
        """Test that scores are in valid range."""
        from site_empathy_analysis.models.empathy_model import EmpathyScorer
        
        scorer = EmpathyScorer()
        
        result = scorer.score("We care about your wellbeing and are here every step of the way.")
        
        assert 0 <= result.empathy_score <= 1
        assert 0 <= result.emotional_reaction <= 1
        assert 0 <= result.interpretation <= 1
        assert 0 <= result.exploration <= 1
    
    def test_levels_valid(self):
        """Test that levels are 0, 1, or 2."""
        from site_empathy_analysis.models.empathy_model import EmpathyScorer
        
        scorer = EmpathyScorer()
        
        result = scorer.score("Any text here for testing.")
        
        assert result.er_level in [0, 1, 2]
        assert result.ip_level in [0, 1, 2]
        assert result.ex_level in [0, 1, 2]
    
    def test_batch_scoring(self):
        """Test batch scoring."""
        from site_empathy_analysis.models.empathy_model import EmpathyScorer
        
        scorer = EmpathyScorer()
        
        texts = [
            "We understand how hard this is.",
            "Click here to submit.",
            "You are not alone in this journey.",
        ]
        
        results = scorer.score_batch(texts)
        
        assert len(results) == 3
        for result in results:
            assert 0 <= result.empathy_score <= 1


class TestEmpathyResult:
    """Tests for EmpathyResult dataclass."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        from site_empathy_analysis.models.empathy_model import EmpathyResult
        
        result = EmpathyResult(
            text="Test text",
            empathy_score=0.5,
            emotional_reaction=0.4,
            interpretation=0.3,
            exploration=0.2,
            er_level=1,
            ip_level=1,
            ex_level=0,
            er_probs=[0.3, 0.5, 0.2],
            ip_probs=[0.4, 0.4, 0.2],
            ex_probs=[0.6, 0.3, 0.1],
            empathic_phrases=["we understand"],
            non_empathic_indicators=[],
        )
        
        d = result.to_dict()
        
        assert "empathy_score" in d
        assert d["empathy_score"] == 0.5
        assert "empathic_phrases" in d

