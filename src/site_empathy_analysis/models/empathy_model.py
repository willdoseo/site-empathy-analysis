"""
Empathy Analysis Model
======================

Implementation of the Sharma et al. (2020) empathy framework for text analysis.

This module provides models for analyzing empathy across three dimensions:
- Emotional Reactions (ER): Warmth, compassion, concern
- Interpretations (IP): Understanding, acknowledging feelings  
- Explorations (EX): Engagement, asking questions

Attribution
-----------
Based on: "A Computational Approach to Understanding Empathy Expressed in 
Text-Based Mental Health Support" - Sharma et al., EMNLP 2020
https://aclanthology.org/2020.emnlp-main.425/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel, RobertaTokenizer
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass
import math
import os
import logging
import warnings
import requests
from pathlib import Path
from tqdm import tqdm

# Suppress transformer warnings about uninitialized weights
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="Some weights of")

# Model hosting configuration
# Models are hosted on HuggingFace Hub
# Users can override with their own URL via environment variable
MODEL_BASE_URL = os.getenv(
    "SITE_EMPATHY_MODEL_URL",
    "https://huggingface.co/willdoseo/site-empathy-models/resolve/main"
)
MODEL_CACHE_DIR = Path.home() / ".cache" / "site-empathy" / "models"

# Empathy dimension definitions
EMPATHY_DIMENSIONS = {
    "ER": {
        "name": "Emotional Reactions",
        "description": "Expressing warmth, compassion, and concern",
        "examples": ["I feel for you", "That must be hard", "I'm sorry you're going through this"],
        "weight": 0.5,  # Most important for perceived empathy
    },
    "IP": {
        "name": "Interpretations", 
        "description": "Acknowledging and naming feelings",
        "examples": ["I understand what you're feeling", "It sounds like you're frustrated"],
        "weight": 0.3,
    },
    "EX": {
        "name": "Explorations",
        "description": "Asking questions, inviting dialogue",
        "examples": ["Tell me more", "How does that make you feel?", "What happened next?"],
        "weight": 0.2,
    },
}


@dataclass
class EmpathyResult:
    """Results from empathy analysis of a single text."""
    
    text: str
    empathy_score: float  # Combined weighted score (0-1)
    
    # Dimension scores (0-1 scale)
    emotional_reaction: float
    interpretation: float
    exploration: float
    
    # Raw predictions (0, 1, 2)
    er_level: int
    ip_level: int
    ex_level: int
    
    # Probabilities for each level
    er_probs: List[float]
    ip_probs: List[float]
    ex_probs: List[float]
    
    # Language analysis
    empathic_phrases: List[str]
    non_empathic_indicators: List[str]
    
    def to_dict(self) -> dict:
        """Convert to dictionary for CSV/JSON export."""
        return {
            "text_preview": self.text[:200] + "..." if len(self.text) > 200 else self.text,
            "empathy_score": round(self.empathy_score, 4),
            "emotional_reaction_score": round(self.emotional_reaction, 4),
            "interpretation_score": round(self.interpretation, 4),
            "exploration_score": round(self.exploration, 4),
            "er_level": self.er_level,
            "ip_level": self.ip_level,
            "ex_level": self.ex_level,
            "empathic_phrases": "|".join(self.empathic_phrases),
            "non_empathic_indicators": "|".join(self.non_empathic_indicators),
        }


class MultiHeadAttention(nn.Module):
    """Cross-attention between seeker and responder representations."""
    
    def __init__(self, heads: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        bs = q.size(0)
        
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attended = torch.matmul(attn_weights, v)
        
        concat = attended.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        
        return output


class EmpathyClassificationHead(nn.Module):
    """Classification head for 3-class empathy prediction."""
    
    def __init__(self, hidden_size: int = 768, num_labels: int = 3, dropout: float = 0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, num_labels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class BiEncoderEmpathyModel(nn.Module):
    """
    Bi-encoder empathy model based on Sharma et al. (2020).
    
    Architecture:
    - Two RoBERTa encoders for context (seeker) and target (responder)
    - Cross-attention from responder attending to seeker
    - Classification head for empathy level (0=none, 1=weak, 2=strong)
    
    Attribution:
    Based on the EMNLP 2020 paper by Sharma et al.
    """
    
    def __init__(
        self, 
        hidden_size: int = 768, 
        num_labels: int = 3,
        dropout: float = 0.2,
        attn_heads: int = 1
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        
        # Load pretrained RoBERTa encoders
        self.seeker_encoder = RobertaModel.from_pretrained('roberta-base')
        self.responder_encoder = RobertaModel.from_pretrained('roberta-base')
        
        # Cross-attention
        self.cross_attention = MultiHeadAttention(attn_heads, hidden_size, dropout)
        self.dropout = nn.Dropout(dropout)
        
        # Classification heads (names must match trained weights)
        self.empathy_classifier = EmpathyClassificationHead(hidden_size, num_labels, dropout)
        self.rationale_classifier = nn.Linear(hidden_size, 2)  # For rationale extraction
    
    def forward(
        self,
        input_ids_SP: torch.Tensor,
        input_ids_RP: torch.Tensor,
        attention_mask_SP: Optional[torch.Tensor] = None,
        attention_mask_RP: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids_SP: Seeker post input IDs [batch, seq_len]
            input_ids_RP: Responder post input IDs [batch, seq_len]
            attention_mask_SP: Seeker attention mask
            attention_mask_RP: Responder attention mask
            
        Returns:
            empathy_logits: Logits for empathy levels [batch, 3]
            cls_output: CLS token representation [batch, hidden]
        """
        # Encode seeker and responder
        seeker_outputs = self.seeker_encoder(
            input_ids_SP,
            attention_mask=attention_mask_SP
        )
        responder_outputs = self.responder_encoder(
            input_ids_RP,
            attention_mask=attention_mask_RP
        )
        
        seeker_hidden = seeker_outputs.last_hidden_state
        responder_hidden = responder_outputs.last_hidden_state
        
        # Cross-attention: responder attends to seeker
        attended = self.cross_attention(responder_hidden, seeker_hidden, seeker_hidden)
        responder_hidden = responder_hidden + self.dropout(attended)
        
        # Classification from [CLS] token
        cls_output = responder_hidden[:, 0, :]
        empathy_logits = self.empathy_classifier(cls_output)
        
        return empathy_logits, cls_output


class EmpathyScorer:
    """
    High-level interface for empathy scoring.
    
    Uses the Sharma et al. (2020) framework to analyze text for empathy
    across three dimensions: Emotional Reactions, Interpretations, and Explorations.
    
    Example
    -------
    >>> scorer = EmpathyScorer()
    >>> result = scorer.score("We understand how difficult this time must be for you.")
    >>> print(f"Empathy Score: {result.empathy_score:.2f}")
    """
    
    # Generic context for analyzing web content
    DEFAULT_CONTEXT = "I'm struggling and looking for help and support."
    
    # Empathic language patterns (from Intent Analysis research)
    EMPATHIC_PHRASES = [
        "every step", "your journey", "we understand", "difficult time",
        "going through", "you deserve", "struggling", "compassion",
        "ready to", "feeling", "we care", "healing", "hope", "together",
        "you may feel", "many people", "it can be", "experience",
        "here for you", "not alone", "reach out", "support you",
        "by your side", "walk with you", "believe in", "proud of",
        "courage", "strength", "brave", "taking this step",
        "whenever you're ready", "at your own pace", "no judgment",
    ]
    
    # Non-empathic indicators (clinical/transactional language)
    NON_EMPATHIC_INDICATORS = [
        "click here", "submit", "required fields", "terms and conditions",
        "policy", "procedure", "compliance", "pursuant to", "hereby",
        "liability", "disclaimer", "copyright", "all rights reserved",
        "contact us for more information", "see our", "visit our",
        "founded in", "established", "serving since", "completion rate",
        "success rate", "statistics show", "studies indicate",
        "evidence-based", "best practices", "industry standard",
    ]
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        context: Optional[str] = None,
    ):
        """
        Initialize the empathy scorer.
        
        Args:
            model_path: Path to trained model weights. If None, uses pretrained RoBERTa.
            device: Device to use ('cuda', 'mps', 'cpu'). Auto-detected if None.
            context: Default context for empathy analysis. Uses help-seeking context if None.
        """
        # Set device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        self.context = context or self.DEFAULT_CONTEXT
        
        # Initialize tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.max_length = 64
        
        # Initialize models (one per dimension)
        self.models = {}
        self._load_models(model_path)
    
    def _download_model(self, dim: str) -> Optional[Path]:
        """Download a model file if not cached."""
        MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        
        filename = f"modern_{dim}.pth"
        cache_path = MODEL_CACHE_DIR / filename
        
        if cache_path.exists():
            return cache_path
        
        url = f"{MODEL_BASE_URL}/{filename}"
        
        print(f"📥 Downloading {dim} model (~960MB)...")
        print(f"   From: {url}")
        
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(cache_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"   {dim}") as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            print(f"   ✓ Saved to {cache_path}")
            return cache_path
            
        except requests.exceptions.RequestException as e:
            print(f"   ✗ Download failed: {e}")
            if cache_path.exists():
                cache_path.unlink()
            return None
    
    def _find_model_weights(self, dim: str, model_path: Optional[str] = None) -> Optional[Path]:
        """Find model weights locally or download if needed."""
        # Local paths to check first
        local_paths = [
            model_path,
            Path(__file__).parent.parent.parent.parent / "models",  # package/models/
            MODEL_CACHE_DIR,
        ]
        
        # Try different naming conventions
        filenames = [
            f"modern_{dim}.pth",
            f"{dim.lower()}_model.pth",
            f"reddit_{dim}.pth",
        ]
        
        # Check local paths first
        for base_path in local_paths:
            if base_path is None:
                continue
            base_path = Path(base_path)
            
            for filename in filenames:
                weights_path = base_path / filename
                if weights_path.exists():
                    return weights_path
        
        # Not found locally - try to download
        return self._download_model(dim)
    
    def _load_models(self, model_path: Optional[str] = None):
        """Load or initialize models for each empathy dimension."""
        models_loaded = 0
        
        print("🔄 Loading empathy models...")
        
        for dim in ["ER", "IP", "EX"]:
            model = BiEncoderEmpathyModel()
            
            # Find weights (local or download)
            weights_path = self._find_model_weights(dim, model_path)
            
            if weights_path and weights_path.exists():
                try:
                    state_dict = torch.load(weights_path, map_location='cpu')
                    model.load_state_dict(state_dict)
                    models_loaded += 1
                except Exception as e:
                    print(f"   ⚠ Failed to load {dim}: {e}")
            
            model.to(self.device)
            model.eval()
            self.models[dim] = model
        
        if models_loaded == 3:
            print(f"✓ Loaded trained empathy models ({models_loaded}/3)")
        elif models_loaded > 0:
            print(f"⚠ Partially loaded empathy models ({models_loaded}/3) - some scores may be inaccurate")
        else:
            print("⚠ No trained models found - using base model (scores will be inaccurate)")
            print("   To download models, ensure you have internet access and try again.")
            print("   Or set SITE_EMPATHY_MODEL_URL to your model hosting location.")
    
    def _encode_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize and encode text."""
        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoded['input_ids'].to(self.device),
            'attention_mask': encoded['attention_mask'].to(self.device),
        }
    
    def _extract_phrases(self, text: str) -> Tuple[List[str], List[str]]:
        """Extract empathic and non-empathic phrases from text."""
        text_lower = text.lower()
        
        empathic = []
        for phrase in self.EMPATHIC_PHRASES:
            if phrase in text_lower:
                empathic.append(phrase)
        
        non_empathic = []
        for phrase in self.NON_EMPATHIC_INDICATORS:
            if phrase in text_lower:
                non_empathic.append(phrase)
        
        return empathic, non_empathic
    
    def score(self, text: str, context: Optional[str] = None) -> EmpathyResult:
        """
        Score a single text for empathy.
        
        Args:
            text: The text to analyze
            context: Optional context (e.g., seeker's message). Uses default if None.
            
        Returns:
            EmpathyResult with scores and analysis
        """
        context = context or self.context
        
        # Encode context and target text
        context_enc = self._encode_text(context)
        text_enc = self._encode_text(text)
        
        results = {}
        
        with torch.no_grad():
            for dim, model in self.models.items():
                logits, _ = model(
                    input_ids_SP=context_enc['input_ids'],
                    input_ids_RP=text_enc['input_ids'],
                    attention_mask_SP=context_enc['attention_mask'],
                    attention_mask_RP=text_enc['attention_mask'],
                )
                
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                pred = int(probs.argmax())
                
                results[dim] = {
                    'probs': probs.tolist(),
                    'pred': pred,
                    # Score: weighted probability of level 1 + level 2
                    'score': float(probs[1] * 0.5 + probs[2] * 1.0),
                }
        
        # Calculate combined empathy score
        empathy_score = (
            results['ER']['score'] * EMPATHY_DIMENSIONS['ER']['weight'] +
            results['IP']['score'] * EMPATHY_DIMENSIONS['IP']['weight'] +
            results['EX']['score'] * EMPATHY_DIMENSIONS['EX']['weight']
        )
        
        # Extract language patterns
        empathic_phrases, non_empathic = self._extract_phrases(text)
        
        return EmpathyResult(
            text=text,
            empathy_score=empathy_score,
            emotional_reaction=results['ER']['score'],
            interpretation=results['IP']['score'],
            exploration=results['EX']['score'],
            er_level=results['ER']['pred'],
            ip_level=results['IP']['pred'],
            ex_level=results['EX']['pred'],
            er_probs=results['ER']['probs'],
            ip_probs=results['IP']['probs'],
            ex_probs=results['EX']['probs'],
            empathic_phrases=empathic_phrases,
            non_empathic_indicators=non_empathic,
        )
    
    def score_batch(
        self, 
        texts: List[str], 
        context: Optional[str] = None,
        batch_size: int = 16
    ) -> List[EmpathyResult]:
        """
        Score multiple texts for empathy.
        
        Args:
            texts: List of texts to analyze
            context: Optional context for all texts
            batch_size: Batch size for processing
            
        Returns:
            List of EmpathyResult objects
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            for text in batch:
                try:
                    result = self.score(text, context)
                    results.append(result)
                except Exception as e:
                    # Return zero scores on error
                    results.append(EmpathyResult(
                        text=text,
                        empathy_score=0.0,
                        emotional_reaction=0.0,
                        interpretation=0.0,
                        exploration=0.0,
                        er_level=0,
                        ip_level=0,
                        ex_level=0,
                        er_probs=[1.0, 0.0, 0.0],
                        ip_probs=[1.0, 0.0, 0.0],
                        ex_probs=[1.0, 0.0, 0.0],
                        empathic_phrases=[],
                        non_empathic_indicators=[f"Error: {str(e)}"],
                    ))
        
        return results

