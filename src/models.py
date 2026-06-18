
"""
Core dataclasses for LLM Cost Autopilot.
All data flowing through the system uses these types.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional
import uuid


@dataclass

class ModelConfig:
    """
    Configuration for a single LLM model in the registry.
    Loaded from config/registry.yaml at startup.
    """
    provider:str
    model_id:str
    cost_per_1k_input_usd:float 
    cost_per_1k_output_usd:float 
    quality_tier :str
    avg_latency_ms : int
    display_name: str
    registry_key: str = ""

    def cost_for_tokens(self , input_tokens:int , output_tokens:int) ->float:
        """Calculate actual cost in USD for a given token count."""
        input_cost=(input_tokens /1000)* self.cost_per_1k_input_usd
        output_cost=(output_tokens /1000)* self.cost_per_1k_output_usd
        return round(input_cost + output_cost , 8)
    
    def to_dict(self)->dict:
        return asdict(self)

@dataclass
class LLMResponse:

    """
    Standarized response returned by each provider.
    The dispatcher always returns this — never a raw SDK object. 
    """
    output_text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    cost_if_highest_quality: float        
    model_id: str
    provider: str
    timestamp: datetime
    request_id: str=field(default_factory=lambda: str(uuid.uuid4())) 

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens
    
    @property 
    def savings_usd(self)-> float :
        """How much cheaper than the highest quality baseline."""
        return round(self.cost_if_highest_quality - self.cost_usd , 8)
    
    @property 
    def savings_pct(self)-> float :
        """How much cheaper than the highest quality baseline."""
        if self.cost_if_highest_quality == 0:
            return 0.0
        return round((self.cost_if_highest_quality - self.cost_usd) / self.cost_if_highest_quality * 100 , 2)
    
    def to_dict(self)->dict:
        d=asdict(self)
        d["timestamp"]=self.timestamp.isoformat()
        d["total_tokens"]=self.total_tokens
        d["savings_usd"]=self.savings_usd
        d["savings_pct"]=self.savings_pct
        return d 
    


@dataclass
class ClassifierResult:
    """
    Output from the complexity classifier.
    Includes confidence so the router can apply safe fallbacks.
    """
    tier: int                   # 1 | 2 | 3
    confidence: float           # 0.0 – 1.0 from predict_proba()
    features: dict              # raw extracted features for logging/debugging
    low_confidence: bool = False  # True if confidence < threshold in routing.yaml

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class QualityScore:
    """
    Output from the LLM-as-judge verifier.
    Populated asynchronously after the user receives their response.
    """
    cheap_score: float          # 1–5
    expensive_score: float      # 1–5
    quality_gap: float          # expensive_score - cheap_score
    routing_correct: bool       # True if gap < threshold AND cheap_score >= min
    failure_reason: Optional[str]  # judge's explanation if routing_correct=False
    judge_cost_usd: float       # cost of the verification call itself

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EscalationResult:
    """
    Outcome of the auto-escalation engine.
    Logged alongside the original request in the DB.
    """
    escalated: bool
    original_model: str
    escalated_model: Optional[str]
    cost_delta_usd: float       # extra cost incurred by escalating
    output: Optional[str]       # new output if escalated, else None
    reason: str                 # human-readable explanation

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ProviderError(Exception):
    """
    Raised by any provider when an API call fails.
    Wraps the original exception with model context.
    """
    model_id: str
    message: str
    original_error: Optional[Exception] = None

    def __str__(self) -> str:
        return f"[{self.model_id}] {self.message}"
