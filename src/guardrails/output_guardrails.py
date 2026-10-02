# ============================================================
# src/guardrails/output_guardrails.py - Output Validation
#
# Learning Note:
#   OUTPUT GUARDRAILS check the LLM's response BEFORE sending
#   it to the user. We check for:
#
#   1. HALLUCINATION DETECTION:
#      Does the answer actually come from the retrieved context?
#      If the LLM invented facts not in the context, it hallucinated.
#      We check by measuring semantic overlap between answer and context.
#
#   2. TOXICITY / HARMFUL CONTENT:
#      The LLM might occasionally produce harmful content.
#      We scan the output before returning it.
#
#   3. PII IN OUTPUT:
#      Even if input is clean, LLM might reproduce PII from documents.
#      We scan and redact PII from responses too.
#
#   4. REFUSAL DETECTION:
#      Sometimes the LLM says "I cannot answer that." We detect this
#      so we can return a proper error message to the user.
# ============================================================

import re
from dataclasses import dataclass, field
from typing import List, Optional
from src.config import settings
from src.observability.logger import get_logger

logger = get_logger(__name__)


# Patterns indicating the LLM refused to answer
REFUSAL_PATTERNS = [
    r"i (cannot|can't|am not able to|am unable to)",
    r"i (don't|do not) have (access|information)",
    r"i (cannot|can't) (help|assist) with that",
    r"that (is|'s) (outside|beyond) my",
    r"i (was not|wasn't) provided",
    r"the (context|document|provided information) (does not|doesn't) (contain|mention|include)",
    r"based on the (provided|given) (context|information), i (cannot|can't)",
]

COMPILED_REFUSAL_PATTERNS = [re.compile(p, re.IGNORECASE) for p in REFUSAL_PATTERNS]

# Patterns indicating potential harmful content
HARMFUL_PATTERNS = [
    r"\b(kill|murder|harm|attack)\s+(yourself|others|people)\b",
    r"\b(bomb|explosive|weapon)\s+(making|building|instructions)\b",
    r"\b(hack|crack|exploit)\s+(password|system|account)\b",
]

COMPILED_HARMFUL_PATTERNS = [re.compile(p, re.IGNORECASE) for p in HARMFUL_PATTERNS]


@dataclass
class OutputValidationResult:
    """Result of output guardrail checks."""
    is_safe: bool
    original_response: str
    sanitized_response: str
    hallucination_score: float        # 0=likely hallucinated, 1=grounded in context
    is_refusal: bool                  # True if LLM said it can't answer
    violations: List[str] = field(default_factory=list)
    pii_entities_found: List[str] = field(default_factory=list)


class GroundednessChecker:
    """
    Checks if the LLM response is grounded in the retrieved context
    (not hallucinated).

    Learning Note:
        HALLUCINATION is when an LLM generates confident-sounding facts
        that are NOT in the source documents. This is dangerous in
        enterprise RAG because users might trust wrong information.

        Our heuristic approach:
          1. Count how many key phrases from the response appear in context
          2. Compute an overlap ratio (0-1)
          3. If ratio < threshold, flag as potential hallucination

        A more robust approach uses a separate NLI (Natural Language
        Inference) model to check entailment - this is what RAGAS does.
    """

    def check_groundedness(self, response: str, context_chunks: List[str]) -> float:
        """
        Check how well the response is grounded in the context.

        Returns:
            float: 0.0 = completely hallucinated, 1.0 = fully grounded
        """
        if not context_chunks:
            # No context available, cannot check groundedness
            return 0.5

        # Combine all context into one string for checking
        full_context = " ".join(context_chunks).lower()
        response_lower = response.lower()

        # Extract meaningful phrases (3+ word sequences) from response
        words = response_lower.split()
        if len(words) < 3:
            return 1.0  # Too short to reliably check

        # Create sliding window of 3-word phrases
        phrases = [" ".join(words[i:i+3]) for i in range(len(words) - 2)]

        if not phrases:
            return 1.0

        # Count how many response phrases appear in the context
        grounded_count = sum(1 for phrase in phrases if phrase in full_context)
        groundedness_score = grounded_count / len(phrases)

        logger.info(
            "groundedness_check",
            score=round(groundedness_score, 3),
            phrases_checked=len(phrases),
            phrases_grounded=grounded_count,
        )

        return groundedness_score


class OutputGuardrails:
    """
    Full output validation pipeline.
    Validates LLM responses before sending to users.
    """

    HALLUCINATION_THRESHOLD = 0.15  # responses with <15% grounding are flagged

    def __init__(self):
        self.groundedness_checker = GroundednessChecker()
        # Reuse PII detector from input guardrails
        from src.guardrails.input_guardrails import PIIDetector
        self.pii_detector = PIIDetector()

    def _check_refusal(self, response: str) -> bool:
        """Check if the LLM refused to answer."""
        for pattern in COMPILED_REFUSAL_PATTERNS:
            if pattern.search(response):
                return True
        return False

    def _check_harmful_content(self, response: str) -> List[str]:
        """Check for harmful content patterns."""
        violations = []
        for pattern in COMPILED_HARMFUL_PATTERNS:
            if pattern.search(response):
                violations.append(f"Potentially harmful content detected")
                break
        return violations

    def validate(
        self,
        response: str,
        context_chunks: Optional[List[str]] = None,
    ) -> OutputValidationResult:
        """
        Run all output guardrail checks.

        Args:
            response: LLM generated response text
            context_chunks: Retrieved document chunks used for generation
                           (used for hallucination detection)

        Returns:
            OutputValidationResult with safety assessment
        """
        if not settings.enable_output_guardrails:
            return OutputValidationResult(
                is_safe=True,
                original_response=response,
                sanitized_response=response,
                hallucination_score=1.0,
                is_refusal=False,
            )

        violations = []

        # Check 1: Refusal detection
        is_refusal = self._check_refusal(response)

        # Check 2: Harmful content
        violations.extend(self._check_harmful_content(response))

        # Check 3: Groundedness / hallucination check
        hallucination_score = self.groundedness_checker.check_groundedness(
            response, context_chunks or []
        )

        if hallucination_score < self.HALLUCINATION_THRESHOLD and context_chunks:
            violations.append(
                f"Low groundedness score ({hallucination_score:.2f}) - response may not be grounded in context"
            )

        # Check 4: PII in output
        sanitized_response, pii_entities = self.pii_detector.detect_and_redact(response)

        is_safe = len([v for v in violations if "harmful" in v.lower()]) == 0

        result = OutputValidationResult(
            is_safe=is_safe,
            original_response=response,
            sanitized_response=sanitized_response,
            hallucination_score=hallucination_score,
            is_refusal=is_refusal,
            violations=violations,
            pii_entities_found=pii_entities,
        )

        if not is_safe:
            logger.warning("output_guardrail_blocked", violations=violations)
        elif violations:
            logger.warning("output_guardrail_warnings", warnings=violations)

        return result


# Singleton instance
output_guardrails = OutputGuardrails()
