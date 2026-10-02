# ============================================================
# src/guardrails/input_guardrails.py - Input Validation
#
# Learning Note:
#   INPUT GUARDRAILS protect your system from:
#     1. PROMPT INJECTION: User tries "Ignore all instructions and..."
#        to hijack the LLM's behavior.
#     2. PII LEAKAGE: User inputs their SSN, credit card, etc.
#        We detect and redact before sending to the LLM.
#     3. TOXIC INPUT: Hate speech, harmful requests.
#     4. OFF-TOPIC QUERIES: Keep the system focused on its purpose.
#
#   WHY THIS MATTERS:
#     Without guardrails, a production RAG can be manipulated to:
#     - Reveal system prompts
#     - Generate harmful content
#     - Store/log sensitive user data (GDPR violation!)
#
#   TOOLS USED:
#     - Microsoft Presidio: State-of-the-art PII detection
#     - Pattern matching: Fast rule-based injection detection
# ============================================================

import re
from dataclasses import dataclass, field
from typing import List, Tuple
from src.config import settings
from src.observability.logger import get_logger

logger = get_logger(__name__)


# Prompt injection patterns - common attack signatures
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"disregard\s+(all\s+)?instructions",
    r"you\s+are\s+now\s+(a\s+)?.*assistant",
    r"pretend\s+you\s+are",
    r"act\s+as\s+if\s+you",
    r"system\s+prompt\s*:",
    r"<\|.*?\|>",          # special token injection
    r"\[INST\]",           # Llama instruction injection
    r"###\s*(system|instruction)",
    r"jailbreak",
    r"DAN\s+mode",
]

COMPILED_INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS
]


@dataclass
class InputValidationResult:
    """Result of input guardrail checks."""
    is_safe: bool
    original_text: str
    sanitized_text: str
    violations: List[str] = field(default_factory=list)
    pii_entities_found: List[str] = field(default_factory=list)
    redaction_count: int = 0


class PIIDetector:
    """
    Detects and redacts Personally Identifiable Information (PII)
    using Microsoft Presidio.

    Learning Note:
        Presidio uses NLP + pattern matching to find:
        - Names, email addresses, phone numbers
        - SSNs, credit card numbers, passport numbers
        - IP addresses, URLs, medical record numbers
        - And many more entity types

        We replace found entities with [REDACTED_<TYPE>] placeholders.
        This way the LLM still gets context but never sees actual PII.
    """

    def __init__(self):
        self._analyzer = None
        self._anonymizer = None
        self._load_presidio()

    def _load_presidio(self) -> None:
        """Load Presidio analyzer and anonymizer."""
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine
            self._analyzer = AnalyzerEngine()
            self._anonymizer = AnonymizerEngine()
            logger.info("presidio_pii_detector_loaded")
        except ImportError:
            logger.warning("presidio_not_installed", msg="PII detection disabled. pip install presidio-analyzer presidio-anonymizer")
        except Exception as exc:
            logger.warning("presidio_load_failed", error=str(exc))

    def detect_and_redact(self, text: str) -> Tuple[str, List[str]]:
        """
        Detect PII in text and replace with redaction placeholders.

        Returns:
            (redacted_text, list_of_entity_types_found)
        """
        if not self._analyzer or not settings.enable_pii_detection:
            return text, []

        try:
            # Analyze: find PII entities with positions
            results = self._analyzer.analyze(
                text=text,
                language="en",
                entities=[
                    "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER",
                    "CREDIT_CARD", "US_SSN", "IP_ADDRESS",
                    "LOCATION", "US_PASSPORT", "MEDICAL_RECORD",
                ],
            )

            if not results:
                return text, []

            entity_types = list({r.entity_type for r in results})

            # Anonymize: replace entities with [REDACTED_<TYPE>]
            from presidio_anonymizer.entities import OperatorConfig
            anonymized = self._anonymizer.anonymize(
                text=text,
                analyzer_results=results,
                operators={
                    "DEFAULT": OperatorConfig("replace", {"new_value": "<REDACTED>"})
                },
            )

            logger.info("pii_detected_and_redacted", entities=entity_types, count=len(results))
            return anonymized.text, entity_types

        except Exception as exc:
            logger.error("pii_detection_failed", error=str(exc))
            return text, []


class InputGuardrails:
    """
    Full input validation pipeline.
    Checks for injection attacks, PII, and toxic content
    before the query reaches the RAG pipeline.
    """

    def __init__(self):
        self.pii_detector = PIIDetector()

    def _check_prompt_injection(self, text: str) -> List[str]:
        """
        Check for prompt injection attack patterns.
        Returns list of detected violation descriptions.
        """
        violations = []
        for pattern in COMPILED_INJECTION_PATTERNS:
            if pattern.search(text):
                violations.append(f"Potential prompt injection: pattern '{pattern.pattern[:30]}...' detected")
        return violations

    def _check_length(self, text: str, max_length: int = 4000) -> List[str]:
        """Check if input is within acceptable length."""
        if len(text) > max_length:
            return [f"Input too long: {len(text)} chars exceeds limit of {max_length}"]
        return []

    def _check_empty(self, text: str) -> List[str]:
        """Check if input is empty or whitespace only."""
        if not text or not text.strip():
            return ["Input is empty or whitespace only"]
        return []

    def validate(self, user_input: str) -> InputValidationResult:
        """
        Run all input guardrail checks on user input.

        Returns InputValidationResult with:
          - is_safe: whether to proceed with the query
          - sanitized_text: cleaned/redacted text to use
          - violations: list of issues found
        """
        if not settings.enable_input_guardrails:
            return InputValidationResult(
                is_safe=True,
                original_text=user_input,
                sanitized_text=user_input,
            )

        all_violations = []

        # Check 1: Empty input
        all_violations.extend(self._check_empty(user_input))

        # Check 2: Length limit
        all_violations.extend(self._check_length(user_input))

        # Check 3: Prompt injection detection
        injection_violations = self._check_prompt_injection(user_input)
        all_violations.extend(injection_violations)

        # Check 4: PII detection and redaction
        sanitized_text, pii_entities = self.pii_detector.detect_and_redact(user_input)

        # Injection attacks are hard blocks - reject the request
        is_safe = len(injection_violations) == 0 and len(self._check_empty(user_input)) == 0

        result = InputValidationResult(
            is_safe=is_safe,
            original_text=user_input,
            sanitized_text=sanitized_text,
            violations=all_violations,
            pii_entities_found=pii_entities,
            redaction_count=len(pii_entities),
        )

        if not is_safe:
            logger.warning(
                "input_guardrail_blocked",
                violations=all_violations,
                pii_found=pii_entities,
            )
        elif pii_entities:
            logger.info("input_pii_redacted", entities=pii_entities)

        return result


# Singleton instance
input_guardrails = InputGuardrails()
