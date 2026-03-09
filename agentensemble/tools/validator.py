"""
Validation Tool

Data validation and quality assurance for agent outputs.
"""

import re
from typing import Any, List, Optional


class ValidationTool:
    """
    Validation tool for agent outputs.

    Provides configurable validation: length, non-empty, regex, content quality.
    """

    def __init__(
        self,
        validation_mode: str = "fast",
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        required_pattern: Optional[str] = None,
        reject_pattern: Optional[str] = None,
    ):
        """
        Initialize validation tool.

        Args:
            validation_mode: "fast" (basic checks) or "deep" (stricter)
            min_length: Minimum content length
            max_length: Maximum content length
            required_pattern: Regex that must match (e.g., for expected format)
            reject_pattern: Regex that must NOT match (e.g., "error", "failed")
        """
        self.name = "validator"
        self.validation_mode = validation_mode
        self.min_length = min_length
        self.max_length = max_length
        self.required_pattern = re.compile(required_pattern) if required_pattern else None
        self.reject_pattern = re.compile(reject_pattern, re.I) if reject_pattern else None

    def run(
        self,
        value: str,
        context: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Validate a value.

        Args:
            value: Value to validate
            context: Optional context for validation
            **kwargs: Additional parameters

        Returns:
            Validation result with valid, confidence, justifications
        """
        justifications: List[str] = []
        valid = True

        # Non-empty check
        if not value or not str(value).strip():
            valid = False
            justifications.append("Value is empty or whitespace-only")
        else:
            text = str(value).strip()
            length = len(text)

            # Length checks
            if self.min_length is not None and length < self.min_length:
                valid = False
                justifications.append(f"Length {length} below minimum {self.min_length}")
            if self.max_length is not None and length > self.max_length:
                valid = False
                justifications.append(f"Length {length} exceeds maximum {self.max_length}")

            # Required pattern
            if self.required_pattern and valid:
                if not self.required_pattern.search(text):
                    valid = False
                    justifications.append(
                        f"Required pattern '{self.required_pattern.pattern}' not found"
                    )

            # Reject pattern (e.g., error indicators)
            if self.reject_pattern and valid:
                if self.reject_pattern.search(text):
                    valid = False
                    justifications.append(
                        f"Rejected pattern '{self.reject_pattern.pattern}' found"
                    )

            # Deep mode: additional quality checks
            if self.validation_mode == "deep" and valid:
                if text.lower() in ("no result", "no result generated", "not found", "n/a"):
                    valid = False
                    justifications.append("Content indicates no useful result")
                if len(text) < 10 and not text.isdigit():
                    valid = False
                    justifications.append("Content too short for meaningful validation")

        confidence = 0.95 if valid else max(0.0, 0.5 - 0.1 * len(justifications))

        return {
            "valid": valid,
            "confidence": round(confidence, 2),
            "justifications": justifications,
            "value": value[:500] + "..." if len(str(value)) > 500 else value,
        }

    def __call__(
        self,
        value: str,
        context: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Make tool callable."""
        return self.run(value, context, **kwargs)
