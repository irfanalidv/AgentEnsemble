"""
Benchmark Suite

Pre-defined test cases for agent evaluation.
"""

from typing import Any


class Benchmark:
    """
    Benchmark suite for agent testing.

    Provides pre-defined test cases for different scenarios.
    """

    def __init__(self, test_cases: list[dict[str, Any]]):
        """
        Initialize benchmark.

        Args:
            test_cases: List of test case dictionaries
        """
        self.test_cases = test_cases

    @classmethod
    def research_tasks(cls) -> "Benchmark":
        """
        Research tasks benchmark.

        Returns:
            Benchmark with research-oriented test cases
        """
        test_cases = [
            {
                "query": "Find the CEO of Stripe",
                "expected_type": "person_name",
                "category": "research",
            },
            {
                "query": "What year was OpenAI founded?",
                "expected_type": "year",
                "category": "research",
            },
            {
                "query": "Find the official email of Anthropic",
                "expected_type": "email",
                "category": "research",
            },
        ]
        return cls(test_cases)

    @classmethod
    def data_extraction_tasks(cls) -> "Benchmark":
        """
        Data extraction tasks benchmark.

        Returns:
            Benchmark with extraction-oriented test cases
        """
        test_cases = [
            {
                "query": "Extract company name and industry from: Stripe is a payments company",
                "expected_type": "structured_data",
                "category": "extraction",
            },
        ]
        return cls(test_cases)

    @classmethod
    def load(cls, name: str) -> "Benchmark":
        """
        Load a benchmark by name.

        Args:
            name: Benchmark name

        Returns:
            Benchmark instance
        """
        if name == "research_tasks":
            return cls.research_tasks()
        if name == "data_extraction_tasks":
            return cls.data_extraction_tasks()
        raise ValueError(f"Unknown benchmark: {name}")
