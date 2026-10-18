"""Check if generated feedback is supported by retrieved context."""

import re

import structlog

logger = structlog.get_logger()

# Word tokens with surrounding punctuation dropped ("Python," -> "python"), while keeping
# punctuation that is part of a tech name: "node.js", "c++", "c#", "scikit-learn".
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*[+#]*")

# Sentence ends: terminal punctuation followed by whitespace or end of text, so the
# dots inside "Node.js" or "3.11" don't split a claim in two.
_SENTENCE_END_PATTERN = re.compile(r"[.!?]+(?=\s|$)")

_STOP_WORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "and",
    "or",
    "but",
    "in",
    "of",
    "to",
    "for",
    "that",
    "has",
    "have",
    "had",
    "with",
    "on",
    "at",
    "by",
    "as",
    "this",
    "they",
    "their",
    "it",
    "its",
}

# Generic resume words that describe *how much* someone knows rather than *what* they
# know. Left in, they pad a claim's word count and dilute the share of words that
# actually name a skill, which skews the fractional support score.
_GENERIC_RESUME_WORDS = {
    "developer",
    "developers",
    "candidate",
    "knows",
    "knowledge",
    "experience",
    "experienced",
    "expertise",
    "expert",
    "skill",
    "skills",
    "skilled",
    "strong",
    "shows",
    "demonstrated",
}

_IGNORED_WORDS = _STOP_WORDS | _GENERIC_RESUME_WORDS

# Minimum support score for _is_supported() to call a claim supported.
_SUPPORT_THRESHOLD = 0.5


class FaithfulnessChecker:
    """Verify that feedback claims are supported by context."""

    def check(self, feedback: str, context_chunks: list[dict]) -> float:
        """Check faithfulness of feedback to context.

        Args:
            feedback: Generated feedback text
            context_chunks: Retrieved context chunks

        Returns:
            Faithfulness score 0.0-1.0 (average support across claims)
        """
        if not feedback or not context_chunks:
            logger.info(
                "faithfulness_empty_input",
                has_feedback=bool(feedback),
                has_chunks=bool(context_chunks),
            )
            return 0.0

        # Extract key claims from feedback (sentences)
        claims = self._extract_claims(feedback)
        if not claims:
            logger.info("faithfulness_no_claims_extracted")
            return 0.5  # Default to neutral if no extractable claims

        # Concatenate context text
        context_text = " ".join([chunk.get("text", "") for chunk in context_chunks])

        # Score each claim by how much of it the context supports. Claims made only of
        # stop words and generic resume words assert nothing checkable, so skip them.
        claim_scores = [
            score
            for claim in claims
            if (score := self._support_score(claim, context_text)) is not None
        ]
        if not claim_scores:
            logger.info("faithfulness_no_checkable_claims", claims_count=len(claims))
            return 0.5  # Default to neutral if no claim names anything checkable

        score = sum(claim_scores) / len(claim_scores)

        logger.info(
            "faithfulness_checked",
            claims_count=len(claims),
            scored_claims_count=len(claim_scores),
            score=score,
        )

        return score

    @staticmethod
    def _extract_claims(text: str) -> list[str]:
        """Extract key claims from feedback text.

        Args:
            text: Feedback text

        Returns:
            List of claims (sentences of two or more words)
        """
        sentences = _SENTENCE_END_PATTERN.split(text)
        claims = [s.strip() for s in sentences if len(_TOKEN_PATTERN.findall(s.lower())) >= 2]
        return claims[:10]  # Limit to 10 claims for scoring

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Split text into lowercase word tokens, ignoring punctuation.

        Args:
            text: Text to tokenize

        Returns:
            Set of unique tokens
        """
        return set(_TOKEN_PATTERN.findall(text.lower()))

    @staticmethod
    def _support_score(claim: str, context: str) -> float | None:
        """Score how much of a claim is supported by context.

        Args:
            claim: Claim text
            context: Context text

        Returns:
            Share (0.0-1.0) of the claim's meaningful words found in context, or None
            if the claim has no meaningful words to check
        """
        claim_tokens = FaithfulnessChecker._tokenize(claim) - _IGNORED_WORDS
        if not claim_tokens:
            return None

        context_tokens = FaithfulnessChecker._tokenize(context)
        return len(claim_tokens & context_tokens) / len(claim_tokens)

    @staticmethod
    def _is_supported(claim: str, context: str) -> bool:
        """Check if a claim is supported by context.

        Args:
            claim: Claim text
            context: Context text

        Returns:
            True if at least half of the claim's meaningful words appear in context
        """
        score = FaithfulnessChecker._support_score(claim, context)
        return score is not None and score >= _SUPPORT_THRESHOLD
