"""Graphbrain exception hierarchy.

This module defines custom exceptions for the graphbrain library,
providing structured error handling with context information.
"""


class GraphbrainError(Exception):
    """Base exception for all graphbrain errors.

    All graphbrain-specific exceptions inherit from this class,
    allowing callers to catch all graphbrain errors with a single
    except clause if desired.
    """
    pass


class EdgeParseError(GraphbrainError):
    """Raised when edge string parsing fails.

    Attributes:
        edge_str: The edge string that failed to parse.
        position: The character position where parsing failed (if known).
    """

    def __init__(self, message, edge_str=None, position=None):
        super().__init__(message)
        self.edge_str = edge_str
        self.position = position


class StorageError(GraphbrainError):
    """Raised when database/storage operations fail.

    Attributes:
        operation: The operation that failed (e.g., 'add', 'remove', 'read').
        key: The key involved in the failed operation (if applicable).
    """

    def __init__(self, message, operation=None, key=None):
        super().__init__(message)
        self.operation = operation
        self.key = key


class TransactionError(StorageError):
    """Raised when transaction operations fail.

    This includes errors like:
    - Starting a transaction when one is already in progress
    - Ending a transaction when none is active
    - Rollback failures
    """
    pass


class ValidationError(GraphbrainError):
    """Raised when input validation fails.

    Attributes:
        value: The invalid value that was provided.
        expected_type: Description of what type/format was expected.
    """

    def __init__(self, message, value=None, expected_type=None):
        super().__init__(message)
        self.value = value
        self.expected_type = expected_type


# ===================
# Parser Exceptions
# ===================

class ParserError(GraphbrainError):
    """Base exception for parser-related errors.

    Attributes:
        text: The text being parsed when the error occurred.
        token: The spaCy token involved (if applicable).
    """

    def __init__(self, message, text=None, token=None):
        super().__init__(message)
        self.text = text
        self.token = token


class SpacyProcessingError(ParserError):
    """Raised when spaCy processing fails.

    This includes errors during tokenization, POS tagging,
    dependency parsing, or named entity recognition.

    Attributes:
        model: The spaCy model being used.
        stage: The processing stage that failed (e.g., 'tokenize', 'parse').
    """

    def __init__(self, message, text=None, model=None, stage=None):
        super().__init__(message, text=text)
        self.model = model
        self.stage = stage


class AtomConstructionError(ParserError):
    """Raised when atom construction from a token fails.

    Attributes:
        token: The spaCy token that couldn't be converted.
        atom_type: The type of atom being constructed.
    """

    def __init__(self, message, token=None, atom_type=None):
        super().__init__(message, token=token)
        self.atom_type = atom_type


class RuleApplicationError(ParserError):
    """Raised when parse rule application fails.

    Attributes:
        rule: The rule that failed to apply.
        sentence: The sentence being parsed.
        position: The position in the sentence.
    """

    def __init__(self, message, rule=None, sentence=None, position=None):
        super().__init__(message)
        self.rule = rule
        self.sentence = sentence
        self.position = position


class ModelNotFoundError(ParserError):
    """Raised when a required language model is not available.

    Attributes:
        model_name: The name of the model that was not found.
        fallbacks_tried: List of fallback models that were also unavailable.
    """

    def __init__(self, message, model_name=None, fallbacks_tried=None):
        super().__init__(message)
        self.model_name = model_name
        self.fallbacks_tried = fallbacks_tried or []


class CoreferenceError(ParserError):
    """Raised when coreference resolution fails.

    Attributes:
        cluster: The coreference cluster involved (if applicable).
    """

    def __init__(self, message, text=None, cluster=None):
        super().__init__(message, text=text)
        self.cluster = cluster
