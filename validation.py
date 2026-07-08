"""General-purpose validation utilities.

Provides small, composable validation functions used throughout the
framework's configuration and I/O layers. Centralizing these checks keeps
error messages consistent and avoids duplicated ``if``/``raise`` boilerplate
across modules.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Container, Optional, Sequence, TypeVar

_logger = logging.getLogger(__name__)

T = TypeVar("T")


class ValidationError(ValueError):
    """Raised when a validation helper detects an invalid value.

    Attributes:
        field_name: Name of the field or parameter that failed validation.
        value: The offending value (repr-safe; may be truncated for logs).
    """

    def __init__(self, message: str, *, field_name: Optional[str] = None, value: Any = None) -> None:
        super().__init__(message)
        self.field_name = field_name
        self.value = value


def validate_positive(value: float, *, field_name: str, allow_zero: bool = False) -> float:
    """Validate that a numeric value is positive (or non-negative).

    Args:
        value: The numeric value to validate.
        field_name: Name of the field, used in the error message.
        allow_zero: If True, zero is accepted; otherwise the value must be
            strictly greater than zero.

    Returns:
        The validated value, unchanged.

    Raises:
        ValidationError: If the value is not positive (or non-negative when
            ``allow_zero`` is True).
    """
    if allow_zero and value < 0:
        raise ValidationError(
            f"'{field_name}' must be non-negative, got {value}", field_name=field_name, value=value
        )
    if not allow_zero and value <= 0:
        raise ValidationError(
            f"'{field_name}' must be positive, got {value}", field_name=field_name, value=value
        )
    return value


def validate_range(
    value: float,
    *,
    field_name: str,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
    inclusive: bool = True,
) -> float:
    """Validate that a numeric value falls within ``[minimum, maximum]``.

    Args:
        value: The numeric value to validate.
        field_name: Name of the field, used in the error message.
        minimum: Optional lower bound. If ``None``, no lower bound is enforced.
        maximum: Optional upper bound. If ``None``, no upper bound is enforced.
        inclusive: Whether the bounds are inclusive (``True``) or exclusive.

    Returns:
        The validated value, unchanged.

    Raises:
        ValidationError: If the value falls outside the specified range.
    """
    if minimum is not None:
        if (inclusive and value < minimum) or (not inclusive and value <= minimum):
            raise ValidationError(
                f"'{field_name}' must be {'>=' if inclusive else '>'} {minimum}, got {value}",
                field_name=field_name,
                value=value,
            )
    if maximum is not None:
        if (inclusive and value > maximum) or (not inclusive and value >= maximum):
            raise ValidationError(
                f"'{field_name}' must be {'<=' if inclusive else '<'} {maximum}, got {value}",
                field_name=field_name,
                value=value,
            )
    return value


def validate_choice(value: T, *, field_name: str, choices: Container[T]) -> T:
    """Validate that a value is one of an allowed set of choices.

    Args:
        value: The value to validate.
        field_name: Name of the field, used in the error message.
        choices: An allowed container of valid values (e.g. a tuple/set).

    Returns:
        The validated value, unchanged.

    Raises:
        ValidationError: If ``value`` is not present in ``choices``.
    """
    if value not in choices:
        raise ValidationError(
            f"'{field_name}' must be one of {choices!r}, got {value!r}",
            field_name=field_name,
            value=value,
        )
    return value


def validate_type(value: Any, *, field_name: str, expected_type: type | tuple[type, ...]) -> Any:
    """Validate that a value is an instance of the expected type(s).

    Args:
        value: The value to validate.
        field_name: Name of the field, used in the error message.
        expected_type: A type or tuple of types.

    Returns:
        The validated value, unchanged.

    Raises:
        ValidationError: If ``value`` is not an instance of ``expected_type``.
    """
    if not isinstance(value, expected_type):
        raise ValidationError(
            f"'{field_name}' must be of type {expected_type!r}, got {type(value)!r}",
            field_name=field_name,
            value=value,
        )
    return value


def validate_path_exists(
    path: str | Path,
    *,
    field_name: str = "path",
    must_be_file: bool = False,
    must_be_dir: bool = False,
) -> Path:
    """Validate that a filesystem path exists (and optionally its kind).

    Args:
        path: The path to validate.
        field_name: Name of the field, used in the error message.
        must_be_file: If True, the path must exist and be a regular file.
        must_be_dir: If True, the path must exist and be a directory.

    Returns:
        The path coerced to a :class:`pathlib.Path`.

    Raises:
        ValidationError: If the path does not exist, or does not match the
            requested kind, or if both ``must_be_file`` and ``must_be_dir``
            are requested simultaneously.
    """
    if must_be_file and must_be_dir:
        raise ValidationError(
            "must_be_file and must_be_dir cannot both be True", field_name=field_name
        )

    resolved = Path(path)
    if not resolved.exists():
        raise ValidationError(
            f"'{field_name}' does not exist: {resolved}", field_name=field_name, value=str(resolved)
        )
    if must_be_file and not resolved.is_file():
        raise ValidationError(
            f"'{field_name}' must be a file: {resolved}", field_name=field_name, value=str(resolved)
        )
    if must_be_dir and not resolved.is_dir():
        raise ValidationError(
            f"'{field_name}' must be a directory: {resolved}",
            field_name=field_name,
            value=str(resolved),
        )
    return resolved


def validate_non_empty_sequence(value: Sequence[T], *, field_name: str) -> Sequence[T]:
    """Validate that a sequence is non-empty.

    Args:
        value: The sequence to validate.
        field_name: Name of the field, used in the error message.

    Returns:
        The validated sequence, unchanged.

    Raises:
        ValidationError: If the sequence has zero length.
    """
    if len(value) == 0:
        raise ValidationError(f"'{field_name}' must not be empty", field_name=field_name, value=value)
    return value


def validate_file_extension(
    path: str | Path, *, field_name: str, allowed_extensions: Sequence[str]
) -> Path:
    """Validate that a path's extension is among the allowed set.

    Args:
        path: The path to validate.
        field_name: Name of the field, used in the error message.
        allowed_extensions: Sequence of allowed extensions, each including
            the leading dot (e.g. ``(".wav", ".flac")``). Matching is
            case-insensitive.

    Returns:
        The path coerced to a :class:`pathlib.Path`.

    Raises:
        ValidationError: If the path's suffix is not in ``allowed_extensions``.
    """
    resolved = Path(path)
    suffix = resolved.suffix.lower()
    normalized_allowed = {ext.lower() for ext in allowed_extensions}
    if suffix not in normalized_allowed:
        raise ValidationError(
            f"'{field_name}' has unsupported extension '{suffix}'; expected one of "
            f"{sorted(normalized_allowed)}",
            field_name=field_name,
            value=str(resolved),
        )
    return resolved
