from __future__ import annotations

import math
from typing import Any, Iterable

from .constants import SEASON_LABELS


def to_season_label(value: Any) -> str | None:
	"""Return a human-readable season label for the given raw identifier."""
	if value is None:
		return None
	if isinstance(value, float) and math.isnan(value):
		return None

	lookup_candidates: Iterable[Any] = ()
	try:
		lookup_candidates = (int(value), str(int(value)))
	except (TypeError, ValueError):
		lookup_candidates = (str(value).strip(),)

	for candidate in lookup_candidates:
		if candidate in SEASON_LABELS:
			return SEASON_LABELS[candidate]

		# Allow matching when dictionary keys are stored as ints but the
		# candidate is a string (or vice versa) without duplicating entries.
		if isinstance(candidate, str):
			try:
				as_int = int(candidate)
			except ValueError:
				as_int = None
			if as_int in SEASON_LABELS:
				return SEASON_LABELS[as_int]

	return str(value)
