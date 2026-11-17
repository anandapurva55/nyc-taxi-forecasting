"""Fixed NYC TLC zone groups used as model features and serving checks.

IDs come from the official taxi_zone_lookup.csv. They are embedded here so the
API can build features without shipping the lookup file.
"""

from __future__ import annotations

# Newark (EWR), JFK and LaGuardia.
AIRPORT_ZONES = frozenset({1, 132, 138})
JFK_ZONE = 132

# Zones with no usable location: 264 "Unknown" and 265 "Outside of NYC".
UNKNOWN_ZONES = frozenset({264, 265})

MANHATTAN_ZONES = frozenset({
    4, 12, 13, 24, 41, 42, 43, 45, 48, 50, 68, 74, 75, 79, 87, 88, 90, 100, 103,
    104, 105, 107, 113, 114, 116, 120, 125, 127, 128, 137, 140, 141, 142, 143, 144,
    148, 151, 152, 153, 158, 161, 162, 163, 164, 166, 170, 186, 194, 202, 209, 211,
    224, 229, 230, 231, 232, 233, 234, 236, 237, 238, 239, 243, 244, 246, 249, 261,
    262, 263,
})


def is_airport_trip(pu_location_id: int, do_location_id: int) -> bool:
    """True when the trip starts or ends at an airport."""
    return pu_location_id in AIRPORT_ZONES or do_location_id in AIRPORT_ZONES


def is_jfk_manhattan(pu_location_id: int, do_location_id: int) -> bool:
    """True for JFK to Manhattan (either direction), which is a flat-fare route."""
    return (pu_location_id == JFK_ZONE and do_location_id in MANHATTAN_ZONES) or (
        do_location_id == JFK_ZONE and pu_location_id in MANHATTAN_ZONES
    )


def has_unknown_zone(pu_location_id: int, do_location_id: int) -> bool:
    """True when either end of the trip has no usable zone."""
    return pu_location_id in UNKNOWN_ZONES or do_location_id in UNKNOWN_ZONES


# Shown with fares for these trips, which had the largest errors in the subgroup audit.
UNKNOWN_ZONE_NOTE = (
    "Pickup or drop-off zone is unknown (264 or 265); these trips have the "
    "largest fare errors in the subgroup audit."
)
