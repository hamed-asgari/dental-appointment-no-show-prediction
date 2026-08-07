"""Frozen hashes for the committed Version 2 synthetic raw benchmark."""

from types import MappingProxyType


FROZEN_V2_RAW_HASHES = MappingProxyType(
    {
        "patients.csv": "37df61e47d4060f1e92af49de224228e02451c9eb72e2820f037285ffa9a8ad6",
        "dentists.csv": "22426232d2fa4051ebe1484b5b495ed5afc2d51e2f5c3f19c654aa1f61cad5e8",
        "appointments.csv": "00d759e69fa51eb5250fafb07e844a7c7ba0cb16dec2b80de47ce78092a162ba",
    }
)
FROZEN_V2_MANIFEST_SHA256 = "7702fa5fa0638c52dd0598e28f35f678fb5d61a886faadf9b38a6e292fdcd561"
FROZEN_V2_DATASET_FINGERPRINT = "d9fdfa1a93091fd15bc34a62d655aef313966e2603d901350a6bd969b4e3c1bf"


__all__ = (
    "FROZEN_V2_DATASET_FINGERPRINT",
    "FROZEN_V2_MANIFEST_SHA256",
    "FROZEN_V2_RAW_HASHES",
)
