"""Hash the configuration, inputs and binaries of an inversion run."""
import hashlib
import json


def fingerprint_hash(fingerprint):
    return hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
