"""Sampling schedule and numerical settings for the four navigation benchmarks."""

import hashlib

import numpy as np


SYSTEMS = ("unicycle", "quad2d", "quad3d", "bicycle")
METHODS = {
    system: ("ours_gat", "ours_fc", "fixed_low", "fixed_high", "optimal_decay")
    + (("optimal_decay_qp",) if system in ("unicycle", "quad2d") else ())
    + ("barriernet",)
    for system in SYSTEMS
}
NOISE = (0.0, 0.5, 1.0, 2.0)
OBSTACLE_COUNTS = (24, 36, 48, 60)
FIELD_FAMILIES = ("random_field", "paired_clusters", "staggered_layers", "random_pockets")
CASES_PER_SYSTEM = 1000
STEPS = 1600
DT = 0.05
MODEL_MANIFEST_SHA256 = "78e7d0fbf2741efa771e73b1f7a4a93dcdd7b34cccaacb1d2c8860db6a08ee44"


def verify_models(directory):
    """Check the trained weights and calibration used by the benchmark."""
    from oa_cbf_jax.model_release import digest, verify

    if digest(directory / "release.json") != MODEL_MANIFEST_SHA256:
        raise ValueError("Benchmark models must match the oa-cbf-models-2026-10 release")
    verify(directory)


def configure_precision(system):
    import jax

    jax.config.update("jax_explicit_x64_dtypes", "allow")
    jax.config.update("jax_enable_x64", system in ("bicycle", "quad3d"))


def _cells(system):
    if system == "unicycle":
        from oa_cbf_jax.scenes import DIVERSE_FAMILIES

        for i, family in enumerate(DIVERSE_FAMILIES):
            for n in range(4):
                for replica in range(31 + (n == i % 4)):
                    yield family, None, n, replica
        return

    for i, family in enumerate(FIELD_FAMILIES):
        for density in range(4):
            extra = {(i + density) % 4, (i + density + 1) % 4}
            if density in (i, (i + 1) % 4):
                extra.add((i + density + 2) % 4)
            for n in range(4):
                for replica in range(15 + (n in extra)):
                    yield family, density, n, replica


def requests(system):
    """Yield the same stratified parent and sensor streams for every method."""
    if system not in SYSTEMS:
        raise ValueError(f"Unknown dynamics {system!r}")
    root = 2026111902 if system in ("quad3d", "unicycle") else 2026111901
    for i, (family, density, n, replica) in enumerate(_cells(system)):
        identity = (
            f"tro_rebuttal_prospective_unique_parent_v1:final:previous:{system}:{root}:{i:04d}"
        )
        seeds = []
        for stream in (0, 1):
            sequence = np.random.SeedSequence([root, SYSTEMS.index(system), 3, 0, i, stream])
            seeds.append(1_000_000 + int(sequence.generate_state(1)[0]) % 3_998_000_000)
        geometry, sensor = seeds
        if system == "unicycle":
            sensor = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:4], "little")
        elif system == "quad2d":
            sensor = geometry + 7193
        yield dict(
            dynamics=system,
            ordinal=i,
            case_id=identity,
            root_seed=root,
            family=family,
            density=density,
            noise_index=n,
            noise_scale=NOISE[n],
            replica=replica,
            geometry_seed=geometry,
            sensor_seed=sensor,
        )
