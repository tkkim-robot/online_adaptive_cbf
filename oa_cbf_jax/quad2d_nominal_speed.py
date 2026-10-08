"""Explicit, bounded speed-only diagnostic around a frozen flight controller."""
from dataclasses import asdict, replace
import math

from .quad2d_control import flight_config_from_contract


def speed_contract(reference, speed):
    if isinstance(speed,bool) or not isinstance(speed,(float,int)) or not math.isfinite(speed):
        raise ValueError('A finite nominal speed is required')
    # This is a bounded sensitivity experiment, not arbitrary controller transfer.
    if not reference.cruise_speed < speed <= 1.25*reference.cruise_speed:
        raise ValueError('Diagnostic speed must increase by at most25percent')
    runtime=replace(reference,cruise_speed=float(speed))
    return runtime,dict(schema='quad2d_modest_nominal_speed_diagnostic_v1',
        reference_config=asdict(reference),cruise_speed=float(speed),
        calibration_coverage_valid=False,final_test=False,
        change='Only nominal cruise speed; unchanged physical limits, route, guidance, gain bank and neural weights.')


def reference_from_manifest(manifest):
    runtime=flight_config_from_contract(manifest['config'])
    diagnostic=manifest.get('diagnostic_nominal_speed')
    if diagnostic is None:return runtime
    if (manifest.get('calibration_coverage_valid') is not False or manifest.get('final_test') is not False
            or 'observation_neighborhood' in manifest):
        raise ValueError('Speed-only diagnosis must not imply calibrated coverage or combine locality')
    reference=flight_config_from_contract(diagnostic['reference_config'])
    expected,contract=speed_contract(reference,diagnostic['cruise_speed'])
    if expected!=runtime or contract!=diagnostic:
        raise ValueError('More than nominal speed changed in the diagnostic')
    return reference
