"""Source-faithful methods used for A-LQR paper comparison."""

from robust_steerability.source_methods.actadd import (
    ActAddSteerer,
    collect_positionwise_mean,
    fit_actadd_direction,
    positionwise_mean,
)
from robust_steerability.source_methods.calibration import (
    ControlCalibration,
    collect_attention_head_activations,
    collect_block_output_activations,
    collect_decoder_states,
    fit_actadd_calibration,
    fit_control_calibration,
    fit_iti_calibration,
    fit_odesteer_calibration,
    fit_transport_stack,
)
from robust_steerability.source_methods.control import (
    LayerSelectedSetpointPolicy,
    SetpointCalibration,
    build_alqr_policy,
    build_spid_policy,
    fit_setpoint_calibration,
)
from robust_steerability.source_methods.iti import ITIFit, fit_iti, register_iti_hooks
from robust_steerability.source_methods.generation import (
    generate_full_texts,
    generate_unsteered,
    generate_with_control_policy,
    source_completions,
)
from robust_steerability.source_methods.modeling import load_source_model, source_model_spec
from robust_steerability.source_methods.odesteer import (
    ODESteerFit,
    fit_odesteer,
    register_odesteer_hook,
    steer,
    vector_field,
)
from robust_steerability.source_methods.protocol import act_module_patterns, protocol_manifest
from robust_steerability.source_methods.transport import (
    LinearTransport,
    MeanTransport,
    PIDTransport,
    fit_linear_transport,
    fit_mean_transport,
    fit_pid_transport,
    matching_module_names,
    register_transport_hooks,
)

__all__ = [
    "ActAddSteerer",
    "ControlCalibration",
    "ITIFit",
    "LayerSelectedSetpointPolicy",
    "LinearTransport",
    "MeanTransport",
    "ODESteerFit",
    "PIDTransport",
    "SetpointCalibration",
    "build_alqr_policy",
    "build_spid_policy",
    "act_module_patterns",
    "collect_attention_head_activations",
    "collect_block_output_activations",
    "collect_decoder_states",
    "collect_positionwise_mean",
    "fit_actadd_calibration",
    "fit_control_calibration",
    "fit_actadd_direction",
    "fit_iti",
    "fit_iti_calibration",
    "fit_linear_transport",
    "fit_mean_transport",
    "fit_odesteer",
    "fit_odesteer_calibration",
    "fit_pid_transport",
    "fit_setpoint_calibration",
    "fit_transport_stack",
    "generate_full_texts",
    "generate_unsteered",
    "generate_with_control_policy",
    "matching_module_names",
    "load_source_model",
    "positionwise_mean",
    "protocol_manifest",
    "register_iti_hooks",
    "register_odesteer_hook",
    "register_transport_hooks",
    "steer",
    "source_model_spec",
    "source_completions",
    "vector_field",
]
