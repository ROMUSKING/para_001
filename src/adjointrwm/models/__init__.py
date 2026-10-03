"""World-model arms (needs PyTorch): the pilot reference model and re-implemented rivals.

Every arm is labelled "-style": these are re-implementations at a matched size on our data
contract, not the official systems. See docs/plans/rival-benchmark-plan.md §2.
"""

from .adjoint_rwm import (
    ALLOCATOR_HEADS,
    AdjointRecursiveWorldModel,
    AdjointRWMConfig,
    sample_candidate_mask,
    sample_single_choice_mask,
    stage1_loss,
)
from .common import (
    INPUT_KEYS,
    TARGET_KEYS,
    ArmDims,
    WorldModel,
    flatten_visual,
    inputs_only,
    prediction_objective,
    variance_covariance_regularizer,
)
from .feature_predictor import FeaturePredictorWorldModel, block_causal_mask
from .hybrid_adjoint import (
    HybridAdjointConfig,
    HybridAdjointRecursiveWorldModel,
    KinematicTransition,
    build_hybrid_adjoint_rwm,
    hybrid_stage1_loss,
)
from .registry import BUILDERS, RECIPE_DEFAULTS, REFERENCE_ARM, WIDTH_GRIDS, build_arm, count_prediction_parameters, match_width
from .rssm import RSSMWorldModel
from .spatial_adapter import SpatialPatchAdapter
from .spatial_adjoint import SpatialAdjointRecursiveWorldModel, build_spatial_adjoint_rwm
from .tdmpc2 import SimNorm, TDMPC2WorldModel

__all__ = [
    "ALLOCATOR_HEADS",
    "AdjointRWMConfig",
    "AdjointRecursiveWorldModel",
    "ArmDims",
    "BUILDERS",
    "FeaturePredictorWorldModel",
    "HybridAdjointConfig",
    "HybridAdjointRecursiveWorldModel",
    "INPUT_KEYS",
    "KinematicTransition",
    "RECIPE_DEFAULTS",
    "REFERENCE_ARM",
    "RSSMWorldModel",
    "SimNorm",
    "SpatialAdjointRecursiveWorldModel",
    "SpatialPatchAdapter",
    "TARGET_KEYS",
    "TDMPC2WorldModel",
    "WIDTH_GRIDS",
    "WorldModel",
    "block_causal_mask",
    "build_arm",
    "build_hybrid_adjoint_rwm",
    "build_spatial_adjoint_rwm",
    "count_prediction_parameters",
    "flatten_visual",
    "hybrid_stage1_loss",
    "inputs_only",
    "match_width",
    "prediction_objective",
    "sample_candidate_mask",
    "sample_single_choice_mask",
    "stage1_loss",
    "variance_covariance_regularizer",
]
