import os
import shutil
import paddle
from typing import Dict, Union, Optional, cast, Tuple, List
from mlagents_envs.exception import UnityPolicyException
from mlagents_envs.logging_util import get_logger
from mlagents.trainers.model_saver.model_saver import BaseModelSaver
from mlagents.trainers.settings import TrainerSettings, SerializationSettings

# Ensure imports point to your converted Paddle classes
from mlagents.trainers.policy.paddle_policy import PaddlePolicy
from mlagents.trainers.optimizer.paddle_optimizer import PaddleOptimizer
from mlagents.trainers.paddle_entities.model_serialization import ModelSerializer


logger = get_logger(__name__)
# Paddle standard extension is .pdparams usually, but we can stick to .pt or .pdparams
DEFAULT_CHECKPOINT_NAME = "checkpoint.pdparams"


class PaddleModelSaver(BaseModelSaver):
    """
    ModelSaver class for PaddlePaddle
    """

    def __init__(
        self, trainer_settings: TrainerSettings, model_path: str, load: bool = False
    ):
        super().__init__()
        self.model_path = model_path
        self.initialize_path = trainer_settings.init_path
        self._keep_checkpoints = trainer_settings.keep_checkpoints
        self.load = load

        self.policy: Optional[PaddlePolicy] = None
        self.exporter: Optional[ModelSerializer] = None
        self.modules: Dict[str, paddle.nn.Layer] = {}

    def register(self, module: Union[PaddlePolicy, PaddleOptimizer]) -> None:
        if isinstance(module, PaddlePolicy) or isinstance(module, PaddleOptimizer):
            self.modules.update(module.get_modules())  # type: ignore
        else:
            raise UnityPolicyException(
                "Registering Object of unsupported type {} to ModelSaver ".format(
                    type(module)
                )
            )
        if self.policy is None and isinstance(module, PaddlePolicy):
            self.policy = module
            self.exporter = ModelSerializer(self.policy)

    def save_checkpoint(self, behavior_name: str, step: int) -> Tuple[str, List[str]]:
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        
        # Determine paths
        checkpoint_path = os.path.join(self.model_path, f"{behavior_name}-{step}")
        paddle_ckpt_path = f"{checkpoint_path}.pdparams"
        export_ckpt_path = f"{checkpoint_path}.onnx"
        
        # Collect State Dicts
        # Note: For optimizers, Paddle uses state_dict() just like Layers
        state_dict = {
            name: module.state_dict() for name, module in self.modules.items()
        }
        
        # Save
        paddle.save(state_dict, paddle_ckpt_path)
        paddle.save(state_dict, os.path.join(self.model_path, DEFAULT_CHECKPOINT_NAME))
        
        # Export ONNX
        self.export(checkpoint_path, behavior_name)
        
        return export_ckpt_path, [paddle_ckpt_path]

    def export(self, output_filepath: str, behavior_name: str) -> None:
        if self.exporter is not None:
            self.exporter.export_policy_model(output_filepath)

    def initialize_or_load(self, policy: Optional[PaddlePolicy] = None) -> None:
        # Initialize/Load registered self.policy by default.
        # If given input argument policy, use the input policy instead.
        # This argument is mainly for initialization of the ghost trainer's fixed policy.
        reset_steps = not self.load
        if self.initialize_path is not None:
            logger.info(f"Initializing from {self.initialize_path}.")
            self._load_model(
                self.initialize_path, policy, reset_global_steps=reset_steps
            )
        elif self.load:
            logger.info(f"Resuming from {self.model_path}.")
            self._load_model(
                os.path.join(self.model_path, DEFAULT_CHECKPOINT_NAME),
                policy,
                reset_global_steps=reset_steps,
            )

    def _load_model(
        self,
        load_path: str,
        policy: Optional[PaddlePolicy] = None,
        reset_global_steps: bool = False,
    ) -> None:
        # paddle.load works for both Layers and state_dict files
        saved_state_dict = paddle.load(load_path)
        
        if policy is None:
            modules = self.modules
            policy = self.policy
        else:
            modules = policy.get_modules()
        policy = cast(PaddlePolicy, policy)

        for name, mod in modules.items():
            try:
                # Paddle uses set_state_dict instead of load_state_dict
                if isinstance(mod, paddle.nn.Layer) or hasattr(mod, 'set_state_dict'):
                    # Paddle doesn't strictly return missing/unexpected keys tuple by default in set_state_dict
                    # but assumes strict=True by default. 
                    # use_structured_name=True is generally safer for complex models.
                    mod.set_state_dict(saved_state_dict[name])
                else:
                    # Fallback or custom objects
                    # If the object has a custom load method, or we try manual assignment
                    if hasattr(mod, "load_state_dict"):
                         mod.load_state_dict(saved_state_dict[name])
                    else:
                         logger.warning(f"Module {name} does not support set_state_dict.")

            # Catching generic errors during loading
            except (KeyError, ValueError, RuntimeError) as err:
                logger.warning(f"Failed to load for module {name}. Initializing")
                logger.debug(f"Module loading error : {err}")

        if reset_global_steps:
            policy.set_step(0)
            logger.info(
                "Starting training from step 0 and saving to {}.".format(
                    self.model_path
                )
            )
        else:
            logger.info(f"Resuming training from step {policy.get_current_step()}.")

    def copy_final_model(self, source_nn_path: str) -> None:
        """
        Copy the .nn (or .pdparams in this case) file at the given source to the destination.
        Also copies the corresponding .onnx file if it exists.
        """
        # source_nn_path might not have extension logic perfect if it was passed from a checkpoint manager
        # expecting .nn vs .pdparams. Usually source_nn_path is the base path or specific file.
        
        final_model_name = os.path.splitext(source_nn_path)[0]

        if SerializationSettings.convert_to_onnx:
            try:
                source_path = f"{final_model_name}.onnx"
                destination_path = f"{self.model_path}.onnx"
                shutil.copyfile(source_path, destination_path)
                logger.info(f"Copied {source_path} to {destination_path}.")
            except OSError:
                pass