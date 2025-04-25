from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner

class RealCoffeeLowdimRunner(BaseLowdimRunner):
    def __init__(self, output_dir):
        super().__init__(output_dir)

    def run(self, policy: BaseLowdimPolicy):
        # No rollout logic implemented yet
        print("[Runner] Running RealCoffeeLowdimRunner (no-op mode)")
        return dict()
