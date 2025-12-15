# In file: UAV-RIS_EnergyHarvesting/gym_foo/__init__.py

from gym.envs.registration import register

# The entry point points to the foo_env.py module within the gym_foo package
# and the SecrecyISACEnv class inside it.
register(
    id='foo-v0',
    entry_point='gym_foo.foo_env:SecrecyISACEnv',
)