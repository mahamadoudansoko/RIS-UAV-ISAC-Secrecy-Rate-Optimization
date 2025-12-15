# In file: gym_foo/__init__.py

from gym.envs.registration import register

# We register our NEW environment class (SecrecyISACEnv)
# under the OLD ID ('foo-v0') that DDPG.py is looking for.
register(
    id='foo-v0',  # <-- CORRECTED ID
    entry_point='gym_foo.foo_env:SecrecyISACEnv',
)