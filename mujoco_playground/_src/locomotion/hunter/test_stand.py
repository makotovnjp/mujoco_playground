#!/usr/bin/env python3
"""Test script for Hunter robot standing task."""

import jax

from mujoco_playground._src.locomotion.hunter import stand


def test_hunter_standing():
    """Test the Hunter standing environment."""
    print("Creating Hunter standing environment...")
    
    # Create environment with default config
    env = stand.Stand()
    
    print(f"Action space size: {env.action_size}")
    print(f"XML path: {env.xml_path}")
    
    # Test reset
    rng = jax.random.PRNGKey(0)
    state = env.reset(rng)
    
    print(f"Initial observation shape: {state.obs.shape}")
    print(f"Initial reward: {state.reward}")
    print(f"Initial done: {state.done}")
    
    # Test a few steps
    for step in range(5):
        rng, action_rng = jax.random.split(rng)
        action = jax.random.normal(action_rng, (env.action_size,)) * 0.1
        
        state = env.step(state, action)
        print(f"Step {step + 1}: reward={state.reward:.4f}, done={state.done}")
        
        if state.done:
            print("Episode terminated early")
            break
    
    print("Test completed successfully!")


if __name__ == "__main__":
    test_hunter_standing()
