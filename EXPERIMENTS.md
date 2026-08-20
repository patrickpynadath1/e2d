# Experiment Notes

## Latent-flow parameterization ablations

Keep normalization, data, initialization, optimizer settings, and evaluation fixed
while comparing:

1. **Direct x0 prediction** (not yet implemented): the final head emits the
   normalized clean latent `x0` directly. This is the literal x0 parameterization
   described in the latent-diffusion paper.
2. **Previous-block-mean residual** (`x0_prev_block_residual`): predict
   `x0 - mean(previous_block)`. The first completion block uses the mean of the
   final valid prompt block; subsequent blocks use the preceding completion block.
3. **Self-conditioning** (`SELF_CONDITIONING=true`): condition the denoiser on a
   stop-gradient preliminary x0 estimate through the learned `2d -> d` fusion head.
   Start with probability `0.5`. One-step sampling uses a bootstrap pass followed
   by the conditioned pass.
4. **Denoiser initialization**: compare the current initialization copied from
   Qwen's preceding transformer layers against training the flow layers from
   scratch. Measure both early loss slope and final converged loss; the pretrained
   weights may encode features useful for language modeling but poorly conditioned
   for the difficult high-noise (`t` roughly 0.75--1.0) denoising regime.

Run the parameterization comparison both without and with self-conditioning. Track
MSE, decoder KL, speculative acceptance rate, expected accepted length, throughput,
and peak training memory. The existing `x0` mode is a noisy-canvas residual
(`x0 - xt`, reconstructed as `xt + prediction`), not literal direct-x0 output, and
should be retained as an additional baseline.

The initialization comparison should use identical data order, noise seeds,
timestep samples, parameterization, and optimizer schedule. Report loss and decoder
KL by timestep bin so an aggregate loss does not hide improvements specifically in
the high-noise region.
