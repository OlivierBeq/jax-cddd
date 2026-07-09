# Model Weights

This branch does not contain source code. It exists solely as a home for the pretrained model weights associated with [jax-cddd](https://github.com/OlivierBeq/jax-cddd).

## Purpose

Weight files are distributed through the [Releases](../../releases) page of this branch rather than committed directly to the repository history. This keeps large binary artifacts out of the main development branches while still providing a stable, versioned location from which weights can be downloaded.

## Usage

1. Go to the [Releases](https://github.com/OlivierBeq/jax-cddd/releases) page.
2. Select the release matching the model version you need.
3. Download the attached weight file(s) from that release's assets.

Please refer to the main branch of [jax-cddd](https://github.com/OlivierBeq/jax-cddd) for installation instructions, usage examples, and documentation on how to load and use these weights.

## Notes

- Do not open pull requests against this branch expecting code review — it is not used for development.
- Each release corresponds to a specific, tagged set of weights. Avoid relying on this branch's tip commit; always reference a specific release/tag for reproducibility. 🔖
