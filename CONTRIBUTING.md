# Contributing to RustSight

We welcome contributions from the community. To ensure a smooth process, please follow these guidelines.

## How to Contribute

1. Fork the repository.
2. Create a new branch for your feature or bug fix: `git checkout -b my-feature`.
3. Make your changes and ensure the code follows existing styles.
4. Commit your changes: `git commit -m "Add feature X"`.
5. Push to your branch: `git push origin my-feature`.
6. Submit a Pull Request to the `main` branch.

## Bug Reports

If you find a bug, please create an issue with the following details:
- Steps to reproduce.
- Expected vs. actual behavior.
- System information (OS, hardware, versions).

## Code Standards

- Follow idiomatic Rust patterns.
- Ensure all code is properly formatted with `cargo fmt`.
- Do not use emojis in code or documentation.
- Do not add comments to the code unless absolutely necessary (project policy).

## Commit Messages

Use clear and descriptive commit messages following the Conventional Commits format for compatibility with our release-please workflow.

Example:
- `feat: add support for new execution provider`
- `fix: correct NMS box calculation`
- `docs: update installation instructions`
- `chore: minor maintenance tasks`
