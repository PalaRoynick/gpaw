# Repository and collaboration rules

The main branch of this repository is `master`. The development branch is `DeepDFT`.

Work exclusively in my repository PalaRoynick/gpaw. Create commits and push them only to my development branch (DeepDFT). Never create pull requests or merge requests. Never push changes to the original GPAW GitLab repository. Do not notify, mention, or request reviews from upstream maintainers.

# Scientific code development requirements

- Preserve GPAW backward compatibility unless the task explicitly requires a breaking change.
- Use ASE/GPAW units consistently, and do not mix eV, Hartree, Å, and Bohr without explicit conversions.
- For changes to energy, force, stress, or derivative calculations, add numerical checks and comparisons against a reference.
- Do not relax test tolerances to hide numerical errors.
- Do not combine large refactorings and changes to physical algorithms in the same commit.
- Do not commit PAW datasets, virtual environments, large calculation output files, or binary artifacts.
