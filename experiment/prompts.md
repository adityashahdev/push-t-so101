# Prompts from Xie, Chen and Goldberg (arXiv 2608.18227)

Copied from the arXiv HTML on Nov 7, 2026. The PDF text could not be extracted, so check both prompts against the PDF before the gym stage.

## Section 3.1, state-based controller

> Task Description: I want you to use code as policy to achieve Push-T. You are not allowed to use a learned policy, but using existing robot data is fine.

## Section 3.2, vision feedback

> Task Description: Add vision feedback to PushT controller instead of directly taking the geometrical states.

## Lines the two tests target, both in Section 3.2

- "One-Time Calibration": the agent "fits a free scale parameter to match letter geometry" and initializes polygon bounds, center of mass, inertia and contact points. The word "letter" appears, but the bullet is in the Push-T vision section.
- "Per-Push Parameter Adaptation": after each pushing segment, the controller "refines the friction gain parameters".

## Their setup

- Claude Code with Fable 5.
- Push-T success: at least 95% target coverage, over 200 fixed seeds. Push-Alphabet uses 90%.
- About $1,500 to $2,000 of API cost and about 221 hours in auto mode with human interaction, for the whole paper.
