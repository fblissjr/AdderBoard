#!/usr/bin/env python3
"""Pure Adam training baseline on gentle-mask architecture.

Quick test to see if the a0 (anchor=0) architecture is trainable with Adam.
"""

import sys
import time

import torch
import numpy as np

from train_cmaes import (
    build_model_for_arch,
    unflatten_weights,
    evaluate_fitness_teacher_forced,
    evaluate_fitness_batched,
    _teacher_forced_loss_grad,
)


def main():
    model, _ = build_model_for_arch('a0')

    # Zero init
    x0 = np.zeros(sum(p.numel() for p in model.parameters()))
    unflatten_weights(model, x0)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    digit_schedule = {0: 1, 500: 2, 1500: 3, 3000: 5, 5000: 10}
    current_digits = 1
    n_pairs = 300

    print('Pure Adam training on a0 (gentle mask), curriculum:')
    for step in range(10001):
        # Curriculum
        for thresh in sorted(digit_schedule.keys()):
            if step >= thresh:
                new_d = digit_schedule[thresh]
                if new_d != current_digits:
                    current_digits = new_d
                    print(f'  === Step {step}: {current_digits}-digit numbers ===')

        optimizer.zero_grad()
        loss = _teacher_forced_loss_grad(model, n_pairs=n_pairs, seed=step, max_digits=current_digits)
        loss.backward()
        optimizer.step()

        if step % 200 == 0:
            with torch.no_grad():
                eval_loss, eval_dacc = evaluate_fitness_teacher_forced(
                    model, n_pairs=500, seed=99, max_digits=current_digits
                )
                autoreg = evaluate_fitness_batched(model, n_pairs=200, seed=99)
            print(
                f'  step {step:5d} | loss {eval_loss:.4f} | '
                f'digit_acc {eval_dacc:.3f} | autoreg {autoreg:.3f} | '
                f'digits {current_digits}'
            )
            sys.stdout.flush()


if __name__ == '__main__':
    main()
