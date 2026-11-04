# Azrael lock solver

Solves the code lock from **Azrael Stash Mod**. Take a screenshot of the lock and the
decoded digits appear in a small always-on-top window and are read out loud.

Runs locally and offline, with no OCR engine or API key needed.

## Download (no Python needed)

Get **AzraelLockSolver.exe** from the [Releases page](../../releases/latest) and double-click it.
It's built automatically by GitHub Actions on every push to `main`
(see `.github/workflows/build-exe.yml`).

For the Russian voice, make a shortcut to the exe and add ` --ru` to the end of its Target.

> Windows SmartScreen may warn about an unsigned app the first time: click
> **More info → Run anyway**.

## Run from source (Windows)

```
pip install -r requirements.txt
python lock_solver.py          # English voice
python lock_solver.py --ru     # Russian voice (if a Russian Windows voice is installed)
```

## Usage

- **Whole screen** (`PrtScn`): one screenshot with both the decode table and the
  `code:` line is solved immediately.
- **Separate snips** (`Win+Shift+S`): snip the decode table and the code line, in either order.

Under the digits, the window shows the symbols it read (e.g. `TG@H>AHB?>`). If an answer
looks wrong, check that line against the lock first.

> If the game runs in exclusive fullscreen the window may be hidden behind it.
> The voice still works. Borderless windowed mode shows both.

Self-test on saved screenshots: `python lock_solver.py --test table.png code.png`

## How it works

Instead of general OCR, it matches glyphs against templates learned from real screenshots
(the font never changes):

1. Builds a text mask that is robust to the dimmed hint band and the "Нажмите…" overlay text.
2. Finds table entries by anchoring on the dashes (`X - 7`).
3. Classifies each symbol and digit against 30 embedded 24×24 templates (`TEMPLATES` in `lock_solver.py`).
4. Uses the rule that every table has 16 distinct symbols to repair a misread symbol.

Checked leave-one-out on 15 tables and 14 code lines (0 errors), plus 3 full-screen screenshots.

## Retraining

If the resolution or the mod's UI changes and it starts misreading:

1. Put screenshots in `training/samples/` (named like the keys in `training/train.py`).
2. Add their labels to `TABLES` / `CODES` in `training/train.py`.
3. Run:
   ```
   python training/train.py           # leave-one-out accuracy check
   python training/train.py --write   # rewrite TEMPLATES in lock_solver.py
   ```
