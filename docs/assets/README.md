# Assets to record manually

These files are intentionally **not** in the repository yet. They must come from the real running application (no mockups).

| File | What to capture |
|---|---|
| `demo.gif` | Full journey: type a business question, agent runs, SQL shown, table/chart appears, business interpretation appears. Keep it under ~25 s, ~860 px wide, ~10-12 fps, ideally under 8 MB. |
| `dashboard.png` | (optional) Main screen with a finished analysis |
| `analysis-history.png` | (optional) History panel with pinned/archived items and projects |

## How to record

1. Start the backend and frontend as described in the main README, with at least one provider key in `.env`.
2. Record the browser window with [ScreenToGif](https://www.screentogif.com/) (Windows). Export as GIF and optimize.
3. Save as `docs/assets/demo.gif`.
4. In `README.md`, uncomment the `<img src="docs/assets/demo.gif" ...>` line in both language sections (search for `demo.gif`) and delete the "pending recording" note.
