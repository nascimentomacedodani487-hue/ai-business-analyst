
# Product demonstration assets

All visual assets in this directory must be captured from the real running application.

Do not use mockups, fabricated screenshots or simulated product results. Keep the files optional until they have actually been recorded and reviewed.

## Assets to capture

| File | Type | What to capture |
|---|---|---|
| `demo.gif` | GIF | Main business analysis workflow: open a saved analysis and show its question, SQL, results table, chart and business interpretation. |
| `evaluation-demo.gif` | GIF | Evaluation Engine workflow: browse benchmark history, open a report and compare compatible benchmark runs. |
| `dashboard.png` | Screenshot | Main screen showing a completed business analysis. |
| `analysis-history.png` | Screenshot | Analysis history with search, filters, projects and pinned or archived records. |
| `evaluation-engine.png` | Screenshot | Evaluation Engine with the benchmark history list. |
| `benchmark-details.png` | Screenshot | Detailed view of an individual benchmark report. |
| `benchmark-comparison.png` | Screenshot | Historical comparison between two compatible benchmark reports. |

## How to capture

### 1. Start the application

Follow the installation and startup instructions in the main [`README.md`](../../README.md).

Start both services:

- Backend: `http://127.0.0.1:8000`
- Frontend: `http://localhost:3000`

### 2. Prefer existing records

Use existing successful analysis records and benchmark reports whenever possible.

The main analysis GIF can show the process of opening a previously saved analysis and displaying its question, SQL, table, chart and interpretation.

The Evaluation Engine GIF can show the history, report details and comparison of existing compatible benchmarks.

This approach avoids unnecessary calls to LLM providers and helps preserve API quotas.

Starting a new analysis or running a new benchmark may consume provider API quota. Only do so when a new execution is genuinely needed for the demonstration.

### 3. Capture screenshots

Recommended files:

- `dashboard.png`
- `analysis-history.png`
- `evaluation-engine.png`
- `benchmark-details.png`
- `benchmark-comparison.png`

Guidelines:

- Use the real application in a normal browser window.
- Keep browser dimensions and zoom consistent across screenshots.
- Prefer PNG format for clarity.
- Keep the relevant application context visible.
- Avoid unnecessary browser chrome and empty space.
- Do not expose API keys, `.env` contents, private information or local credentials.

### 4. Record GIF demonstrations

Recommended tools:

- [ScreenToGif](https://www.screentogif.com/) — Windows screen recording and GIF editing.
- [ShareX](https://getsharex.com/) — screenshot and screen recording utility.

Recommended GIF settings:

| Setting | Recommendation |
|---|---|
| Width | Approximately 860 px |
| Frame rate | 10–12 FPS |
| Duration | Up to 25 seconds per GIF |
| File size | Ideally below 8 MB |
| Format | Optimized GIF |

Keep each recording focused on one workflow.

**`demo.gif`**

1. Open the main application.
2. Open a previously saved successful analysis.
3. Show the original business question.
4. Display the generated SQL.
5. Show the results table and chart.
6. Finish with the written business interpretation.

**`evaluation-demo.gif`**

1. Open the Evaluation Engine.
2. Show the benchmark history.
3. Open an existing benchmark report.
4. Display its evaluation details.
5. Select two compatible reports.
6. Show the historical comparison and metric differences.

### 5. Review the assets

Before committing the files, verify that:

- The screenshots come from the real application.
- The GIFs show actual application behavior.
- No credentials or sensitive information are visible.
- Text and charts are readable at the displayed size.
- GIF playback is smooth and the file sizes are reasonable.
- Each image represents a feature that is implemented.

### 6. Include the assets in the main README

Only reference images after the corresponding files exist.

For example:

```markdown
![AI Business Analyst dashboard](docs/assets/dashboard.png)

![Analysis history](docs/assets/analysis-history.png)

![Evaluation Engine](docs/assets/evaluation-engine.png)

![Benchmark details](docs/assets/benchmark-details.png)

![Benchmark comparison](docs/assets/benchmark-comparison.png)
```

For the animated demonstrations:

```markdown
![AI Business Analyst demonstration](docs/assets/demo.gif)

![Evaluation Engine demonstration](docs/assets/evaluation-demo.gif)
```

Do not uncomment or add image references to the main README before the actual files have been recorded and checked.

---

**Reminder:** All demonstrations must represent the current application implementation. Do not present planned features as completed functionality.
