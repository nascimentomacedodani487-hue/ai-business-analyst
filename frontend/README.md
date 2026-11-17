
# Frontend — AI Business Analyst

Frontend do **AI Business Analyst**, desenvolvido com Next.js, React e TypeScript.

A interface conecta o usuário à API FastAPI e disponibiliza dois fluxos principais:

- **Business Analyst:** perguntas de negócio em linguagem natural, consultas SQL, resultados, gráficos e interpretações.
- **AI Evaluation Engine:** histórico de benchmarks, visualização de relatórios e comparação histórica entre avaliações compatíveis.

## Technology stack

| Technology | Purpose |
|---|---|
| Next.js 16.4.6 | React framework and application routing |
| React 19 | User interface |
| TypeScript | Static typing |
| Tailwind CSS | Styling |
| Recharts | Data visualization |
| ESLint | Code quality |
| npm | Dependency management |

## Project structure

```text
frontend/
├── app/
│   ├── page.tsx
│   └── evaluation/
│       └── page.tsx
├── components/
│   └── evaluation/
│       └── benchmark-history.tsx
├── lib/
│   └── evaluation.ts
├── public/
├── package.json
├── package-lock.json
└── next.config.ts
```

### Main routes

| Route | Description |
|---|---|
| `/` | Main Business Analyst interface |
| `/evaluation` | AI Evaluation Engine and benchmark history |

## Requirements

- Node.js 20.9 or newer
- npm
- Backend API running locally

## Configuration

The frontend communicates with the backend through the `NEXT_PUBLIC_API_URL` environment variable.

Create a `.env.local` file in this directory:

```powershell
"NEXT_PUBLIC_API_URL=http://127.0.0.1:8000" | Out-File -Encoding ascii .env.local
```

The backend must allow requests from `http://localhost:3000` through its CORS configuration.

**Security:** Do not place Gemini, OpenAI or Anthropic API keys in frontend environment variables prefixed with `NEXT_PUBLIC_`. Provider credentials belong exclusively in the backend environment.

## Run locally

From the repository root, first initialize the backend according to the main [`README.md`](../README.md).

Then open a second terminal:

```powershell
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```

Open:

**http://localhost:3000**

The Evaluation Engine is available at:

**http://localhost:3000/evaluation**

## Quality checks

Run these commands from the `frontend/` directory.

```powershell
# ESLint
npx eslint .

# TypeScript
npx tsc --noEmit

# Production build
npm run build
```

The project has been locally validated with ESLint, TypeScript and a successful Next.js production build.

## Related documentation

- [Main project README](../README.md) — architecture, backend, AI orchestration, Evaluation Engine, security, installation and testing.
- [Product demonstration assets](../docs/assets/README.md) — instructions for recording application screenshots and demonstration GIFs.

---

**AI Business Analyst** · Frontend documentation
