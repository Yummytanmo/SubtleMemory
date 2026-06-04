import { Fragment, useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

type MetricKey =
  | "all"
  | "complementary"
  | "multiEvidence"
  | "anyOne"
  | "nuanced"
  | "temporal"
  | "contextual"
  | "contradictory";

type LeaderboardRow = {
  id: string;
  table: string;
  model: string;
  method: string;
  methodLabel: string;
  setting: string;
  category: string;
  metricValues: Record<MetricKey, number>;
};

type LeaderboardData = {
  rows: LeaderboardRow[];
};

type IntegrationRow = {
  model: string;
  system: string;
  base: Record<"all" | "complementary" | "nuanced" | "contradictory", number>;
  plugin: Record<"all" | "complementary" | "nuanced" | "contradictory", number>;
  delta: Record<"all" | "complementary" | "nuanced" | "contradictory", number>;
};

type IntegrationData = {
  rows: IntegrationRow[];
};

type DiagnosticMetricKey = "complementary" | "nuanced" | "contradictory" | "overall";

type DiagnosticScore = {
  pct: number;
};

type DiagnosticRow = {
  rank: number;
  system: string;
  systemLabel: string;
} & Record<DiagnosticMetricKey, DiagnosticScore>;

type DiagnosticData = {
  model: string;
  metrics: Array<{ key: DiagnosticMetricKey; label: string }>;
  memoryPreservation: DiagnosticRow[];
  retrievalGivenPreservation: DiagnosticRow[];
};

type DiagnosticSortKey = "main" | "preservation" | "retrieval";

type SiteData = {
  leaderboard: LeaderboardData;
  integration: IntegrationData;
  diagnostic: DiagnosticData;
};

const baseUrl = import.meta.env.BASE_URL;

const mainOrder = [
  "memobase",
  "mirix",
  "memos",
  "mem0",
  "evermemos",
  "a-mem",
  "metaclaw",
  "openclaw",
  "memos-openclaw",
  "mem0-openclaw",
  "evermemos-openclaw",
  "oracle",
];

const resultMetricColumns: Array<{ key: MetricKey; label: string }> = [
  { key: "multiEvidence", label: "Multi-evidence" },
  { key: "anyOne", label: "Any-one" },
  { key: "complementary", label: "Overall" },
  { key: "temporal", label: "Temporal" },
  { key: "contextual", label: "Contextual" },
  { key: "nuanced", label: "Overall" },
  { key: "contradictory", label: "Contradictory" },
  { key: "all", label: "All" },
];

const integrationMetrics = [
  { key: "complementary", label: "Comp." },
  { key: "nuanced", label: "Nuanced" },
  { key: "contradictory", label: "Contr." },
  { key: "all", label: "All" },
] as const;

const diagnosticMetrics: Array<{ key: DiagnosticMetricKey; label: string }> = [
  { key: "overall", label: "Overall" },
  { key: "complementary", label: "Complementary" },
  { key: "nuanced", label: "Nuanced" },
  { key: "contradictory", label: "Contradictory" },
];

const integrationOrder = ["MemOS", "Mem0", "EverMemOS"];

const diagnosticMainOrder = [
  "memobase",
  "mirix",
  "memos",
  "mem0",
  "evermemos",
  "amem",
  "metaclaw",
  "openclaw",
  "openclaw_memos_plugin",
  "openclaw_mem0_plugin",
  "openclaw_evermemos_plugin",
];

async function loadJson<T>(fileName: string): Promise<T> {
  const response = await fetch(`${baseUrl}data/${fileName}`);
  if (!response.ok) {
    throw new Error(`Failed to load ${fileName}`);
  }
  return response.json() as Promise<T>;
}

function formatPct(value: number): string {
  return `${value.toFixed(1)}%`;
}

function formatPoint(value: number): string {
  return value.toFixed(1);
}

function formatDelta(value: number): string {
  return `${value >= 0 ? "+" : ""}${value.toFixed(1)}`;
}

function assetPath(path: string): string {
  return `${baseUrl}assets/${path}`;
}

function orderedRows(rows: LeaderboardRow[], order: string[]) {
  const byMethod = new Map(rows.map((row) => [row.method, row]));
  return order.map((method) => byMethod.get(method)).filter(Boolean) as LeaderboardRow[];
}

function highlightMap(rows: LeaderboardRow[]) {
  const result = new Map<string, "best" | "second">();
  for (const { key } of resultMetricColumns) {
    const candidates = rows
      .filter((row) => row.category !== "Oracle reference")
      .map((row) => ({ id: row.id, value: row.metricValues[key] }))
      .sort((a, b) => b.value - a.value);
    const distinct = Array.from(new Set(candidates.map((item) => item.value))).sort((a, b) => b - a);
    const best = distinct[0];
    const second = distinct[1];
    for (const item of candidates) {
      if (item.value === best) result.set(`${item.id}:${key}`, "best");
      else if (item.value === second) result.set(`${item.id}:${key}`, "second");
    }
  }
  return result;
}

function App() {
  const [data, setData] = useState<SiteData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([
      loadJson<LeaderboardData>("leaderboard.json"),
      loadJson<IntegrationData>("integration_effect.json"),
      loadJson<DiagnosticData>("diagnostic_waterfall.json"),
    ])
      .then(([leaderboard, integration, diagnostic]) => {
        setData({ leaderboard, integration, diagnostic });
      })
      .catch((loadError: unknown) => {
        setError(loadError instanceof Error ? loadError.message : String(loadError));
      });
  }, []);

  if (error) {
    return (
      <main className="page">
        <section className="status-panel">
          <h1>SubtleMemory</h1>
          <p>{error}</p>
        </section>
      </main>
    );
  }

  if (!data) {
    return (
      <main className="page">
        <section className="status-panel">
          <h1>SubtleMemory</h1>
          <p>Loading benchmark results.</p>
        </section>
      </main>
    );
  }

  return (
    <main className="page">
      <Hero />
      <Narrative />
      <MainResults data={data} />
      <Integration data={data.integration} />
      <DiagnosticAnalysis data={data.diagnostic} />
      <Resources />
    </main>
  );
}

function Hero() {
  return (
    <header className="hero" id="top">
      <nav className="top-nav" aria-label="Primary">
        <a className="brand" href="#top">SubtleMemory</a>
        <a href="#benchmark">Benchmark</a>
        <a href="#design">Design</a>
        <a href="#evaluation">Evaluation</a>
        <a href="#results">Results</a>
        <a href="#integration">Integration</a>
        <a href="#diagnostics">Diagnostics</a>
        <a href="#resources">Resources</a>
      </nav>
      <div className="hero-intro">
        <div className="hero-copy">
          <p className="venue">Long-Horizon Agent Memory Evaluation</p>
          <h1 className="hero-title">
            <span>Subtle</span>
            <span className="title-accent">Memory</span>
          </h1>
          <p className="hero-lead">
            A benchmark for fine-grained relational memory discrimination in long-running AI agents.
          </p>
          <div className="hero-actions">
            <a className="primary-link" href="#results">
              <Icon name="chart" />
              Results
            </a>
            <a className="secondary-link" href="https://github.com/qzds/SubtleMemory">
              <Icon name="github" />
              Code
            </a>
            <a className="secondary-link" href="https://arxiv.org/abs/TODO">
              <Icon name="arxiv" />
              Preprint
            </a>
            <a className="secondary-link" href="https://huggingface.co/papers/TODO">
              <Icon name="huggingface" />
              Hugging Face
            </a>
          </div>
        </div>
      </div>
      <div className="hero-result-board" aria-label="Core benchmark findings">
        <div>
          <span>Oracle reference</span>
          <strong>85.4%</strong>
          <p>GPT-5.4 with target evidence.</p>
        </div>
        <div>
          <span>Best non-oracle</span>
          <strong>71.3%</strong>
          <p>Mem0 + OpenClaw under GPT-5.4.</p>
        </div>
        <div>
          <span>Hardest relation</span>
          <strong>Contradictory</strong>
          <p>Conflict preservation remains the clearest gap.</p>
        </div>
      </div>
      <div className="stat-strip" aria-label="Benchmark scale">
        <Stat label="Long histories" value="10" />
        <Stat label="Relation-controlled sets" value="1,090" />
        <Stat label="Evaluation instances" value="1,522" />
        <Stat label="Evaluated system categories" value="3" />
      </div>
    </header>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="stat">
      <strong>{value}</strong>
      <span>{label}</span>
    </div>
  );
}

type IconName =
  | "arxiv"
  | "chart"
  | "diagnostics"
  | "github"
  | "huggingface"
  | "pipeline"
  | "protocol"
  | "relations"
  | "results"
  | "runtime";

function Icon({ name }: { name: IconName }) {
  if (name === "arxiv") {
    return (
      <img
        className="brand-image-icon platform-icon platform-icon-arxiv"
        src={assetPath("arxiv-logo.svg")}
        alt=""
        aria-hidden="true"
      />
    );
  }

  if (name === "huggingface") {
    return (
      <img
        className="brand-image-icon platform-icon platform-icon-huggingface"
        src={assetPath("hf-logo.svg")}
        alt=""
        aria-hidden="true"
      />
    );
  }

  const paths: Record<Exclude<IconName, "arxiv" | "huggingface">, ReactNode> = {
    chart: (
      <>
        <path d="M4 19V5" />
        <path d="M4 19h16" />
        <path d="M8 16v-5" />
        <path d="M12 16V8" />
        <path d="M16 16v-9" />
      </>
    ),
    diagnostics: (
      <>
        <path d="M5 6h14" />
        <path d="M5 12h14" />
        <path d="M5 18h14" />
        <path d="M8 6v12" />
        <path d="M15 6v12" />
        <path d="M8 12h7" />
      </>
    ),
    github: (
      <path
        d="M12 2.5c-5.25 0-9.5 4.25-9.5 9.5 0 4.19 2.72 7.74 6.49 8.99.47.09.65-.2.65-.45v-1.64c-2.64.57-3.19-1.13-3.19-1.13-.43-1.09-1.05-1.38-1.05-1.38-.86-.59.07-.58.07-.58.95.07 1.45.98 1.45.98.84 1.44 2.21 1.03 2.75.78.09-.61.33-1.03.6-1.27-2.11-.24-4.33-1.06-4.33-4.7 0-1.04.37-1.88.98-2.55-.1-.24-.42-1.21.09-2.51 0 0 .8-.26 2.61.97.76-.21 1.57-.32 2.38-.32s1.62.11 2.38.32c1.81-1.23 2.61-.97 2.61-.97.51 1.3.19 2.27.09 2.51.61.67.98 1.51.98 2.55 0 3.65-2.22 4.46-4.34 4.7.34.29.65.88.65 1.77v2.46c0 .25.17.54.66.45A9.51 9.51 0 0 0 21.5 12c0-5.25-4.25-9.5-9.5-9.5Z"
        fill="currentColor"
        stroke="none"
      />
    ),
    pipeline: (
      <>
        <circle cx="5" cy="12" r="2.5" />
        <circle cx="12" cy="12" r="2.5" />
        <circle cx="19" cy="12" r="2.5" />
        <path d="M7.5 12h2" />
        <path d="M14.5 12h2" />
        <path d="M12 5v4.5" />
        <path d="M12 14.5V19" />
      </>
    ),
    protocol: (
      <>
        <path d="m5 7 2 2 4-4" />
        <path d="M13 7h6" />
        <path d="m5 14 2 2 4-4" />
        <path d="M13 14h6" />
        <path d="M7 20h12" />
      </>
    ),
    relations: (
      <>
        <circle cx="12" cy="12" r="2.5" />
        <circle cx="5" cy="7" r="2.5" />
        <circle cx="19" cy="7" r="2.5" />
        <circle cx="12" cy="19" r="2.5" />
        <path d="M7.2 8.6 10 10.6" />
        <path d="m16.8 8.6-2.8 2" />
        <path d="M12 14.5V16.5" />
      </>
    ),
    results: (
      <>
        <rect x="4" y="5" width="16" height="14" rx="2" />
        <path d="M4 10h16" />
        <path d="M9 5v14" />
        <path d="M14 5v14" />
        <path d="M6.5 14h1" />
        <path d="M11.5 14h1" />
        <path d="M16.5 14h1" />
      </>
    ),
    runtime: (
      <>
        <path d="M8 8h5a3 3 0 0 1 0 6h-2" />
        <path d="M16 16h-5a3 3 0 0 1 0-6h2" />
        <path d="M9 4v3" />
        <path d="M15 17v3" />
        <path d="M4 12h3" />
        <path d="M17 12h3" />
      </>
    ),
  };

  const platformIcons = new Set<IconName>(["github"]);
  const platformClass = platformIcons.has(name)
    ? ` platform-icon platform-icon-${name}`
    : "";

  return (
    <svg className={`link-icon${platformClass}`} viewBox="0 0 24 24" aria-hidden="true">
      {paths[name]}
    </svg>
  );
}

function SectionTitle({ icon, children }: { icon: IconName; children: ReactNode }) {
  return (
    <h2>
      <span className="section-title-row">
        <span className="section-title-icon">
          <Icon name={icon} />
        </span>
        <span>{children}</span>
      </span>
    </h2>
  );
}

function Narrative() {
  return (
    <>
      <section className="section narrative" id="benchmark">
        <div className="section-heading">
          <SectionTitle icon="relations">Benchmark Overview</SectionTitle>
          <p>
            SubtleMemory evaluates whether persistent assistants preserve and use
            relations among related memories, rather than only retrieving isolated facts.
          </p>
        </div>
        <div className="narrative-grid">
          <p>
            As user-agent histories grow, related memories may reinforce one another,
            diverge across time or context, or directly conflict. Correct assistance
            therefore depends on relation-sensitive memory use.
          </p>
          <div className="relation-card">
            <span>Complementary</span>
            <p>Use mutually compatible memories together, or use any sufficient equivalent memory.</p>
          </div>
          <div className="relation-card">
            <span>Nuanced</span>
            <p>Distinguish similar memories by temporal, contextual, role, or condition cues.</p>
          </div>
          <div className="relation-card">
            <span>Contradictory</span>
            <p>Preserve unresolved conflict instead of silently selecting one incompatible record.</p>
          </div>
        </div>
      </section>

      <section className="section construction" id="design">
        <div className="section-heading">
          <SectionTitle icon="pipeline">Benchmark Design</SectionTitle>
          <p>
            The benchmark turns semantic seeds into relation-preserving variants,
            task-oriented sessions, evaluation instances, and chronological histories.
          </p>
        </div>
        <figure className="figure-shell construction-figure">
          <img
            src={assetPath("data_construction.png")}
            alt="SubtleMemory data construction pipeline"
          />
        </figure>
        <div className="process-row">
          <ProcessStep index="1" title="Semantic seeds" text="Start from target information needs." />
          <ProcessStep index="2" title="Relation variants" text="Create complementary, nuanced, and contradictory variants." />
          <ProcessStep index="3" title="Session embedding" text="Embed variants into natural multi-turn interactions." />
          <ProcessStep index="4" title="Evaluation queries" text="Ask later tasks that require relation-aware resolution." />
          <ProcessStep index="5" title="Long histories" text="Assemble chronological persona-level histories." />
        </div>
      </section>

      <section className="section protocol" id="evaluation">
        <div className="section-heading">
          <SectionTitle icon="protocol">Evaluation Protocol</SectionTitle>
          <p>
            Systems run under the same add, finalize, search, answer, and evaluate
            pipeline. Oracle Setting and Perfect Retrieval Setting isolate answer
            generation, memory preservation, and retrieval effects.
          </p>
        </div>
        <div className="protocol-grid">
          <ProtocolItem title="Default Setting" text="The system writes memory, retrieves by query, and answers from the retrieved evidence." />
          <ProtocolItem title="Oracle Setting" text="The answer model receives the original target evidence sessions, bypassing memory formation and retrieval." />
          <ProtocolItem title="Perfect Retrieval Setting" text="The answer model receives memory objects written from the target evidence sessions, bypassing retrieval while preserving memory-construction effects." />
        </div>
      </section>
    </>
  );
}

function ProcessStep({ index, title, text }: { index: string; title: string; text: string }) {
  return (
    <div className="process-step">
      <span>{index}</span>
      <h3>{title}</h3>
      <p>{text}</p>
    </div>
  );
}

function ProtocolItem({ title, text }: { title: string; text: string }) {
  return (
    <div className="protocol-item">
      <h3>{title}</h3>
      <p>{text}</p>
    </div>
  );
}

function MainResults({ data }: { data: SiteData }) {
  const gpt54Rows = useMemo(
    () =>
      orderedRows(
        data.leaderboard.rows.filter((row) => row.table === "main_table" && row.model === "gpt-5.4"),
        mainOrder,
      ),
    [data.leaderboard.rows],
  );
  const gptOssRows = useMemo(
    () =>
      orderedRows(
        data.leaderboard.rows.filter((row) => row.table === "main_table" && row.model === "gpt-oss-120b"),
        mainOrder,
      ),
    [data.leaderboard.rows],
  );
  return (
    <section className="section results-section" id="results">
      <div className="section-heading">
        <SectionTitle icon="results">Main Results</SectionTitle>
        <p>
          Default Setting results are reported across answer-generation backbones,
          standalone memory systems, framework-native agents, and OpenClaw plugin
          memory systems; Oracle Setting isolates answer generation from memory and
          retrieval effects.
        </p>
      </div>
      <Takeaways
        items={[
          {
            title: "Substantial oracle gap",
            text: "Under GPT-5.4, A-Mem achieves 70.0% overall, followed by Mem0 at 69.0% and EverMemOS at 68.1%, while Oracle reaches 85.4%.",
          },
          {
            title: "Agent context organization matters",
            text: "Integrating strong memory plugins raises OpenClaw to 69.1% with EverMemOS and 71.3% with Mem0 under GPT-5.4.",
          },
          {
            title: "Contradictions remain hard",
            text: "Under GPT-5.4, A-Mem achieves the strongest contradictory performance at 50.4%, but still trails Oracle by 18.3 points.",
          },
        ]}
      />
      <PaperResultsTable
        title="Main results on SubtleMemory"
        blocks={[
          { label: "Base Model: GPT-5.4", rows: gpt54Rows },
          { label: "Base Model: GPT-OSS-120B", rows: gptOssRows },
        ]}
      />
    </section>
  );
}

function Takeaways({ items }: { items: Array<{ title: string; text: string }> }) {
  return (
    <div className="takeaway-grid" aria-label="Key findings">
      {items.map((item) => (
        <div className="takeaway-card" key={item.title}>
          <span>Key finding</span>
          <h3>{item.title}</h3>
          <p>{item.text}</p>
        </div>
      ))}
    </div>
  );
}

function PaperResultsTable({
  title,
  blocks,
}: {
  title: string;
  blocks: Array<{ label: string; rows: LeaderboardRow[] }>;
}) {
  return (
    <div className="table-wrap paper-table-wrap" aria-label={title}>
      <table className="paper-results-table">
        <thead>
          <tr>
            <th rowSpan={2}>Method</th>
            <th colSpan={3}>Complementary</th>
            <th colSpan={3}>Nuanced</th>
            <th rowSpan={2}>Contradictory</th>
            <th rowSpan={2}>All</th>
          </tr>
          <tr>
            <th>Multi-evidence</th>
            <th>Any-one</th>
            <th>Overall</th>
            <th>Temporal</th>
            <th>Contextual</th>
            <th>Overall</th>
          </tr>
        </thead>
        <tbody>
          {blocks.map((block) => {
            const highlights = highlightMap(block.rows);
            return (
              <TableBlock
                key={block.label}
                label={block.label}
                rows={block.rows}
                highlights={highlights}
              />
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function TableBlock({
  label,
  rows,
  highlights,
}: {
  label: string;
  rows: LeaderboardRow[];
  highlights: Map<string, "best" | "second">;
}) {
  return (
    <>
      <tr className="block-row">
        <td colSpan={9}>{label}</td>
      </tr>
      {rows.map((row, index) => {
        const breakBefore = [6, 8, 11].includes(index);
        return (
          <tr
            key={row.id}
            className={`${breakBefore ? "category-break" : ""} ${
              row.category === "Oracle reference" ? "oracle-row" : ""
            }`}
          >
            <td className="method-cell">{row.methodLabel}</td>
            {resultMetricColumns.map(({ key }) => {
              const highlight = highlights.get(`${row.id}:${key}`);
              return (
                <td key={key} className={highlight ? `highlight-${highlight}` : ""}>
                  {formatPct(row.metricValues[key])}
                </td>
              );
            })}
          </tr>
        );
      })}
    </>
  );
}

function DiagnosticAnalysis({ data }: { data: DiagnosticData }) {
  const [sortBy, setSortBy] = useState<DiagnosticSortKey>("main");
  const preservationBySystem = new Map(data.memoryPreservation.map((row) => [row.system, row]));
  const retrievalBySystem = new Map(data.retrievalGivenPreservation.map((row) => [row.system, row]));
  const mainOrderIndex = new Map(diagnosticMainOrder.map((system, index) => [system, index]));
  const rows = diagnosticMainOrder
    .map((system) => {
      const preservation = preservationBySystem.get(system);
      const retrieval = retrievalBySystem.get(system);
      return preservation && retrieval ? { preservation, retrieval } : null;
    })
    .filter(Boolean) as Array<{ preservation: DiagnosticRow; retrieval: DiagnosticRow }>;
  const sortedRows = [...rows].sort((a, b) => {
    if (sortBy === "preservation") {
      return b.preservation.overall.pct - a.preservation.overall.pct;
    }
    if (sortBy === "retrieval") {
      return b.retrieval.overall.pct - a.retrieval.overall.pct;
    }
    return (
      (mainOrderIndex.get(a.preservation.system) ?? Number.MAX_SAFE_INTEGER) -
      (mainOrderIndex.get(b.preservation.system) ?? Number.MAX_SAFE_INTEGER)
    );
  });

  return (
    <section className="section diagnostics-section" id="diagnostics">
      <div className="section-heading">
        <SectionTitle icon="diagnostics">Diagnostic Analysis</SectionTitle>
        <p>
          The diagnostic waterfall analysis decomposes performance into memory
          preservation success and retrieval success conditioned on successful
          preservation under GPT-5.4.
        </p>
      </div>
      <Takeaways
        items={[
          {
            title: "End accuracy cannot localize failures",
            text: "Because memory preservation and retrieval operate sequentially, end-task accuracy alone cannot localize failure sources.",
          },
          {
            title: "Both stages correlate with final accuracy",
            text: "Memory preservation success and conditional retrieval success are generally correlated with final accuracy.",
          },
          {
            title: "Bottlenecks differ by system",
            text: "MemoBase has low preservation but relatively strong retrieval, while OpenClaw has strong contradictory preservation but weak contradictory retrieval.",
          },
        ]}
      />
      <div className="metric-definition-grid" aria-label="Diagnostic metric definitions">
        <div>
          <strong>P<sub>preserve</sub></strong>
          <p>
            Let S<sub>O</sub> be instances answered correctly under the Oracle
            Setting, filtering out answer-generation failures. Let S<sub>P</sub>{" "}
            ⊂ S<sub>O</sub> be instances that remain correct under the Perfect
            Retrieval Setting, indicating that sufficient information is preserved.
            P<sub>preserve</sub> = |S<sub>P</sub>| / |S<sub>O</sub>|.
          </p>
        </div>
        <div>
          <strong>P<sub>retrieve</sub></strong>
          <p>
            Let S<sub>D</sub> ⊂ S<sub>P</sub> be instances that remain correct
            under the Default Setting. Since instances in S<sub>P</sub> already
            contain sufficient preserved information, failures at this stage
            primarily reflect retrieval deficiencies. P<sub>retrieve</sub> =
            |S<sub>D</sub>| / |S<sub>P</sub>|.
          </p>
        </div>
      </div>
      <DiagnosticCombinedTable rows={sortedRows} sortBy={sortBy} onSortChange={setSortBy} />
    </section>
  );
}

function DiagnosticCombinedTable({
  rows,
  sortBy,
  onSortChange,
}: {
  rows: Array<{ preservation: DiagnosticRow; retrieval: DiagnosticRow }>;
  sortBy: DiagnosticSortKey;
  onSortChange: (sortBy: DiagnosticSortKey) => void;
}) {
  const sortOptions: Array<{ key: DiagnosticSortKey; label: string }> = [
    { key: "main", label: "Main Results order" },
    { key: "preservation", label: "Memory Preservation Success" },
    { key: "retrieval", label: "Retrieval Success Given Preservation" },
  ];

  return (
    <div className="diagnostic-paper-table">
      <div className="diagnostic-toolbar">
        <div className="diagnostic-stage-legend" aria-label="Diagnostic stage legend">
          <span><strong>P</strong> Memory Preservation Success</span>
          <span><strong>R</strong> Retrieval Success Given Preservation</span>
        </div>
        <div className="diagnostic-sort-control" aria-label="Diagnostic table sorting">
          <span>Sort by</span>
          <div>
            {sortOptions.map((option) => (
              <button
                key={option.key}
                type="button"
                className={sortBy === option.key ? "active" : ""}
                aria-pressed={sortBy === option.key}
                onClick={() => onSortChange(option.key)}
              >
                {option.label}
              </button>
            ))}
          </div>
          <p>Metric-based sorting is high to low.</p>
        </div>
      </div>
      <div className="table-wrap diagnostic-table-wrap">
        <table className="diagnostic-table" aria-label="Diagnostic waterfall comparison">
          <thead>
            <tr>
              <th>Baseline</th>
              {diagnosticMetrics.map((metric) => (
                <th key={metric.key}>{metric.label}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map(({ preservation, retrieval }) => (
              <tr key={preservation.system}>
                <td className="method-cell">{preservation.systemLabel}</td>
                {diagnosticMetrics.map((metric) => (
                  <DiagnosticStageCell
                    key={metric.key}
                    preservation={preservation[metric.key]}
                    retrieval={retrieval[metric.key]}
                  />
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function DiagnosticStageCell({
  preservation,
  retrieval,
}: {
  preservation: DiagnosticScore;
  retrieval: DiagnosticScore;
}) {
  return (
    <td>
      <div className="diagnostic-stage-cell">
        <DiagnosticBar label="P" value={preservation.pct} />
        <DiagnosticBar label="R" value={retrieval.pct} />
      </div>
    </td>
  );
}

function DiagnosticBar({ label, value }: { label: string; value: number }) {
  return (
    <div className={`diagnostic-bar-row diagnostic-bar-${label.toLowerCase()}`}>
      <span className="diagnostic-bar-label">{label}</span>
      <span className="diagnostic-bar-track" aria-hidden="true">
        <span style={{ width: `${Math.max(0, Math.min(100, value))}%` }} />
      </span>
      <strong>{formatPct(value)}</strong>
    </div>
  );
}

function Integration({ data }: { data: IntegrationData }) {
  const rowsByModel = {
    "gpt-5.4": orderedIntegrationRows(data.rows.filter((row) => row.model === "gpt-5.4")),
    "gpt-oss-120b": orderedIntegrationRows(data.rows.filter((row) => row.model === "gpt-oss-120b")),
  };

  return (
    <section className="section integration-section" id="integration">
      <div className="section-heading">
        <SectionTitle icon="runtime">Agent-Runtime Integration</SectionTitle>
        <p>
          This table reports the effect of using OpenClaw context organization with
          the same external memory backend.
        </p>
      </div>
      <Takeaways
        items={[
          {
            title: "Context organization changes behavior",
            text: "OpenClaw integration is an agent-context intervention: recalled memories are injected through the runtime rather than a flat benchmark context list.",
          },
          {
            title: "Benefits are conditional",
            text: "With GPT-5.4, Mem0 + OpenClaw improves overall accuracy to 71.3%, but gains are concentrated in complementary and nuanced cases.",
          },
          {
            title: "Weaker backbones can suffer",
            text: "Under GPT-OSS-120B, adding the agent layer is generally flat or harmful, suggesting runtime-organized context can make evidence use harder.",
          },
        ]}
      />
      <div className="table-wrap">
        <table className="integration-table">
          <thead>
            <tr>
              <th>System</th>
              <th>Setting</th>
              <th>Comp.</th>
              <th>Nuanced</th>
              <th>Contr.</th>
              <th>All</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(rowsByModel).map(([model, rows]) => (
              <IntegrationBlock key={model} model={model} rows={rows} />
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function orderedIntegrationRows(rows: IntegrationRow[]) {
  const bySystem = new Map(rows.map((row) => [row.system, row]));
  return integrationOrder.map((system) => bySystem.get(system)).filter(Boolean) as IntegrationRow[];
}

function IntegrationBlock({ model, rows }: { model: string; rows: IntegrationRow[] }) {
  return (
    <>
      <tr className="block-row">
        <td colSpan={6}>Base Model: {model === "gpt-5.4" ? "GPT-5.4" : "GPT-OSS-120B"}</td>
      </tr>
      {rows.map((row) => (
        <Fragment key={`${row.model}-${row.system}`}>
          <tr key={`${row.model}-${row.system}-base`}>
            <td rowSpan={3} className="method-cell">{row.system}</td>
            <td>Base</td>
            {integrationMetrics.map(({ key }) => (
              <td key={key}>{formatPoint(row.base[key])}</td>
            ))}
          </tr>
          <tr key={`${row.model}-${row.system}-plugin`}>
            <td>+OpenClaw</td>
            {integrationMetrics.map(({ key }) => (
              <td key={key}>{formatPoint(row.plugin[key])}</td>
            ))}
          </tr>
          <tr key={`${row.model}-${row.system}-delta`} className="delta-table-row">
            <td>Delta</td>
            {integrationMetrics.map(({ key }) => (
              <td key={key} className={row.delta[key] >= 0 ? "gain-text" : "loss-text"}>
                {formatDelta(row.delta[key])}
              </td>
            ))}
          </tr>
        </Fragment>
      ))}
    </>
  );
}

function Resources() {
  return (
    <footer className="site-footer" id="resources">
      <div className="footer-brand">
        <a className="footer-logo" href="#top">SubtleMemory</a>
        <p>Long-horizon agent memory evaluation for relation-sensitive memory use.</p>
      </div>
      <div className="footer-links" aria-label="Project resources">
        <a href="https://github.com/qzds/SubtleMemory">
          <Icon name="github" />
          Code repository
        </a>
        <a href="https://arxiv.org/abs/TODO">
          <Icon name="arxiv" />
          Preprint
        </a>
        <a href="https://huggingface.co/papers/TODO">
          <Icon name="huggingface" />
          Hugging Face
        </a>
      </div>
    </footer>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
