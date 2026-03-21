function escapeHtml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function renderChips(items, className = 'pill') {
  return items.map(item => `<span class="${className}">${escapeHtml(item)}</span>`).join('');
}

function renderKeyValueMetrics(metrics) {
  const entries = [
    ['J', metrics.J],
    ['Expected gain', metrics.expected_gain],
    ['Replay', metrics.replay_alignment],
    ['Repro', metrics.reproducibility],
    ['Memory GB', metrics.memory_gb],
    ['Runtime min', metrics.runtime_min],
  ];

  return entries.map(([label, value]) => `
    <div class="metric">
      <div class="metric-label">${escapeHtml(label)}</div>
      <div class="metric-value">${escapeHtml(value)}</div>
    </div>
  `).join('');
}

function renderChecks(checks) {
  return Object.entries(checks).map(([rule, passed]) => `
    <li>${escapeHtml(rule)}: <strong>${passed ? 'pass' : 'miss'}</strong></li>
  `).join('');
}

function renderMemory(memories) {
  if (!memories.length) {
    return '<p class="muted">No matching experience retrieved for this task.</p>';
  }

  return `
    <ul class="memory-list">
      ${memories.map(memory => `
        <li>
          <strong>${escapeHtml(memory.experience_id)}</strong><br />
          <span class="small">${escapeHtml(memory.successful_strategy)}</span><br />
          <span class="small">score ${escapeHtml(memory.retrieval_score)} · delta_J ${escapeHtml(memory.delta_J)}</span>
        </li>
      `).join('')}
    </ul>
  `;
}

function renderCandidate(candidate, winnerAgent) {
  const isWinner = candidate.agent === winnerAgent;
  const supportIds = candidate.supporting_memory_ids.length
    ? renderChips(candidate.supporting_memory_ids, 'pill good')
    : '<span class="pill warn">no replay support</span>';

  return `
    <article class="candidate-card ${isWinner ? 'winner' : ''}">
      <div class="chip">${isWinner ? 'winner' : 'candidate'} · ${escapeHtml(candidate.agent)}</div>
      <h3>${escapeHtml(candidate.label)}</h3>
      <p class="body-text">${escapeHtml(candidate.strategy)}</p>
      <div class="metric-row">
        ${renderKeyValueMetrics(candidate.metrics)}
      </div>
      <div class="score-strip">
        <div class="score-card"><span class="small">compatibility</span><strong>${escapeHtml(candidate.metrics.compatibility)}</strong></div>
        <div class="score-card"><span class="small">budget</span><strong>${escapeHtml(candidate.metrics.budget_pass)}</strong></div>
        <div class="score-card"><span class="small">test pass</span><strong>${escapeHtml(candidate.metrics.test_pass)}</strong></div>
      </div>
      <h4>Program patch</h4>
      <ul class="patch-list">
        ${candidate.proposal.program_patch.map(line => `<li>${escapeHtml(line)}</li>`).join('')}
      </ul>
      <h4>Artifacts</h4>
      <ul class="artifact-list">
        ${candidate.proposal.artifacts.map(item => `<li>${escapeHtml(item)}</li>`).join('')}
      </ul>
      <h4>Replay support</h4>
      <div class="pill-row">${supportIds}</div>
      <h4>Required checks</h4>
      <ul class="check-list">
        ${renderChecks(candidate.metrics.required_checks)}
      </ul>
    </article>
  `;
}

function renderRun(run, index) {
  return `
    <section class="panel run-section">
      <div class="run-header">
        <div>
          <div class="eyebrow">task ${index + 1} · ${escapeHtml(run.task.profile.target_device)}</div>
          <h2 class="task-title">${escapeHtml(run.task.title)}</h2>
          <p class="body-text">${escapeHtml(run.task.description)}</p>
        </div>
        <div>
          <div class="winner-banner">${escapeHtml(run.winner.agent)} selected</div>
          <div class="small">delta_J ${escapeHtml(run.delta_J)} · memory ${escapeHtml(run.memory_before_count)} -> ${escapeHtml(run.memory_after_count)}</div>
        </div>
      </div>

      <div class="run-grid">
        <article class="task-block">
          <h3 class="section-title">Retrieved Experience</h3>
          <p class="muted">${escapeHtml(run.plan.selection_policy)}</p>
          ${renderMemory(run.retrieved_memories)}
        </article>

        <article class="task-block">
          <h3 class="section-title">Planner</h3>
          <p class="body-text">${escapeHtml(run.plan.objective)}</p>
          <div class="pill-row">${renderChips(run.plan.active_rules.length ? run.plan.active_rules : ['no active rules yet'])}</div>
          <ul class="patch-list">
            ${run.plan.priority_checks.map(line => `<li>${escapeHtml(line)}</li>`).join('')}
          </ul>
        </article>

        <article class="task-block">
          <h3 class="section-title">Baseline vs Write-back</h3>
          <p class="muted">${escapeHtml(run.selection_reason)}</p>
          <div class="baseline-grid">
            <div class="score-card">
              <span class="small">baseline J</span>
              <strong>${escapeHtml(run.baseline.metrics.J)}</strong>
            </div>
            <div class="score-card">
              <span class="small">winner J</span>
              <strong>${escapeHtml(run.winner.metrics.J)}</strong>
            </div>
            <div class="score-card">
              <span class="small">write back</span>
              <strong>${escapeHtml(run.should_write_memory)}</strong>
            </div>
          </div>
          <h4>New experience</h4>
          ${
            run.new_experience
              ? `<ul class="memory-list">
                  <li><strong>${escapeHtml(run.new_experience.experience_id)}</strong><br /><span class="small">${escapeHtml(run.new_experience.successful_strategy)}</span></li>
                  <li>failure pattern: ${escapeHtml(run.new_experience.failure_pattern)}</li>
                  <li>rules: ${escapeHtml(run.new_experience.reusable_rules.join(', '))}</li>
                </ul>`
              : '<p class="muted">No new experience was added for this task.</p>'
          }
        </article>
      </div>

      <div class="candidate-grid">
        ${run.candidates.map(candidate => renderCandidate(candidate, run.winner.agent)).join('')}
      </div>
    </section>
  `;
}

function renderApp(data) {
  const winnerMix = Object.entries(data.summary.winner_agents)
    .map(([agent, count]) => `<span class="pill good">${escapeHtml(agent)} × ${escapeHtml(count)}</span>`)
    .join('');

  return `
    <section class="panel hero">
      <div class="hero-top">
        <div class="hero-copy">
          <div class="eyebrow">local-first autoresearch flywheel</div>
          <h1>macOS research loops that write back only validated experience.</h1>
          <p>
            This demo merges the fixed-budget experiment style of autoresearch, the Apple Silicon constraints
            of the macOS fork, and OpenEvolve-style proposal competition into one deterministic local prototype.
          </p>
        </div>
        <div class="mini-card">
          <div class="small">generated</div>
          <strong>${escapeHtml(data.summary.generated_at)}</strong>
        </div>
      </div>

      <div class="hero-stats">
        <div class="stat-card">
          <div class="small">tasks</div>
          <div class="stat-value">${escapeHtml(data.summary.num_tasks)}</div>
        </div>
        <div class="stat-card">
          <div class="small">memory growth</div>
          <div class="stat-value">${escapeHtml(data.summary.initial_memory_count)} -> ${escapeHtml(data.summary.memory_size_after_run)}</div>
        </div>
        <div class="stat-card">
          <div class="small">write-backs</div>
          <div class="stat-value">${escapeHtml(data.summary.write_backs)}</div>
        </div>
      </div>

      <div>
        <div class="small">flywheel</div>
        <div class="flywheel">
          ${data.summary.flywheel.map(step => `<div class="flywheel-step">${escapeHtml(step)}</div>`).join('')}
        </div>
      </div>

      <div>
        <div class="small">winner mix</div>
        <div class="pill-row">${winnerMix}</div>
      </div>
    </section>

    <section class="panel run-section">
      <h2 class="section-title">Roadmap</h2>
      <div class="roadmap">
        ${data.roadmap.map(item => `
          <article class="roadmap-card">
            <span class="eyebrow">${escapeHtml(item.status)}</span>
            <strong>${escapeHtml(item.phase)}</strong>
            <span>${escapeHtml(item.detail)}</span>
          </article>
        `).join('')}
      </div>
    </section>

    <section class="panel run-section">
      <h2 class="section-title">Reference Mapping</h2>
      <div class="reference-grid">
        ${data.reference_mapping.map(item => `
          <article class="mini-card">
            <strong>${escapeHtml(item.source)}</strong>
            <span>${escapeHtml(item.takeaway)}</span>
          </article>
        `).join('')}
      </div>
    </section>

    ${data.runs.map(renderRun).join('')}

    <p class="footer-note">The frontend is intentionally read-only. The backend generates the latest run artifact and the UI renders the flywheel for review.</p>
  `;
}

async function main() {
  const root = document.getElementById('app');
  const res = await fetch('/api/latest-run');
  if (!res.ok) {
    throw new Error(`Request failed with status ${res.status}`);
  }
  const data = await res.json();
  root.innerHTML = renderApp(data);
}

main().catch(error => {
  document.getElementById('app').innerHTML = `
    <section class="panel loading-panel">
      <p class="eyebrow">load failure</p>
      <h1>Failed to load the latest flywheel artifact.</h1>
      <p class="body-text">${escapeHtml(error)}</p>
    </section>
  `;
});
