function escapeHtml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function metric(label, value) {
  return `
    <div class="metric">
      <div class="metric-label">${escapeHtml(label)}</div>
      <div class="metric-value">${escapeHtml(value)}</div>
    </div>
  `;
}

function renderSummary(summary) {
  return `
    <section class="panel">
      <div class="summary-grid">
        <article class="summary-card">
          <div class="small">tasks in current run</div>
          <div class="summary-value">${escapeHtml(summary.num_tasks)}</div>
        </article>
        <article class="summary-card">
          <div class="small">memory growth</div>
          <div class="summary-value">${escapeHtml(summary.initial_memory_count)} -> ${escapeHtml(summary.memory_size_after_run)}</div>
        </article>
        <article class="summary-card">
          <div class="small">write backs</div>
          <div class="summary-value">${escapeHtml(summary.write_backs)}</div>
        </article>
      </div>
    </section>
  `;
}

function renderTaskCatalog(taskCatalog, activeTaskId) {
  return `
    <section class="panel">
      <div class="eyebrow">tasks</div>
      <h2>Runnable local tasks</h2>
      <p class="muted">Each task loads a real baseline Python function, evaluates candidate mutations, and writes memory only if the winner actually beats the baseline.</p>
      <div class="task-list">
        ${taskCatalog.map(task => `
          <article class="task-card">
            <button class="task-button ${task.id === activeTaskId ? 'active' : ''}" data-task="${escapeHtml(task.id)}">Run ${escapeHtml(task.id)}</button>
            <h3>${escapeHtml(task.title)}</h3>
            <p class="muted">${escapeHtml(task.description)}</p>
            <div class="small">${escapeHtml(task.baseline_path)}</div>
          </article>
        `).join('')}
      </div>
    </section>
  `;
}

function renderCandidate(candidate, winnerAgent) {
  const isWinner = candidate.agent === winnerAgent;
  const benchmarkValue = candidate.metrics.benchmark_ms === null ? 'n/a' : `${candidate.metrics.benchmark_ms} ms`;
  const speedupValue = candidate.metrics.speedup_vs_baseline === 0 ? 'n/a' : `${candidate.metrics.speedup_vs_baseline}x`;
  const memoryBadge = candidate.supporting_memory_ids.length
    ? `<span class="pill good">memory: ${escapeHtml(candidate.supporting_memory_ids.join(', '))}</span>`
    : `<span class="pill warn">no memory support</span>`;

  return `
    <article class="candidate-card ${isWinner ? 'winner' : ''}">
      <div class="pill ${isWinner ? 'good' : ''}">${isWinner ? 'winner' : 'candidate'} · ${escapeHtml(candidate.agent)}</div>
      <h3>${escapeHtml(candidate.label)}</h3>
      <p class="muted">${escapeHtml(candidate.strategy)}</p>
      <div class="metric-grid">
        ${metric('J', candidate.metrics.J)}
        ${metric('benchmark', benchmarkValue)}
        ${metric('speedup', speedupValue)}
        ${metric('tests', `${candidate.metrics.passed_tests}/${candidate.metrics.total_tests}`)}
        ${metric('stability', candidate.metrics.stability)}
        ${metric('lines', candidate.metrics.line_count)}
      </div>
      <div class="toolbar">${memoryBadge}</div>
      <ul class="list">
        ${candidate.notes.map(note => `<li>${escapeHtml(note)}</li>`).join('')}
      </ul>
      <pre class="code-block"><code>${escapeHtml(candidate.code)}</code></pre>
    </article>
  `;
}

function renderRun(run) {
  const baselineBenchmark = run.baseline.metrics.benchmark_ms === null ? 'n/a' : `${run.baseline.metrics.benchmark_ms} ms`;
  return `
    <section class="panel stack">
      <div>
        <div class="eyebrow">${escapeHtml(run.task.id)}</div>
        <h2>${escapeHtml(run.task.title)}</h2>
        <p class="muted">${escapeHtml(run.task.description)}</p>
      </div>

      <div class="run-grid">
        <article class="code-card">
          <h3>Baseline</h3>
          <p class="muted">${escapeHtml(run.task.baseline_path)}</p>
          <div class="metric-grid">
            ${metric('baseline J', run.baseline.metrics.J)}
            ${metric('benchmark', baselineBenchmark)}
            ${metric('tests', `${run.baseline.metrics.passed_tests}/${run.baseline.metrics.total_tests}`)}
          </div>
          <pre class="code-block"><code>${escapeHtml(run.baseline.code)}</code></pre>
        </article>

        <article class="code-card">
          <h3>Selection</h3>
          <p>${escapeHtml(run.selection_reason)}</p>
          <div class="metric-grid">
            ${metric('delta_J', run.delta_J)}
            ${metric('memory', `${run.memory_before_count} -> ${run.memory_after_count}`)}
            ${metric('write back', run.should_write_memory)}
          </div>
          <h3 style="margin-top:16px;">Retrieved memory</h3>
          <ul class="list">
            ${run.retrieved_memories.map(memory => `
              <li>
                <strong>${escapeHtml(memory.experience_id)}</strong>: ${escapeHtml(memory.successful_strategy)}
              </li>
            `).join('') || '<li>No retrieved memory.</li>'}
          </ul>
        </article>
      </div>

      <div class="candidate-grid">
        ${run.candidates.map(candidate => renderCandidate(candidate, run.winner.agent)).join('')}
      </div>
    </section>
  `;
}

function renderApp(data, activeTaskId) {
  return `
    <section class="panel">
      <div class="header-grid">
        <div>
          <div class="eyebrow">real local runner</div>
          <h1>Task definitions that actually execute.</h1>
          <p class="muted">This page is no longer a blueprint. It runs concrete Python optimization tasks on your Mac, benchmarks the candidates, and stores the winning strategy as reusable experience.</p>
          <div class="toolbar">
            <button class="action-button primary" id="run-sequence">Run full sequence</button>
            <span class="small">generated at ${escapeHtml(data.summary.generated_at)}</span>
          </div>
        </div>
        <div class="summary-card">
          <div class="small">flywheel</div>
          <ul class="list">
            ${data.summary.flywheel.map(step => `<li>${escapeHtml(step)}</li>`).join('')}
          </ul>
        </div>
      </div>
    </section>

    ${renderSummary(data.summary)}
    ${renderTaskCatalog(data.task_catalog, activeTaskId)}
    ${data.runs.map(renderRun).join('')}
  `;
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    throw new Error(`request failed with status ${response.status}`);
  }
  return response.json();
}

async function loadTask(taskId) {
  return fetchJson(`/api/latest-run?task_id=${encodeURIComponent(taskId)}`);
}

async function runTask(taskId) {
  return fetchJson(`/api/run-task?task_id=${encodeURIComponent(taskId)}`, { method: 'POST' });
}

async function runSequence() {
  return fetchJson('/api/run-sequence', { method: 'POST' });
}

async function main() {
  const root = document.getElementById('app');
  let activeTaskId = 'contains-duplicates';
  let data = await loadTask(activeTaskId);

  async function repaint(nextData) {
    data = nextData;
    root.innerHTML = renderApp(data, activeTaskId);

    const taskButtons = root.querySelectorAll('[data-task]');
    for (const button of taskButtons) {
      button.addEventListener('click', async () => {
        activeTaskId = button.getAttribute('data-task');
        root.innerHTML = '<section class="loading-panel"><p class="eyebrow">running</p><h1>Executing selected task...</h1></section>';
        repaint(await runTask(activeTaskId));
      });
    }

    const sequenceButton = root.querySelector('#run-sequence');
    if (sequenceButton) {
      sequenceButton.addEventListener('click', async () => {
        root.innerHTML = '<section class="loading-panel"><p class="eyebrow">running</p><h1>Executing full sequence...</h1></section>';
        activeTaskId = '';
        repaint(await runSequence());
      });
    }
  }

  await repaint(data);
}

main().catch(error => {
  document.getElementById('app').innerHTML = `
    <section class="loading-panel">
      <p class="eyebrow">load failure</p>
      <h1>Failed to load the task runner.</h1>
      <p class="muted">${escapeHtml(error)}</p>
    </section>
  `;
});
