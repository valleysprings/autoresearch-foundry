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
  const matches = summary.karpathy_alignment.matches.map(item => `<li>${escapeHtml(item)}</li>`).join('');
  const gaps = summary.karpathy_alignment.gaps.map(item => `<li>${escapeHtml(item)}</li>`).join('');

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
      <div class="run-grid" style="margin-top:14px;">
        <article class="summary-card">
          <h3>Karpathy Alignment</h3>
          <ul class="list">${matches}</ul>
        </article>
        <article class="summary-card">
          <h3>Current Gaps</h3>
          <ul class="list">${gaps}</ul>
        </article>
      </div>
    </section>
  `;
}

function renderFormulas(formulas) {
  return `
    <section class="panel">
      <div class="eyebrow">scoring</div>
      <h2>How J and delta_J are computed</h2>
      <div class="run-grid">
        <article class="summary-card">
          <h3>J</h3>
          <pre class="code-block"><code>${escapeHtml(formulas.J)}</code></pre>
        </article>
        <article class="summary-card">
          <h3>Speed Score</h3>
          <pre class="code-block"><code>${escapeHtml(formulas.speed_score)}</code></pre>
        </article>
      </div>
      <article class="summary-card" style="margin-top:14px;">
        <h3>delta_J</h3>
        <pre class="code-block"><code>${escapeHtml(formulas.delta_J)}</code></pre>
      </article>
    </section>
  `;
}

function renderTaskCatalog(taskCatalog, activeTaskId) {
  return `
    <section class="panel">
      <div class="eyebrow">tasks</div>
      <h2>Runnable local tasks</h2>
      <p class="muted">Each task loads a real baseline Python function, evaluates candidate mutations, and writes memory only if the winner actually beats the baseline. You can switch tasks or run the whole sequence from the page.</p>
      <div class="task-list">
        ${taskCatalog.map(task => `
          <article class="task-card">
            <button class="task-button ${task.id === activeTaskId ? 'active' : ''}" data-task="${escapeHtml(task.id)}">Run ${escapeHtml(task.id)}</button>
            <h3>${escapeHtml(task.title)}</h3>
            <p class="muted">${escapeHtml(task.description)}</p>
            <div class="small">${escapeHtml(task.family)} · ${escapeHtml(task.baseline_path)}</div>
          </article>
        `).join('')}
      </div>
    </section>
  `;
}

function renderLiveStatus(liveState) {
  if (!liveState) {
    return '';
  }

  const events = liveState.events.map(event => `
    <li>
      <strong>${escapeHtml(event.phase)}</strong>
      ${event.candidate ? ` · ${escapeHtml(event.candidate)}` : ''}
      ${event.architecture ? ` · ${escapeHtml(event.architecture)}` : ''}
      <br />
      <span class="small">${escapeHtml(event.message)}</span>
    </li>
  `).join('');

  return `
    <section class="panel">
      <div class="eyebrow">live run</div>
      <h2>${liveState.status === 'running' ? 'Executing in real time' : 'Last live run'}</h2>
      <p class="muted">${liveState.taskId ? `task: ${liveState.taskId}` : 'full sequence'}</p>
      <ul class="list">
        ${events || '<li>No events yet.</li>'}
      </ul>
    </section>
  `;
}

function renderCandidate(candidate, winnerAgent) {
  const isWinner = candidate.agent === winnerAgent;
  const benchmarkValue = candidate.metrics.benchmark_ms === null ? 'n/a' : `${candidate.metrics.benchmark_ms} ms`;
  const speedupValue = candidate.metrics.speedup_vs_baseline === 0 ? 'n/a' : `${candidate.metrics.speedup_vs_baseline}x`;
  const memoryBadge = candidate.supporting_memory_ids.length
    ? `<span class="pill good">memory: ${escapeHtml(candidate.supporting_memory_ids.join(', '))}</span>`
    : '<span class="pill warn">no memory support</span>';

  return `
    <article class="candidate-card ${isWinner ? 'winner' : ''}">
      <div class="pill ${isWinner ? 'good' : ''}">${isWinner ? 'winner' : 'candidate'} · ${escapeHtml(candidate.agent)}</div>
      <h3>${escapeHtml(candidate.label)}</h3>
      <p class="muted">${escapeHtml(candidate.strategy)}</p>
      <div class="toolbar">
        <span class="pill">${escapeHtml(candidate.architecture_family)}</span>
        ${memoryBadge}
      </div>
      <div class="metric-grid">
        ${metric('J', candidate.metrics.J)}
        ${metric('benchmark', benchmarkValue)}
        ${metric('speedup', speedupValue)}
        ${metric('speed score', candidate.metrics.speed_score)}
        ${metric('tests', `${candidate.metrics.passed_tests}/${candidate.metrics.total_tests}`)}
        ${metric('stability', candidate.metrics.stability)}
      </div>
      <ul class="list">
        ${candidate.notes.map(note => `<li>${escapeHtml(note)}</li>`).join('')}
      </ul>
      <pre class="code-block"><code>${escapeHtml(candidate.code)}</code></pre>
    </article>
  `;
}

function renderArchitectureComparison(run) {
  return `
    <article class="code-card">
      <h3>Architecture Comparison</h3>
      <ul class="list">
        ${run.architectures.map(item => `
          <li>
            <strong>${escapeHtml(item.agent)}</strong>
            <span class="small"> · ${escapeHtml(item.family)} · ${escapeHtml(item.label)}</span><br />
            <span class="small">J=${escapeHtml(item.J)} · benchmark=${escapeHtml(item.benchmark_ms)} ms · speedup=${escapeHtml(item.speedup_vs_baseline)}x</span>
          </li>
        `).join('')}
      </ul>
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
            ${metric('speed score', run.baseline.metrics.speed_score)}
            ${metric('tests', `${run.baseline.metrics.passed_tests}/${run.baseline.metrics.total_tests}`)}
            ${metric('family', run.baseline.architecture_family)}
            ${metric('delta_J', run.delta_J)}
          </div>
          <pre class="code-block"><code>${escapeHtml(run.baseline.code)}</code></pre>
        </article>

        <article class="code-card">
          <h3>Selection</h3>
          <p>${escapeHtml(run.selection_reason)}</p>
          <div class="metric-grid">
            ${metric('memory', `${run.memory_before_count} -> ${run.memory_after_count}`)}
            ${metric('write back', run.should_write_memory)}
            ${metric('winner', run.winner.agent)}
            ${metric('winner family', run.winner.architecture_family)}
            ${metric('winner J', run.winner.metrics.J)}
            ${metric('winner speedup', `${run.winner.metrics.speedup_vs_baseline}x`)}
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

      ${renderArchitectureComparison(run)}

      <div class="candidate-grid">
        ${run.candidates.map(candidate => renderCandidate(candidate, run.winner.agent)).join('')}
      </div>
    </section>
  `;
}

function renderApp(data, activeTaskId, liveState) {
  return `
    <section class="panel">
      <div class="header-grid">
        <div>
          <div class="eyebrow">real local runner</div>
          <h1>Task definitions that actually execute.</h1>
          <p class="muted">This page now behaves more like a tiny autoresearch workbench: it compares multiple architecture families, writes back experience only on measured gains, and can show a live timeline as a task runs.</p>
          <div class="toolbar">
            <button class="action-button primary" id="run-sequence">Run full sequence live</button>
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
    ${renderFormulas(data.formulas)}
    ${renderLiveStatus(liveState)}
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

async function startJob(taskId) {
  const suffix = taskId ? `?task_id=${encodeURIComponent(taskId)}` : '';
  const url = taskId ? `/api/run-task${suffix}` : '/api/run-sequence';
  return fetchJson(url, { method: 'POST' });
}

async function pollJob(jobId) {
  while (true) {
    const job = await fetchJson(`/api/job?job_id=${encodeURIComponent(jobId)}`);
    if (job.status === 'completed' || job.status === 'failed') {
      return job;
    }
    await new Promise(resolve => setTimeout(resolve, 220));
  }
}

async function main() {
  const root = document.getElementById('app');
  let activeTaskId = 'contains-duplicates';
  let liveState = null;
  let data = await loadTask(activeTaskId);

  async function repaint(nextData) {
    data = nextData;
    root.innerHTML = renderApp(data, activeTaskId, liveState);

    for (const button of root.querySelectorAll('[data-task]')) {
      button.addEventListener('click', async () => {
        activeTaskId = button.getAttribute('data-task');
        liveState = { status: 'running', taskId: activeTaskId, events: [] };
        root.innerHTML = renderApp(data, activeTaskId, liveState);
        const jobStart = await startJob(activeTaskId);
        let job = await fetchJson(`/api/job?job_id=${encodeURIComponent(jobStart.job_id)}`);
        while (job.status === 'running') {
          liveState = { status: job.status, taskId: activeTaskId, events: job.events };
          root.innerHTML = renderApp(data, activeTaskId, liveState);
          await new Promise(resolve => setTimeout(resolve, 220));
          job = await fetchJson(`/api/job?job_id=${encodeURIComponent(jobStart.job_id)}`);
        }
        if (job.status === 'failed') {
          throw new Error(job.error || 'job failed');
        }
        liveState = { status: job.status, taskId: activeTaskId, events: job.events };
        await repaint(job.payload);
      });
    }

    const sequenceButton = root.querySelector('#run-sequence');
    if (sequenceButton) {
      sequenceButton.addEventListener('click', async () => {
        activeTaskId = '';
        liveState = { status: 'running', taskId: null, events: [] };
        root.innerHTML = renderApp(data, activeTaskId, liveState);
        const jobStart = await startJob(null);
        let job = await fetchJson(`/api/job?job_id=${encodeURIComponent(jobStart.job_id)}`);
        while (job.status === 'running') {
          liveState = { status: job.status, taskId: null, events: job.events };
          root.innerHTML = renderApp(data, activeTaskId, liveState);
          await new Promise(resolve => setTimeout(resolve, 220));
          job = await fetchJson(`/api/job?job_id=${encodeURIComponent(jobStart.job_id)}`);
        }
        if (job.status === 'failed') {
          throw new Error(job.error || 'job failed');
        }
        liveState = { status: job.status, taskId: null, events: job.events };
        await repaint(job.payload);
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
