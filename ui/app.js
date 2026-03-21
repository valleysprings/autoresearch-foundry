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

function shortPath(path) {
  return path ? path.replace(/^runs\//, '') : 'n/a';
}

function renderErrorPanel(errorPayload) {
  if (!errorPayload) {
    return '';
  }
  return `
    <section class="panel error-panel">
      <div class="eyebrow">terminal failure</div>
      <h2>${escapeHtml(errorPayload.error_type || 'runtime_error')}</h2>
      <p class="muted">${escapeHtml(errorPayload.error || 'Unknown error')}</p>
      <div class="metric-grid">
        ${metric('terminal', String(Boolean(errorPayload.terminal)))}
        ${metric('model', errorPayload.model || 'n/a')}
      </div>
    </section>
  `;
}

function renderSummary(summary, audit) {
  const proposalEngine = summary.proposal_engine || {};
  return `
    <section class="panel">
      <div class="summary-grid">
        <article class="summary-card">
          <div class="small">mode</div>
          <div class="summary-value">${escapeHtml(summary.run_mode)}</div>
        </article>
        <article class="summary-card">
          <div class="small">active model</div>
          <div class="summary-value">${escapeHtml(summary.active_model)}</div>
        </article>
        <article class="summary-card">
          <div class="small">tasks</div>
          <div class="summary-value">${escapeHtml(summary.num_tasks)}</div>
        </article>
        <article class="summary-card">
          <div class="small">generations</div>
          <div class="summary-value">${escapeHtml(summary.total_generations)}</div>
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
          <h3>Flywheel</h3>
          <ul class="list">${summary.flywheel.map(step => `<li>${escapeHtml(step)}</li>`).join('')}</ul>
        </article>
        <article class="summary-card">
          <h3>Audit</h3>
          <div class="metric-grid">
            ${metric('git commit', summary.git_commit)}
            ${metric('session id', audit.session_id || 'n/a')}
            ${metric('workspace', audit.workspace_root || 'n/a')}
          </div>
          <ul class="list">
            <li><strong>source repo</strong>: ${escapeHtml(summary.source_repo)}</li>
            <li><strong>upstream target</strong>: ${escapeHtml(summary.upstream_target)}</li>
            <li><strong>api base</strong>: ${escapeHtml(proposalEngine.api_base || 'n/a')}</li>
          </ul>
        </article>
      </div>
    </section>
  `;
}

function renderFormulas(formulas) {
  return `
    <section class="panel">
      <div class="eyebrow">scoring</div>
      <h2>Deterministic Verifier</h2>
      <div class="run-grid">
        <article class="summary-card">
          <h3>J</h3>
          <pre class="code-block"><code>${escapeHtml(formulas.J)}</code></pre>
        </article>
        <article class="summary-card">
          <h3>Objective</h3>
          <pre class="code-block"><code>${escapeHtml(formulas.objective)}</code></pre>
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
      <h2>Strict Codegen Workbench</h2>
      <p class="muted">The model must generate candidate function bodies. The verifier materializes those candidates under <code>runs/</code>, runs fixed tests and benchmarks, and accepts no fallback of any kind.</p>
      <div class="task-list">
        ${taskCatalog.map(task => `
          <article class="task-card">
            <button class="task-button ${task.id === activeTaskId ? 'active' : ''}" data-task="${escapeHtml(task.id)}">Run ${escapeHtml(task.id)}</button>
            <h3>${escapeHtml(task.title)}</h3>
            <p class="muted">${escapeHtml(task.description)}</p>
            <div class="small">${escapeHtml(task.family)} · ${escapeHtml(task.function_name)} · ${escapeHtml(task.objective_direction)} ${escapeHtml(task.objective_label)}</div>
            <div class="toolbar">
              <span class="pill">${escapeHtml(task.generation_budget)} generations</span>
              <span class="pill">${escapeHtml(task.candidate_budget)} candidates</span>
            </div>
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
  const events = (liveState.events || []).map(event => `
    <li>
      <strong>${escapeHtml(event.phase || event.event_type || 'event')}</strong>
      ${event.generation ? ` · g${escapeHtml(event.generation)}` : ''}
      ${event.candidate ? ` · ${escapeHtml(event.candidate)}` : ''}
      <br />
      <span class="small">${escapeHtml(event.timestamp || '')} · ${escapeHtml(event.message || '')}</span>
    </li>
  `).join('');
  const failure = liveState.status === 'failed'
    ? `<p class="muted">terminal=${escapeHtml(String(Boolean(liveState.terminal)))} · ${escapeHtml(liveState.error_type || 'runtime_error')} · ${escapeHtml(liveState.error || '')}</p>`
    : '';

  return `
    <section class="panel">
      <div class="eyebrow">live run</div>
      <h2>${liveState.status === 'running' ? 'Executing strict codegen loop' : 'Last live run'}</h2>
      <p class="muted">${liveState.taskId ? `task: ${liveState.taskId}` : 'full sequence'}</p>
      ${failure}
      <ul class="list">${events || '<li>No events yet.</li>'}</ul>
    </section>
  `;
}

function renderCandidate(candidate, isWinner) {
  const benchmark = candidate.metrics.benchmark_ms == null ? 'n/a' : `${candidate.metrics.benchmark_ms} ms`;
  return `
    <article class="candidate-card ${isWinner ? 'winner' : ''}">
      <div class="toolbar">
        <span class="pill ${isWinner ? 'good' : ''}">${isWinner ? 'winner' : 'candidate'}</span>
        <span class="pill">${escapeHtml(candidate.agent)}</span>
        <span class="pill">${escapeHtml(candidate.verifier_status)}</span>
        <span class="pill">${escapeHtml(candidate.proposal_model || 'n/a')}</span>
      </div>
      <h3>${escapeHtml(candidate.label)}</h3>
      <p class="muted">${escapeHtml(candidate.strategy)}</p>
      <p class="small">${escapeHtml(candidate.candidate_summary)}</p>
      <div class="metric-grid">
        ${metric('objective', candidate.metrics.objective)}
        ${metric('J', candidate.metrics.J)}
        ${metric('benchmark', benchmark)}
        ${metric('speedup', `${candidate.metrics.speedup_vs_baseline}x`)}
        ${metric('tests', `${candidate.metrics.passed_tests}/${candidate.metrics.total_tests}`)}
        ${metric('workspace', shortPath(candidate.workspace_path))}
      </div>
      <p class="small">Rationale: ${escapeHtml(candidate.rationale)}</p>
      <pre class="code-block code-scroll"><code>${escapeHtml(candidate.source_code)}</code></pre>
    </article>
  `;
}

function renderGeneration(generation) {
  return `
    <section class="panel">
      <div class="toolbar">
        <span class="pill">generation ${escapeHtml(generation.generation)}</span>
        <span class="pill ${generation.winner_accepted ? 'good' : 'warn'}">${generation.winner_accepted ? 'accepted' : 'rejected'}</span>
        <span class="pill ${generation.wrote_memory ? 'good' : 'warn'}">${generation.wrote_memory ? 'memory write-back' : 'no write-back'}</span>
      </div>
      <h3>${escapeHtml(generation.winner.label)}</h3>
      <p class="muted">${escapeHtml(generation.winner.candidate_summary)}</p>
      <div class="metric-grid">
        ${metric('winner objective', generation.winner.metrics.objective)}
        ${metric('winner J', generation.winner.metrics.J)}
        ${metric('delta_J', generation.delta_J)}
      </div>
      <h3 style="margin-top:16px;">Retrieved Memory Fragments</h3>
      <ul class="list">
        ${generation.retrieved_memories.map(memory => `
          <li>
            <strong>${escapeHtml(memory.experience_id)}</strong>: ${escapeHtml(memory.prompt_fragment || '')}
            <br />
            <span class="small">${escapeHtml(memory.strategy_hypothesis || '')}</span>
          </li>
        `).join('') || '<li>No retrieved memory.</li>'}
      </ul>
      <div class="candidate-grid" style="margin-top:16px;">
        ${generation.candidates.map(candidate => renderCandidate(candidate, candidate.candidate_id === generation.winner.candidate_id)).join('')}
      </div>
    </section>
  `;
}

function renderRun(run) {
  const manifest = run.handoff_bundle ? run.handoff_bundle.manifest : null;
  const artifactItems = manifest ? `
    <ul class="list artifact-list">
      <li><strong>manifest</strong>: ${escapeHtml(shortPath(run.handoff_bundle.manifest_path))}</li>
      <li><strong>payload</strong>: ${escapeHtml(shortPath(manifest.artifact_paths.payload))}</li>
      <li><strong>trace</strong>: ${escapeHtml(shortPath(manifest.artifact_paths.trace))}</li>
      <li><strong>llm trace</strong>: ${escapeHtml(shortPath(manifest.artifact_paths.llm_trace_jsonl))}</li>
      <li><strong>memory markdown</strong>: ${escapeHtml(shortPath(manifest.artifact_paths.memory_markdown))}</li>
    </ul>
  ` : '<p class="muted">No artifact manifest yet.</p>';

  return `
    <section class="panel stack">
      <div>
        <div class="eyebrow">${escapeHtml(run.task.id)}</div>
        <h2>${escapeHtml(run.task.title)}</h2>
        <p class="muted">${escapeHtml(run.task.description)}</p>
        <div class="toolbar">
          <span class="pill">${escapeHtml(run.run_mode)}</span>
          <span class="pill">${escapeHtml(run.active_model)}</span>
          <span class="pill">${escapeHtml(run.llm_traces.length)} llm traces</span>
        </div>
      </div>
      <div class="run-grid">
        <article class="code-card">
          <h3>Baseline</h3>
          <p class="muted">${escapeHtml(run.baseline.candidate_summary)}</p>
          <div class="metric-grid">
            ${metric('objective', run.baseline.metrics.objective)}
            ${metric('benchmark', run.baseline.metrics.benchmark_ms == null ? 'n/a' : `${run.baseline.metrics.benchmark_ms} ms`)}
            ${metric('J', run.baseline.metrics.J)}
          </div>
          <pre class="code-block code-scroll"><code>${escapeHtml(run.baseline.source_code)}</code></pre>
        </article>
        <article class="code-card">
          <h3>Winner</h3>
          <p class="muted">${escapeHtml(run.selection_reason)}</p>
          <div class="metric-grid">
            ${metric('winner candidate', run.winner.agent)}
            ${metric('objective', run.winner.metrics.objective)}
            ${metric('delta_J', run.delta_J)}
          </div>
          <p class="small">${escapeHtml(run.winner.candidate_summary)}</p>
          <pre class="code-block code-scroll"><code>${escapeHtml(run.winner.source_code)}</code></pre>
        </article>
      </div>
      <section class="panel">
        <div class="eyebrow">artifacts</div>
        <h3>Handoff Bundle</h3>
        ${artifactItems}
      </section>
      <section class="panel">
        <div class="eyebrow">memory</div>
        <h3>Prompt-Ready Strategy Ledger</h3>
        <pre class="code-block memory-ledger"><code>${escapeHtml(run.memory_markdown)}</code></pre>
      </section>
      ${run.generations.map(renderGeneration).join('')}
    </section>
  `;
}

function renderApp(data, activeTaskId, liveState, terminalError) {
  return `
    <section class="panel">
      <div class="header-grid">
        <div>
          <div class="eyebrow">llm-required codegen</div>
          <h1>Deterministic verification around direct code generation.</h1>
          <p class="muted">The configured model proposes candidate function bodies. The verifier materializes each candidate in an ignored workspace, runs fixed tests and benchmarks, and fails the run immediately on config, runtime, or LLM errors.</p>
          <div class="toolbar">
            <button class="action-button primary" id="run-sequence">Run full sequence live</button>
            <span class="small">generated at ${escapeHtml(data.summary.generated_at)}</span>
          </div>
        </div>
        <div class="summary-card">
          <div class="small">engine</div>
          <ul class="list">
            <li><strong>mode</strong>: ${escapeHtml(data.summary.proposal_engine.mode)}</li>
            <li><strong>model</strong>: ${escapeHtml(data.summary.proposal_engine.active_model)}</li>
            <li><strong>temperature</strong>: ${escapeHtml(data.summary.proposal_engine.temperature)}</li>
            <li><strong>max tokens</strong>: ${escapeHtml(data.summary.proposal_engine.max_tokens)}</li>
            <li><strong>timeout</strong>: ${escapeHtml(data.summary.proposal_engine.timeout_s)} s</li>
          </ul>
        </div>
      </div>
    </section>
    ${renderErrorPanel(terminalError)}
    ${renderSummary(data.summary, data.audit)}
    ${renderFormulas(data.formulas)}
    ${renderLiveStatus(liveState)}
    ${renderTaskCatalog(data.task_catalog, activeTaskId)}
    ${data.runs.map(renderRun).join('')}
  `;
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  const text = await response.text();
  let payload = {};
  try {
    payload = text ? JSON.parse(text) : {};
  } catch (error) {
    throw new Error(`request failed with status ${response.status}: ${text}`);
  }
  if (!response.ok) {
    const message = payload.error_type
      ? `${payload.error_type}: ${payload.error || ''}`.trim()
      : `request failed with status ${response.status}`;
    const failure = new Error(message);
    failure.payload = payload;
    throw failure;
  }
  return payload;
}

async function loadTask(taskId) {
  const query = taskId ? `?task_id=${encodeURIComponent(taskId)}` : '';
  return fetchJson(`/api/latest-run${query}`);
}

async function startJob(taskId) {
  const suffix = taskId ? `?task_id=${encodeURIComponent(taskId)}` : '';
  const url = taskId ? `/api/run-task${suffix}` : '/api/run-sequence';
  return fetchJson(url, { method: 'POST' });
}

async function main() {
  const root = document.getElementById('app');
  let liveState = null;
  let terminalError = null;
  let data;

  try {
    data = await loadTask();
  } catch (error) {
    terminalError = error.payload || { terminal: true, error_type: 'runtime_error', error: String(error), model: null };
    data = {
      summary: {
        generated_at: 'n/a',
        run_mode: 'llm-required',
        active_model: 'n/a',
        num_tasks: 0,
        total_generations: 0,
        initial_memory_count: 0,
        memory_size_after_run: 0,
        write_backs: 0,
        source_repo: 'n/a',
        git_commit: 'n/a',
        upstream_target: 'n/a',
        flywheel: [],
        proposal_engine: { mode: 'llm-required', active_model: 'n/a', temperature: 'n/a', max_tokens: 'n/a', timeout_s: 'n/a', api_base: 'n/a' },
      },
      formulas: { J: '', objective: '', delta_J: '' },
      audit: { workspace_root: 'n/a', session_id: 'n/a' },
      task_catalog: await fetchJson('/api/tasks').then(result => result.tasks).catch(() => []),
      runs: [],
    };
  }

  let activeTaskId = data.runs[0]?.task?.id || data.task_catalog[0]?.id || '';

  async function repaint(nextData) {
    data = nextData;
    root.innerHTML = renderApp(data, activeTaskId, liveState, terminalError);

    for (const button of root.querySelectorAll('[data-task]')) {
      button.addEventListener('click', async () => {
        activeTaskId = button.getAttribute('data-task');
        liveState = { status: 'running', taskId: activeTaskId, events: [] };
        terminalError = null;
        root.innerHTML = renderApp(data, activeTaskId, liveState, terminalError);
        const jobStart = await startJob(activeTaskId);
        let job = await fetchJson(`/api/job?job_id=${encodeURIComponent(jobStart.job_id)}`);
        while (job.status === 'running') {
          liveState = { status: job.status, taskId: activeTaskId, events: job.events };
          root.innerHTML = renderApp(data, activeTaskId, liveState, terminalError);
          await new Promise(resolve => setTimeout(resolve, 220));
          job = await fetchJson(`/api/job?job_id=${encodeURIComponent(jobStart.job_id)}`);
        }
        if (job.status === 'failed') {
          liveState = { ...job, taskId: activeTaskId };
          terminalError = job;
          root.innerHTML = renderApp(data, activeTaskId, liveState, terminalError);
          return;
        }
        liveState = { status: job.status, taskId: activeTaskId, events: job.events };
        terminalError = null;
        await repaint(job.payload);
      });
    }

    const sequenceButton = root.querySelector('#run-sequence');
    if (sequenceButton) {
      sequenceButton.addEventListener('click', async () => {
        activeTaskId = '';
        liveState = { status: 'running', taskId: null, events: [] };
        terminalError = null;
        root.innerHTML = renderApp(data, activeTaskId, liveState, terminalError);
        const jobStart = await startJob(null);
        let job = await fetchJson(`/api/job?job_id=${encodeURIComponent(jobStart.job_id)}`);
        while (job.status === 'running') {
          liveState = { status: job.status, taskId: null, events: job.events };
          root.innerHTML = renderApp(data, activeTaskId, liveState, terminalError);
          await new Promise(resolve => setTimeout(resolve, 220));
          job = await fetchJson(`/api/job?job_id=${encodeURIComponent(jobStart.job_id)}`);
        }
        if (job.status === 'failed') {
          liveState = { ...job, taskId: null };
          terminalError = job;
          root.innerHTML = renderApp(data, activeTaskId, liveState, terminalError);
          return;
        }
        liveState = { status: job.status, taskId: null, events: job.events };
        terminalError = null;
        await repaint(job.payload);
      });
    }
  }

  await repaint(data);
}

main().catch(error => {
  document.getElementById('app').innerHTML = `
    <section class="loading-panel">
      <p class="eyebrow">terminal failure</p>
      <h1>Failed to load the codegen workbench.</h1>
      <p class="muted">${escapeHtml(error)}</p>
    </section>
  `;
});
