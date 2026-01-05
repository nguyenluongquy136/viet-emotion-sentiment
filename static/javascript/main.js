// Dark mode removed on index page – always light theme

// App logic
document.addEventListener('DOMContentLoaded', async () => {
  // Refs
  const form = document.getElementById('sentimentForm');
  const modelSelect = document.getElementById('model');
  const textArea = document.getElementById('text');
  const getAnalyzeButton = () => document.querySelector('#sentimentForm button[type="submit"], #sentimentForm .btn.btn-primary');
  const singleResults = document.getElementById('single-results');
  const batchResults = document.getElementById('batch-results');
  const fileResults = document.getElementById('file-results');
  const statusMessage = document.getElementById('status-message');
  const batchTextArea = document.getElementById('texts-batch');
  const batchAnalyzeBtn = document.getElementById('analyze-batch');
  const csvFileInput = document.getElementById('csv-file');
  const csvColumnInput = document.getElementById('csv-column');
  const csvDelimiterInput = document.getElementById('csv-delimiter');
  const analyzeFileBtn = document.getElementById('analyze-file');
  const MODELS = ['lstm', 'gru', 'transformers'];

  const visible = (el, v) => { if (el) el.style.display = v ? '' : 'none'; };

  // Hide result blocks initially
  visible(singleResults, false);
  visible(batchResults, false);
  visible(fileResults, false);

  // Utils
  function translateLabel(label) {
    const map = { pos: 'Tích cực', neg: 'Tiêu cực', neu: 'Trung tính', positive: 'Tích cực', negative: 'Tiêu cực', neutral: 'Trung tính' };
    return map[String(label || '').toLowerCase()] || label;
  }
  function getEmotionClass(label) {
    const lower = String(label || '').toLowerCase();
    if (['tích cực','positive','vui','hạnh phúc','tốt','pos'].some(x => lower.includes(x))) return 'bg-success';
    if (['tiêu cực','negative','buồn','xấu','tệ','neg'].some(x => lower.includes(x))) return 'bg-danger';
    if (['trung tính','neutral','neu'].some(x => lower.includes(x))) return 'bg-warning';
    return 'bg-secondary';
  }
  const bar = (pct, width=120) => `
    <div class="d-flex align-items-center">
      <div class="confidence-bar rounded-pill me-2" style="width: ${width}px;">
        <div class="confidence-fill rounded-pill" style="width: ${pct}%"></div>
      </div>
      <span class="confidence-text">${Number(pct).toFixed(2)}%</span>
    </div>`;
  const clip = (t) => `<div class="cell-clip" title="${t}">${t}</div>`;

  async function postJson(url, body) {
    const res = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  }
  async function postForm(url, fd) {
    const res = await fetch(url, { method: 'POST', body: fd });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  }

  // Load models
  try {
    const res = await fetch('/api/models');
    const data = await res.json();
    if (data.models && Array.isArray(data.models)) {
      const opts = data.models.map(m => `<option value="${m}">${m.toUpperCase()}</option>`).join('') + `<option value="all">Tất cả</option>`;
      modelSelect.innerHTML = opts;
    }
  } catch (e) {
    console.error(e);
    if (statusMessage) {
      statusMessage.textContent = 'Đã xảy ra lỗi. Vui lòng thử lại sau.';
      statusMessage.style.backgroundColor = '#dc3545';
      statusMessage.style.display = 'block';
      setTimeout(() => { statusMessage.style.display = 'none'; }, 5000);
    }
  }

  // Toggle expand normalized text
  document.addEventListener('click', (e) => {
    const t = e.target;
    if (t && t.classList && t.classList.contains('toggle-normalized')) {
      const card = t.closest('.card');
      const span = card ? card.querySelector('.normalized-text') : null;
      if (span) {
        span.classList.toggle('expanded');
        t.textContent = span.classList.contains('expanded') ? 'Thu gọn' : 'Xem thêm';
      }
    }
  });

  // Renderers for single text
  function fillSingleSummary(result, modelName) {
    const card = document.getElementById('single-summary');
    if (!card) return;
    const lbl = translateLabel(result.label);
    const pct = (() => {
      if (!result.probs) return 0;
      const low = String(result.label || '').toLowerCase();
      const v = result.probs[low] ?? result.probs[result.label] ?? Math.max(...Object.values(result.probs));
      return Math.max(0, Math.min(100, Number((v || 0) * 100)));
    })();
    const modelEl = document.getElementById('single-model-name');
    const badge = document.getElementById('single-label-badge');
    const fill = document.getElementById('single-confidence-fill');
    const txt = document.getElementById('single-confidence-text');
    if (modelEl) modelEl.textContent = String(modelName || '').toUpperCase();
    if (badge) { badge.textContent = lbl; badge.className = `emotion-badge ${getEmotionClass(lbl)}`; }
    if (fill) fill.style.width = `${pct.toFixed(2)}%`;
    if (txt) txt.textContent = `${pct.toFixed(2)}%`;
    visible(card, true);
  }

  function fillSingleDetail(result) {
    const card = document.getElementById('single-detail');
    if (!card) return;
    const normEl = document.getElementById('single-norm');
    const toggleBtn = document.getElementById('single-toggle');
    const norm = result.expanded || 'Không có';
    if (normEl) { normEl.textContent = norm; normEl.classList.remove('expanded'); }
    if (toggleBtn) { visible(toggleBtn, norm.length > 120); toggleBtn.textContent = 'Xem thêm'; }
    const tbody = document.getElementById('single-detail-body');
    if (tbody) {
      const rows = Object.entries(result.probs || {})
        .sort(([,a],[,b]) => b - a)
        .map(([label, prob]) => {
          const pct = (Number(prob) * 100).toFixed(2);
          const tl = translateLabel(label);
          return `<tr><td><span class="emotion-badge ${getEmotionClass(tl)}">${tl}</span></td><td>${bar(pct)}</td></tr>`;
        }).join('');
      tbody.innerHTML = rows;
    }
    visible(card, true);
  }

  function renderAllModelsSingle(results, targetEl, showAll) {
    const norm = results[0]?.expanded || 'Không có';
    const needToggle = norm.length > 120;
    let html = `
      <h3 class="h2 fw-bold mb-4">Kết quả phân tích - Tất cả mô hình</h3>
      <div class="card p-4 mb-3">
        <div class="row"><div class="col-12">
          <strong>Văn bản sau chuẩn hóa:</strong>
          <span class="text-muted ms-2 normalized-text">${norm}</span>
          ${needToggle ? '<button type="button" class="btn btn-link btn-sm p-0 ms-2 toggle-normalized">Xem thêm</button>' : ''}
        </div></div>
      </div>`;
    results.forEach(r => {
      if (r.error) {
        html += `<div class="card p-4 mb-3"><h4 class="h6 fw-bold mb-3">${r.model}</h4><div class="alert alert-danger mb-0">Lỗi: ${r.error}</div></div>`;
        return;
      }
      if (showAll && r.probs) {
        const rows = Object.entries(r.probs).sort(([,a],[,b]) => b - a).map(([lb,pr]) => {
          const tl = translateLabel(lb); const pct = (Number(pr)*100).toFixed(2);
          return `<tr><td><span class="emotion-badge ${getEmotionClass(tl)}">${tl}</span></td><td>${bar(pct)}</td></tr>`;
        }).join('');
        html += `
          <div class="card p-4 mb-3">
            <h4 class="h6 fw-bold mb-3">${r.model}</h4>
            <div class="results-scroll"><div class="table-responsive">
              <table class="table table-hover">
                <thead class="table-light"><tr><th>Nhãn cảm xúc</th><th>Độ tin cậy</th></tr></thead>
                <tbody>${rows}</tbody>
              </table>
            </div></div>
          </div>`;
      } else {
        const tl = translateLabel(r.label); const cls = getEmotionClass(tl);
        const pct = r.probs ? (Math.max(...Object.values(r.probs))*100).toFixed(2) : '0.00';
        html += `
          <div class="card p-4 mb-3">
            <h4 class="h6 fw-bold mb-3">${r.model}</h4>
            <div class="d-flex align-items-center justify-content-between mb-2">
              <span class="emotion-badge ${cls}">${tl}</span>
              ${bar(pct)}
            </div>
          </div>`;
      }
    });
    targetEl.innerHTML = html; visible(targetEl, true); targetEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  function renderBatchTable(title, texts, labels, probs, showAll) {
    const head = `<thead class="table-light"><tr><th>#</th><th>Văn bản sau chuẩn hóa</th>${showAll ? '<th>Các nhãn</th>' : '<th>Nhãn</th><th>Độ tin cậy</th>'}</tr></thead>`;
    let body = '';
    for (let i=0;i<texts.length;i++){
      const rp = Array.isArray(probs)? probs[i] : undefined;
      const lbl = translateLabel(labels[i] ?? '');
      const cls = getEmotionClass(lbl);
      if (showAll && rp && typeof rp === 'object'){
        const list = Object.entries(rp).sort(([,a],[,b])=>b-a).map(([lb,p])=>{
          const t=translateLabel(lb); const c=getEmotionClass(t); const pct=(Number(p)*100).toFixed(2);
          return `<div class="d-flex align-items-center mb-1"><span class="emotion-badge ${c} me-2" style="min-width:90px; text-align:center;">${t}</span><div class="confidence-bar rounded-pill me-2" style="width:120px;"><div class="confidence-fill rounded-pill" style="width:${pct}%"></div></div><span class="confidence-text">${pct}%</span></div>`;
        }).join('');
        body += `<tr><td>${i+1}</td><td>${clip(texts[i])}</td><td>${list}</td></tr>`;
      } else {
        let pct='N/A';
        if (rp && typeof rp==='object'){
          const lower=String(labels[i]||'').toLowerCase();
          const val=rp[lower] ?? rp[labels[i]] ?? Math.max(...Object.values(rp));
          pct=((Number(val)||0)*100).toFixed(2)+'%';
        }
        body += `<tr><td>${i+1}</td><td>${clip(texts[i])}</td><td><span class="emotion-badge ${cls}">${lbl}</span></td><td>${pct}</td></tr>`;
      }
    }
    return `<div class="card p-4 mb-3"><h4 class="h6 fw-bold mb-3">${title}</h4><div class="results-scroll"><div class="table-responsive"><table class="table table-hover">${head}<tbody>${body}</tbody></table></div></div></div>`;
  }

  // Submit: single text
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const text = (textArea?.value || '').trim(); if (!text) return alert('Vui lòng nhập văn bản!');
    const model = modelSelect.value;
    const analyzeButton = getAnalyzeButton();
    if (analyzeButton && analyzeButton.disabled) return; // prevent double submit
    if (analyzeButton) {
      analyzeButton.disabled = true;
      analyzeButton.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Đang phân tích...';
    }
    try {
      // Prepare result areas
      visible(document.getElementById('single-summary'), false);
      visible(document.getElementById('single-detail'), false);
      const showAll = document.getElementById('highest-confidence')?.checked === true;
      if (model === 'all'){
        // Use batchResults as the target container for rendering ALL models
        visible(singleResults, false);
        visible(batchResults, true);
        const all = await Promise.all(MODELS.map(async m => {
          try { const r = await postJson(`/api/predict?model_name=${m}`, { text, return_probs: true }); return { model: m.toUpperCase(), ...r }; }
          catch(err){ return { model: m.toUpperCase(), error: String(err.message||err) }; }
        }));
        renderAllModelsSingle(all, batchResults, showAll);
      } else {
        // Hide ALL-models container when focusing on a single model
        visible(batchResults, false);
        visible(singleResults, true);
        const r = await postJson(`/api/predict?model_name=${model}`, { text, return_probs: true });
        if (showAll) fillSingleDetail(r); else fillSingleSummary(r, model);
      }
    } catch(err){ console.error(err); alert('Có lỗi xảy ra khi phân tích.'); }
    finally {
      const btn = getAnalyzeButton();
      if (btn) { btn.disabled = false; btn.innerHTML = 'Phân tích'; }
    }
  });

  // Analyze: many texts
  if (batchAnalyzeBtn) batchAnalyzeBtn.addEventListener('click', async () => {
    const raw = (batchTextArea?.value || '').trim(); if (!raw) return alert('Vui lòng nhập ít nhất một câu.');
    const texts = raw.split(/\r?\n/).map(s=>s.trim()).filter(Boolean); if (!texts.length) return alert('Không có câu hợp lệ.');
    const model = modelSelect.value; batchAnalyzeBtn.disabled = true; batchAnalyzeBtn.textContent = 'Đang phân tích...'; visible(batchResults,false);
    try {
      const showAll = document.getElementById('highest-confidence')?.checked === true;
      if (model === 'all'){
        const all = await Promise.all(MODELS.map(async m => { try { const r = await postJson(`/api/predict_batch?model_name=${m}`, { texts, return_probs: true }); return { model: m.toUpperCase(), ...r }; } catch(err){ return { model: m.toUpperCase(), error: String(err.message||err) }; } }));
        let html = `<h3 class="h2 fw-bold mb-4">Kết quả phân tích - Nhiều câu (Tất cả mô hình)</h3>`;
        all.forEach(block => { html += block.error ? `<div class="card p-4 mb-3"><h4 class="h6 fw-bold mb-3">${block.model}</h4><div class="alert alert-danger mb-0">Lỗi: ${block.error}</div></div>` : renderBatchTable(block.model, block.expanded||[], block.labels||[], block.probs||[], showAll); });
        batchResults.innerHTML = html; visible(batchResults,true); batchResults.scrollIntoView({behavior:'smooth', block:'nearest'});
      } else {
        const r = await postJson(`/api/predict_batch?model_name=${model}`, { texts, return_probs: true });
        const html = `<h3 class="h2 fw-bold mb-4">Kết quả phân tích nhiều câu - ${model.toUpperCase()}</h3>` + renderBatchTable(model.toUpperCase(), r.expanded||[], r.labels||[], r.probs||[], showAll);
        batchResults.innerHTML = html; visible(batchResults,true); batchResults.scrollIntoView({behavior:'smooth', block:'nearest'});
      }
    } catch(err){ console.error(err); alert('Có lỗi xảy ra khi phân tích nhiều câu.'); }
    finally { batchAnalyzeBtn.disabled = false; batchAnalyzeBtn.textContent = 'Phân tích nhiều câu'; }
  });

  // Analyze: file
  if (analyzeFileBtn) analyzeFileBtn.addEventListener('click', async () => {
    const f = csvFileInput?.files?.[0]; if (!f) return alert('Vui lòng chọn tệp CSV/TXT.');
    const column = (csvColumnInput?.value || 'text').trim();
    const delimiter = (csvDelimiterInput?.value || ',').slice(0,1);
    const model = modelSelect.value; analyzeFileBtn.disabled = true; analyzeFileBtn.textContent = 'Đang phân tích...'; visible(fileResults,false);
    try {
      const showAll = document.getElementById('highest-confidence')?.checked === true;
      if (model === 'all'){
        const all = await Promise.all(MODELS.map(async m => { const fd = new FormData(); fd.append('file', f); fd.append('model_name', m); fd.append('column', column); fd.append('delimiter', delimiter); try { const r = await postForm('/api/predict_file', fd); return { model: m.toUpperCase(), ...r }; } catch(err){ return { model: m.toUpperCase(), error: String(err.message||err) }; } }));
        let html = `<h3 class="h2 fw-bold mb-4">Kết quả phân tích - Tệp (Tất cả mô hình)</h3>`;
        all.forEach(block => { html += block.error ? `<div class="card p-4 mb-3"><h4 class="h6 fw-bold mb-3">${block.model}</h4><div class="alert alert-danger mb-0">Lỗi: ${block.error}</div></div>` : renderBatchTable(block.model, block.expanded||[], block.labels||[], block.probs||[], showAll); });
        fileResults.innerHTML = html; visible(fileResults,true); fileResults.scrollIntoView({behavior:'smooth', block:'nearest'});
      } else {
        const fd = new FormData(); fd.append('file', f); fd.append('model_name', model); fd.append('column', column); fd.append('delimiter', delimiter);
        const r = await postForm('/api/predict_file', fd);
        const html = `<h3 class="h2 fw-bold mb-4">Kết quả phân tích tệp - ${model.toUpperCase()}</h3>` + renderBatchTable(model.toUpperCase(), r.expanded||[], r.labels||[], r.probs||[], showAll);
        fileResults.innerHTML = html; visible(fileResults,true); fileResults.scrollIntoView({behavior:'smooth', block:'nearest'});
      }
    } catch(err){ console.error(err); alert('Có lỗi xảy ra khi phân tích tệp.'); }
    finally { analyzeFileBtn.disabled = false; analyzeFileBtn.textContent = 'Phân tích từ tệp'; }
  });
});

