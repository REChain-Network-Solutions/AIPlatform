<div id="ai-insights" class="mb-3">
  <div class="card border-0 shadow-sm">
    <div class="card-body">
      <div class="d-flex justify-content-between align-items-center mb-2">
        <strong>AI Insights (last 30 days)</strong>
        <button type="button" class="btn btn-sm btn-outline-secondary" id="ai-insights-refresh">Refresh</button>
      </div>
      <div id="ai-insights-body"><em>Loading insights…</em></div>
    </div>
  </div>
</div>
<script>
(async function(){
  const el = document.getElementById('ai-insights-body');
  const btn = document.getElementById('ai-insights-refresh');
  async function load(){
    try{
      const resp = await fetch("{{ route('feedback.insights') }}", {headers: {'Accept':'application/json'}});
      const j = await resp.json();
      const s = j.summary || j.data?.summary || {};
      if(!s.enabled){
        el.innerHTML = '<span class="text-muted">AI disabled: ' + (s.reason || 'no key') + '</span>';
        return;
      }
      const bullets = (s.bullets || []).map(b => '<li>'+b+'</li>').join('');
      el.innerHTML = '<ul class="m-0">'+ bullets +'</ul>';
    }catch(e){
      el.innerHTML = '<span class="text-danger">Failed to load insights.</span>';
    }
  }
  btn?.addEventListener('click', load);
  load();
})();
</script>
