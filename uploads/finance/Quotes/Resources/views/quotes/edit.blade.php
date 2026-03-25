@extends('layouts.app')

@section('content')
<div class="container py-4">
  <h3>Edit Estimate / Quote</h3>
  <form method="post" action="{{ route('quotes.update', $quote->id) }}">
    @csrf @method("put")
    <div class="row">
      <div class="col-md-3 mb-3">
        <label class="form-label">Client ID (optional)</label>
        <input type="number" name="client_id" class="form-control" />
      </div>
      <div class="col-md-4 mb-3">
        <label class="form-label">Currency</label>
        <input type="text" name="currency" id="quote-currency" class="form-control" value="{{ $currency }}" />
      </div>
      <div class="col-md-4 mb-3">
        <label class="form-label">Valid Until</label>
        <input type="date" name="valid_until" class="form-control" />
      </div>
      <div class="col-md-3 mb-3">
        <label class="form-label">Price List (optional)</label>
        <select name="price_list_id" id="price-list" class="form-select"></select>
      </div>
    </div>
    <div class="mb-3">
      <label class="form-label">Notes</label>
      <textarea name="notes" class="form-control" rows="2"></textarea>
    </div>

    <h5>Items</h5>
    <div class="mb-2"><input id="item-search" class="form-control" placeholder="Search items (from Items module or Price Lists)" /></div>
    <div id="items"></div>
    <button type="button" id="add-item" class="btn btn-sm btn-outline-primary mb-3">Add Item</button>

    <button class="btn btn-success">Create</button>
  </form>
</div>

<script>
(function(){
  const wrap = document.getElementById('items');
  const add = document.getElementById('add-item');
  function row(idx){
    return `<div class="row g-2 align-items-end mb-2">
      <div class="col-md-6">
        <label class="form-label">Description</label>
        <input name="items[${idx}][description]" class="form-control" required>
      </div>
      <div class="col-md-2">
        <label class="form-label">Qty</label>
        <input name="items[${idx}][qty]" type="number" step="0.01" class="form-control" value="1" required>
      </div>
      <div class="col-md-2">
        <label class="form-label">Unit Price</label>
        <input name="items[${idx}][unit_price]" type="number" step="0.01" class="form-control" value="0" required>
      </div>
      <div class="col-md-2">
        <label class="form-label">Tax Rate</label>
        <input name="items[${idx}][tax_rate]" type="number" step="0.0001" class="form-control" value="0">
      </div>
      <div class="col-md-3 mb-3">
        <label class="form-label">Price List (optional)</label>
        <select name="price_list_id" id="price-list" class="form-select"></select>
      </div>
    </div>`;
  }
  let i = 0;
  function addRow(){ wrap.insertAdjacentHTML('beforeend', row(i++)); }
  add.addEventListener('click', addRow);
  addRow();
})();

  const search = document.getElementById('item-search');
  const currencyEl = document.getElementById('quote-currency');
  const priceListEl = document.getElementById('price-list');

  async function loadPriceLists(){
    const r = await fetch('/api/quotes/pricelists?currency='+encodeURIComponent(currencyEl.value||''));
    const lists = await r.json();
    priceListEl.innerHTML = '<option value="">-- none --</option>' + lists.map(l=>`<option value="${l.id}">${l.name} (${l.currency})</option>`).join('');
  }
  currencyEl.addEventListener('change', loadPriceLists);
  loadPriceLists();
  let timer = null;
  const dropdown = document.createElement('div');
  dropdown.className = 'list-group position-absolute shadow';
  dropdown.style.zIndex = 1000;
  search.parentNode.style.position = 'relative';
  search.parentNode.appendChild(dropdown);

  search.addEventListener('input', () => {
    const q = search.value.trim();
    clearTimeout(timer);
    if (q.length < 1) { dropdown.innerHTML=''; return; }
    timer = setTimeout(async () => {
      const r = await fetch('/api/quotes/items/search?term='+encodeURIComponent(q)+'&currency='+encodeURIComponent(currencyEl.value||'')+'&price_list_id='+(priceListEl.value||''));
      const items = await r.json();
      dropdown.innerHTML = '';
      items.forEach((it) => {
        const a = document.createElement('a');
        a.href = '#';
        a.className = 'list-group-item list-group-item-action';
        a.textContent = `${it.label} — ${Number(it.unit_price).toFixed(2)}`;
        a.addEventListener('click', (e) => {
          e.preventDefault();
          addRow();
          const idx = i-1;
          document.querySelector(`[name="items[${idx}][description]"]`).value = it.label;
          document.querySelector(`[name="items[${idx}][unit_price]"]`).value = it.unit_price;
          document.querySelector(`[name="items[${idx}][tax_rate]"]`).value = it.tax_rate || 0;
          dropdown.innerHTML='';
          search.value='';
        });
        dropdown.appendChild(a);
      });
    }, 250);
  });
</script>

@endsection
