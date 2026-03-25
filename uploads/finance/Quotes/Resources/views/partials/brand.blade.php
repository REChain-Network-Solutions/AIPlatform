@php
$brand = config('quotes.brand');
@endphp
<div class="d-flex align-items-center gap-3 mb-3">
  @if(!empty($brand['logo_url']))
    <img src="{{ $brand['logo_url'] }}" alt="Logo" style="height:48px">
  @endif
  <div>
    <h4 class="mb-0">{{ $brand['company_name'] ?? '' }}</h4>
    @if(!empty($brand['terms']))
      <div class="small text-muted">{{ $brand['terms'] }}</div>
    @endif
  </div>
</div>
<hr class="my-3">
