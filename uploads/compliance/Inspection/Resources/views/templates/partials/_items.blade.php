@if($template->items->count() === 0)
  <div class="text-muted mb-3">{{ __('inspection::templates.no_items') }}</div>
@else
  <ul class="list-group mb-3">
    @foreach($template->items as $item)
      <li class="list-group-item d-flex justify-content-between align-items-center">
        <div>
          <div><strong>{{ $item->item_name }}</strong></div>
          @if($item->standard)<div class="text-muted">{{ $item->standard }}</div>@endif
        </div>
        <form method="POST" action="{{ route('inspection-templates.items.destroy', [$template->id, $item->id]) }}">
          @csrf
          @method('DELETE')
          <button class="btn btn-sm btn-danger">{{ __('inspection::buttons.remove') }}</button>
        </form>
      </li>
    @endforeach
  </ul>
@endif

<form method="POST" action="{{ route('inspection-templates.items.store', $template->id) }}">
  @csrf
  <div class="row g-2">
    <div class="col-md-6">
      <input type="text" name="item_name" class="form-control" placeholder="{{ __('inspection::templates.fields.item_name') }}" required>
    </div>
    <div class="col-md-4">
      <input type="text" name="standard" class="form-control" placeholder="{{ __('inspection::templates.fields.standard') }}">
    </div>
    <div class="col-md-2 d-grid">
      <button class="btn btn-primary">{{ __('inspection::templates.add_item') }}</button>
    </div>
  </div>
</form>
