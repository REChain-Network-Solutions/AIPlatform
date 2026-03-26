<table class="table table-striped">
  <thead>
    <tr>
      <th>{{ __('inspection::templates.fields.name') }}</th>
      <th>{{ __('inspection::templates.fields.trade') }}</th>
      <th>{{ __('inspection::templates.fields.active') }}</th>
      <th class="text-end">{{ __('inspection::templates.actions') }}</th>
    </tr>
  </thead>
  <tbody>
    @foreach($templates as $tpl)
      <tr>
        <td><a href="{{ route('inspection-templates.show', $tpl->id) }}">{{ $tpl->name }}</a></td>
        <td>{{ $tpl->trade }}</td>
        <td>{{ $tpl->is_active ? __('inspection::templates.yes') : __('inspection::templates.no') }}</td>
        <td class="text-end">
          <a class="btn btn-sm btn-light" href="{{ route('inspection-templates.edit', $tpl->id) }}">{{ __('inspection::buttons.edit') }}</a>
        </td>
      </tr>
    @endforeach
  </tbody>
</table>
