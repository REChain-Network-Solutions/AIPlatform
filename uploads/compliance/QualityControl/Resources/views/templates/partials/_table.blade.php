<table class="table table-striped">
  <thead>
    <tr>
      <th>{{ __('quality_control::templates.fields.name') }}</th>
      <th>{{ __('quality_control::templates.fields.trade') }}</th>
      <th>{{ __('quality_control::templates.fields.active') }}</th>
      <th class="text-end">{{ __('quality_control::templates.actions') }}</th>
    </tr>
  </thead>
  <tbody>
    @foreach($templates as $tpl)
      <tr>
        <td><a href="{{ route('inspection-templates.show', $tpl->id) }}">{{ $tpl->name }}</a></td>
        <td>{{ $tpl->trade }}</td>
        <td>{{ $tpl->is_active ? __('quality_control::templates.yes') : __('quality_control::templates.no') }}</td>
        <td class="text-end">
          <a class="btn btn-sm btn-light" href="{{ route('inspection-templates.edit', $tpl->id) }}">{{ __('quality_control::buttons.edit') }}</a>
        </td>
      </tr>
    @endforeach
  </tbody>
</table>
