@php
    $currentVersion = \Illuminate\Support\Facades\File::get($module->getPath() . '/version.txt');
    $plugin = $plugins->where('envato_id', config(strtolower($module) . '.envato_item_id'))->first();
    $latestVersion = $plugin?->version;
@endphp
@if ($plugin)
    @if (version_compare($latestVersion, $currentVersion, '>'))
        <span class="badge badge-danger" data-toggle="tooltip"
              data-original-title="@lang('app.moduleUpdateMessage', [
                            'name' => $module->getName(),
                            'version' => $latestVersion,
        ])">
            {{ $currentVersion }}
        </span>
    @else
        <span class="badge badge-success">
            {{ $currentVersion }}
        </span>
    @endif
@else
    <span class="badge badge-success">{{ $currentVersion }}</span>
@endif
