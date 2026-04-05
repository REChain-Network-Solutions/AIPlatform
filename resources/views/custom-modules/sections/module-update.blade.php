@if ($fetchSetting?->purchase_code && $fetchSetting?->supported_until)
    @if (\Carbon\Carbon::parse($fetchSetting->supported_until)->isPast())
        <button type="button" class="btn btn-outline-secondary btn-sm refreshModule f-11"
                data-toggle="tooltip"
                data-original-title="This will fetch the latest support date from codecanyon. Click on this button only when you have renewed the support and the new support date is not reflecting"
                data-module-name="{{ $module }}">
            <i class="fa fa-sync-alt mr-1"></i>Refresh
        </button>
        <a target="_blank"
           href="{{ Froiden\Envato\Helpers\FroidenApp::renewSupportUrl(config(strtolower($module) . '.envato_item_id')) }}"
           class="ml-1 btn btn-primary btn-sm px-2 py-1 f-11" data-module-name="{{ $module }}">
            <i class="fa fa-shopping-cart mr-1"></i>Renew support now
        </a>
{{--    Show only for universal Bundle       --}}
    @elseif(str_contains($module, 'UniversalBundle'))
        @if (\Carbon\Carbon::parse($fetchSetting->supported_until)->diffInDays() < 60)
            <button type="button" class="btn btn-outline-secondary btn-sm refreshModule f-11"
                    data-module-name="{{ $module }}">
                <i class="fa fa-sync-alt mr-1"></i>Refresh Support
            </button>
            <a target="_blank"
               href="{{ Froiden\Envato\Helpers\FroidenApp::extendSupportUrl(config(strtolower($module) . '.envato_item_id')) }}"
               class="ml-1 btn btn-primary btn-sm px-2 py-1 f-11" data-module-name="{{ $module }}">
                <i class="fa fa-shopping-cart mr-1"></i>Extend Support
            </a>
        @endif
    @endif
@endif
@if($plugins->where('envato_id', config(strtolower($module) . '.envato_item_id'))->pluck('version')->first() > \Illuminate\Support\Facades\File::get($module->getPath() . '/version.txt'))
    <button type="button" class="ml-1 btn btn-success btn-sm update-module f-11"
            data-product-url="{{ Froiden\Envato\Helpers\FroidenApp::renewSupportUrl(config(strtolower($module) . '.envato_item_id')) }}"
            data-module-name="{{ $module }}">
        <i class="fa fa-download"></i> @lang('app.update')
    </button>
@endif
