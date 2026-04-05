@if ($fetchSetting->purchase_code)
    <span class="blur-code purchase-code f-12">{{ $fetchSetting->purchase_code }}</span>
    <div class="show-hide-purchase-code d-inline" data-toggle="tooltip"
        data-original-title="{{ __('messages.showHidePurchaseCode') }}">
        <i class="icon far fa-eye-slash cursor-pointer"></i>
    </div>
    <div class="verify-module d-inline" data-toggle="tooltip" data-original-title="{{ __('messages.changePurchaseCode') }}"
        data-module="{{ strtolower($module) }}">
        <i class="ml-1 icon far fa-edit cursor-pointer"></i>
    </div>
@else
    <a href="javascript:;" class="verify-module f-w-500" data-module="{{ strtolower($module) }}">@lang('app.verifyEnvato')</a>
@endif
