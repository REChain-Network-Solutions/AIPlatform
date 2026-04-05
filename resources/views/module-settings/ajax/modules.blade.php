<x-cards.data class="w-100">
    <div class="row">
        @foreach ($modulesData as $setting)
            <div class="col-lg-3 col-md-4 col-6">
                <div class="form-group mb-4">
                    <x-forms.label :fieldId="'module-'.$setting->id"
                                   :fieldLabel="__('modules.module.'.$setting->module_name)">
                    </x-forms.label>

                    <div class="custom-control custom-switch">
                        <input type="checkbox"
                               @if ($setting->status == 'active') checked @endif
                               @if($setting->module_name == 'settings') disabled @endif
                               class="cursor-pointer custom-control-input change-module-setting"
                               id="module-{{ $setting->id }}"
                               data-setting-id="{{ $setting->id }}"
                               data-module-name="{{ $setting->module_name }}"
                        >
                        <label class="custom-control-label cursor-pointer" for="module-{{ $setting->id }}"></label>
                    </div>
                </div>
            </div>
        @endforeach
    </div>
</x-cards.data>
