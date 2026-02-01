@extends('panel.layout.settings')
@section('title', __(\App\Domains\Engine\Enums\EngineEnum::PEBBLELY->label() . ' Settings'))
@section('titlebar_actions', '')

@section('settings')
    <form
        action="{{ route('dashboard.admin.settings.pebblely.update') }}"
        method="post"
    >
        @csrf
        <x-card
            class="mb-2 max-md:text-center"
            szie="lg"
        >
            <div
                class="form-control mb-3 border-none p-0 [&_.select2-selection--multiple]:!rounded-[--tblr-border-radius] [&_.select2-selection--multiple]:!border-[--tblr-border-color] [&_.select2-selection--multiple]:!p-[1em_1.23em]">
                <label class="form-label">{{ __(':label API Key', ['label' => \App\Domains\Engine\Enums\EngineEnum::PEBBLELY->label()]) }}
                    <x-alert class="mt-2">
                        <x-button
                            variant="link"
                            href="https://pebblely.com/docs/"
                            target="_blank"
                        >
                            {{ __('Get an API key') }}
                        </x-button>
                    </x-alert>
                </label>
                <input
                    class="form-control"
                    id="pebblely_key"
                    type="text"
                    name="pebblely_key"
                    value="{{ $app_is_demo ? '**********************' : $setting->pebblely_key }}"
                >
                <x-alert
                    class="mt-2"
                    variant="lg"
                >
                    <p>
                        {{ __('Please ensure that your :label api key is fully functional and billing defined on your :label account.', ['label' => \App\Domains\Engine\Enums\EngineEnum::PEBBLELY->label()]) }}
                    </p>
                </x-alert>
            </div>

        </x-card>
        <button
            class="btn btn-primary w-full"
            type="submit"
        >
            {{ __('Save') }}
        </button>
    </form>
@endsection

@push('script')
@endpush
