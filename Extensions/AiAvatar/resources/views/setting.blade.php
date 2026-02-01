@extends('panel.layout.settings')
@section('title', __('Synthesia Settings'))
@section('titlebar_actions', '')
@section('titlebar_subtitle', __('This API key is used for these features: AI Avatar'))

@section('additional_css')
@endsection

@section('settings')
    <form
        method="post"
        enctype="multipart/form-data"
        action="{{ route('dashboard.admin.settings.synthesia.update') }}"
    >
        @csrf
        <x-card
            class="mb-2 max-md:text-center"
            szie="lg"
        >

            <div
                class="form-control mb-3 border-none p-0 [&_.select2-selection--multiple]:!rounded-[--tblr-border-radius] [&_.select2-selection--multiple]:!border-[--tblr-border-color] [&_.select2-selection--multiple]:!p-[1em_1.23em]">
                <label class="form-label">{{ 'Synthesia API Key' }}
                    <x-alert class="mt-2">
                        <x-button
                            variant="link"
                            href="https://www.synthesia.io/?via=magicai"
                            target="_blank"
                        >
                            {{ __('Get an API key') }}
                        </x-button>
                    </x-alert>
                </label>
                <input
                    class="form-control"
                    id="synthesia_secret_key"
                    type="text"
                    name="synthesia_secret_key"
                    value="{{ $app_is_demo ? '*********************' : $setting->synthesia_secret_key }}"
                    required
                >
                <x-alert
                    class="mt-2"
                    variant="lg"
                >
                    <p>
                        {{ __('Please ensure that your Synthesia api key is fully functional and billing defined on your Synthesia account.') }}
                    </p>
                </x-alert>
            </div>

        </x-card>
        <button
            class="btn btn-primary w-full"
        >
            {{ __('Save') }}
        </button>
    </form>
@endsection

@push('script')
@endpush
