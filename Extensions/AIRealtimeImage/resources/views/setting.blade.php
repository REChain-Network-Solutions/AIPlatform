@extends('panel.layout.settings')
@section('title', __('Together Settings'))
@section('titlebar_actions', '')
@section('titlebar_subtitle', __('This API key is used for these features: AI Realtime Image'))

@section('additional_css')
@endsection

@section('settings')
    <form
        method="post"
        enctype="multipart/form-data"
        action="{{ route('dashboard.admin.settings.together.update') }}"
    >
        @csrf
        <x-card
            class="mb-2 max-md:text-center"
            szie="lg"
        >

            <div
                class="form-control mb-3 border-none p-0 [&_.select2-selection--multiple]:!rounded-[--tblr-border-radius] [&_.select2-selection--multiple]:!border-[--tblr-border-color] [&_.select2-selection--multiple]:!p-[1em_1.23em]">
                <label class="form-label">{{ 'Together API Key' }}
                </label>
                <input
                    class="form-control"
                    id="together_api_key"
                    type="text"
                    name="together_api_key"
                    value="{{ $app_is_demo ? '*********************' : setting('together_api_key') }}"
                    required
                >
                <x-alert
                    class="mt-2"
                    variant="lg"
                >
                    <p>
                        {{ __('Please ensure that your Together api key is fully functional and billing defined on your Together account.') }}
                        <x-button
                            variant="link"
                            href="https://api.together.xyz/signin?redirectUrl=/settings/api-keys"
                            target="_blank"
                        >
                            {{ __('Get an API key') }}
                        </x-button>
                    </p>
                </x-alert>
            </div>

        </x-card>
        <button class="btn btn-primary w-full">
            {{ __('Save') }}
        </button>
    </form>
@endsection

@push('script')
@endpush
