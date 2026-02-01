@extends('panel.layout.settings')
@section('title', __('Clipdrop API Settings'))
@section('titlebar_actions', '')
@section('titlebar_subtitle', __('This API key is used for these features: AI Photo Studio, Advanced Image Editor'))

@section('settings')
    <form
        action="{{ route('dashboard.admin.settings.clipdrop.update') }}"
        enctype="multipart/form-data"
        method="POST"
    >
        <h3 class="mb-[25px] text-[20px]">{{ __('Clipdrop API Settings') }}</h3>
        <div class="row">
            <!-- TODO Serper api key -->
            <div class="col-md-12">
                <div
                    class="form-control mb-3 border-none p-0 [&_.select2-selection--multiple]:!rounded-[--tblr-border-radius] [&_.select2-selection--multiple]:!border-[--tblr-border-color] [&_.select2-selection--multiple]:!p-[1em_1.23em]">
                    <label class="form-label">{{ __('Clipdrop API Key') }}</label>
                    <input
                        class="form-control"
                        id="clipdrop_api_key"
                        type="text"
                        name="clipdrop_api_key"
                        value="{{ $app_is_demo ? '*********************' : setting('clipdrop_api_key') }}"
                        required
                    >
                    <x-alert class="mt-2">
                        <p>
                            {{ __('Please ensure that your Clipdrop api key is fully functional and billing defined on your Clipdrop account.') }}
                        </p>
                    </x-alert>
                </div>
            </div>
        </div>
        <button
            class="btn btn-primary w-full"
        >
            {{ __('Save') }}
        </button>
    </form>
@endsection
