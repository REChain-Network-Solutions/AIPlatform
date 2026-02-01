@extends('panel.layout.settings')
@section('title', __('Maintenance Settings'))
@section('titlebar_actions', '')

@section('settings')
    <form action="{{ route('dashboard.admin.settings.maintenance.index') }}" method="post" enctype="multipart/form-data">
        @csrf
        <h3 class="mb-[25px] text-[20px]">{{ __('Maintenance Settings') }}</h3>
        <div class="row">
            <div class="col-md-12">
                <div class="mb-3">
                    <label class="form-label">{{ __('Header') }}</label>
                    <input
                            class="form-control"
                            id="header"
                            type="text"
                            name="header"
                            value="{{ data_get($maintenance, 'header') ?: "We'll Be Back Soon" }}"
                    >
                </div>
            </div>
            <div class="col-md-12">
                <div class="mb-3">
                    <label class="form-label">{{ __('Paragraph') }}</label>
                    <input
                            class="form-control"
                            id="paragraph"
                            type="text"
                            name="paragraph"
                            value="{{ data_get($maintenance, 'paragraph') ?: "Our website is currently undergoing scheduled maintenance to bring you a better online experience. We’re working hard to enhance our features and improve our services." }}"
                    >
                </div>
            </div>
            <div class="col-md-12">
                <div class="mb-3">
                    <label class="form-label">{{ __('Image') }}</label>
                    <input
                            class="form-control"
                            id="image"
                            type="file"
                            name="image"
                    >
                </div>
            </div>

            <div class="col-md-12">
                <div class="mb-3">
                    <label class="form-label">
                        {{ __('Footer') }}
                        <x-info-tooltip text="{{ __('You can also use html code for footer input.') }}" />
                    </label>
                    <input
                            class="form-control"
                            id="footer"
                            type="text"
                            name="footer"
                            value="{!! data_get($maintenance, 'footer') ?: "We apologize for any inconvenience and appreciate your understanding." !!}"
                    >
                </div>
            </div>
            <div class="col-md-12">
                <hr class="mb-3">
            </div>
            <div class="col-md-12">

                <div class="mb-3">
                    <x-forms.input
                            name="maintenance_mode"
                            type="checkbox"
                            switcher
                            type="checkbox"
                            :checked="(bool) data_get($maintenance, 'maintenance_mode')"
                            label="{{ __('Maintenance Mode') }}"
                            tooltip="{{ __('If you turn on Maintenance Mode, your site will be closed for visitors.') }}"
                    />
                </div>
            </div>
        </div>
        <button
                class="btn btn-primary w-full"
                type="submit"
        >
            {{ __('Save') }}
        </button>
    </form>
@endsection