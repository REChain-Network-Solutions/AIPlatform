@extends('panel.layout.settings')
@section('title', __('Mailchimp Newsletter'))
@section('titlebar_subtitle', __('You can add new users who sign up to your Mailchimp subscribers by integrating your Mailchimp account.'))

@section('content')
    <div class="page-body pt-6">
        <div class="container-xl">
            <div class="row">
                <div class="col-md-5 mx-auto">
                    <form method="POST" action="{{route("dashboard.admin.mailchimp-newsletter.store")}}">
                        @csrf
                        <h3 class="mb-[25px] text-[20px]">{{ __('Mailchimp Newsletter Settings') }}</h3>

                        <div class="row">

                            <div class="col-md-12">
                                <div class="mb-3">
                                    <label class="form-label">{{__('MAILCHIMP API KEY')}}</label>
                                    <input type="text" class="form-control"
                                           value="{{ setting('mailchimp_api_key', old('mailchimp_api_key')) }}"
                                           name="mailchimp_api_key">
                                </div>
                            </div>
                            <div class="col-md-12">
                                <div class="mb-3">
                                    <label class="form-label">{{__('MAILCHIMP LIST ID')}}</label>
                                    <input type="text" class="form-control"
                                           value="{{ setting('mailchimp_list_id', old('mailchimp_list_id')) }}"
                                           name="mailchimp_list_id">
                                </div>
                            </div>

                            <div class="col-md-12">
                                <div class="mb-3">
                                    <x-forms.input
                                            class:container="mb-2"
                                            type="checkbox"
                                            name="mailchimp_register"
                                            :checked=" (bool) setting('mailchimp_register', old('mailchimp_register'))"
                                            label="{{ __('Register Active') }}"
                                            switcher
                                    />
                                </div>
                            </div>

                            <div class="col-md-12">
                                <button type="submit" class="btn btn-primary w-100 btn-block">
                                    {{__('Save')}}
                                </button>
                            </div>

                        </div>

                    </form>
                </div>
            </div>
        </div>
    </div>
@endsection
