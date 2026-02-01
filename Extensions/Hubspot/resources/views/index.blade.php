@extends('panel.layout.settings')
@section('title', __('Hubspot'))
@section('titlebar_subtitle', __('Take advantage of our HubSpot integration to simplify your business processes and increase customer satisfaction.'))

@section('content')
    <div class="page-body pt-6">
        <div class="container-xl">
            <div class="row">
                <div class="col-md-5 mx-auto">
                    <form method="POST" action="{{route("dashboard.admin.hubspot.store")}}">
                        @csrf
                        <h3 class="mb-[25px] text-[20px]">{{ __('Hubspot Settings') }}</h3>

                        <div class="row">

                            <div class="col-md-12">
                                <div class="mb-3">
                                    <label class="form-label">{{__('HUBSPOT ACCESS TOKEN')}}</label>
                                    <input type="text" class="form-control"
                                           value="{{ setting('hubspot_access_token') }}"
                                           name="hubspot_access_token">
                                </div>
                            </div>

                            <div class="col-md-12">
                                <div class="mb-3">

                                    <label class="form-label">{{ __('CRM Contact Register') }}
                                        <x-info-tooltip
                                                text="{{ __('Activating this will automatically add new users who register to your HubSpot CRM contact list.') }}"/>
                                    </label>
                                    <select
                                            class="form-select"
                                            name="hubspot_crm_contact_register"
                                    >
                                        <option
                                                value="1"
                                                {{ setting('hubspot_crm_contact_register') == 1 ? 'selected' : '' }}
                                        >
                                            {{ __('Active') }}</option>
                                        <option
                                                value="0"
                                                {{ setting('hubspot_crm_contact_register') == 0 ? 'selected' : '' }}
                                        >
                                            {{ __('Passive') }}</option>
                                    </select>
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
