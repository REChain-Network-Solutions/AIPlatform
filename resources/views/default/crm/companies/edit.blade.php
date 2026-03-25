@extends('panel.layout.app', ['disable_tblr' => true])
@section('title', __('Edit Company — CRM'))
@section('titlebar_title', __('Edit :name', ['name' => $company->name]))
@section('titlebar_back', route('dashboard.crm.companies.show', $company))

@section('content')
    <div class="mx-auto max-w-2xl py-10">
        <form
            method="POST"
            action="{{ route('dashboard.crm.companies.update', $company) }}"
            class="space-y-6"
        >
            @csrf
            @method('PUT')
            @include('crm.companies._form', ['company' => $company])
            <div class="flex items-center gap-3">
                <x-button
                    type="submit"
                    variant="primary"
                >{{ __('Update Company') }}</x-button>
                <a
                    href="{{ route('dashboard.crm.companies.show', $company) }}"
                    class="text-sm text-muted-foreground hover:underline"
                >{{ __('Cancel') }}</a>
            </div>
        </form>
    </div>
@endsection
