@extends('panel.layout.app', ['disable_tblr' => true])
@section('title', __('New Company — CRM'))
@section('titlebar_title', __('New Company'))
@section('titlebar_back', route('dashboard.crm.companies.index'))

@section('content')
    <div class="mx-auto max-w-2xl py-10">
        <form
            method="POST"
            action="{{ route('dashboard.crm.companies.store') }}"
            class="space-y-6"
        >
            @csrf
            @include('crm.companies._form', ['company' => null])
            <div class="flex items-center gap-3">
                <x-button
                    type="submit"
                    variant="primary"
                >{{ __('Save Company') }}</x-button>
                <a
                    href="{{ route('dashboard.crm.companies.index') }}"
                    class="text-sm text-muted-foreground hover:underline"
                >{{ __('Cancel') }}</a>
            </div>
        </form>
    </div>
@endsection
