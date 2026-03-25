@extends('panel.layout.app', ['disable_tblr' => true])

@section('title', __('New Lead — CRM'))
@section('titlebar_title', __('New Lead'))
@section('titlebar_back', route('dashboard.crm.leads.index'))

@section('content')
    <div class="mx-auto max-w-2xl py-10">
        <form
            method="POST"
            action="{{ route('dashboard.crm.leads.store') }}"
            class="space-y-6"
        >
            @csrf
            @include('crm.leads._form', ['lead' => null])

            <div class="flex items-center gap-3">
                <x-button
                    type="submit"
                    variant="primary"
                >{{ __('Save Lead') }}</x-button>
                <a
                    href="{{ route('dashboard.crm.leads.index') }}"
                    class="text-sm text-muted-foreground hover:underline"
                >{{ __('Cancel') }}</a>
            </div>
        </form>
    </div>
@endsection
