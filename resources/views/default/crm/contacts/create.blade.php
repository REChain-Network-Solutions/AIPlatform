@extends('panel.layout.app', ['disable_tblr' => true])

@section('title', __('New Contact — CRM'))
@section('titlebar_title', __('New Contact'))
@section('titlebar_back', route('dashboard.crm.contacts.index'))

@section('content')
    <div class="mx-auto max-w-2xl py-10">
        <form
            method="POST"
            action="{{ route('dashboard.crm.contacts.store') }}"
            class="space-y-6"
        >
            @csrf
            @include('crm.contacts._form', ['contact' => null])

            <div class="flex items-center gap-3">
                <x-button
                    type="submit"
                    variant="primary"
                >{{ __('Save Contact') }}</x-button>
                <a
                    href="{{ route('dashboard.crm.contacts.index') }}"
                    class="text-sm text-muted-foreground hover:underline"
                >{{ __('Cancel') }}</a>
            </div>
        </form>
    </div>
@endsection
