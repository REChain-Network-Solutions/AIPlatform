@extends('panel.layout.app', ['disable_tblr' => true])

@section('title', __('Edit Contact — CRM'))
@section('titlebar_title', __('Edit Contact'))
@section('titlebar_back', route('dashboard.crm.contacts.show', $contact))

@section('content')
    <div class="mx-auto max-w-2xl py-10">
        <form
            method="POST"
            action="{{ route('dashboard.crm.contacts.update', $contact) }}"
            class="space-y-6"
        >
            @csrf
            @method('PUT')
            @include('crm.contacts._form', ['contact' => $contact])

            <div class="flex items-center gap-3">
                <x-button
                    type="submit"
                    variant="primary"
                >{{ __('Update Contact') }}</x-button>
                <a
                    href="{{ route('dashboard.crm.contacts.show', $contact) }}"
                    class="text-sm text-muted-foreground hover:underline"
                >{{ __('Cancel') }}</a>
            </div>
        </form>
    </div>
@endsection
