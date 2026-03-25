@extends('panel.layout.app', ['disable_tblr' => true])

@section('title', __('Edit Lead — CRM'))
@section('titlebar_title', __('Edit Lead'))
@section('titlebar_back', route('dashboard.crm.leads.show', $lead))

@section('content')
    <div class="mx-auto max-w-2xl py-10">
        <form
            method="POST"
            action="{{ route('dashboard.crm.leads.update', $lead) }}"
            class="space-y-6"
        >
            @csrf
            @method('PUT')
            @include('crm.leads._form', ['lead' => $lead])

            <div class="flex items-center gap-3">
                <x-button
                    type="submit"
                    variant="primary"
                >{{ __('Update Lead') }}</x-button>
                <a
                    href="{{ route('dashboard.crm.leads.show', $lead) }}"
                    class="text-sm text-muted-foreground hover:underline"
                >{{ __('Cancel') }}</a>
            </div>
        </form>
    </div>
@endsection
