@extends('panel.layout.app', ['disable_tblr' => true])

@section('title', __('Edit Pipeline — CRM'))
@section('titlebar_title', __('Edit Pipeline'))
@section('titlebar_subtitle', $pipeline->name)

@section('content')
    <div class="py-10">
        <form method="POST" action="{{ route('dashboard.crm.pipelines.update', $pipeline) }}" id="pipeline-form">
            @csrf @method('PUT')
            @include('crm.pipelines._form', ['pipeline' => $pipeline])

            <div class="mt-6 flex items-center gap-3">
                <x-button type="submit" variant="primary">{{ __('Save Changes') }}</x-button>
                <x-button href="{{ route('dashboard.crm.pipelines.index') }}" variant="ghost">{{ __('Cancel') }}</x-button>
            </div>
        </form>
    </div>
@endsection

@push('script')
    @include('crm.pipelines._stage_js')
@endpush
