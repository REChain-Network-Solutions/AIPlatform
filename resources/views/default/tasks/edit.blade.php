@extends('panel.layout.app', ['disable_tblr' => true])

@section('title', __('Edit Task'))
@section('titlebar_title', __('Edit Task'))
@section('titlebar_subtitle', $task->title)

@section('content')
    <div class="py-10">
        <form method="POST" action="{{ route('dashboard.tasks.update', $task) }}" id="task-form">
            @csrf @method('PUT')
            @include('tasks._form', ['task' => $task])
            <div class="mt-6 flex items-center gap-3">
                <x-button type="submit" variant="primary">{{ __('Save Changes') }}</x-button>
                <x-button href="{{ route('dashboard.tasks.show', $task) }}" variant="ghost">{{ __('Cancel') }}</x-button>
            </div>
        </form>
    </div>
@endsection
