@extends('panel.layout.settings')
@section('title', __('AI Engines'))
@section('titlebar_actions', '')
@section('settings')
    <div x-data="{ 'activeFilter': 'All' }">
        <form
            class="flex flex-col gap-5"
            action="{{ route('dashboard.admin.ai-chat-model.update.engine.images') }}"
            method="POST"
            enctype="multipart/form-data"
        >
            @csrf
            <h4 class="mb-0">
                {{ __('AI Engines Management') }}
            </h4>
            <label>
                {{ __('Manage AI engine logos and images. Update engine logos that will be displayed across all models for each engine.') }}
            </label>

            <div class="overflow-x-auto">
                <table class="w-full border-collapse">
                    <thead>
                        <tr class="border-b">
                            <th class="text-left p-3 font-semibold">{{ __('Engine') }}</th>
                            <th class="text-left p-3 font-semibold">{{ __('Models Count') }}</th>
                            <th class="text-left p-3 font-semibold">{{ __('Upload Logo') }}</th>
                            <th class="text-left p-3 font-semibold">{{ __('Preview') }}</th>
                        </tr>
                    </thead>
                    <tbody>
                        @foreach ($engines as $engine)
                            <tr class="border-b hover:bg-gray-50">
                                <td class="p-3">
                                    <span class="font-medium">{{ $engine->engine }}</span>
                                </td>
                                <td class="p-3">
                                    <span class="px-2 py-1 bg-blue-100 text-blue-800 rounded text-xs font-medium">
                                        {{ $engine->entity_count ?? 0 }} {{ __('models') }}
                                    </span>
                                </td>
                                <td class="p-3">
                                    <x-forms.input
                                        type="file"
                                        name="engine_logo[{{ $engine->engine }}]"
                                        label="{{ __('Engine Logo') }}"
                                        accept="image/*"
                                        class="w-full"
                                    />
                                </td>
                                <td class="p-3">
                                    @if($engine->first_image)
                                        <img src="{{ asset($engine->first_image) }}" alt="{{ $engine->engine }}" class="size-12 object-contain rounded">
                                    @else
                                        <span class="text-gray-400 text-xs">{{ __('No Image') }}</span>
                                    @endif
                                </td>
                            </tr>
                        @endforeach
                    </tbody>
                </table>
            </div>

            @if ($engines->isEmpty())
                <div class="text-center py-8">
                    <p class="text-gray-500">{{ __('No AI engines found.') }}</p>
                </div>
            @endif

            <div class="mt-6">
                @if ($app_is_demo)
                    <x-button
                        type="button"
                        onclick="return toastr.info('This feature is disabled in Demo version.');"
                    >
                        {{ __('Save Engine Images') }}
                    </x-button>
                @else
                    <x-button
                        type="submit"
                        size="lg"
                    >
                        {{ __('Save Engine Images') }}
                    </x-button>
                @endif
            </div>
        </form>
    </div>
@endsection
