
@extends('panel.layout.app', ['disable_tblr' => true])
@section('title', __('Chat Categories'))
@section('titlebar_subtitle', __('Manage Chat Categories'))
@section('titlebar_actions')
    <x-button
            onclick="{{ $app_is_demo ? 'return toastr.info(\'This feature is disabled in Demo version.\')' : '' }}"
            href="{{ $app_is_demo ? '#' : LaravelLocalization::localizeUrl(route('dashboard.user.chat-setting.chat-category.create')) }}"
    >
        {{ __('Add Category') }}
    </x-button>
@endsection
@section('content')
    <div class="py-6">
        <x-table class="table">
            <x-slot:head>
                <tr>
                    <th>
                        {{ __('Name') }}
                    </th>
                    <th class="text-end">
                        {{ __('Action') }}
                    </th>
                </tr>
            </x-slot:head>

            <x-slot:body>
                @foreach ($list as $entry)
                    <tr>
                        <td>
                            {{ $entry->name }}
                        </td>
                        <td class="text-end">
                            @if ($app_is_demo)
                                <x-button
                                        class="h-9 w-9"
                                        variant="ghost-shadow"
                                        size="none"
                                        onclick="return toastr.info('This feature is disabled in Demo version.')"
                                        title="{{ __('Edit') }}"
                                >
                                    <x-tabler-pencil class="h-4 w-4" />
                                </x-button>
                                <x-button
                                        class="h-9 w-9"
                                        variant="ghost-shadow"
                                        hover-variant="danger"
                                        size="none"
                                        onclick="return toastr.info('This feature is disabled in Demo version.')"
                                        title="{{ __('Delete') }}"
                                >
                                    <x-tabler-x class="h-4 w-4" />
                                </x-button>
                            @else
                                <x-button
                                        class="h-9 w-9"
                                        variant="ghost-shadow"
                                        size="none"
                                        href="{{ LaravelLocalization::localizeUrl(route('dashboard.user.chat-setting.chat-category.edit', $entry->id)) }}"
                                        title="{{ __('Edit') }}"
                                >
                                    <x-tabler-pencil class="h-4 w-4" />
                                </x-button>
                                <x-button
                                        class="h-9 w-9"
                                        variant="ghost-shadow"
                                        hover-variant="danger"
                                        size="none"
                                        href="{{ LaravelLocalization::localizeUrl(route('dashboard.user.chat-setting.chat-category.destroy', $entry->id)) }}"
                                        onclick="return confirm('{{ __('Are you sure? This is permanent.') }}')"
                                        title="{{ __('Delete') }}"
                                >
                                    <x-tabler-x class="h-4 w-4" />
                                </x-button>
                            @endif
                        </td>
                    </tr>
                @endforeach

            </x-slot:body>
        </x-table>
    </div>
@endsection