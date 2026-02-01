@php
    $announcements = cache('public_announcements');
@endphp

<div class="flex w-full flex-col gap-5 rounded-lg border sm:p-5">
    <div class="w-full">
        <x-button
            class="w-full"
            href="{{ route('dashboard.admin.public-announcement.create') }}"
        >
            <x-tabler-plus class="size-4" />
            {{ __('New') }}
        </x-button>
    </div>
    @if (!empty($announcements))
        <x-table>
            <x-slot:head>
                <tr>
                    <th>
                        {{ __('Title') }}
                    </th>
                    <th>
                        {{ __('Type') }}
                    </th>
                    <th>
                        {{ __('Active') }}
                    </th>
                    <th class="text-end">
                        {{ __('Action') }}
                    </th>
                </tr>
            </x-slot:head>

            <x-slot:body>
                @foreach ($announcements as $entry)
                    <tr>
                        <td>
                            <span class="line-clamp-3">{{ $entry->title }}</span>
                        </td>
                        <td>
                            {{ $entry->type->label() }}
                        </td>
                        <td>
                            @if ($entry->active)
                                <x-badge
                                    class="text-3xs group-[&.active]:block"
                                    variant="primary"
                                >
                                    {{ __('Enabled') }}
                                </x-badge>
                            @else
                                <x-badge
                                    class="text-3xs group-[&.passive]:block"
                                    variant="danger"
                                >
                                    {{ __('Disabled') }}
                                </x-badge>
                            @endif
                        </td>
                        <td class="whitespace-nowrap text-end">
                            <x-button
                                class="size-9"
                                variant="ghost-shadow"
                                size="none"
                                href="{{ route('dashboard.admin.public-announcement.edit', $entry->id) }}"
                                title="{{ __('Edit') }}"
                            >
                                <x-tabler-pencil class="size-4" />
                            </x-button>
                            @if ($app_is_demo)
                                <x-button
                                    class="size-9"
                                    variant="ghost-shadow"
                                    hover-variant="danger"
                                    size="none"
                                    onclick="return toastr.info('This feature is disabled in Demo version.')"
                                    title="{{ __('Delete') }}"
                                >
                                    <x-tabler-x class="size-4" />
                                </x-button>
                            @else
                                <x-button
                                    class="size-9"
                                    variant="ghost-shadow"
                                    hover-variant="danger"
                                    size="none"
                                    onclick="return confirm('{{ __('Are you sure? This is permanent and will delete all documents related to user.') }}')"
                                    href="{{ route('dashboard.admin.public-announcement.delete', $entry->id) }}"
                                    title="{{ __('Delete') }}"
                                >
                                    <x-tabler-x class="size-4" />
                                </x-button>
                            @endif
                        </td>
                    </tr>
                @endforeach
            </x-slot:body>
        </x-table>
    @else
        <span class="text-center text-lg font-semibold sm:text-xl">There is no public announcement yet</span>
    @endif
</div>
