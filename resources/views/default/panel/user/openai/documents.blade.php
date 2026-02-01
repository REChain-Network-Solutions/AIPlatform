@php
    $sort_buttons = [
        [
            'label' => __('Date'),
            'sort' => 'created_at',
        ],
        [
            'label' => __('Title'),
            'sort' => 'title',
        ],
        [
            'label' => __('Type'),
            'sort' => 'openai_id',
        ],
        [
            'label' => __('Cost'),
            'sort' => 'credits',
        ],
    ];

    $filter_buttons = [
        [
            'label' => __('All'),
            'filter' => 'all',
        ],
        [
            'label' => __('Favorites'),
            'filter' => 'favorites',
        ],
        [
            'label' => __('Text'),
            'filter' => 'text',
        ],
        [
            'label' => __('Image'),
            'filter' => 'image',
        ],
        [
            'label' => __('Video'),
            'filter' => 'video',
        ],
        [
            'label' => __('Code'),
            'filter' => 'code',
        ],
    ];
@endphp

@extends('panel.layout.app', ['disable_tblr' => true])
@section('title', __('My Documents'))
@section('titlebar_title')
    {{ $currfolder?->name ? __("Folder: $currfolder?->name") : __('My Documents') }}
@endsection

{{-- Filter list --}}
@if ($items && count($items) > 0)
    @section('titlebar_after')
        <div class="flex flex-wrap items-center justify-between gap-2 lg:flex-nowrap">
            @if (blank($currfolder))
                <div class="flex flex-wrap items-center gap-1">
                    <x-dropdown.dropdown
                        class="pe-3"
                        offsetY="1rem"
                    >
                        <x-slot:trigger
                            class="whitespace-nowrap py-1.5"
                            variant="link"
                            size="xs"
                        >
                            {{ __('Sort by:') }}
                            <x-tabler-arrows-sort class="size-4" />
                        </x-slot:trigger>

                        <x-slot:dropdown
                            class="overflow-hidden text-2xs font-medium"
                        >
                            <form
                                class="lqd-sort-list flex flex-col"
                                action="{{ route('dashboard.user.openai.documents.all', ['id' => $currfolder?->id, 'listOnly' => 'true']) }}"
                                method="GET"
                                x-init
                                x-target="lqd-docs-container"
                                @submit="$store.documentsFilter.changePage('1')"
                            >
                                <input
                                    type="hidden"
                                    name="filter"
                                    :value="$store.documentsFilter.filter"
                                >
                                <input
                                    type="hidden"
                                    name="page"
                                    value="1"
                                >
                                <input
                                    type="hidden"
                                    name="sortAscDesc"
                                    :value="$store.documentsFilter.sortAscDesc"
                                >
                                @foreach ($sort_buttons as $button)
                                    <button
                                        class="group flex w-full items-center gap-1 px-3 py-2 hover:bg-foreground/5 [&.active]:bg-foreground/5"
                                        :class="$store.documentsFilter.sort === '{{ $button['sort'] }}' && 'active'"
                                        name="sort"
                                        value="{{ $button['sort'] }}"
                                        @click="$store.documentsFilter.changeSort('{{ $button['sort'] }}')"
                                    >
                                        {{ $button['label'] }}
                                        <x-tabler-caret-down-filled
                                            class="size-3 opacity-0 transition-all group-[&.active]:opacity-80"
                                            ::class="$store.documentsFilter.sortAscDesc === 'asc' && 'rotate-180'"
                                        />
                                    </button>
                                @endforeach
                            </form>
                        </x-slot:dropdown>
                    </x-dropdown.dropdown>

                    <form
                        class="lqd-filter-list flex flex-wrap items-center gap-x-4 gap-y-2 text-heading-foreground max-sm:gap-3"
                        action="{{ route('dashboard.user.openai.documents.all', ['id' => $currfolder?->id, 'listOnly' => 'true']) }}"
                        method="GET"
                        x-init
                        x-target="lqd-docs-container"
                        @submit="$store.documentsFilter.changePage('1')"
                    >
                        <input
                            type="hidden"
                            name="sort"
                            :value="$store.documentsFilter.sort"
                        >
                        <input
                            type="hidden"
                            name="page"
                            value="1"
                        >
                        <input
                            type="hidden"
                            name="sortAscDesc"
                            :value="$store.documentsFilter.sortAscDesc"
                        >
                        @foreach ($filter_buttons as $button)
                            <x-button
                                @class([
                                    'lqd-filter-btn inline-flex px-2.5 py-0.5 transition-colors hover:bg-foreground/5 [&.active]:bg-foreground/5 hover:translate-y-0 text-2xs leading-tight',
                                    // 'active' => $filter == $button['filter'],
                                ])
                                tag="button"
                                type="submit"
                                name="filter"
                                value="{{ $button['filter'] }}"
                                variant="ghost"
                                ::class="$store.documentsFilter.filter === '{{ $button['filter'] }}' && 'active'"
                                @click="$store.documentsFilter.changeFilter('{{ $button['filter'] }}')"
                            >
                                {{ $button['label'] }}
                            </x-button>
                        @endforeach
                    </form>
                </div>

                <div class="lqd-posts-view-toggle lqd-docs-view-toggle lqd-view-toggle relative z-1 flex items-center gap-2 lg:ms-auto lg:justify-end">
                    <button
                        class="lqd-view-toggle-trigger inline-flex size-7 items-center justify-center rounded-md transition-colors hover:bg-foreground/5 [&.active]:bg-foreground/5"
                        :class="$store.docsViewMode.docsViewMode === 'list' && 'active'"
                        x-init
                        @click="$store.docsViewMode.change('list')"
                        title="List view"
                    >
                        <x-tabler-list
                            class="size-5"
                            stroke-width="1.5"
                        />
                    </button>
                    <button
                        class="lqd-view-toggle-trigger inline-flex size-7 items-center justify-center rounded-md transition-colors hover:bg-foreground/5 [&.active]:bg-foreground/5"
                        :class="$store.docsViewMode.docsViewMode === 'grid' && 'active'"
                        x-init
                        @click="$store.docsViewMode.change('grid')"
                        title="Grid view"
                    >
                        <x-tabler-layout-grid
                            class="size-5"
                            stroke-width="1.5"
                        />
                    </button>
                </div>
            @endif
        </div>
    @endsection
@endif

@section('content')
    <div class="py-10">
        {{-- Folders row --}}
        @if ($currfolder == null)
            <div class="mb-6 flex flex-col justify-between gap-4 md:flex-row md:items-center">
                @if (isset(auth()->user()->folders) && count(auth()->user()->folders) > 0)
                    <div class="grid grow grid-cols-3 !gap-5 max-md:grid-cols-1">
                        @foreach (auth()->user()->folders ?? [] as $folder)
                            <x-documents.folder :$folder />
                        @endforeach
                    </div>
                @endif

                <x-dropdown.dropdown
                    class="shrink-0 md:ms-auto"
                    class:dropdown-dropdown="max-lg:end-auto max-lg:start-0"
                    :teleport="false"
                    offsetY="1rem"
                    anchor="end"
                >
                    <x-slot:trigger
                        class="px-2 py-1"
                        variant="link"
                        size="xs"
                    >
                        {{ __('Options') }}
                        <x-tabler-dots class="size-5" />
                    </x-slot:trigger>
                    <x-slot:dropdown
                        class="p-1"
                    >
                        <form
                            class="w-full"
                            x-data="{ selectedAction: 'delete' }"
                            @submit.prevent="
								if (selectedAction === 'delete') {
									$store.documentsSelection.bulkDelete('all', {
										confirmSelectedMessage: '{{ __('Are you sure you want to delete all documents?') }}',
										deleteUrl: '{{ route('dashboard.user.openai.documents.bulkDelete') }}'
									});
								}"
                        >
                            <x-button
                                class="w-full rounded-md hover:bg-rose-500 hover:text-white"
                                variant="none"
                                type="submit"
                            >
                                <x-tabler-trash class="size-4" />
                                {{ __('Delete All') }}
                            </x-button>
                        </form>
                    </x-slot:dropdown>
                </x-dropdown.dropdown>
            </div>
        @else
            <div class="mb-6 flex items-center gap-3">
                <x-button
                    class="aspect-square rounded-lg"
                    href="{{ route('dashboard.user.openai.documents.all') }}"
                    variant="secondary"
                    title="{{ __('Back to documents') }}"
                >
                    <x-tabler-arrow-left />
                </x-button>
                <x-documents.folder
                    :folder="$currfolder"
                    folder-single-view="{{ true }}"
                />
            </div>
        @endif

        {{-- Documents row --}}
        @if (!$items || count($items) === 0)
            @include('panel.user.openai.documents_empty')
        @else
            @include('panel.user.openai.documents_container')
        @endif

    </div>
@endsection
