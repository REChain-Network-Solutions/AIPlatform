@php
    $announcements = cache('announcements');
@endphp

<x-card
    class="flex w-full flex-col lg:w-[48%]"
    id="add-new"
    size="md"
>
    <x-slot:head>
        <h4 class="m-0 text-[17px]">{{ __('Announcements') }}</h4>
    </x-slot:head>
    <ul class="flex flex-col overflow-hidden">
        @forelse ($announcements as $announcement)
            <li class="flex flex-nowrap items-center justify-between gap-2 py-4 sm:gap-16">
                <div class="flex flex-nowrap items-center gap-3">
                    <div
                        class="{{ $announcement->is($announcements->last()) ? '' : 'after:absolute after:w-px after:-bottom-12 after:start-1/2 after:h-12 after:-z-1 after:bg-foreground/10' }} relative flex-shrink-0">
                        <x-dynamic-component
                            class="size-8 rounded-full stroke-white p-1"
                            style="background: {{ $announcement->type->color() }}"
                            stroke-width="1.5"
                            :component="'tabler-' . $announcement->type->image()"
                        />
                    </div>
                    <span class="line-clamp-2 font-semibold text-heading-foreground">
                        {{ $announcement->title }}
                    </span>
                </div>
                <div class="flex flex-shrink-0 flex-col items-end gap-1">
                    <span class="font-semibold text-heading-foreground">{{ $announcement->type->label() }}</span>
                    <span>{{ $announcement->created_at->format('M d,Y') }}</span>
                </div>
            </li>
        @empty
            <h4 class="mx-auto">@lang('There is no announcement')</h4>
        @endforelse
    </ul>
</x-card>
