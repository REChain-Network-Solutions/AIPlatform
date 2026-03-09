@php
    $generatedContent = cache()->get('generated_content');
    $all = $generatedContent->count();
    $all = $all == 0 ? 1 : $all;

    $contents = [
        [
            'type' => 'text',
            'label' => 'Text',
            'color' => '#3C82F6',
            'percent' => round(($generatedContent->filter(fn($entity) => $entity->generator?->type == 'text')->count() / $all) * 100, 0),
        ],
        [
            'type' => 'image',
            'label' => 'Image',
            'color' => '#9F77F8',
            'percent' => round(($generatedContent->filter(fn($entity) => $entity->generator?->type == 'image')->count() / $all) * 100, 0),
        ],
        [
            'type' => 'audio',
            'label' => 'Audio',
            'color' => '#60A5FA',
            'percent' => round(($generatedContent->filter(fn($entity) => $entity->generator?->type == 'audio')->count() / $all) * 100, 0),
        ],
        [
            'type' => 'video',
            'label' => 'Video',
            'color' => '#20C69F',
            'percent' => round(($generatedContent->filter(fn($entity) => $entity->generator?->type == 'video')->count() / $all) * 100, 0),
        ],
        [
            'type' => 'code',
            'label' => 'Code',
            'color' => '#E0B43E',
            'percent' => round(($generatedContent->filter(fn($entity) => $entity->generator?->type == 'code')->count() / $all) * 100, 0),
        ],
    ];

    /**
     * sort the content using percent
     * @param array $contents
     */
    usort($contents, function ($a, $b) {
        return $a['percent'] < $b['percent'];
    });
@endphp
<x-card
    class="flex flex-col"
    class:body="flex flex-col justify-center grow"
    id="{{ 'admin-card-' . ($widget?->name?->value ?? 'generated-content') }}"
>
    <x-slot:head
        class="flex items-center justify-between px-5 py-3.5"
    >
        <div class="flex items-center gap-4">
            <x-lqd-icon class="bg-background text-heading-foreground dark:bg-foreground/5">
                <x-tabler-clipboard-text
                    class="size-6"
                    stroke-width="1.5"
                />
            </x-lqd-icon>
            <h4 class="m-0 flex items-center gap-1 text-base font-medium">
                {{ __('Generated Content') }}
                <x-info-tooltip text="{{ __('Track how much and what kind of content users are creating.') }}" />
            </h4>
        </div>
    </x-slot:head>

    <div class="flex flex-col gap-4">
        <div class="flex w-full rounded-7xl border p-2.5">
            <div class="flex h-3 w-full flex-nowrap gap-0.5 overflow-hidden rounded-7xl">
                @foreach ($contents as $content)
                    @if ($content['percent'] != 0)
                        <span style="width: {{ $content['percent'] }}%; background-color: {{ $content['color'] }};"></span>
                    @endif
                @endforeach
            </div>
        </div>
        <ul class="flex flex-col gap-2.5">
            @foreach ($contents as $content)
                <li class="flex items-center justify-between border-b border-card-border py-2">
                    <div class="flex items-center gap-2.5">
                        <span
                            class="size-2.5 rounded-sm"
                            style="background-color: {{ $content['color'] }}"
                        ></span>
                        <p class="mb-0 text-[15px] font-medium text-foreground">{{ __($content['label']) }}</p>
                    </div>
                    <span class="text-center text-foreground/50">{{ $content['percent'] }}%</span>
                </li>
            @endforeach
        </ul>
    </div>
</x-card>
