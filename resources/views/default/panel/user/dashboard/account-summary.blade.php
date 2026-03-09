@php
	$userId = auth()->id();

	$docs = cache()->get("user:{$userId}:user_docs");
	$docsCount = count($docs);
	$hoursSaved = $docsCount * 3;

	$textDocsCount = $docs->filter(fn($entity) => $entity->generator?->type == 'text')?->count();
	$imageDocsCount = $docs->filter(fn($entity) => $entity->generator?->type == 'image')?->count();
	$audioDocsCount = $docs->filter(fn($entity) => $entity->generator?->type == 'audio')?->count();
	$otherDocsCount = $docsCount - $textDocsCount - $imageDocsCount - $audioDocsCount;

	$sum = $textDocsCount + $imageDocsCount + $audioDocsCount;

	$chatbots = cache()->get("user:{$userId}:user_chatbots");
	$chatbotCount = count($chatbots);
@endphp

<x-card
    class="w-full"
    id="summary"
    size="lg"
>
    <div class="flex justify-between max-lg:flex-wrap lg:mb-7">
        <h3 class="items-center text-[17px] leading-6 lg:mb-0">
            @lang('Account Summary')
        </h3>
        <div class="flex w-full justify-between max-lg:flex-wrap lg:w-3/5">
            <div class="relative flex grow flex-col justify-center lg:ps-12 lg:after:absolute lg:after:right-0 lg:after:h-[80%] lg:after:w-px lg:after:bg-border">
                <p class="text-nowrap text-sm leading-5">@lang('Hours Saved')</p>
                <h2>{{ $hoursSaved }}</h2>
            </div>
            <div class="relative flex grow flex-col justify-center lg:ps-12 lg:after:absolute lg:after:right-0 lg:after:h-[80%] lg:after:w-px lg:after:bg-border">
                <p class="text-nowrap text-sm leading-5">@lang('Documents')</p>
                <h2> {{ $docsCount }} </h2>
            </div>
            <div class="flex grow flex-col justify-center lg:ps-12">
                <p class="text-nowrap text-sm leading-5">@lang('Chatbots')</p>
                <h2>{{ $chatbotCount }}</h2>
            </div>
        </div>
    </div>

    <hr>

    <div class="flex flex-col gap-4 sm:py-6">
        <h4 class="text-foreground/80">@lang('Document Overview')</h4>
        <div class="flex flex-nowrap items-center gap-10 max-sm:flex-wrap max-sm:gap-4">
            <div class="flex h-[10px] w-full flex-nowrap gap-0.5 overflow-hidden rounded-lg">
                <span
                    class="{{ $textDocsCount == 0 ? 'hidden' : '' }} bg-accent"
                    style="width: {{ $sum == 0 ? '100' : ($textDocsCount / $sum) * 100 }}%"
                ></span>
                <span
                    class="{{ $imageDocsCount == 0 ? 'hidden' : '' }} bg-[#1CA685]"
                    style="width: {{ $sum == 0 ? '100' : ($imageDocsCount / $sum) * 100 }}%"
                ></span>
                <span
                    class="{{ $audioDocsCount == 0 ? 'hidden' : '' }} bg-[#667085]"
                    style="width: {{ $sum == 0 ? '100' : ($audioDocsCount / $sum) * 100 }}%"
                ></span>
            </div>
            <x-button
                variant="link"
                href="{{ route('dashboard.user.openai.documents.all') }}"
            >
                <span class="text-nowrap font-bold text-foreground">@lang('View All')</span>
                <x-tabler-chevron-right class="size-4 rtl:rotate-180" />
            </x-button>
        </div>
        <div class="inline-flex flex-wrap gap-7 px-2 pt-1 max-sm:gap-3">
            <div class="inline-flex items-center gap-2">
                <span class="size-2.5 rounded-sm bg-accent"></span>
                <span class="text-sm leading-5 text-heading-foreground">@lang('Text')</span>
                <span class="leading-5 text-foreground/70">{{ $textDocsCount }}</span>
            </div>
            <div class="inline-flex items-center gap-2">
                <span class="size-2.5 rounded-sm bg-[#1CA685]"></span>
                <span class="text-sm leading-5 text-heading-foreground">@lang('Image')</span>
                <span class="leading-5 text-foreground/70">{{ $imageDocsCount }}</span>
            </div>
            <div class="inline-flex items-center gap-2">
                <span class="size-2.5 rounded-sm bg-[#667085]"></span>
                <span class="text-sm leading-5 text-heading-foreground">@lang('Audio')</span>
                <span class="leading-5 text-foreground/70">{{ $audioDocsCount }}</span>
            </div>
            <div class="inline-flex items-center gap-2">
                <span class="size-2.5 rounded-sm bg-[#E6E7E9]"></span>
                <span class="text-sm leading-5 text-heading-foreground">@lang('Other')</span>
                <span class="leading-5 text-foreground/70">{{ $otherDocsCount }}</span>
            </div>
        </div>
    </div>
</x-card>
