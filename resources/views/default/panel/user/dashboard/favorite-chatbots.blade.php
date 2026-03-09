@php
	$userId = auth()->id();
	$favoriteChabots = cache()->get("user:{$userId}:favorite_chatbots");
@endphp

<x-card
    class="flex w-full flex-col lg:w-[48%]"
    id="favorite-chatbots"
    size="md"
>
    <x-slot:head
        @class(['pb-0 pt-5 border-0' => filled($favoriteChabots)])
    >
        <div class="flex items-center justify-between">
            <h4 class="m-0 text-[17px]">{{ __('Favorite Chatbots') }}</h4>
            <x-button
                variant="link"
                href="/dashboard/user/openai/chat/ai-chat-list"
            >
                <span class="text-nowrap font-bold text-foreground"> {{ __('View All') }} </span>
                <x-tabler-chevron-right class="size-4 rtl:rotate-180" />
            </x-button>
        </div>
    </x-slot:head>
    <div class="flex w-full flex-wrap justify-between gap-y-4">
        @forelse ($favoriteChabots as $chatbot)
            <x-card
                class="flex w-full flex-col lg:w-[48%]"
                class:body="space-y-3"
                size="sm"
            >
                <a
                    @class([
                        'absolute left-0 top-0 z-2 block h-full w-full',
                        'border-[3px] border-secondary' =>
                            $chatbot->openaiGeneratorChatCategory?->plan == 'premium',
                    ])
                    href="{{ route('dashboard.user.openai.chat.chat', $chatbot->openaiGeneratorChatCategory?->slug) }}"
                ></a>

                <div
                    class="!mt-0 flex items-center overflow-hidden rounded-full max-sm:mx-auto"
                    style="width: 61px; height: 61px; background: {{ $chatbot->openaiGeneratorChatCategory?->color }};"
                >
                    @if ($chatbot->openaiGeneratorChatCategory?->slug === 'ai-chat-bot')
                        <img
                            class="lqd-chat-avatar-img h-full w-full object-cover object-center"
                            src="{{ custom_theme_url('/assets/img/chat-default.jpg') }}"
                            alt="{{ __($chatbot->openaiGeneratorChatCategory?->name) }}"
                        >
                    @elseif ($chatbot->openaiGeneratorChatCategory?->image)
                        <img
                            class="lqd-chat-avatar-img h-full w-full object-cover object-center"
                            src="{{ custom_theme_url($chatbot->openaiGeneratorChatCategory?->image, true) }}"
                            alt="{{ __($chatbot->openaiGeneratorChatCategory?->name) }}"
                        >
                    @else
                        <span
                            class="block w-full overflow-hidden overflow-ellipsis whitespace-nowrap text-center text-2xl"
                        >
                            {{ __($chatbot->openaiGeneratorChatCategory?->short_name) }}
                        </span>
                    @endif
                </div>
                <h4 class="font-medium max-sm:text-center">{{ $chatbot->openaiGeneratorChatCategory?->name }}</h4>
                <div class="flex w-fit rounded-xl border px-2 py-1 max-sm:mx-auto">
                    <span
                        class="text-center">{{ ucwords(str_replace('_', ' ', $chatbot->openaiGeneratorChatCategory?->description)) }}</span>
                </div>
            </x-card>
        @empty
            <h3 class="mx-auto text-foreground">
                {{ __("You don't have favorite chatbot") }}
            </h3>
        @endforelse
    </div>
</x-card>
