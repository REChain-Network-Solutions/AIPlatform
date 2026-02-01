<x-card
    class:body="px-5 lg:px-10 py-6 lg:py-11 flex items-center flex-wrap gap-y-5"
    class="mb-9"
>
    <div class="w-full shrink lg:basis-1/3">
        <p class="mb-0 font-heading text-xl font-semibold text-heading-foreground">
            @lang('What’s New.')
        </p>
    </div>

    <div class="w-full lg:basis-2/3">
        <div class="flex flex-col gap-y-4 md:flex-row">

			@if(\App\Helpers\Classes\MarketplaceHelper::isRegistered('chatbot-agent'))
				<a
					class="group flex grow flex-col gap-1 border-b pb-4 text-heading-foreground transition-all md:border-b-0 md:border-e md:pb-0 md:pe-3 xl:px-10"
					href="{{ route('dashboard.chatbot-agent.index') }}"
				>
                <span class="group-hover:text-primary group-hover:underline">
                    {{ __('New Agent Messages') }}

                    <x-tabler-chevron-right class="ms-1 inline size-4 -translate-x-1 opacity-0 transition-all group-hover:translate-x-0 group-hover:opacity-100" />
                </span>
					<span class="flex font-heading text-[23px]/none font-semibold">
                    {{$unreadAgentMessagesCount}}
                </span>
				</a>
			@endif


            <a
                class="group flex grow flex-col gap-1 border-b pb-4 text-heading-foreground transition-all md:border-b-0 md:border-e md:px-3 md:pb-0 xl:px-10"
                href="#"
				@click.prevent="$store.externalChatbotHistory.setOpen(true)"
            >
                <span class="group-hover:text-primary group-hover:underline">
                    {{ __('New AI Messages') }}

                    <x-tabler-chevron-right class="ms-1 inline size-4 -translate-x-1 opacity-0 transition-all group-hover:translate-x-0 group-hover:opacity-100" />
                </span>
                <span class="flex font-heading text-[23px]/none font-semibold">
                    {{$unreadAiBotMessagesCount}}
                </span>
            </a>

            <a
                class="group flex grow flex-col gap-1 pb-4 text-heading-foreground transition-all md:px-3 md:pb-0 xl:px-10"
                href="javascript:void(0)"
            >
                <span class="group-hover:text-primary group-hover:underline">
                    {{ __('Total Messages') }}
				{{-- <x-tabler-chevron-right class="ms-1 inline size-4 -translate-x-1 opacity-0 transition-all group-hover:translate-x-0 group-hover:opacity-100" />--}}
                </span>
                <span class="flex font-heading text-[23px]/none font-semibold">
                    {{$allMessagesCount}}
                </span>
            </a>
        </div>
    </div>
</x-card>

<div class="grid grid-cols-1 gap-5 lg:grid-cols-2 lg:gap-11">
    {{-- Add new chatbot card --}}
    <x-card
        class:body="lg:p-16 p-9"
        class="text-center"
    >
        <figure class="mx-auto mb-6 inline-grid size-40 place-items-center rounded-full bg-heading-foreground/[3%]">
            <img
                src="{{ asset('vendor/chatbot/images/chatbot-create.png') }}"
                alt="Add New Chatbot"
            >
        </figure>
        <p class="mx-auto mb-6 max-w-[370px] font-heading text-xl font-semibold leading-[1.3em] text-heading-foreground">
            @lang('Create and configure a chatbot that interacts with your users.')
        </p>
        <x-button
            @click.prevent="setActiveChatbot('new_chatbot', 1, true);"
            variant="ghost-shadow"
            href="#"
        >
            <x-tabler-plus class="size-4" />
            @lang('Add New Chatbot')
        </x-button>
    </x-card>

    {{-- Show history card --}}
    <x-card
        class:body="lg:p-16 p-9"
        class="text-center"
    >
        <figure class="mx-auto mb-6 inline-grid size-40 place-items-center rounded-full bg-heading-foreground/[3%]">
            <img
                src="{{ asset('vendor/chatbot/images/chatbot-history.png') }}"
                alt="View Chat History"
            >
        </figure>
        <p class="mx-auto mb-6 max-w-[370px] font-heading text-xl font-semibold leading-[1.3em] text-heading-foreground">
            @lang('Explore recent conversations from your users.')
        </p>

        <div class="flex flex-wrap justify-center gap-x-3 gap-y-1">
            <x-button
                variant="ghost-shadow"
                href="#"
                @click.prevent="$store.externalChatbotHistory.setOpen(true)"
            >
                <x-tabler-robot
                    class="size-5"
                    stroke-width="1.5"
                />
                @lang('AI Bot Messages')
            </x-button>
			@if(\App\Helpers\Classes\MarketplaceHelper::isRegistered('chatbot-agent'))
				<x-button
					variant="ghost-shadow"
					href="{{ route('dashboard.chatbot-agent.index') }}"
				>
					<x-tabler-user
						class="size-5"
						stroke-width="1.5"
					/>
					@lang('Agent Messages')
				</x-button>
			@endif
        </div>
    </x-card>
</div>