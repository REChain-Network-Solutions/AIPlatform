{{-- Chatbots List --}}
<div class="py-14">
    <h2 class="mb-9">
        @lang('Active Chatbots')
    </h2>

    <div class="grid grid-cols-1 gap-5 md:grid-cols-2 lg:grid-cols-3">
        {{-- Newly added items --}}
        <template
            x-for="chatbot in chatbots.data.filter(c => c.id !== 'new_chatbot')"
            :key="chatbot.id"
        >
            <x-card size="md">
                <x-slot:head
                    class="flex items-center justify-between gap-4 border-none px-5 py-[18px]"
                >
                    <figure>
                        <img
                            class="size-10 rounded-full object-cover object-center"
                            width="40"
                            height="40"
                            :src="`${window.location.origin}/${chatbot.avatar}`"
                            :alt="chatbot.title"
                        />
                    </figure>

                    <x-dropdown.dropdown
                        class:dropdown-dropdown="max-lg:end-0 max-lg:start-auto"
                        anchor="end"
                    >
                        <x-slot:trigger
                            class="size-10"
                        >
                            <svg
                                width="3"
                                height="13"
                                viewBox="0 0 3 13"
                                fill="currentColor"
                                xmlns="http://www.w3.org/2000/svg"
                            >
                                <path
                                    d="M3 11.5C3 12.3 2.3 13 1.5 13C0.7 13 0 12.3 0 11.5C0 10.7 0.7 10 1.5 10C2.3 10 3 10.7 3 11.5ZM3 6.5C3 7.3 2.3 8 1.5 8C0.7 8 0 7.3 0 6.5C0 5.7 0.7 5 1.5 5C2.3 5 3 5.7 3 6.5ZM3 1.5C3 2.3 2.3 3 1.5 3C0.7 3 0 2.3 0 1.5C0 0.7 0.7 0 1.5 0C2.3 0 3 0.7 3 1.5Z"
                                />
                            </svg>
                            <span class="sr-only">
                                @lang('Chatbot Options')
                            </span>
                        </x-slot:trigger>
                        <x-slot:dropdown
                            class="min-w-[170px]"
                        >
                            @php
                                $dropdown_items = [
                                    [
                                        'label' => __('Edit'),
                                        'link' => '#',
                                        'attrs' => [
                                            '@click.prevent' => 'setActiveChatbot(chatbot.id, 1, true);',
                                        ],
                                    ],
                                    [
                                        'label' => __('Customize'),
                                        'link' => '#',
                                        'attrs' => [
                                            '@click.prevent' => 'setActiveChatbot(chatbot.id, 2, true);',
                                        ],
                                    ],
                                    [
                                        'label' => __('Train'),
                                        'link' => '#',
                                        'attrs' => [
                                            '@click.prevent' => 'setActiveChatbot(chatbot.id, 3);',
                                        ],
                                    ],
                                    [
                                        'label' => __('Test & Embed'),
                                        'link' => '#',
                                        'attrs' => [
                                            '@click.prevent' => 'setActiveChatbot(chatbot.id, 4, true);',
                                        ],
                                    ]
                                ];

								if (\App\Extensions\Chatbot\System\Helpers\ChatbotHelper::existChannels()){
									$dropdown_items[] = [
										'label' => __('Channel'),
                                        'link' => '#',
                                        'attrs' => [
                                            '@click.prevent' => 'setActiveChatbot(chatbot.id, 5, true);',
                                        ],
									];
								}

                            @endphp
                            <ul class="py-1 text-xs font-medium">
                                @foreach ($dropdown_items as $dropdown_item)
                                    <li>
                                        <a
                                            class="flex px-5 py-2 text-heading-foreground transition-colors hover:bg-heading-foreground/[3%]"
                                            href="{{ $dropdown_item['link'] }}"
                                            @foreach ($dropdown_item['attrs'] as $attr => $value)
                                            {{ $attr }}="{{ $value }}" @endforeach
                                        >
                                            @lang($dropdown_item['label'])
                                        </a>
                                    </li>
                                @endforeach
                                <li :class="{ 'opacity-50': submittingData, 'pointer-events-none': submittingData }">
                                    <x-forms.input
                                        class="h-[18px] w-[34px] [background-size:0.625rem]"
                                        class:label="py-2 px-5 flex-row-reverse justify-between text-xs font-medium text-heading-foreground hover:bg-heading-foreground/[3%]"
                                        label="{{ __('Activate') }}"
                                        type="checkbox"
                                        switcher
                                        ::id="`active-chatbot-${chatbot.id}`"
                                        ::checked="chatbot.active"
                                        @change="toggleChatbotActivation(chatbot.id);"
                                        x-model="chatbot.active"
                                        x-init="$el.closest('label').setAttribute('for', `active-chatbot-${chatbot.id}`)"
                                    />
                                </li>
                                <li :class="{ 'opacity-50': submittingData, 'pointer-events-none': submittingData }">
                                    <form
                                        action="{{ route('dashboard.chatbot.delete') }}"
                                        @submit.prevent="deleteChatbot"
                                    >
                                        <input
                                            type="hidden"
                                            :value="chatbot.id"
                                            name="id"
                                        >
                                        <x-button
                                            class="w-full justify-between rounded-none px-5 py-2 text-start text-xs font-medium text-heading-foreground hover:translate-y-0"
                                            variant="ghost"
                                            hover-variant="danger"
                                            type="submit"
                                        >
                                            @lang('Delete')
                                            <x-tabler-trash
                                                class="size-4"
                                                aria-hidden="true"
                                            />
                                        </x-button>
                                    </form>
                                </li>
                            </ul>
                        </x-slot:dropdown>
                    </x-dropdown.dropdown>
                </x-slot:head>

                <h3
                    class="mb-2.5"
                    x-text="chatbot.title"
                ></h3>
                <p
                    class="mb-2.5 text-sm font-medium text-heading-foreground/50"
                    x-data="{ diff: Math.floor((new Date() - new Date(chatbot.created_at)) / 1000) }"
                    x-init="if (Math.floor((new Date() - new Date(chatbot.created_at)) / 1000) < 60) { setInterval(() => { diff = Math.floor((new Date() - new Date(chatbot.created_at)) / 1000); }, 1000); }"
                >
                    @lang('Created')
                    <span
                        x-text="
                            diff < 60 ? diff + ' {{ __('seconds ago') }}' :
                            diff < 3600 ? (Math.floor(diff / 60) === 1 ? '1 {{ __('minute ago') }}' : Math.floor(diff / 60) + ' {{ __('minutes ago') }}') :
                            diff < 86400 ? (Math.floor(diff / 3600) === 1 ? '1 {{ __('hour ago') }}' : Math.floor(diff / 3600) + ' {{ __('hours ago') }}') :
                            Math.floor(diff / 86400) === 1 ? '1 {{ __('day ago') }}' : Math.floor(diff / 86400) + ' {{ __('days ago') }}'
                        "
                    ></span>
                </p>

                <div
                    class="inline-flex items-center gap-1.5 rounded-full border px-1.5 py-1 text-[12px] font-medium leading-none transition-all [&.lqd-active]:text-green-500 [&.lqd-passive]:bg-heading-foreground/5 [&.lqd-passive]:text-heading-foreground"
                    :class="{
                        'lqd-active': chatbot.active,
                        'lqd-passive': !chatbot.active
                    }"
                >
                    <x-tabler-check
                        class="size-4"
                        ::class="{ hidden: !chatbot.active }"
                    />
                    <span
                        class="inline-flex min-h-4 items-center"
                        :class="{ hidden: !chatbot.active }"
                    >
                        @lang('Active')
                    </span>
                    <span
                        class="inline-flex min-h-4 items-center"
                        :class="{ hidden: chatbot.active }"
                    >
                        @lang('Passive')
                    </span>
                </div>
            </x-card>
        </template>
    </div>
</div>
