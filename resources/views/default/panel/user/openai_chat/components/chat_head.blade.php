@php
    use App\Domains\Entity\Enums\EntityEnum;

    $currentUrl = url()->current();
    $previousUrl = url()->previous();

    $is_chat_pro =
        \App\Helpers\Classes\MarketplaceHelper::isRegistered('ai-chat-pro') &&
        (route('dashboard.user.openai.chat.pro.index') === $currentUrl ||
            route('chat.pro') === $currentUrl ||
            route('dashboard.user.openai.chat.pro.index') === $previousUrl ||
            route('chat.pro') === $previousUrl);
    $is_social_media_agent_chat =
        \App\Helpers\Classes\MarketplaceHelper::isRegistered('social-media-agent') &&
        Route::has('dashboard.user.social-media.agent.chat.index') &&
        (route('dashboard.user.social-media.agent.chat.index') === $currentUrl || route('dashboard.user.social-media.agent.chat.index') === $previousUrl);

    $isOtherCategories = isset($category) && ($category->slug == 'ai_vision' || $category->slug == 'ai_pdf' || $category->slug == 'ai_chat_image');
    $realtimeHiddenIn = ['ai_pdf', 'ai_vision', 'ai_chat_image'];
@endphp

<div
    class="lqd-chat-head sticky -top-px z-30 flex min-h-20 items-center justify-between gap-2 rounded-se-[inherit] border-b bg-background/80 px-5 py-3 backdrop-blur-lg backdrop-saturate-150 [interpolate-size:allow-keywords] max-md:bg-background/95 max-md:px-4">
    <div class="flex flex-col items-start justify-center text-sm">
        @include('panel.user.openai_chat.components.chat_category_dropdown')
    </div>

    <div class="flex grow items-center justify-end gap-4">
        <div class="flex flex-wrap items-center justify-end gap-2">
            <div
                class="lqd-chat-mobile-more-options-wrap max-md:absolute max-md:inset-x-0 max-md:top-full max-md:z-10 max-md:h-0 max-md:max-h-[var(--chats-container-height,400px)] max-md:overflow-y-auto max-md:border-y max-md:bg-background max-md:transition-all md:contents max-md:[&.active]:h-auto"
                :class="{ 'active': typeof mobileHeaderMoreOptionsShow !== 'undefined' && mobileHeaderMoreOptionsShow }"
            >
                <div class="contents max-md:flex max-md:flex-col max-md:gap-4 max-md:px-4 max-md:py-6">
                    @includeFirst(['chat-share::share-button-include', 'panel.user.openai_chat.includes.share-button-include', 'vendor.empty'])

                    @if (view()->hasSection('chat_head_actions'))
                        @yield('chat_head_actions')
                    @else
                        @auth
                            @if (!$is_chat_pro && !$is_social_media_agent_chat)
                                <div x-data="realtimeToggle">
                                    <x-forms.input
                                        class="max-md:hidden"
                                        class:label="w-full text-xs font-medium text-foreground gap-2 justify-normal"
                                        id="realtime"
                                        container-class="{{ in_array($category->slug, $realtimeHiddenIn, true) ? 'hidden' : 'flex' }}"
                                        label="{{ __('Real-Time Data') }}"
                                        type="checkbox"
                                        name="realtime"
                                        @change="handleRealtimeChange($event)"
                                        switcher
                                    >
                                        <x-tabler-world-download
                                            class="order-first size-5 md:hidden"
                                            stroke-width="1.5"
                                        />

                                        <x-tabler-check class="invisible order-last size-5 peer-checked:visible md:hidden" />
                                    </x-forms.input>
                                </div>
                            @endif
                        @else
                            <x-forms.input
                                class="max-md:hidden"
                                class:label="text-xs font-medium text-foreground flex-row-reverse gap-2"
                                id="realtime"
                                container-class="{{ in_array($category->slug, $realtimeHiddenIn, true) ? 'hidden' : 'flex' }}"
                                label="{{ __('Real-Time Data') }}"
                                type="checkbox"
                                name="realtime"
                                onchange="toastr.warning('{{ __('Login to use Real-Time search') }}'); document.querySelector('#realtime').checked = false; return false;"
                                switcher
                            >
                                <x-tabler-world-download
                                    class="size-5 md:hidden"
                                    stroke-width="1.5"
                                />
                            </x-forms.input>
                        @endauth
                    @endif
                    <div
                        class="group relative inline-flex flex-wrap items-center self-center max-md:w-full md:justify-center"
                        id="show_export_btns"
                        x-data="{ show: false }"
                    >
                        <x-button
                            class="p-0 text-xs font-medium max-md:w-full max-md:justify-start max-md:!border-none max-md:!bg-transparent max-md:p-0 max-md:text-start max-md:!text-foreground max-md:!shadow-none"
                            variant="none"
                            @click.prevent="show = !show"
                        >
                            <x-tabler-clipboard-copy class="size-5 md:size-6 md:[stroke-width:1.5]" />
                            <span class="md:hidden">
                                {{ __('Export') }}
                            </span>

                            <x-tabler-chevron-down
                                class="hidden size-4 transition max-md:block"
                                ::class="{ 'rotate-180': show }"
                            />
                        </x-button>
                        <div
                            class="flex rounded-lg transition-all group-focus-within:visible group-focus-within:translate-y-0 group-focus-within:scale-100 group-focus-within:opacity-100 group-hover:visible group-hover:translate-y-0 group-hover:scale-100 group-hover:opacity-100 max-md:h-0 max-md:w-full max-md:overflow-clip md:invisible md:absolute md:-end-4 md:bottom-full md:translate-y-2 md:scale-95 md:items-center md:justify-center md:bg-primary md:text-primary-foreground md:opacity-0 max-md:[&.active]:h-auto"
                            id="export_btns"
                            :class="{ 'active': show }"
                        >
                            <div class="max-md:flex max-md:flex-col max-md:gap-2 max-md:pt-3 md:contents">
                                <button
                                    class="chat-download flex items-center gap-2 border-none py-1 text-xs font-medium max-md:justify-start max-md:border-none max-md:text-start md:px-3 md:text-3xs"
                                    id="export_pdf"
                                    data-doc-type="pdf"
                                >
                                    <x-tabler-file-type-pdf class="size-5 md:hidden" />
                                    {{ __('PDF') }}
                                </button>
                                <button
                                    class="chat-download flex items-center gap-2 border-x border-x-primary-foreground/20 py-1 text-xs font-medium max-md:justify-start max-md:border-none max-md:text-start md:px-2.5 md:text-3xs"
                                    id="export_word"
                                    data-doc-type="doc"
                                >
                                    <x-tabler-file-type-doc class="size-5 md:hidden" />
                                    {{ __('Word') }}
                                </button>
                                <button
                                    class="chat-download flex items-center gap-2 py-1 text-xs font-medium max-md:justify-start max-md:border-none max-md:text-start md:px-3 md:text-3xs"
                                    id="export_txt"
                                    data-doc-type="txt"
                                >
                                    <x-tabler-file-type-txt class="size-5 md:hidden" />
                                    {{ __('Txt') }}
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            @if (view()->hasSection('chat_sidebar_actions'))
                @yield('chat_sidebar_actions')
            @else
                @if (isset($category) && $category->slug == 'ai_pdf')
                    {{-- #selectDocInput is present in chat_sidebar component. no need to duplicate it here --}}
                    <x-button
                        class="lqd-upload-doc-trigger group size-8 shrink-0 grid-flow-row place-items-center rounded-full shadow-md max-md:grid md:hidden"
                        variant="none"
                        size="none"
                        href="javascript:void(0);"
                        onclick="return $('#selectDocInput').click();"
                    >
                        <x-tabler-plus class="size-5" />
                        <span class="sr-only">
                            {{ __('Upload Document') }}
                        </span>
                    </x-button>
                @else
                    <x-button
                        class="lqd-new-chat-trigger group size-8 shrink-0 grid-flow-row place-items-center rounded-full shadow-md max-md:grid md:hidden"
                        variant="none"
                        size="none"
                        href="javascript:void(0);"
                        onclick="{!! $disable_actions
                            ? 'return toastr.info(\'{{ __('This feature is disabled in Demo version.') }}\')'
                            : (auth()->check()
                                ? 'return startNewChat(\'{{ $category?->id }}\', \'{{ LaravelLocalization::getCurrentLocale() }}\', \'chatpro\')'
                                : 'return window.location.reload();') !!}"
                    >
                        <x-tabler-plus class="size-5" />
                        <span class="sr-only">
                            {{ __('New Conversation') }}
                        </span>
                    </x-button>
                @endif

                @if ($is_chat_pro)
                    <div
                        class="lqd-chat-mobile-prompt-nav-trigger self-center md:hidden"
                        x-show="typeof promptNavVisible !== 'undefined' ? promptNavVisible : false"
                        x-cloak
                    >
                        <button
                            class="group grid size-8 shrink-0 place-items-center rounded-full shadow-md"
                            :class="{ 'active': typeof mobilePromptNavShow !== 'undefined' && mobilePromptNavShow }"
                            @click.prevent="toggleMobilePromptNav()"
                            type="button"
                        >
                            <x-tabler-list-numbers
                                class="col-start-1 row-start-1 size-5 transition-all group-[&.active]:rotate-45 group-[&.active]:scale-75 group-[&.active]:opacity-0"
                            />
                            <x-tabler-x
                                class="col-start-1 row-start-1 size-4.5 -rotate-45 opacity-0 transition-all group-[&.active]:rotate-0 group-[&.active]:!opacity-100"
                                stroke-width="2"
                            />
                        </button>
                    </div>
                @endif

                @if (!$isOtherCategories && $is_chat_pro)
                    <x-button
                        class="lqd-mobile-model-modal-trigger group size-8 shrink-0 grid-flow-row place-items-center rounded-full shadow-md max-md:grid md:hidden"
                        variant="none"
                        size="none"
                        x-data="{}"
                        @click.prevent="document.querySelector('.select-ai-model-modal') && Alpine.$data(document.querySelector('.select-ai-model-modal')).toggleModal()"
                        ::title="$store.modelList?.selectedModelLabel || '{{ __('None') }}'"
                    >
                        <x-tabler-brand-openai class="size-5" />
                    </x-button>
                @endif

                <div class="lqd-chat-mobile-sidebar-trigger self-center">
                    <button
                        class="group size-8 shrink-0 grid-flow-row place-items-center rounded-full shadow-md max-md:grid md:hidden"
                        :class="{ 'active': mobileSidebarShow }"
                        @click.prevent="toggleMobileSidebar"
                        type="button"
                        title="{{ __('Chat history') }}"
                    >
                        <x-tabler-history class="col-start-1 row-start-1 size-5 transition-all group-[&.active]:rotate-45 group-[&.active]:scale-75 group-[&.active]:opacity-0" />
                        <x-tabler-x
                            class="col-start-1 row-start-1 size-4.5 -rotate-45 opacity-0 transition-all group-[&.active]:rotate-0 group-[&.active]:!opacity-100"
                            stroke-width="2"
                        />
                    </button>
                </div>
            @endif

            <button
                class="lqd-chat-mobile-more-trigger group size-8 shrink-0 grid-flow-row place-items-center rounded-full shadow-md max-md:grid md:hidden"
                :class="{ 'active': typeof mobileHeaderMoreOptionsShow !== 'undefined' && mobileHeaderMoreOptionsShow }"
                @click.prevent="toggleMobileHeaderMoreOptions"
                type="button"
                title="{{ __('More options') }}"
            >
                <x-tabler-dots class="col-start-1 row-start-1 size-5 transition-all group-[&.active]:rotate-45 group-[&.active]:scale-75 group-[&.active]:opacity-0" />
                <x-tabler-x
                    class="col-start-1 row-start-1 size-4.5 -rotate-45 opacity-0 transition-all group-[&.active]:rotate-0 group-[&.active]:!opacity-100"
                    stroke-width="2"
                />
            </button>
        </div>
    </div>
</div>

@pushOnce('script')
    <script>
        document.addEventListener('alpine:init', () => {
            Alpine.data('realtimeToggle', () => ({
                init() {},
                handleRealtimeChange(event) {
                    if (event.target.checked) {
                        toastr.success('{{ __('Real-Time data activated') }}');
                    } else {
                        toastr.warning('{{ __('Real-Time data deactivated') }}');
                    }
                }
            }));
        });
    </script>
@endPushOnce
