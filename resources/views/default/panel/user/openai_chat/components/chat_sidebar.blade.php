@php
    $disable_actions = $app_is_demo && (isset($category) && ($category->slug == 'ai_vision' || $category->slug == 'ai_pdf' || $category->slug == 'ai_chat_image'));
    $trap_search = isset($trap_search) ? $trap_search : 'false';
@endphp
<x-card
    class="chats-list-container flex h-[inherit] w-full shrink-0 grow-0 flex-col overflow-hidden rounded-e-none border-e-0 max-md:absolute max-md:start-0 max-md:top-0 max-md:z-50 max-md:h-full max-md:overflow-hidden max-md:border-none max-md:bg-background/95 max-md:backdrop-blur-lg max-md:backdrop-saturate-150 max-md:transition-all max-md:duration-300 md:!flex"
    class:body="flex flex-col h-full"
    id="chats-list-container"
    size="none"
    ::class="{ 'active': mobileSidebarShow }"
>
    @auth
        <div class="chats-search flex gap-x-2 border-b p-5 max-xl:p-2.5 lg:h-20">
            <form
                class="chats-search-form relative grow"
                action="#"
            >
                <x-forms.input
                    class="navbar-search-input peer ps-10"
                    id="chat_search_word"
                    data-category-id="{{ $category->id }}"
                    data-website-url="{{ $website_url ?? null }}"
                    type="search"
                    onkeydown="return event.key != 'Enter';"
                    placeholder="{{ __('Search') }}"
                    aria-label="{{ __('Search in conversations') }}"
                    x-trap="{{ $trap_search }}"
                />
                <x-tabler-search class="pointer-events-none absolute start-3 top-1/2 size-5 -translate-y-1/2 opacity-80" />
            </form>
            <x-button
                class="lqd-chat-clear-all aspect-square min-w-10 shrink-0 bg-foreground/5 text-foreground"
                size="sm"
                hover-variant="danger"
                href="javascript:void(0);"
                onclick="{!! $app_is_demo ? 'return toastr.info(\'{{ __('This feature is disabled in Demo version.') }}\')' : 'return deleteAllConv(\'{{ isset($category) ? $category->id : 0 }}\')' !!}"
            >
                <x-tabler-trash class="size-5" />
            </x-button>
        </div>
        <div
            class="chats-list grow-0 overflow-hidden"
            id="chat_sidebar_container"
        >
            @if (view()->hasSection('chat_sidebar_list'))
                @yield('chat_sidebar_list')
            @else
                @include('panel.user.openai_chat.components.chat_sidebar_list', ['website_url' => $website_url ?? null])
            @endif
        </div>

        <div class="chats-new mt-auto flex gap-2 px-6 py-8 max-md:hidden">
            @if (view()->hasSection('chat_sidebar_actions'))
                @yield('chat_sidebar_actions')
            @else
                @if (isset($category) && $category->slug == 'ai_pdf')
                    <input
                        id="selectDocInput"
                        type="file"
                        style="display: none;"
                        accept="file/*, .pdf, .csv, .docx, .xlsx, .xls"
                    />
                    <x-button
                        class="lqd-upload-doc-trigger grow text-[0.8rem]"
                        href="javascript:void(0);"
                        onclick="return $('#selectDocInput').click();"
                    >
                        <x-tabler-plus class="size-4" />
                        {{ __('Upload Document') }}
                    </x-button>
                @else
                    <x-button
                        class="lqd-new-chat-trigger grow text-[0.8rem]"
                        href="javascript:void(0);"
                        onclick="{!! $disable_actions
                            ? 'return toastr.info(\'{{ __('This feature is disabled in Demo version.') }}\')'
                            : 'return startNewChat(\'{{ $category->id }}\', \'{{ LaravelLocalization::getCurrentLocale() }}\')' !!}"
                    >
                        <x-tabler-plus class="size-4" />
                        {{ __('New Conversation') }}
                    </x-button>
                @endif
            @endif
        </div>
    @else
        <div class="flex h-full w-full flex-col items-center justify-center p-5 text-center lg:p-7">
            <svg
                class="mx-auto mb-4"
                width="111"
                height="111"
                viewBox="0 0 111 111"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
            >
                <path
                    d="M55.5 111C86.1518 111 111 86.1518 111 55.5C111 24.8482 86.1518 0 55.5 0C24.8482 0 0 24.8482 0 55.5C0 86.1518 24.8482 111 55.5 111Z"
                    fill="hsl(var(--heading-foreground))"
                    fill-opacity="0.1"
                />
                <path
                    d="M88.7992 111H22.1992V39.22C25.3383 39.2165 28.3478 37.9679 30.5675 35.7483C32.7871 33.5286 34.0357 30.5191 34.0392 27.38H76.9592C76.9559 28.935 77.261 30.4753 77.8568 31.9116C78.4527 33.3479 79.3275 34.6519 80.4306 35.7479C81.5266 36.8513 82.8306 37.7264 84.267 38.3224C85.7035 38.9184 87.244 39.2235 88.7992 39.22V111Z"
                    fill="hsl(var(--background))"
                />
                <path
                    d="M65.1202 79.92H45.8802C44.6541 79.92 43.6602 80.914 43.6602 82.14C43.6602 83.3661 44.6541 84.36 45.8802 84.36H65.1202C66.3462 84.36 67.3402 83.3661 67.3402 82.14C67.3402 80.914 66.3462 79.92 65.1202 79.92Z"
                    fill="hsl(var(--heading-foreground))"
                    fill-opacity="0.1"
                />
                <path
                    d="M65.1202 48.84H45.8802C44.6541 48.84 43.6602 49.8339 43.6602 51.06C43.6602 52.286 44.6541 53.28 45.8802 53.28H65.1202C66.3462 53.28 67.3402 52.286 67.3402 51.06C67.3402 49.8339 66.3462 48.84 65.1202 48.84Z"
                    fill="hsl(var(--heading-foreground))"
                    fill-opacity="0.1"
                />
                <path
                    d="M71.78 88.8H39.22C37.9939 88.8 37 89.794 37 91.02C37 92.2461 37.9939 93.24 39.22 93.24H71.78C73.0061 93.24 74 92.2461 74 91.02C74 89.794 73.0061 88.8 71.78 88.8Z"
                    fill="hsl(var(--heading-foreground))"
                    fill-opacity="0.1"
                />
                <path
                    d="M71.78 57.72H39.22C37.9939 57.72 37 58.7139 37 59.94C37 61.166 37.9939 62.16 39.22 62.16H71.78C73.0061 62.16 74 61.166 74 59.94C74 58.7139 73.0061 57.72 71.78 57.72Z"
                    fill="hsl(var(--heading-foreground))"
                    fill-opacity="0.1"
                />
            </svg>

            <h3 class="mx-auto lg:w-9/12">
                <span class="opacity-40">
                    {{ __('Temporary Chat') }}
                </span>
                <br>
                {{ __('Login to save your chat history.') }}
            </h3>
        </div>
    @endauth
</x-card>
