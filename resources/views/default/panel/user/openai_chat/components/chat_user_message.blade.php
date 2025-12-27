@if ($message->input != null && (empty($message->shared_uuid) || isset($multi_model_message_pairs[$message->shared_uuid])))
    <div class="lqd-chat-user-bubble mb-2.5 flex max-w-full flex-row-reverse content-end gap-2 last:mb-0 lg:ms-auto">
        @php
            $avatarUrl = isset(Auth::user()->avatar) ? Auth::user()->avatar : url('/assets/img/auth/default-avatar.png');
            if (str_starts_with(Auth::user()->avatar, 'upload') || str_starts_with(Auth::user()->avatar, 'assets')) {
                $avatarUrl = '/' . Auth::user()->avatar;
            }
        @endphp
        <div class="lqd-chat-sender flex items-center gap-2.5">
            <span
                class="lqd-chat-avatar mt-0.5 inline-block size-6 shrink-0 rounded-full bg-cover bg-center"
                style="background-image: url('{{ url(custom_theme_url($avatarUrl)) }}')"
            ></span>
            <span class="lqd-chat-sender-name sr-only">
                @lang('You')
            </span>
        </div>
        <div
            class="chat-content-container group relative max-w-[calc(100%-64px)] rounded-[2em] bg-secondary text-secondary-foreground dark:bg-zinc-700 dark:text-primary-foreground">
            <div class="chat-content px-5 py-3.5 max-md:break-all">
                {{ $message->input }}
            </div>
            <div
                class="lqd-chat-actions-wrap pointer-events-auto invisible absolute -start-5 bottom-0 flex flex-col gap-2 leading-5 opacity-0 transition-all group-hover:!visible group-hover:!opacity-100">
                <div class="lqd-clipboard-copy-wrap group/copy-wrap flex flex-col gap-2 transition-all">
                    <button
                        class="lqd-clipboard-copy group/btn relative inline-flex size-10 items-center justify-center rounded-full border-none bg-white p-0 text-[12px] text-black shadow-lg transition-all hover:-translate-y-[2px] hover:scale-110"
                        data-copy-options='{ "content": ".chat-content", "contentIn": "<.chat-content-container" }'
                        title="{{ __('Copy to clipboard') }}"
                    >
                        <span
                            class="absolute end-full top-1/2 me-1 inline-block -translate-y-1/2 translate-x-1 whitespace-nowrap rounded-full bg-white px-3 py-1 font-medium leading-5 opacity-0 shadow-lg transition-all group-hover/btn:translate-x-0 group-hover/btn:opacity-100"
                        >
                            {{ __('Copy to clipboard') }}
                        </span>
                        <x-tabler-copy class="size-4" />
                    </button>
                </div>
            </div>
        </div>
    </div>

    @if ($message->pdfPath != null && $message->pdfPath != '')
        <div class="lqd-chat-user-bubble mb-2.5 flex !w-auto max-w-full flex-row-reverse content-end gap-2 !p-0 last:mb-0 lg:ms-auto lg:justify-self-end">
            <div class="chat-content-container group relative rounded-[2em] bg-secondary !px-3 !py-2.5 text-secondary-foreground dark:bg-zinc-700 dark:text-primary-foreground">
                <a
                    class="flex items-center gap-1.5 underline underline-offset-2"
                    href="{{ $message->pdfPath }}"
                    target="_blank"
                >
                    <svg
                        class="shrink-0 opacity-50"
                        width="15"
                        height="19"
                        viewBox="0 0 15 19"
                        fill="currentColor"
                        xmlns="http://www.w3.org/2000/svg"
                    >
                        <path
                            d="M1.66667 18.5C1.20833 18.5 0.815972 18.3042 0.489583 17.9125C0.163194 17.5208 0 17.05 0 16.5V2.5C0 1.95 0.163194 1.47917 0.489583 1.0875C0.815972 0.695833 1.20833 0.5 1.66667 0.5H10L15 6.5V16.5C15 17.05 14.8368 17.5208 14.5104 17.9125C14.184 18.3042 13.7917 18.5 13.3333 18.5H1.66667ZM3.33333 14.5H11.6667V12.5H3.33333V14.5ZM3.33333 10.5H11.6667V8.5H3.33333V10.5ZM3.33333 6.5H9.16667V4.5H3.33333V6.5Z"
                        />
                    </svg>
                    <span>
                        {{ $message->pdfName ?? basename($message->pdfPath) }}
                    </span>
                </a>
            </div>
        </div>
    @endif

    @if ($message->images != null)
        @foreach (explode(',', $message->images) as $image)
            <div class="lqd-chat-image-bubble mb-2 flex !w-auto max-w-[50%] flex-row-reverse content-end gap-2 !px-3 !py-2.5 last:mb-0 lg:ms-auto lg:justify-self-end">
                <a
                    class="flex items-center gap-1.5 underline underline-offset-2"
                    data-fslightbox="gallery"
                    data-type="image"
                    href="{{ $image }}"
                >
                    <img
                        class="img-content rounded-3xl"
                        loading="lazy"
                        src={{ $image }}
                    />
                </a>
            </div>
        @endforeach
    @endif
@endif
