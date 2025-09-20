@php
    $model_label = '';
    try {
        $model_label = \App\Domains\Entity\Enums\EntityEnum::fromSlug($message->model_slug ?? '')->label();
    } catch (RuntimeException | Exception $exception) {
    }
@endphp

<div
    data-message-id="{{ $message->id }}"
    data-title="{{ $message->tiptapContent?->title ?? 'Document' }}"
    data-model="{{ $message->model_slug ?? '' }}"
    @class([
        'lqd-chat-ai-bubble mb-2.5 flex max-w-full content-start items-start gap-2 animating-words-done',
        'w-full' => $is_multi_model_message,
        'w-auto' => !$is_multi_model_message,
    ])
>
    @if ($message->output != null)
        <div class="lqd-chat-sender flex items-center gap-2.5">
            <span
                class="lqd-chat-avatar mt-0.5 inline-block size-6 shrink-0 rounded-full bg-cover bg-center"
                style="background-image: url('{{ !empty($chat->category?->image) ? custom_theme_url($chat->category?->image, true) : url(custom_theme_url('/assets/img/auth/default-avatar.png')) }}')"
            ></span>
            <span class="lqd-chat-sender-name sr-only">
                @lang('AI Assistant')
            </span>
        </div>
        <div class="chat-content-container group relative max-w-[calc(100%-64px)] rounded-[2em] bg-clay text-heading-foreground dark:bg-white/[2%]">
            @php
                $has_canvas_content = $canvas_enabled && $message->tiptapContent?->output;
                $output = $has_canvas_content ? $message->tiptapContent?->output : $message->output;

                $output = str_replace(['<br>', '<br/>', '<br >', '<br />'], "\n", $output);
                $output = str_replace('/http', 'http', $output);
            @endphp

            @if ($has_canvas_content && $is_chat_pro)
                <div class="mb-3 w-full">
                    <button
                        class="lqd-chat-bubble-canvas-trigger group/btn flex items-center gap-2 rounded-md border px-3 py-2 text-2xs font-medium transition-all hover:border-foreground hover:bg-foreground hover:text-background group-[&.loading]:pointer-events-none group-[&.streaming-on]:pointer-events-none group-[&.loading]:opacity-50 group-[&.streaming-on]:opacity-50"
                        type="button"
                        @click.prevent="setCanvasActive(true);"
                    >
                        <span
                            class="pointer-events-none inline-grid size-9 place-items-center rounded-full border-none bg-surface-background p-0 text-foreground shadow-lg shadow-black/5 transition-all group-hover/btn:scale-110 group-[&.loading]:scale-90 group-[&.streaming-on]:scale-90"
                        >
                            <x-tabler-pencil class="size-4" />
                        </span>
                        {{ __('Open in Canvas') }}
                    </button>
                </div>
            @endif

            @if ($is_multi_model_message && $is_chat_pro)
                <div class="multi-model-response-head mb-3 hidden w-full items-center gap-4">
                    <svg
                        class="shrink-0"
                        width="15"
                        height="14"
                        viewBox="0 0 15 14"
                        fill="currentColor"
                        xmlns="http://www.w3.org/2000/svg"
                    >
                        <path
                            d="M4.76586 11.495L5.08728 11.4297C5.1773 11.4117 5.25828 11.363 5.31647 11.292C5.37466 11.221 5.40645 11.132 5.40645 11.0402C5.40645 10.9484 5.37466 10.8594 5.31647 10.7884C5.25828 10.7174 5.1773 10.6688 5.08728 10.6507L4.76586 10.5854C4.36954 10.505 4.00569 10.3097 3.71974 10.0237C3.43379 9.7378 3.23842 9.37397 3.15801 8.97767L3.09275 8.65626C3.07471 8.56625 3.02605 8.48525 2.95503 8.42706C2.88402 8.36888 2.79504 8.3371 2.70323 8.3371C2.61142 8.3371 2.52245 8.36888 2.45143 8.42706C2.38042 8.48525 2.33175 8.56625 2.3137 8.65626L2.24844 8.97767C2.16804 9.37397 1.97266 9.7378 1.68671 10.0237C1.40076 10.3097 1.03692 10.505 0.640595 10.5854L0.319189 10.6507C0.229171 10.6688 0.148173 10.7174 0.0899825 10.7884C0.0317923 10.8594 0 10.9484 0 11.0402C0 11.132 0.0317923 11.221 0.0899825 11.292C0.148173 11.363 0.229171 11.4117 0.319189 11.4297L0.640595 11.495C1.03692 11.5754 1.40076 11.7708 1.68671 12.0567C1.97266 12.3426 2.16804 12.7065 2.24844 13.1028L2.3137 13.4242C2.33175 13.5142 2.38042 13.5952 2.45143 13.6534C2.52245 13.7116 2.61142 13.7433 2.70323 13.7433C2.79504 13.7433 2.88402 13.7116 2.95503 13.6534C3.02605 13.5952 3.07471 13.5142 3.09275 13.4242L3.15801 13.1028C3.23842 12.7065 3.43379 12.3426 3.71974 12.0567C4.00569 11.7708 4.36954 11.5754 4.76586 11.495Z"
                        />
                        <path
                            d="M12.5567 5.67479L13.7396 5.43497C13.8576 5.41083 13.9637 5.34666 14.0399 5.25332C14.1161 5.15998 14.1577 5.04318 14.1577 4.92269C14.1577 4.80221 14.1161 4.68542 14.0399 4.59208C13.9637 4.49873 13.8576 4.43457 13.7396 4.41042L12.5567 4.1706C11.9869 4.05496 11.4637 3.77405 11.0526 3.36291C10.6414 2.95178 10.3605 2.42865 10.2449 1.85884L10.005 0.67604C9.98131 0.557759 9.91735 0.451342 9.82403 0.374886C9.73071 0.29843 9.61379 0.256653 9.49315 0.256653C9.37251 0.256653 9.25559 0.29843 9.16228 0.374886C9.06896 0.451342 9.00499 0.557759 8.98126 0.67604L8.74143 1.85884C8.62589 2.4287 8.345 2.95188 7.93384 3.36303C7.52267 3.77418 6.99947 4.05506 6.42959 4.1706L5.24674 4.41042C5.12869 4.43457 5.02259 4.49873 4.9464 4.59208C4.87022 4.68542 4.8286 4.80221 4.8286 4.92269C4.8286 5.04318 4.87022 5.15998 4.9464 5.25332C5.02259 5.34666 5.12869 5.41083 5.24674 5.43497L6.42959 5.67479C6.99947 5.79032 7.52267 6.07121 7.93384 6.48236C8.345 6.89351 8.62589 7.4167 8.74143 7.98656L8.98126 9.16936C9.00499 9.28764 9.06896 9.39404 9.16228 9.4705C9.25559 9.54695 9.37251 9.58874 9.49315 9.58874C9.61379 9.58874 9.73071 9.54695 9.82403 9.4705C9.91735 9.39404 9.98131 9.28764 10.005 9.16936L10.2449 7.98656C10.3605 7.41674 10.6414 6.89361 11.0526 6.48248C11.4637 6.07135 11.9869 5.79042 12.5567 5.67479Z"
                        />
                    </svg>

                    <span class="multi-model-response-name inline-block max-w-full truncate text-[12px] font-medium underline underline-offset-4">
                        {{ $model_label }}
                    </span>

                    {{-- <div class="multi-model-response-actions contents shrink-0">
                        <x-button
                            class="multi-model-response-regenerate size-8 shrink-0 rounded-full p-0"
							data-message-id="{{ $message->id }}"
							data-model="{{ $message->model_slug ?? '' }}"
                            size="none"
                            variant="outline"
                        >
                            <x-tabler-rotate class="size-4" />
                        </x-button>
                    </div> --}}
                </div>
            @endif

            <pre @class([
                'chat-content prose relative w-full max-w-none !whitespace-pre-wrap px-6 py-3.5 indent-0 font-[inherit] text-xs font-normal text-current [word-break:break-word] empty:hidden [&_*]:text-current',
                'is-html' => $has_canvas_content,
            ])>{{ $has_canvas_content ? str()->of($output)->toHtmlString() : $output }}</pre>

            <div
                class="lqd-chat-actions-wrap pointer-events-auto invisible absolute -end-5 bottom-0 flex flex-col gap-2 opacity-0 transition-all group-hover:!visible group-hover:!opacity-100">
                <div class="lqd-clipboard-copy-wrap group/copy-wrap flex flex-col gap-2 transition-all">
                    <button
                        class="lqd-clipboard-copy group/btn relative inline-flex size-10 items-center justify-center rounded-full border-none bg-white p-0 text-[12px] text-black shadow-lg transition-all hover:-translate-y-[2px] hover:scale-110"
                        data-copy-options='{ "content": ".chat-content", "contentIn": "<.chat-content-container" }'
                        title="{{ __('Copy to clipboard') }}"
                    >
                        <span
                            class="pointer-events-none absolute end-full top-1/2 me-1 inline-block -translate-y-1/2 translate-x-1 whitespace-nowrap rounded-full bg-white px-3 py-1 font-medium leading-5 opacity-0 shadow-lg transition-all group-hover/btn:translate-x-0 group-hover/btn:opacity-100"
                        >
                            {{ __('Copy to clipboard') }}
                        </span>
                        <x-tabler-copy class="size-4" />
                    </button>
                </div>

                @if (\App\Helpers\Classes\MarketplaceHelper::isRegistered('canvas') && (bool) setting('ai_chat_pro_canvas', 1) && $is_chat_pro)
                    <button
                        class="lqd-chat-bubble-canvas-trigger group/btn inline-flex size-10 items-center justify-center rounded-full border-none bg-white p-0 text-[12px] text-black shadow-lg transition-all hover:-translate-y-[2px] hover:scale-110"
                        type="button"
                        @click.prevent="setCanvasActive(true);"
                    >
                        <span
                            class="pointer-events-none absolute end-full top-1/2 me-1 inline-block -translate-y-1/2 translate-x-1 whitespace-nowrap rounded-full bg-white px-3 py-1 font-medium leading-5 opacity-0 shadow-lg transition-all group-hover/btn:translate-x-0 group-hover/btn:opacity-100"
                        >
                            {{ __('Open in Canvas') }}
                        </span>
                        <x-tabler-edit class="size-4" />
                    </button>
                @endif
            </div>

            @if ($is_multi_model_message && $is_chat_pro)
                <div class="multi-model-response-foot hidden">
                    <x-button
                        class="multi-model-response-accept mt-3 p-0 text-[12px] underline underline-offset-4"
                        data-message-id="{{ $message->id }}"
                        data-model="{{ $message->model_slug ?? '' }}"
                        size="none"
                        variant="none"
                    >
                        <x-tabler-thumb-up class="size-4" />
                        {{ __('I prefer this response') }}
                    </x-button>
                </div>
            @endif
        </div>
    @endif
</div>

@if ($message->outputImage != null && $message->outputImage != '')
    <div class="lqd-chat-image-bubble mb-2.5 flex gap-2 lg:ms-auto">
        <div class="flex w-4/5 justify-end rounded-[2em] text-heading-foreground md:w-1/2">
            <a
                data-fslightbox="gallery"
                data-type="image"
                href="{{ $message->outputImage }}"
            >
                <img
                    class="img-content rounded-3xl"
                    loading="lazy"
                    src={{ $message->outputImage }}
                />
            </a>
        </div>
    </div>
@endif
