@php
    $councilResponse = is_array($message->council_response ?? null) ? $message->council_response : [];
    $meta = (array) ($councilResponse['meta'] ?? []);
    $labels = (array) ($meta['labels'] ?? []);
    $titles = (array) ($councilResponse['titles'] ?? []);
    $confidencePercent = (int) ($meta['confidence_percent'] ?? ($councilResponse['confidence_percent'] ?? 0));
    $respondedModelsCount = (int) ($meta['responded_models_count'] ?? ($councilResponse['responded_models_count'] ?? 0));
    $respondedModelsLabel = (string) ($labels['responded_models'] ?? __('Models Responded'));
    $confidenceLabel = (string) ($labels['confidence'] ?? __('Confidence'));

    $agreementAnalysis = collect((array) ($councilResponse['agreement_analysis'] ?? []))->filter(fn($item) => is_string($item) && trim($item) !== '')->values();
    $disagreements = collect((array) ($councilResponse['disagreements'] ?? []))->filter(fn($item) => is_string($item) && trim($item) !== '')->values();
    $uniqueDiscoveries = collect((array) ($councilResponse['unique_discoveries'] ?? []))->filter(fn($item) => is_string($item) && trim($item) !== '')->values();

    $councilReplies = $message->relationLoaded('councilResponses') ? $message->councilResponses : $message->councilResponses()->get();

    $agreementRows = (array) ($councilResponse['agreement_table_rows'] ?? []);
@endphp

<div
    class="lqd-chat-ai-bubble model-council-bubble animating-words-done mb-2.5 flex max-w-full content-start items-start gap-2"
    data-message-id="{{ $message->id }}"
    data-model="{{ $message->model_slug ?? '' }}"
>
    <div class="lqd-chat-sender flex items-center gap-2.5">
        <span
            class="lqd-chat-avatar mt-0.5 inline-block size-6 shrink-0 rounded-full bg-cover bg-center"
            style="background-image: url('{{ !empty($chat->category?->image) ? custom_theme_url($chat->category?->image, true) : url(custom_theme_url('/assets/img/auth/default-avatar.png')) }}')"
        ></span>
        <span class="lqd-chat-sender-name sr-only">
            {{ __($chat?->category?->name ?? 'AI Assistant') }}
        </span>
    </div>

    <div
        class="council-content-container chat-content-container group relative max-w-[calc(100%-64px)] rounded-[2em] bg-clay px-6 py-3.5 text-heading-foreground dark:bg-white/[2%]">
        @if ($respondedModelsCount > 0 || $confidencePercent > 0)
            <p class="mb-4 text-2xs text-heading-foreground/60">
                @if ($respondedModelsCount > 0)
                    <span>{{ $respondedModelsCount }} {{ $respondedModelsLabel }}</span>
                @endif
                @if ($respondedModelsCount > 0 && $confidencePercent > 0)
                    <span class="mx-1">•</span>
                @endif
                @if ($confidencePercent > 0)
                    <span>{{ $confidenceLabel }} {{ $confidencePercent }}%</span>
                @endif
            </p>
        @endif

        @if ($message->output != null)
            <h4 class="mb-2 text-sm font-semibold">
                {{ $titles['final_answer'] ?? __('Final Synthesized Answer') }}
            </h4>
            <pre
                class="model-council-final-answer chat-content prose relative w-full max-w-none indent-0 font-[inherit] text-xs font-normal text-current [word-break:break-word] empty:hidden [&_*]:text-current">{{ str_replace(['<br>', '<br/>', '<br >', '<br />'], "\n", (string) $message->output) }}</pre>
        @endif

        @if (count($agreementRows) > 0)
            <div class="mb-4 overflow-hidden rounded-xl border border-heading-foreground/10">
                <div class="grid grid-cols-2 border-b border-heading-foreground/10 bg-heading-foreground/[0.03] px-4 py-2 text-2xs font-semibold">
                    <span>{{ $titles['agreement_level'] ?? __('Agreement Level') }}</span>
                    <span>{{ $titles['confidence_impact'] ?? __('Confidence Impact') }}</span>
                </div>
                @foreach ($agreementRows as $row)
                    <div class="grid grid-cols-2 border-t border-heading-foreground/10 px-4 py-2 text-2xs">
                        <span>{{ $row['level'] ?? '' }}</span>
                        <span>{{ $row['impact'] ?? '' }}</span>
                    </div>
                @endforeach
            </div>
        @endif

        @if ($agreementAnalysis->isNotEmpty() || $disagreements->isNotEmpty() || $uniqueDiscoveries->isNotEmpty())
            <div class="space-y-4 text-xs">
                @if ($agreementAnalysis->isNotEmpty())
                    <div>
                        <h5 class="mb-1 text-xs font-semibold">{{ $titles['agreement_analysis'] ?? __('Agreement Analysis') }}</h5>
                        <ul class="m-0 space-y-1 text-heading-foreground/80">
                            @foreach ($agreementAnalysis as $item)
                                <li class="ms-4 list-disc">{{ $item }}</li>
                            @endforeach
                        </ul>
                    </div>
                @endif

                @if ($disagreements->isNotEmpty())
                    <div>
                        <h5 class="mb-1 text-xs font-semibold">{{ $titles['disagreements'] ?? __('Models Disagree About') }}</h5>
                        <ul class="m-0 space-y-1 text-heading-foreground/80">
                            @foreach ($disagreements as $item)
                                <li class="ms-4 list-disc">{{ $item }}</li>
                            @endforeach
                        </ul>
                    </div>
                @endif

                @if ($uniqueDiscoveries->isNotEmpty())
                    <div>
                        <h5 class="mb-1 text-xs font-semibold">{{ $titles['discoveries'] ?? __('Unique Discoveries') }}</h5>
                        <ul class="m-0 space-y-1 text-heading-foreground/80">
                            @foreach ($uniqueDiscoveries as $item)
                                <li class="ms-4 list-disc">{{ $item }}</li>
                            @endforeach
                        </ul>
                    </div>
                @endif
            </div>
        @endif

        @if ($councilReplies->isNotEmpty())
            <div class="mt-5 w-full">
                <h5 class="mb-4 text-sm font-semibold">
                    {{ $titles['model_replies'] ?? __('Model Replies') }}
                </h5>
                <div class="model-council-replies w-full space-y-4">
                    @foreach ($councilReplies as $reply)
                        <div class="prose prose-sm w-full max-w-none rounded-[20px] border border-foreground/5 p-5.5 transition-border dark:prose-invert">
                            @php
                                $replyText = $reply->failed ? __('Failed to Respond.') : (string) $reply->response_text;
                                $shortReplyText = (string) str($replyText)->limit(320);
                                $isLongReply = str($replyText)->length() > 320;
                            @endphp
                            <p class="mb-3 flex items-center gap-4">
                                {{-- blade-formatter-disable --}}
								<svg class="shrink-0" width="15" height="14" viewBox="0 0 15 14" fill="currentColor" xmlns="http://www.w3.org/2000/svg" > <path d="M4.76586 11.495L5.08728 11.4297C5.1773 11.4117 5.25828 11.363 5.31647 11.292C5.37466 11.221 5.40645 11.132 5.40645 11.0402C5.40645 10.9484 5.37466 10.8594 5.31647 10.7884C5.25828 10.7174 5.1773 10.6688 5.08728 10.6507L4.76586 10.5854C4.36954 10.505 4.00569 10.3097 3.71974 10.0237C3.43379 9.7378 3.23842 9.37397 3.15801 8.97767L3.09275 8.65626C3.07471 8.56625 3.02605 8.48525 2.95503 8.42706C2.88402 8.36888 2.79504 8.3371 2.70323 8.3371C2.61142 8.3371 2.52245 8.36888 2.45143 8.42706C2.38042 8.48525 2.33175 8.56625 2.3137 8.65626L2.24844 8.97767C2.16804 9.37397 1.97266 9.7378 1.68671 10.0237C1.40076 10.3097 1.03692 10.505 0.640595 10.5854L0.319189 10.6507C0.229171 10.6688 0.148173 10.7174 0.0899825 10.7884C0.0317923 10.8594 0 10.9484 0 11.0402C0 11.132 0.0317923 11.221 0.0899825 11.292C0.148173 11.363 0.229171 11.4117 0.319189 11.4297L0.640595 11.495C1.03692 11.5754 1.40076 11.7708 1.68671 12.0567C1.97266 12.3426 2.16804 12.7065 2.24844 13.1028L2.3137 13.4242C2.33175 13.5142 2.38042 13.5952 2.45143 13.6534C2.52245 13.7116 2.61142 13.7433 2.70323 13.7433C2.79504 13.7433 2.88402 13.7116 2.95503 13.6534C3.02605 13.5952 3.07471 13.5142 3.09275 13.4242L3.15801 13.1028C3.23842 12.7065 3.43379 12.3426 3.71974 12.0567C4.00569 11.7708 4.36954 11.5754 4.76586 11.495Z" /> <path d="M12.5567 5.67479L13.7396 5.43497C13.8576 5.41083 13.9637 5.34666 14.0399 5.25332C14.1161 5.15998 14.1577 5.04318 14.1577 4.92269C14.1577 4.80221 14.1161 4.68542 14.0399 4.59208C13.9637 4.49873 13.8576 4.43457 13.7396 4.41042L12.5567 4.1706C11.9869 4.05496 11.4637 3.77405 11.0526 3.36291C10.6414 2.95178 10.3605 2.42865 10.2449 1.85884L10.005 0.67604C9.98131 0.557759 9.91735 0.451342 9.82403 0.374886C9.73071 0.29843 9.61379 0.256653 9.49315 0.256653C9.37251 0.256653 9.25559 0.29843 9.16228 0.374886C9.06896 0.451342 9.00499 0.557759 8.98126 0.67604L8.74143 1.85884C8.62589 2.4287 8.345 2.95188 7.93384 3.36303C7.52267 3.77418 6.99947 4.05506 6.42959 4.1706L5.24674 4.41042C5.12869 4.43457 5.02259 4.49873 4.9464 4.59208C4.87022 4.68542 4.8286 4.80221 4.8286 4.92269C4.8286 5.04318 4.87022 5.15998 4.9464 5.25332C5.02259 5.34666 5.12869 5.41083 5.24674 5.43497L6.42959 5.67479C6.99947 5.79032 7.52267 6.07121 7.93384 6.48236C8.345 6.89351 8.62589 7.4167 8.74143 7.98656L8.98126 9.16936C9.00499 9.28764 9.06896 9.39404 9.16228 9.4705C9.25559 9.54695 9.37251 9.58874 9.49315 9.58874C9.61379 9.58874 9.73071 9.54695 9.82403 9.4705C9.91735 9.39404 9.98131 9.28764 10.005 9.16936L10.2449 7.98656C10.3605 7.41674 10.6414 6.89361 11.0526 6.48248C11.4637 6.07135 11.9869 5.79042 12.5567 5.67479Z" /> </svg>
								{{-- blade-formatter-enable --}}

                                <span class="inline-block max-w-full truncate text-[12px] font-medium underline underline-offset-4">
                                    {{ $reply->model_label ?: $reply->model_slug }}
                                </span>
                            </p>
                            @if ($isLongReply)
                                <div
                                    class="[interpolate-size:allow-keywords]"
                                    x-data="{ expanded: false }"
                                >
                                    <div
                                        class="model-council-reply-content m-0 h-[6lh] overflow-hidden transition-all [mask-image:linear-gradient(to_top,transparent,black_1.5lh)] [&.expanded]:h-auto [&.expanded]:[mask-image:none]"
                                        :class="{ 'expanded': expanded }"
                                    >
                                        {{ $replyText }}
                                    </div>
                                    <button
                                        class="mt-4 w-fit cursor-pointer rounded-full border px-3 py-1.5 text-3xs font-medium text-primary transition hover:border-primary hover:bg-primary hover:text-primary-foreground"
                                        @click="expanded = !expanded"
                                        x-text="expanded ? '{{ __('Show Less') }}' : '{{ __('View Full Response') }}'"
                                    ></button>
                                </div>
                            @else
                                <div class="model-council-reply-content m-0">
                                    {{ $replyText }}
                                </div>
                            @endif
                        </div>
                    @endforeach
                </div>
            </div>
        @endif

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
        </div>
    </div>
</div>
