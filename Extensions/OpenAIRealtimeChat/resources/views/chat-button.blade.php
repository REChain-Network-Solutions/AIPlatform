@php
    $compact = $compact ?? false;
    $messages = isset($messages) ? $messages : [];
    $category_slug = $category_slug ?? '';
@endphp

@if (setting('realtime_voice_chat', 0))
    <div @class([
        'pointer-events-auto flex items-center max-md:flex-col max-md:items-start max-md:gap-4' => $compact,
        'grid place-items-center absolute inset-0 pointer-events-none' => !$compact,
    ])>
        <x-button
            data-compact="{{ $compact ? 'true' : 'false' }}"
            @class([
                'lqd-realtime-chat-button group/realtime pointer-events-auto z-2',
                'data-[compact=true]:[&.conversation-not-started]:hidden data-[compact=false]:[&.conversation-started]:hidden' =>
                    $category_slug === 'ai_realtime_voice_chat',
                'conversation-not-started' => count($messages) <= 1,
                'conversation-started' => count($messages) > 1,
                'md:bg-secondary md:text-secondary-foreground md:hover:bg-primary md:hover:text-primary-foreground [&.active]:hover:bg-red-500 [&.active]:hover:text-white' => $compact,
                'flex size-48 flex-col items-center gap-3 bg-background/80 backdrop-blur-lg text-center text-heading-foreground shadow-[0_2px_55px_hsl(var(--heading-foreground)/10%)] hover:translate-y-0 hover:scale-105 hover:bg-background hover:text-heading-foreground active:scale-95 md:size-[200px] [&.active]:invisible [&.active]:scale-90 [&.active]:opacity-0 hover:shadow-[0_5px_65px_hsl(var(--heading-foreground)/7%)] motion-scale-in-[0.8] motion-blur-in-[12px] motion-duration-[0.45s] motion-ease-spring-bouncier' => !$compact,
            ])
            size="none"
            variant="ghost"
            x-data="openaiRealtime('{{ $apikeyPart1 }}', '{{ $apikeyPart2 }}', '{{ $apikeyPart3 }}')"
            @click.prevent="!$store.realtimeChatStatus.active ? start() : stop()"
            @audio-vis.window="$data[$event.detail.action]()"
        >
            <span @class([
                'grid place-content-center place-items-center',
                'size-6 md:size-10' => $compact,
                'size-14' => !$compact,
            ])>
                <svg
                    @class([
                        'col-start-1 col-end-1 row-start-1 row-end-1 transition-all',
                        'size-14 group-hover/realtime:scale-105 duration-200' => !$compact,
                    ])
                    width="22"
                    height="22"
                    viewBox="0 0 22 22"
                    fill="currentColor"
                    xmlns="http://www.w3.org/2000/svg"
                >
                    <defs>
                        <linearGradient
                            id="paint0_linear_8997_923"
                            x1="0.265625"
                            y1="11.905"
                            x2="45.8863"
                            y2="52.1425"
                            gradientUnits="userSpaceOnUse"
                        >
                            <stop stop-color="#82E2F4" />
                            <stop
                                offset="0.3"
                                stop-color="#8A8AED"
                            />
                            <stop
                                offset="0.7"
                                stop-color="#6977DE"
                            />
                        </linearGradient>
                    </defs>
                    <g @if (!$compact) fill="url(#paint0_linear_8997_923)" @endif>
                        <path
                            d="M11.75 7C11.75 6.58579 11.4142 6.25 11 6.25C10.5858 6.25 10.25 6.58579 10.25 7V15C10.25 15.4142 10.5858 15.75 11 15.75C11.4142 15.75 11.75 15.4142 11.75 15V7Z"
                        />
                        <path
                            d="M8.75 8C8.75 7.58579 8.41421 7.25 8 7.25C7.58579 7.25 7.25 7.58579 7.25 8L7.25 14C7.25 14.4142 7.58579 14.75 8 14.75C8.41421 14.75 8.75 14.4142 8.75 14V8Z"
                        />
                        <path
                            d="M14.75 8C14.75 7.58579 14.4142 7.25 14 7.25C13.5858 7.25 13.25 7.58579 13.25 8V14C13.25 14.4142 13.5858 14.75 14 14.75C14.4142 14.75 14.75 14.4142 14.75 14V8Z"
                        />
                        <path
                            d="M5.75 10C5.75 9.58579 5.41421 9.25 5 9.25C4.58579 9.25 4.25 9.58579 4.25 10V12C4.25 12.4142 4.58579 12.75 5 12.75C5.41421 12.75 5.75 12.4142 5.75 12V10Z"
                        />
                        <path
                            d="M17.75 10C17.75 9.58579 17.4142 9.25 17 9.25C16.5858 9.25 16.25 9.58579 16.25 10V12C16.25 12.4142 16.5858 12.75 17 12.75C17.4142 12.75 17.75 12.4142 17.75 12V10Z"
                        />
                        <path
                            fill-rule="evenodd"
                            clip-rule="evenodd"
                            d="M11 0.25C5.06294 0.25 0.25 5.06294 0.25 11C0.25 16.9371 5.06294 21.75 11 21.75C16.9371 21.75 21.75 16.9371 21.75 11C21.75 5.06294 16.9371 0.25 11 0.25ZM1.75 11C1.75 5.89137 5.89137 1.75 11 1.75C16.1086 1.75 20.25 5.89137 20.25 11C20.25 16.1086 16.1086 20.25 11 20.25C5.89137 20.25 1.75 16.1086 1.75 11Z"
                        />
                    </g>
                </svg>
                <svg
                    class="invisible col-start-1 col-end-1 row-start-1 row-end-1 -rotate-12 scale-50 opacity-0 transition-all duration-200 group-[&.active]/realtime:visible group-[&.active]/realtime:rotate-0 group-[&.active]/realtime:scale-100 group-[&.active]/realtime:opacity-100"
                    xmlns="http://www.w3.org/2000/svg"
                    width="36"
                    height="36"
                    viewBox="0 0 24 24"
                    stroke-width="1"
                    stroke="currentColor"
                    fill="none"
                    stroke-linecap="round"
                    stroke-linejoin="round"
                >
                    <path d="M7 5l10 14" />
                </svg>
            </span>
            @if ($compact)
                <span class="md:hidden">
                    @lang('Real-Time Chat')
                </span>
            @else
                <span class="text-xs font-medium">
                    @lang('Start Voice Chat')
                </span>
            @endif
        </x-button>
    </div>
@endif
