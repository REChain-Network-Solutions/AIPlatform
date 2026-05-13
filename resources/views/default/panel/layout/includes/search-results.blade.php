@php
    $auth = Auth::user();
    $plan = $auth->activePlan();
    $plan_type = 'regular';
    $upgrade = false;
@endphp
@if (count($template_search) > 0)
    <ul class="font-medium">
        @foreach ($template_search as $item)
            @php
                $upgrade = false;
                if ($app_is_demo) {
                    if ($item->premium == 1 && $plan_type === 'regular') {
                        $upgrade = true;
                    }
                } else {
                    if (!$auth->isAdmin() && $item->premium == 1 && $plan_type === 'regular') {
                        $upgrade = true;
                    }
                }
                if ($upgrade) {
                    $href = 'dashboard.user.payment.subscription';
                } else {
                    $href = 'dashboard.user.openai.generator.workbook';
                    switch ($item->type) {
                        case 'image':
                        case 'video':
                        case 'audio':
                        case 'isolator':
                        case 'voiceover':
                            $href = 'dashboard.user.openai.generator';
                            break;
                    }
                    switch ($item->slug) {
                        case 'ai_webchat':
                            $href = 'dashboard.user.openai.webchat.workbook';
                            break;
                        case 'ai_rewriter':
                            $href = 'dashboard.user.openai.rewriter';
                            break;
                    }
                }
            @endphp
            <li class="border-b px-3 py-2 transition-colors last:border-b-0 hover:bg-foreground/5">
                <a
                    class="flex items-center gap-2 text-heading-foreground"
                    href="{{ route($href, $item->slug) }}"
                >
                    <x-lqd-icon
                        size="lg"
                        style="background: {{ $item->color }}"
                    >
                        <span class="flex size-5">
                            @if ($item->image !== 'none')
                                {!! html_entity_decode($item->image) !!}
                            @endif

                            @if ($item->active == 1)
                                <span class="absolute bottom-0 end-0 inline-block size-3 rounded-full border-2 border-background bg-green-500"></span>
                            @else
                                <span class="absolute bottom-0 end-0 inline-block size-3 rounded-full border-2 border-background bg-red-500"></span>
                            @endif
                        </span>
                    </x-lqd-icon>
                    {{ $item->title }}
                    <small class="ms-auto text-foreground/50">{{ __('Template') }}</small>
                </a>
            </li>
        @endforeach
    </ul>
@endif

@if (count($ai_chat_search) > 0)
    <ul class="font-medium">
        @foreach ($ai_chat_search as $item)
            <li class="border-b px-3 py-2 transition-colors last:border-b-0 hover:bg-foreground/10">
                <a
                    class="flex items-center gap-2 text-heading-foreground"
                    href="{{ route('dashboard.user.openai.chat.chat', $item->slug) }}"
                >
                    <x-lqd-icon
                        size="lg"
                        style="background: {{ $item->color }}"
                    >
                        <span class="flex size-5">
                            @if ($item->slug == 'ai-chat-bot')
                                <x-tabler-messages class="size-5" />
                            @else
                                {{ $item->short_name }}
                            @endif
                        </span>
                    </x-lqd-icon>
                    {{ $item->name }}
                    <small class="ms-auto text-foreground/50">{{ __('AI Chat Template') }}</small>
                </a>
            </li>
        @endforeach
    </ul>
@endif

@if (count($video_search) > 0)
    <ul class="font-medium">
        @foreach ($video_search as $item)
            @php
                $words = explode(' ', trim($item->prompt ?? ''));
                $videoTitle = implode(' ', array_slice($words, 0, 6));
                if (count($words) > 6) {
                    $videoTitle .= '...';
                }
                $videoSlug = \Illuminate\Support\Str::slug($videoTitle ?: __('Untitled Video'));
            @endphp
            <li class="border-b px-3 py-2 transition-colors last:border-b-0 hover:bg-foreground/5">
                <a
                    class="flex items-center gap-2 text-heading-foreground"
                    href="{{ route('dashboard.user.openai.documents.single', $videoSlug) }}"
                >
                    <x-lqd-icon
                        size="lg"
                        style="background: darkgrey"
                    >
                        <span class="flex size-5">
                            <x-tabler-video class="size-5" />
                        </span>
                    </x-lqd-icon>
                    {{ \Illuminate\Support\Str::limit($item->prompt, 50) }}
                    <small class="ms-auto text-foreground/50">{{ __('Video') }}</small>
                </a>
            </li>
        @endforeach
    </ul>
@endif

@if (count($ai_image_pro_search) > 0)
    <ul class="font-medium">
        @foreach ($ai_image_pro_search as $item)
            <li class="border-b px-3 py-2 transition-colors last:border-b-0 hover:bg-foreground/5">
                <a
                    class="flex items-center gap-2 text-heading-foreground"
                    href="{{ route('dashboard.user.openai.documents.single', 'ai-image-pro-' . $item->id . '-0') }}"
                >
                    <x-lqd-icon
                        size="lg"
                        style="background: #22c55e"
                    >
                        <span class="flex size-5">
                            <x-tabler-photo class="size-5" />
                        </span>
                    </x-lqd-icon>
                    {{ \Illuminate\Support\Str::limit($item->prompt, 50) }}
                    <small class="ms-auto text-foreground/50">{{ __('AI Image Pro') }}</small>
                </a>
            </li>
        @endforeach
    </ul>
@endif

@if (count($ai_chat_pro_image_search) > 0)
    <ul class="font-medium">
        @foreach ($ai_chat_pro_image_search as $item)
            <li class="border-b px-3 py-2 transition-colors last:border-b-0 hover:bg-foreground/5">
                <a
                    class="flex items-center gap-2 text-heading-foreground"
                    href="{{ route('dashboard.user.openai.documents.single', 'ai-chat-pro-image-chat-' . $item->id . '-0') }}"
                >
                    <x-lqd-icon
                        size="lg"
                        style="background: #22c55e"
                    >
                        <span class="flex size-5">
                            <x-tabler-photo class="size-5" />
                        </span>
                    </x-lqd-icon>
                    {{ \Illuminate\Support\Str::limit($item->prompt, 50) }}
                    <small class="ms-auto text-foreground/50">{{ __('Chat Pro Image') }}</small>
                </a>
            </li>
        @endforeach
    </ul>
@endif

@if (count($workbook_search) > 0)
    <h3 class="m-0 border-b px-3 py-3 text-base font-medium">
        {{ __('Documents') }}
    </h3>
    <ul>
        @foreach ($workbook_search as $item)
            <li class="border-b px-3 py-2 transition-colors last:border-b-0 hover:bg-foreground/5">
                <a
                    class="flex items-center gap-2 text-heading-foreground"
                    href="{{ route('dashboard.user.openai.documents.single', $item->slug) }}"
                >
                    <x-lqd-icon
                        size="lg"
                        style="background: {{ $item->generator->color }}"
                    >
                        <span class="flex size-5">
                            @if ($item->generator->image !== 'none')
                                {!! html_entity_decode($item->generator->image) !!}
                            @endif
                        </span>
                    </x-lqd-icon>

                   {{ $item->title ?: \Illuminate\Support\Str::limit($item->output ?: $item->input, 30) }}
                    <small class="ms-auto text-foreground/50">{{ $item->generator->type == 'text' ? __('Document') : __(ucfirst($item->generator->type)) }}</small>
                </a>
            </li>
        @endforeach
    </ul>
@endif

@if ($result === 'null')
    <div class="p-6 text-center font-medium text-heading-foreground">
        <h3 class="mb-2">{{ __('No results.') }}</h3>
        <p class="opacity-70">{{ __('Please try with another word.') }}</p>
    </div>
@endif
