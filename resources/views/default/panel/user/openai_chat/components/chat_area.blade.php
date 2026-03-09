@php
    $currentUrl = url()->current();
    $previousUrl = url()->previous();
    $canvas_enabled = \App\Helpers\Classes\MarketplaceHelper::isRegistered('canvas') && (bool) setting('ai_chat_pro_canvas', 1);
    $is_chat_pro =
        \App\Helpers\Classes\MarketplaceHelper::isRegistered('ai-chat-pro') &&
        (route('dashboard.user.openai.chat.pro.index') === $currentUrl ||
            route('chat.pro') === $currentUrl ||
            route('dashboard.user.openai.chat.pro.index') === $previousUrl ||
            route('chat.pro') === $previousUrl);
    $is_chat_pro_image = isset($website_url) && $website_url === 'chatpro-image' && \App\Helpers\Classes\MarketplaceHelper::isRegistered('ai-chat-pro-image-chat');
    $messages = $chat?->messages ?? [];

    // Eager load relationships based on chat type
    $eagerLoads = [];
    if ($canvas_enabled) {
        $eagerLoads[] = 'tiptapContent';
    }
    if ($is_chat_pro_image) {
        $eagerLoads[] = 'aiChatProImages';
    }

    if (!empty($eagerLoads)) {
        $messages = $chat?->messages()->with($eagerLoads)->get() ?? [];
    }

	$multi_model_message_pairs = [];

	// Group messages by shared_uuid
	if ($is_chat_pro) {
		foreach ($messages as $message) {
			if (isset($message->shared_uuid) && !empty($message->shared_uuid)) {
				$multi_model_message_pairs[$message->shared_uuid][] = $message;
			}
		}
	}
@endphp

@foreach ($messages ?? [] as $messageIndex => $message)
    {{-- to prevent showing first 'Hi, ...' message on ai vision chat --}}
    @continue(isset($category) && ($category?->slug == 'ai_vision' || $category?->slug === 'ai_realtime_voice_chat') && count($chat?->messages) === 1)

    @php
        $is_multi_model_message = $is_chat_pro && isset($message->shared_uuid) && !empty($message->shared_uuid);
        $isLastMessage = $loop->last;
    @endphp

	@include('panel.user.openai_chat.components.chat_user_message')

    @if ($is_chat_pro)
        @if (!$is_multi_model_message)
            @include('panel.user.openai_chat.components.chat_ai_message', ['website_url' => $website_url ?? null, 'isLastMessage' => $isLastMessage])
        @else
            <div class="multi-model-response-wrap grid grid-cols-1 gap-x-6 lg:grid-cols-2">
                @php
                    $current_shared_uuid = $message->shared_uuid;
                    $paired_messages = $multi_model_message_pairs[$current_shared_uuid] ?? [];
                @endphp

                @foreach ($paired_messages as $pair_message)
                    @include('panel.user.openai_chat.components.chat_ai_message', ['message' => $pair_message, 'website_url' => $website_url ?? null, 'isLastMessage' => $isLastMessage && $loop->last])
                @endforeach

                @php
                    unset($multi_model_message_pairs[$current_shared_uuid]);
                @endphp
            </div>
        @endif
    @else
        @include('panel.user.openai_chat.components.chat_ai_message', ['website_url' => $website_url ?? null, 'isLastMessage' => $isLastMessage])
    @endif
@endforeach

@if ($chat?->category?->slug !== 'ai_realtime_voice_chat' && count($chat?->messages ?? []) === 0)
	<div class="lqd-chat-ai-bubble mb-2.5 flex max-w-full content-start items-start gap-2 group-[&.lqd-chat-v2]/body:first:hidden">
		<div class="chat-content-container group relative max-w-[calc(100%-64px)] rounded-[2em] bg-clay text-heading-foreground dark:bg-white/[2%]">
			@php
				$output = __('You have no message... Please start typing.');
				$output = str_replace(array('<br>','<br/>','<br >','<br />','/http'), array("\n","\n","\n","\n",'http'), $output);
			@endphp
			<pre
				class="chat-content prose relative w-full max-w-none !whitespace-pre-wrap px-6 py-3.5 indent-0 font-[inherit] text-xs font-normal text-current [word-break:break-word] empty:hidden [&_*]:text-current">{{ $output }}</pre>
		</div>
	</div>
@endif
