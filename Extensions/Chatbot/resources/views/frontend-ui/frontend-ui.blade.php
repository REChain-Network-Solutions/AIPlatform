{{-- This is the frontend ui --}}

@if ($is_editor)
	@push('before-head-close')
		@vite('app/Extensions/Chatbot/resources/assets/scss/external-chatbot.scss')
	@endpush
@else
	@vite('app/Extensions/Chatbot/resources/assets/scss/external-chatbot.scss')
@endif
@php
	$style = '';

	if (!$is_editor) {
		if (!empty($chatbot['color'])) {
			$style .= '--lqd-ext-chat-primary: ' . $chatbot['color'] . ';';
		}
	}
@endphp

<div
	class="lqd-ext-chatbot"
	data-pos-x="{{ !$is_editor ? $chatbot['position'] ?? 'right' : 'right' }}"
	data-pos-y="{{ !$is_editor ? $chatbot['position_y'] ?? 'bottom' : 'bottom' }}"
	data-window-state="{{ $is_editor || $is_iframe ? 'open' : 'close' }}"
	data-embedded="{{ $is_editor ? 'false' : 'true' }}"
	x-data="externalChatbot"
	{{-- blade-formatter-disable --}}
	@if ($is_editor)
		:data-pos-x="activeChatbot.position"
	:style="{
            '--lqd-ext-chat-primary': activeChatbot.color,
            '--lqd-ext-chat-trigger-background': activeChatbot.trigger_background,
            '--lqd-ext-chat-window-w': `${testIframeWidth}px`,
            '--lqd-ext-chat-window-h': `${testIframeHeight}px`,
        }"
	@else
		data-fetching="true"
	:data-fetching="fetching ? 'true' : 'false'"
	@if($style)
		style="{{ $style }}"
	@endif
	@endif
	{{-- blade-formatter-enable --}}
>
	<div class="lqd-ext-chatbot-window">
		<div class="lqd-ext-chatbot-window-head">
			@if ($is_editor || (!$is_editor && $chatbot['logo'] && $chatbot['show_logo']))
				<img
					class="lqd-ext-chatbot-window-head-logo"
					{{-- blade-formatter-disable --}}
					@if ($is_editor)
						:src="activeChatbot.logo"
					x-show="activeChatbot.show_logo && activeChatbot.logo"
					@else
						alt="{{ $chatbot['title'] }}"
					@if ($chatbot['logo'] && $chatbot['show_logo'])
						src="{{ $chatbot['logo'] }}"
					@endif
					@endif
					{{-- blade-formatter-enable --}}
					width="25"
					height="25"
				/>
			@endif
			@if ($is_editor || (!$is_editor && !$chatbot['logo'] && $chatbot['show_logo']))
				<svg
					class="lqd-ext-chatbot-window-head-logo"
					width="25"
					height="25"
					viewBox="0 0 25 25"
					fill="currentColor"
					xmlns="http://www.w3.org/2000/svg"
					@if ($is_editor) x-show="activeChatbot.show_logo && !activeChatbot.logo" @endif
				>
					<path
						d="M18.2404 21.8333L14.1279 17.6917L15.7612 16.0583L18.2404 18.5375L23.1987 13.5792L24.832 15.2417L18.2404 21.8333ZM0.332031 24.1667V3.16668C0.332031 2.52501 0.560503 1.9757 1.01745 1.51876C1.47439 1.06182 2.0237 0.833344 2.66536 0.833344H21.332C21.9737 0.833344 22.523 1.06182 22.9799 1.51876C23.4369 1.9757 23.6654 2.52501 23.6654 3.16668V11.3333H11.9987V19.5H4.9987L0.332031 24.1667Z"
					/>
				</svg>
			@endif
			<h4
				class="lqd-ext-chatbot-window-head-title"
				@if ($is_editor) x-text="activeChatbot.title" @endif
			>
				@if (!$is_editor)
					{{ $chatbot['title'] }}
				@endif
			</h4>

			<button
				class="lqd-ext-chatbot-window-head-back-btn"
				type="button"
				title="{{ __('Back') }}"
				x-show="currentView === 'conversation-messages'"
				x-transition
				@click.prevent="toggleView('conversations-list')"
			>
				<svg
					xmlns="http://www.w3.org/2000/svg"
					viewBox="0 0 24 24"
					fill="none"
					stroke="currentColor"
					stroke-linecap="round"
					stroke-linejoin="round"
					width="22"
					height="22"
					stroke-width="2"
				>
					<path d="M15 6l-6 6l6 6"></path>
				</svg>
			</button>
		</div>

		<div class="lqd-ext-chatbot-window-conversations-wrap">
			<div
				class="lqd-ext-chatbot-window-conversations-list"
				x-show="currentView === 'conversations-list'"
				x-transition.opacity.duration.150ms
			>
				@if ($is_editor)
					@include('chatbot::frontend-ui.editor-demo-conversation-list')
				@else
					<template x-for="conversation in conversations">
						<div
							class="lqd-ext-chatbot-window-conversations-list-item"
							:data-id="conversation.id"
							@click.prevent="openConversation(conversation.id, true)"
							:key="conversation.id"
						>
							<figure class="lqd-ext-chatbot-window-conversations-list-item-fig">
								<img
									:src="activeChatbot.avatar"
									:alt="activeChatbot.title"
									width="27"
									height="27"
								>
							</figure>
							<div class="lqd-ext-chatbot-window-conversations-list-item-info">
								<p
									class="lqd-ext-chatbot-window-conversations-list-item-info-name"
									x-text="activeChatbot.title"
								></p>
								<p
									class="lqd-ext-chatbot-window-conversations-list-item-info-last-message"
									x-text="conversation.last_message"
								></p>
							</div>
							<div
								class="lqd-ext-chatbot-window-conversations-list-item-time"
								x-data="{ diff: Math.floor((new Date() - new Date(conversation.updated_at || conversation.created_at)) / 1000) }"
								{{-- x-init="if (Math.floor((new Date() - new Date(conversation.updated_at || conversation.created_at)) / 1000) < 60) { setInterval(() => { diff = Math.floor((new Date() - new Date(conversation.updated_at || conversation.created_at)) / 1000); }, 1000); }" --}}
								x-text="
                                    {{-- diff < 60 ? diff + ' {{ __('seconds ago') }}' : --}}
                                    diff < 60 ? '{{ __('just now') }}' :
                                    diff < 3600 ? (Math.floor(diff / 60) === 1 ? '1 {{ __('minute ago') }}' : Math.floor(diff / 60) + ' {{ __('minutes ago') }}') :
                                    diff < 86400 ? (Math.floor(diff / 3600) === 1 ? '1 {{ __('hour ago') }}' : Math.floor(diff / 3600) + ' {{ __('hours ago') }}') :
                                    Math.floor(diff / 86400) === 1 ? '1 {{ __('day ago') }}' : Math.floor(diff / 86400) + ' {{ __('days ago') }}'
                                "
							></div>
						</div>
					</template>
					<h4
						class="lqd-ext-chatbot-window-conversations-list-no-conversations"
						x-show="!fetching && !conversations.length"
						style="display: none"
					>
						{{ __('No conversations yet.') }}
					</h4>
				@endif
			</div>

			<div
				class="lqd-ext-chatbot-window-conversation-messages"
				x-ref="conversationMessages"
				x-show="currentView === 'conversation-messages'"
				x-transition.opacity.duration.150ms
			>
				@if ($is_editor)
					@include('chatbot::frontend-ui.editor-demo-conversation')
				@else
					<template x-for="(message, index) in messages">
						<div
							class="lqd-ext-chatbot-window-conversation-message"
							:data-type="message.role"
							:data-id="message.id"
							:key="message.id"
						>
							<figure
								class="lqd-ext-chatbot-window-conversation-message-avatar"
								x-show="message.role === 'assistant' || message.role === 'loader'"
							>
								<img
									:src="activeChatbot.avatar"
									:alt="activeChatbot.title"
									width="27"
									height="27"
								>
							</figure>
							<div class="lqd-ext-chatbot-window-conversation-message-content-wrap">
								<div class="lqd-ext-chatbot-window-conversation-message-content">
                                    <pre
										x-ref="conversationMessage"
										:data-index="index"
										x-html="addMessage(message.message, $el)"
									></pre>
									<template x-if="message.role === 'loader'">
										<div class="lqd-ext-chatbot-window-conversation-message-loader">
											<span></span>
											<span></span>
											<span></span>
										</div>
									</template>
								</div>
								@if ($is_editor || (!$is_editor && $chatbot['show_date_and_time']))
									<div
										class="lqd-ext-chatbot-window-conversation-message-time"
										x-data="{ diff: Math.floor((new Date() - new Date(message.created_at)) / 1000) }"
										{{-- x-init="if (Math.floor((new Date() - new Date(message.created_at)) / 1000) < 60) { setInterval(() => { diff = Math.floor((new Date() - new Date(message.created_at)) / 1000); }, 1000); }" --}}
										x-text="
                                        {{-- diff < 60 ? diff + ' {{ __('seconds ago') }}' : --}}
                                        diff < 60 ? '{{ __('just now') }}' :
                                        diff < 3600 ? (Math.floor(diff / 60) === 1 ? '1 {{ __('minute ago') }}' : Math.floor(diff / 60) + ' {{ __('minutes ago') }}') :
                                        diff < 86400 ? (Math.floor(diff / 3600) === 1 ? '1 {{ __('hour ago') }}' : Math.floor(diff / 3600) + ' {{ __('hours ago') }}') :
                                        Math.floor(diff / 86400) === 1 ? '1 {{ __('day ago') }}' : Math.floor(diff / 86400) + ' {{ __('days ago') }}'
                                    "
									></div>
								@endif
							</div>
						</div>
					</template>
				@endif
				@if(
					isset($chatbot)
					&& $chatbot->getAttribute('interaction_type') === \App\Extensions\Chatbot\System\Enums\InteractionType::SMART_SWITCH
					&& \App\Helpers\Classes\MarketplaceHelper::isRegistered('chatbot-agent')
				)
					<div
						 x-show="connect_agent_at === null && showConnectButton"
					>
						<div
							class="connect-agent"
							x-show="showConnectButtonStepOne"
						>

							<!-- Thanks Button -->
							<button  @click="dontConnectToAgent"  class="button thanks-button">
								<svg
									xmlns="http://www.w3.org/2000/svg"
									width="24"
									height="24"
									viewBox="0 0 24 24"
									fill="none"
									stroke="#3B82F6"
									stroke-width="2"
									stroke-linecap="round"
									stroke-linejoin="round"
								>
									<path d="M7 11v8a1 1 0 0 1 -1 1h-2a1 1 0 0 1 -1 -1v-7a1 1 0 0 1 1 -1h3a4 4 0 0 0 4 -4v-1a2 2 0 0 1 4 0v5h3a2 2 0 0 1 2 2l-1 5a2 3 0 0 1 -2 2h-7a3 3 0 0 1 -3 -3" />
								</svg>
								@lang('This answered my question')
							</button>
							<!-- Talk to an Agent Button -->
							<button @click="connectToAgentStepOne" class="button agent-button">
								<svg  xmlns="http://www.w3.org/2000/svg"  width="24"  height="24"  viewBox="0 0 24 24"  fill="none"  stroke="currentColor"  stroke-width="2"  stroke-linecap="round"  stroke-linejoin="round"  class="icon icon-tabler icons-tabler-outline icon-tabler-help"><path stroke="none" d="M0 0h24v24H0z" fill="none"/><path d="M12 12m-9 0a9 9 0 1 0 18 0a9 9 0 1 0 -18 0" /><path d="M12 17l0 .01" /><path d="M12 13.5a1.5 1.5 0 0 1 1 -1.5a2.6 2.6 0 1 0 -3 -4" /></svg>							@lang('Get more help')
							</button>
						</div>

						<div
							class="connect-agent"
							x-show="showConnectButtonStepTwo"
						>
							<!-- Thanks Button -->
							<button  @click="dontConnectToAgent"  class="button thanks-button">
{{--								<svg--}}
{{--									xmlns="http://www.w3.org/2000/svg"--}}
{{--									width="24"--}}
{{--									height="24"--}}
{{--									viewBox="0 0 24 24"--}}
{{--									fill="none"--}}
{{--									stroke="#3B82F6"--}}
{{--									stroke-width="2"--}}
{{--									stroke-linecap="round"--}}
{{--									stroke-linejoin="round"--}}
{{--								>--}}
{{--									<path d="M7 11v8a1 1 0 0 1 -1 1h-2a1 1 0 0 1 -1 -1v-7a1 1 0 0 1 1 -1h3a4 4 0 0 0 4 -4v-1a2 2 0 0 1 4 0v5h3a2 2 0 0 1 2 2l-1 5a2 3 0 0 1 -2 2h-7a3 3 0 0 1 -3 -3" />--}}
{{--								</svg>--}}
								@lang('Could you explain again?')
							</button>

							<!-- Talk to an Agent Button -->
							<button @click="connectToAgent" class="button agent-button">
								<img class="icon" src="{{ asset('vendor/chatbot/icons/agent-button.svg') }}" alt="icon">
								@lang('Connect to agent')
							</button>
						</div>
					</div>

				@endif
			</div>
			@if (!$is_editor)
				<div class="lqd-ext-chatbot-window-loader">
					<svg
						xmlns="http://www.w3.org/2000/svg"
						viewBox="0 0 24 24"
						fill="none"
						stroke="currentColor"
						stroke-linecap="round"
						stroke-linejoin="round"
						width="24"
						height="24"
						stroke-width="1.5"
					>
						<path d="M12 3a9 9 0 1 0 9 9"></path>
					</svg>
				</div>
			@endif
		</div>

		<div
			class="lqd-ext-chatbot-window-form-wrap"
			{{-- blade-formatter-disable --}}
			@if (!$is_editor)
				:style="{ 'opacity': fetching ? 0 : 1, visibility: fetching ? 'hidden' : 'visible' }"
			@endif
			{{-- blade-formatter-enable --}}
		>
			<form
				class="lqd-ext-chatbot-window-form"
				@submit.prevent="onSendMessage"
				x-show="currentView === 'conversation-messages'"
				x-transition
			>
                <textarea
					id="message"
					name="message"
					cols="30"
					rows="4"
					placeholder="{{ __('Message...') }}"
					@keydown.enter.prevent="onMessageFieldHitEnter"
					@input="onMessageFieldInput"
					@input.throttle.50ms="$el.scrollTop = $el.scrollHeight"
					x-ref="message"
				></textarea>
				<button
					type="submit"
					title="{{ __('Send') }}"
					x-ref="submitBtn"
					:disabled="!$refs.message.value.trim()"
				>
					<svg
						width="19"
						height="16"
						viewBox="0 0 19 16"
						fill="currentColor"
						xmlns="http://www.w3.org/2000/svg"
					>
						<path d="M0 16V10L8 8L0 6V0L19 8L0 16Z" />
					</svg>
				</button>
			</form>

			<div
				class="lqd-ext-chatbot-window-start-new"
				x-show="currentView === 'conversations-list'"
				x-transition
			>
				<button
					class="lqd-ext-chatbot-window-start-new-btn"
					type="button"
					@click.prevent="startNewConversation({{ $is_editor ? 'false' : 'true' }})"
				>
					<svg
						width="16"
						height="17"
						viewBox="0 0 16 17"
						fill="currentColor"
						xmlns="http://www.w3.org/2000/svg"
					>
						<path
							d="M13.25 4.5C12.625 4.5 12.0938 4.28125 11.6562 3.84375C11.2188 3.40625 11 2.875 11 2.25C11 1.625 11.2188 1.09375 11.6562 0.65625C12.0938 0.21875 12.625 0 13.25 0C13.875 0 14.4062 0.21875 14.8438 0.65625C15.2812 1.09375 15.5 1.625 15.5 2.25C15.5 2.875 15.2812 3.40625 14.8438 3.84375C14.4062 4.28125 13.875 4.5 13.25 4.5ZM0.5 16.5V3C0.5 2.5875 0.646875 2.23438 0.940625 1.94063C1.23438 1.64688 1.5875 1.5 2 1.5H9.575C9.525 1.75 9.5 2 9.5 2.25C9.5 2.5 9.525 2.75 9.575 3C9.75 3.875 10.1812 4.59375 10.8687 5.15625C11.5562 5.71875 12.35 6 13.25 6C13.65 6 14.0438 5.9375 14.4313 5.8125C14.8188 5.6875 15.175 5.5 15.5 5.25V12C15.5 12.4125 15.3531 12.7656 15.0594 13.0594C14.7656 13.3531 14.4125 13.5 14 13.5H3.5L0.5 16.5Z"
						/>
					</svg>
					{{ __('Start a New Conversation ') }}
				</button>
			</div>
		</div>

		<div class="lqd-ext-chatbot-window-foot">
			{{-- TODO: Using collapsed navbar logo. maybe we need to change --}}
			<img
				width="16"
				height="16"
				src="{{ custom_theme_url($setting->logo_collapsed_path, true) }}"
				@if (isset($setting->logo_collapsed_2x_path) && !empty($setting->logo_collapsed_2x_path)) srcset="/{{ $setting->logo_collapsed_2x_path }} 2x" @endif
				alt="{{ $setting->site_name }}"
			>
			<p>
				@lang('Powered by')
				<u>
					<a href="{{ $chatbot->footer_link ?? request()->getSchemeAndHttpHost() }}" target="_blank">{{ $setting->site_name }}</a>
				</u>
			</p>
		</div>
	</div>

	@if ($is_editor)
		<div class="lqd-ext-chatbot-welcome-bubble">
			<p @if ($is_editor) x-text="activeChatbot.bubble_message" @endif>
				@if (!$is_editor)
					{{ $chatbot['welcome_message'] }}
				@endif
			</p>
		</div>
	@endif

	@if (!$is_iframe)
		<button
			class="lqd-ext-chatbot-trigger"
			type="button"
			@click.prevent="toggleWindowState()"
			{{-- blade-formatter-disable --}}
			@if(!$is_editor && @filled($chatbot['trigger_background']))
				style="background-color: {{ $chatbot['trigger_background'] }}"
			@endif
			{{-- blade-formatter-enable --}}
		>
			<img
				class="lqd-ext-chatbot-trigger-img"
				{{-- blade-formatter-disable --}}
				@if ($is_editor)
					:src="() => activeChatbot.avatar ? `${window.location.origin}/${activeChatbot.avatar}` : ''"
				:style="{ 'width': parseInt(activeChatbot.trigger_avatar_size, 10) + 'px' }"
				@else
					src="/{{$chatbot['avatar']}}"
				alt="{{$chatbot['title']}}"
				@if(!empty($chatbot['trigger_avatar_size']))
					style="width: {{ (int)$chatbot['trigger_avatar_size'] }}px"
				@endif
				@endif
				{{-- blade-formatter-enable --}}
				width="60"
				height="60"
			/>
			<span class="lqd-ext-chatbot-trigger-icon">
                <svg
					width="16"
					height="10"
					viewBox="0 0 16 10"
					fill="currentColor"
					xmlns="http://www.w3.org/2000/svg"
				>
                    <path d="M8 9.07814L0.75 1.82814L2.44167 0.136475L8 5.69481L13.5583 0.136475L15.25 1.82814L8 9.07814Z" />
                </svg>
            </span>
		</button>
	@endif
</div>

@if ($is_editor)
	<template id="lqd-ext-chatbot-user-msg-temp">
		<div
			class="lqd-ext-chatbot-window-conversation-message"
			data-type="user"
		>
			<div class="lqd-ext-chatbot-window-conversation-message-content-wrap">
				<div class="lqd-ext-chatbot-window-conversation-message-content">
					<p></p>
				</div>
				@if ($is_editor || (!$is_editor && $chatbot['show_date_and_time']))
					<div
						class="lqd-ext-chatbot-window-conversation-message-time"
						@if ($is_editor) x-show="activeChatbot.show_date_and_time" @endif
					></div>
				@endif
			</div>
		</div>
	</template>

	<template id="lqd-ext-chatbot-assistant-msg-temp">
		<div
			class="lqd-ext-chatbot-window-conversation-message"
			data-type="assistant"
		>
			<figure class="lqd-ext-chatbot-window-conversation-message-avatar">
				<img
					@if ($is_editor) :src="() => activeChatbot.avatar ? `${window.location.origin}/${activeChatbot.avatar}` : ''"
					@else
						src="" @endif
					alt=""
					width="27"
					height="27"
				>
			</figure>
			<div class="lqd-ext-chatbot-window-conversation-message-content-wrap">
				<div class="lqd-ext-chatbot-window-conversation-message-content">
					<pre></pre>
				</div>
				@if ($is_editor || (!$is_editor && $chatbot['show_date_and_time']))
					<div
						class="lqd-ext-chatbot-window-conversation-message-time"
						@if ($is_editor) x-show="activeChatbot.show_date_and_time" @endif
					></div>
				@endif
			</div>
		</div>
	</template>

	<template id="lqd-ext-chatbot-chat-list-item-temp">
		<div class="lqd-ext-chatbot-window-conversations-list-item">
			<figure class="lqd-ext-chatbot-window-conversations-list-item-fig">
				<img
					src=""
					alt=""
					width="27"
					height="27"
				>
			</figure>
			<div class="lqd-ext-chatbot-window-conversations-list-item-info">
				<p class="lqd-ext-chatbot-window-conversations-list-item-info-name"></p>
				<p class="lqd-ext-chatbot-window-conversations-list-item-info-last-message"></p>
			</div>
			<div class="lqd-ext-chatbot-window-conversations-list-item-time"></div>
			<a
				class="lqd-ext-chatbot-window-conversations-list-item-link"
				data-id=""
				href="#"
				title="{{ __('View Messages') }}"
				@click.prevent="toggleView('conversation-messages')"
			>
			</a>
		</div>
	</template>
@endif

{{--
    Because we are moving the editor dom element, if we declare the <script> tag it would not work
    so we are moving the <script> tag to another place so it would not detach from the dom
--}}
@if ($is_editor)
	@push('script')
		@include('chatbot::frontend-ui.frontend-ui-scripts', ['is_editor' => $is_editor])
	@endpush
@else
	<link
		rel="stylesheet"
		href="{{ custom_theme_url('/assets/libs/prism/prism.css') }}"
	/>
	<script src="{{ custom_theme_url('/assets/libs/prism/prism.js') }}"></script>
	<script src="{{ custom_theme_url('/assets/libs/beautify-html.min.js') }}"></script>
	<script src="{{ custom_theme_url('/assets/libs/markdown-it.min.js') }}"></script>
	<script src="{{ custom_theme_url('/assets/libs/turndown.js') }}"></script>
	<script
		defer
		src="{{ asset('vendor/chatbot/js/alpine.min.js') }}"
	></script>

	@if(\App\Helpers\Classes\MarketplaceHelper::isRegistered('chatbot-agent'))
		<script src="https://cdn.ably.com/lib/ably.min-1.js" type="text/javascript"></script>
	@endif

	@include('chatbot::frontend-ui.frontend-ui-scripts', ['is_editor' => $is_editor])
@endif
