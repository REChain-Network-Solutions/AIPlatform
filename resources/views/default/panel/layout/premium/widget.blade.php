@php
    use Illuminate\Support\Facades\Auth;

    if (\App\Helpers\Classes\Helper::appIsDemo()) {
        return false;
    }

    if (!Auth::user()?->isAdmin()) {
        return false;
    }

    if (!Auth::user()?->isSuperAdmin()) {
        if (!Auth::user()?->checkPermission('VIP_CHAT_WIDGET')) {
            return false;
        }
    }
@endphp

<style>
    .lqd-premium-chatbot {
        @keyframes lqd-ext-chat-spin {
            0% {
                transform: rotate(0deg);
            }

            100% {
                transform: rotate(360deg);
            }
        }

        @keyframes lqd-ext-chat-message-loader {
            0% {
                opacity: 0.5;
                transform: translateY(0);
            }

            50% {
                opacity: 1;
                transform: translateY(-3px);
            }

            100% {
                opacity: 0.5;
                transform: translateY(0);
            }
        }

        .lqd-ext-chatbot-premium {
            --lqd-ext-chat-primary: #017be5;
            --lqd-ext-chat-primary-foreground: #fff;
            --lqd-ext-chat-font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
            --lqd-ext-chat-offset-y: 30px;
            --lqd-ext-chat-offset-x: 30px;
            --lqd-ext-chat-trigger-w: 60px;
            --lqd-ext-chat-trigger-h: 60px;
            --lqd-ext-chat-trigger-background: var(--lqd-ext-chat-primary);
            --lqd-ext-chat-trigger-foreground: var(--lqd-ext-chat-primary-foreground);
            --lqd-ext-chat-window-w: 420px;
            --lqd-ext-chat-window-h: min(min(80vh, 720px), max(250px, calc(100vh - var(--lqd-ext-chat-trigger-h) - (var(--lqd-ext-chat-window-y-offset) * 2) - 20px)));
            --lqd-ext-chat-window-bg: #fff;
            --lqd-ext-chat-window-foreground: #000;
            --lqd-ext-chat-window-y-offset: 20px;
            --lqd-ext-chat-form-h: 70px;
            --lqd-ext-chat-foot-bg: #f8f8f8;
            --lqd-ext-chat-chat-assistant-bg: #f4f5f5;
            --lqd-ext-chat-chat-assistant-color: #000;
            --lqd-ext-chat-position: fixed;
            --lqd-ext-chat-flex-direction: column;
            display: flex;
            flex-direction: var(--lqd-ext-chat-flex-direction);
            gap: var(--lqd-ext-chat-window-y-offset);
            overflow: visible;
            position: fixed;
            bottom: var(--lqd-ext-chat-offset-y);
            right: var(--lqd-ext-chat-offset-x);
            z-index: 9999;
            font-family: var(--lqd-ext-chat-font-family);
            pointer-events: none;
        }

        @media (max-width: 991px) {
            .lqd-ext-chatbot-premium {
                --lqd-ext-chat-offset-y: calc(4rem + 20px);
            }
        }

        .lqd-ext-chatbot-premium,
        .lqd-ext-chatbot-premium *,
        .lqd-ext-chatbot-premium *:before,
        .lqd-ext-chatbot-premium *:after {
            box-sizing: border-box;
        }

        .lqd-ext-chatbot-premium h1,
        .lqd-ext-chatbot-premium h2,
        .lqd-ext-chatbot-premium h3,
        .lqd-ext-chatbot-premium h4,
        .lqd-ext-chatbot-premium h5,
        .lqd-ext-chatbot-premium h6 {
            margin: 0;
        }

        .lqd-ext-chatbot-premium .connect-agent {
            display: flex;
            gap: 16px;
            padding-left: 35px;
        }

        .lqd-ext-chatbot-premium .connect-agent .button {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 16px;
            border-radius: 9999px;
            cursor: pointer;
            transition: background 0.2s;
            font-family: Arial, sans-serif;
            font-size: 14px;
        }

        .lqd-ext-chatbot-premium .connect-agent .thanks-button {
            border: 1px solid #D1D5DB;
            color: #374151;
            background-color: white;
        }

        .lqd-ext-chatbot-premium .connect-agent .thanks-button:hover {
            background-color: #F3F4F6;
        }

        .lqd-ext-chatbot-premium .connect-agent .agent-button {
            border: 1px solid #D1D5DB;
            background-color: #DBEAFE;
            color: #1E40AF;
        }

        .lqd-ext-chatbot-premium .connect-agent .agent-button:hover {
            background-color: #BFDBFE;
        }

        .lqd-ext-chatbot-premium .connect-agent .icon {
            width: 20px;
            height: 20px;
            color: #3B82F6;
        }

        .lqd-ext-chatbot-premium-trigger {
            display: inline-grid;
            place-items: center;
            place-content: center;
            width: var(--lqd-ext-chat-trigger-w);
            height: var(--lqd-ext-chat-trigger-h);
            position: relative;
            background-color: var(--lqd-ext-chat-trigger-background);
            color: var(--lqd-ext-chat-trigger-foreground);
            border-radius: var(--lqd-ext-chat-trigger-w);
            border: none;
            overflow: hidden;
            transition: all 0.15s;
            cursor: pointer;
            backdrop-filter: blur(12px) saturate(120%);
            pointer-events: auto;
        }

        .lqd-ext-chatbot-premium-trigger:before {
            content: '';
            display: inline-block;
            width: 100%;
            height: 100%;
            position: absolute;
            top: 0;
            left: 0;
            background-color: var(--lqd-ext-chat-primary);
            opacity: 0;
            transform: translateY(3px);
            transition: all 0.15s;
        }

        .lqd-ext-chatbot-premium-trigger-img,
        .lqd-ext-chatbot-premium-trigger-icon {
            grid-row: 1 / 1;
            grid-column: 1 / 1;
            transition: all 0.15s;
            position: relative;
            z-index: 1;
        }

        .lqd-ext-chatbot-premium-trigger-img {
            width: 35px;
            height: 35px;
            /*border-radius: 50%;*/
            object-fit: cover;
        }

        .lqd-ext-chatbot-premium-trigger-icon {
            opacity: 0;
            transform: translateY(3px);
        }

        .lqd-ext-chatbot-premium-trigger:active {
            transform: scale(0.9);
        }

        .lqd-ext-chatbot-premium-welcome-bubble {
            padding: 12px 16px;
            border-radius: 12px;
            position: absolute;
            bottom: calc(var(--lqd-ext-chat-trigger-h) + var(--lqd-ext-chat-window-y-offset));
            left: 0;
            color: hsl(var(--foreground));
            font-size: 14px;
            font-weight: 500;
            line-height: 1.2em;
            backdrop-filter: blur(12px) saturate(120%);
            transition: all 0.15s;
            white-space: nowrap;
        }

        .lqd-ext-chatbot-premium-welcome-bubble:before {
            content: '';
            display: inline-block;
            width: 100%;
            height: 100%;
            position: absolute;
            top: 0;
            left: 0;
            background-color: hsl(var(--foreground));
            opacity: 0.05;
            border-radius: inherit;
        }

        .lqd-ext-chatbot-premium-welcome-bubble p {
            position: relative;
            z-index: 1;
            margin: 0;
        }

        .lqd-ext-chatbot-premium-window {
            display: flex;
            flex-direction: column;
            width: var(--lqd-ext-chat-window-w);
            height: var(--lqd-ext-chat-window-h);
            background-color: var(--lqd-ext-chat-window-bg);
            color: var(--lqd-ext-chat-window-foreground);
            border-radius: 12px;
            overflow: hidden;
            pointer-events: none;
            transform-origin: bottom left;
            transform: scale(0.975) translateY(6px);
            opacity: 0;
            visibility: hidden;
            transition: all 0.1s;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
            pointer-events: none;
        }

        .lqd-ext-chatbot-premium-window-head {
            display: flex;
            align-items: center;
            min-height: 75px;
            gap: 10px;
            padding: 10px 28px;
            padding-inline-end: 17px;
            flex-shrink: 0;
            background-color: var(--lqd-ext-chat-primary);
            color: var(--lqd-ext-chat-primary-foreground);
            transition: background-color 0.15s;
        }

        .lqd-ext-chatbot-premium-window-head h4 {
            margin: 0;
            font-size: 19px;
            font-weight: 700;
            letter-spacing: -0.01em;
            color: inherit;
        }

        .lqd-ext-chatbot-premium-window-head-back-btn {
            display: inline-grid;
            place-items: center;
            width: 44px;
            height: 44px;
            padding: 0;
            background: none;
            border: none;
            color: inherit;
            margin-inline-start: auto;
            cursor: pointer;
            transition: all 0.3s;
        }

        .lqd-ext-chatbot-premium-window-head-back-btn:active {
            transform: translateX(-3px);
        }

        .lqd-ext-chatbot-premium-window-conversations-wrap {
            display: grid;
            flex-grow: 1;
            overflow: hidden;
            position: relative;
        }

        .lqd-ext-chatbot-premium-window-conversations-list,
        .lqd-ext-chatbot-premium-window-conversation-messages {
            display: flex;
            flex-direction: column;
            height: 100%;
            padding: 30px;
            overflow-y: auto;
            grid-column: 1 / 1;
            grid-row: 1 / 1;
        }

        .lqd-ext-chatbot-premium-window-conversation-messages {
            gap: 16px;
            font-size: 14px;
            line-height: 1.43em;
            font-weight: 400;
        }

        .lqd-ext-chatbot-premium-window-conversation-message {
            display: flex;
            gap: 10px;
        }

        .lqd-ext-chatbot-premium-window-conversation-message-avatar {
            width: 27px;
            height: 27px;
            overflow: hidden;
            border-radius: 20px;
            flex-shrink: 0;
            margin: 8px 0 0 0;
        }

        .lqd-ext-chatbot-premium-window-conversation-message-avatar img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }

        .lqd-ext-chatbot-premium-window-conversation-message-content-wrap {
            display: flex;
            flex-wrap: wrap;
            flex-grow: 1;
            max-width: 100%;
        }

        .lqd-ext-chatbot-premium-window-conversation-message-content {
            padding: 12px;
            border-radius: 11px;
            background-color: var(--lqd-ext-chat-chat-assistant-bg);
            color: var(--lqd-ext-chat-chat-assistant-color);
            transition: background-color 0.15s;
            max-width: 100%;
        }

        .lqd-ext-chatbot-premium-window-conversation-message-content p {
            margin: 0;
        }

        .lqd-ext-chatbot-premium-window-conversation-message-time {
            width: 100%;
            margin-top: 6px;
            font-size: 11px;
            font-weight: 500;
            opacity: 0.5;
        }

        .lqd-ext-chatbot-premium-window-conversation-message[data-type=user] {
            justify-content: end;
            text-align: end;
        }

        .lqd-ext-chatbot-premium-window-conversation-message[data-type=user] .lqd-ext-chatbot-premium-window-conversation-message-content-wrap,
        .lqd-ext-chatbot-premium-window-conversation-message[data-type=user] .lqd-ext-chatbot-premium-window-conversation-message-content {
            margin-inline-start: auto;
        }

        .lqd-ext-chatbot-premium-window-conversation-message[data-type=user] .lqd-ext-chatbot-premium-window-conversation-message-content {
            background-color: var(--lqd-ext-chat-primary);
            color: var(--lqd-ext-chat-primary-foreground);
        }

        .lqd-ext-chatbot-premium-window-conversation-message[data-type=user]+[data-type=user] .lqd-ext-chatbot-premium-window-conversation-message-avatar {
            visibility: hidden;
        }

        .lqd-ext-chatbot-premium-window-conversation-message[data-type=assistant]+[data-type=assistant] .lqd-ext-chatbot-premium-window-conversation-message-avatar {
            visibility: hidden;
        }

        .lqd-ext-chatbot-premium-window-form-wrap {
            display: grid;
            border-top: 1px solid hsl(0 0% 0% / 5%);
            flex-shrink: 0;
            height: var(--lqd-ext-chat-form-h);
            transition: opacity 0.3s, visibility 0.3s;
        }

        .lqd-ext-chatbot-premium-window-form {
            display: flex;
            width: 100%;
            height: 100%;
            grid-column: 1 / 1;
            grid-row: 1 / 1;
            margin: 0;
        }

        .lqd-ext-chatbot-premium-window-form textarea {
            resize: none;
            width: 100%;
            height: var(--lqd-ext-chat-form-h);
            border: none;
            padding: 20px;
            flex-grow: 1;
            font: inherit;
            outline: none;
            color: currentColor;
        }

        .lqd-ext-chatbot-premium-window-form button {
            display: inline-flex;
            width: var(--lqd-ext-chat-form-h);
            height: var(--lqd-ext-chat-form-h);
            align-items: center;
            justify-content: center;
            background: none;
            border: none;
            padding: 0;
            flex-shrink: 0;
            color: #000;
            transition: all 0.3s;
            cursor: pointer;
        }

        .lqd-ext-chatbot-premium-window-form button[disabled] {
            pointer-events: none;
            opacity: 0.35;
        }

        .lqd-ext-chatbot-premium-window-form button:hover {
            transform: scale(1.1);
        }

        .lqd-ext-chatbot-premium-window-foot {
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 12px;
            gap: 10px;
            text-align: center;
            font-size: 11px;
            line-height: 1.2em;
            color: hsl(0 0% 0% / 50%);
            background-color: var(--lqd-ext-chat-foot-bg);
        }

        .lqd-ext-chatbot-premium-window-foot p {
            margin: 0;
        }

        .lqd-ext-chatbot-premium-window-foot img {
            width: 16px;
            height: 16px;
        }

        .lqd-ext-chatbot-premium-window-foot a {
            color: inherit;
            text-decoration: none;
        }

        .lqd-ext-chatbot-premium[data-pos-x=right] {
            align-items: end;
        }

        .lqd-ext-chatbot-premium[data-pos-x=right] .lqd-ext-chatbot-premium-window {
            transform-origin: bottom right;
        }

        .lqd-ext-chatbot-premium[data-pos-x=right] .lqd-ext-chatbot-premium-welcome-bubble {
            left: auto;
            right: 0;
        }

        .lqd-ext-chatbot-premium[data-window-state=open] .lqd-ext-chatbot-premium-window {
            transform: scale(1) translateY(0);
            opacity: 1;
            visibility: visible;
            pointer-events: auto;
        }

        .lqd-ext-chatbot-premium[data-window-state=open] .lqd-ext-chatbot-premium-welcome-bubble {
            transform: scale(0.95);
            opacity: 0;
            visibility: hidden;
        }

        .lqd-ext-chatbot-premium[data-window-state=open] .lqd-ext-chatbot-premium-trigger:before {
            transform: translateY(0);
            opacity: 1;
        }

        .lqd-ext-chatbot-premium[data-window-state=open] .lqd-ext-chatbot-premium-trigger .lqd-ext-chatbot-premium-trigger-icon {
            opacity: 1;
            transform: translateY(0);
        }

        .lqd-ext-chatbot-premium[data-window-state=open] .lqd-ext-chatbot-premium-trigger .lqd-ext-chatbot-premium-trigger-img {
            opacity: 0;
            transform: translateY(-3px);
        }
    }

    @media (max-width: 767px) {
        .lqd-premium-chatbot {
            display: none !important;
        }
    }
</style>

<div class="lqd-premium-chatbot">
    <div class="lqd-chatbot-preview">
        <div
            class="lqd-ext-chatbot-premium"
            data-pos-x="right"
            data-pos-y="bottom"
            data-window-state="close"
            data-embedded="false"
            x-data="externalChatbotSupport"
            style="--lqd-ext-chat-primary: #272733; --lqd-ext-chat-trigger-background: #007bff;"
        >
            <div class="lqd-ext-chatbot-premium-window">
                <div class="lqd-ext-chatbot-premium-window-head">
                    <svg
                        class="lqd-ext-chatbot-premium-window-head-logo"
                        width="25"
                        height="25"
                        viewBox="0 0 25 25"
                        fill="currentColor"
                        xmlns="http://www.w3.org/2000/svg"
                    >
                        <path
                            d="M18.2404 21.8333L14.1279 17.6917L15.7612 16.0583L18.2404 18.5375L23.1987 13.5792L24.832 15.2417L18.2404 21.8333ZM0.332031 24.1667V3.16668C0.332031 2.52501 0.560503 1.9757 1.01745 1.51876C1.47439 1.06182 2.0237 0.833344 2.66536 0.833344H21.332C21.9737 0.833344 22.523 1.06182 22.9799 1.51876C23.4369 1.9757 23.6654 2.52501 23.6654 3.16668V11.3333H11.9987V19.5H4.9987L0.332031 24.1667Z"
                        />
                    </svg>
                    <h4 class="lqd-ext-chatbot-premium-window-head-title">
                        Help
                    </h4>
                </div>

                <div class="lqd-ext-chatbot-premium-window-conversations-wrap">
                    <div
                        class="lqd-ext-chatbot-premium-window-conversation-messages"
                        x-ref="conversationMessages"
                        x-show="currentView === 'conversation-messages'"
                        x-transition.opacity.duration.150ms
                        style="max-height: 100%; overflow-y: auto;"
                    >
                        <div
                            class="lqd-ext-chatbot-premium-window-conversation-message"
                            data-type="assistant"
                        >
                            <figure class="lqd-ext-chatbot-premium-window-conversation-message-avatar">
                                <img
                                    src="https://static.intercomassets.com/avatars/7780340/square_128/51-1728642351.jpg"
                                    alt="Bot Avatar"
                                    width="27"
                                    height="27"
                                >
                            </figure>
                            <div class="lqd-ext-chatbot-premium-window-conversation-message-content-wrap">
                                <div class="lqd-ext-chatbot-premium-window-conversation-message-content">
                                    <p>
                                        Hi, how can I help you?
                                    </p>
                                </div>
                                <div class="lqd-ext-chatbot-premium-window-conversation-message-time">
                                    MagicAI Help, just now
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="lqd-ext-chatbot-premium-window-form-wrap">
                    <form
                        class="lqd-ext-chatbot-premium-window-form"
                        @submit.prevent="onSendMessage"
                        x-show="currentView === 'conversation-messages'"
                        x-transition
                    >
                        <textarea
                            id="message"
                            name="message"
                            cols="30"
                            rows="3"
                            placeholder="@lang('Type your message here...')"
                            @keydown.enter.prevent="onMessageFieldHitEnter"
                            @input="onMessageFieldInput"
                            @input.throttle.50ms="$el.scrollTop = $el.scrollHeight"
                            x-ref="message"
                            style="min-height: 60px; max-height: 120px;"
                        ></textarea>
                        <button
                            type="submit"
                            title="@lang('Send message')"
                            x-ref="submitBtn"
                            :disabled="!$refs.message || !$refs.message.value.trim()"
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
                </div>

                <div class="lqd-ext-chatbot-premium-window-foot">
                    <img
                        width="16"
                        height="16"
                        src="https://demo.magicproject.ai/themes/default/assets/img/logo/magicAI-logo-Collapsed.png"
                        alt="MagicAI"
                    >
                    <p>
                        @lang('Powered by')
                        <u>
                            <a
                                href="#"
                                target="_blank"
                            >MagicAI</a>
                        </u>
                    </p>
                </div>
            </div>

            <div class="lqd-ext-chatbot-premium-welcome-bubble">
                <p>{{ __('Hi, how can I help you?') }}</p>
            </div>

            <button
                class="lqd-ext-chatbot-premium-trigger"
                type="button"
                @click.prevent="toggleWindowState()"
                style="background-color: black;"
            >
                <svg
                    class="lqd-ext-chatbot-premium-trigger-img"
                    fill="white"
                    xmlns="http://www.w3.org/2000/svg"
                    viewBox="0 0 28 32"
                >
                    <path
                        d="M28 32s-4.714-1.855-8.527-3.34H3.437C1.54 28.66 0 27.026 0 25.013V3.644C0 1.633 1.54 0 3.437 0h21.125c1.898 0 3.437 1.632 3.437 3.645v18.404H28V32zm-4.139-11.982a.88.88 0 00-1.292-.105c-.03.026-3.015 2.681-8.57 2.681-5.486 0-8.517-2.636-8.571-2.684a.88.88 0 00-1.29.107 1.01 1.01 0 00-.219.708.992.992 0 00.318.664c.142.128 3.537 3.15 9.762 3.15 6.226 0 9.621-3.022 9.763-3.15a.992.992 0 00.317-.664 1.01 1.01 0 00-.218-.707z"
                    ></path>
                </svg>

                <span class="lqd-ext-chatbot-premium-trigger-icon">
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
        </div>
    </div>
</div>

<!-- Static Templates -->
<template id="lqd-ext-chatbot-premium-user-msg-temp">
    <div
        class="lqd-ext-chatbot-premium-window-conversation-message"
        data-type="user"
    >
        <div class="lqd-ext-chatbot-premium-window-conversation-message-content-wrap">
            <div class="lqd-ext-chatbot-premium-window-conversation-message-content">
                <p></p>
            </div>
            <div class="lqd-ext-chatbot-premium-window-conversation-message-time"></div>
        </div>
    </div>
</template>

<template id="lqd-ext-chatbot-premium-assistant-msg-temp">
    <div
        class="lqd-ext-chatbot-premium-window-conversation-message"
        data-type="assistant"
    >
        <figure class="lqd-ext-chatbot-premium-window-conversation-message-avatar">
            <img
                src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='%23007bff'%3E%3Cpath d='M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z'/%3E%3C/svg%3E"
                alt="Bot Avatar"
                width="27"
                height="27"
            >
        </figure>
        <div class="lqd-ext-chatbot-premium-window-conversation-message-content-wrap">
            <div class="lqd-ext-chatbot-premium-window-conversation-message-content">
                <x-modal type="page">
                    <x-slot:trigger
                        custom
                    >
                        <p class="!mb-3">
                            {{ __('Need instant help? It\'s not available in your current package.') }}
                        </p>
                        <x-button
                            class="p-0 text-current underline"
                            href="#"
                            variant="link"
                            @click.prevent="toggleWindowState('close'); toggleModal();"
                        >
                            {{ __('Upgrade Now To Unlock Priority Support') }}
                            <x-tabler-bolt
                                class="size-5 shrink-0 fill-current"
                                stroke-width="0"
                            />
                        </x-button>
                    </x-slot:trigger>
                    <x-slot:modal>
                        @includeIf('premium-support.index')
                    </x-slot:modal>
                </x-modal>
            </div>
            <div class="lqd-ext-chatbot-premium-window-conversation-message-time"></div>
        </div>
    </div>
</template>
