<div
    class="lqd-ext-chatbot-window-conversation-message"
    data-type="assistant"
>
    <figure class="lqd-ext-chatbot-window-conversation-message-avatar">
        <img
            :src="() => activeChatbot.avatar ? `${window.location.origin}/${activeChatbot.avatar}` : ''"
            width="27"
            height="27"
        >
    </figure>
    <div class="lqd-ext-chatbot-window-conversation-message-content-wrap">
        <div class="lqd-ext-chatbot-window-conversation-message-content">
            <p @if ($is_editor) x-text="activeChatbot.welcome_message" @endif>@lang('Hi, how can I help you?')</p>
        </div>
        <div
            class="lqd-ext-chatbot-window-conversation-message-time"
            @if ($is_editor) x-show="activeChatbot.show_date_and_time" @endif
        >
            @lang('SupportHub, 3 min ago')
        </div>
    </div>
</div>

<div
    class="lqd-ext-chatbot-window-conversation-message"
    data-type="user"
>
    <div class="lqd-ext-chatbot-window-conversation-message-content-wrap">
        <div class="lqd-ext-chatbot-window-conversation-message-content">
            <p>@lang('I need to make a refund.')</p>
        </div>
        <div
            class="lqd-ext-chatbot-window-conversation-message-time"
            @if ($is_editor) x-show="activeChatbot.show_date_and_time" @endif
        >
            @lang('You, 3 min ago')
        </div>
    </div>
</div>

<div
    class="lqd-ext-chatbot-window-conversation-message"
    data-type="assistant"
>
    <figure class="lqd-ext-chatbot-window-conversation-message-avatar">
        <img
            :src="() => activeChatbot.avatar ? `${window.location.origin}/${activeChatbot.avatar}` : ''"
            width="27"
            height="27"
        >
    </figure>
    <div class="lqd-ext-chatbot-window-conversation-message-content-wrap">
        <div class="lqd-ext-chatbot-window-conversation-message-content">
            <p>@lang('A refund will be provided after we process your return item at our facilities. It may take additional time for your financial institution to  process the refund.')🤔</p>
        </div>
        <div
            class="lqd-ext-chatbot-window-conversation-message-time"
            @if ($is_editor) x-show="activeChatbot.show_date_and_time" @endif
        >
            @lang('20.08.2024 / 15:29:21')
        </div>
    </div>
</div>

<div
    class="lqd-ext-chatbot-window-conversation-message"
    data-type="assistant"
>
    <figure class="lqd-ext-chatbot-window-conversation-message-avatar">
        <img
            :src="() => activeChatbot.avatar ? `${window.location.origin}/${activeChatbot.avatar}` : ''"
            width="27"
            height="27"
        >
    </figure>
    <div class="lqd-ext-chatbot-window-conversation-message-content-wrap">
        <div class="lqd-ext-chatbot-window-conversation-message-content">
            <p>🤔</p>
        </div>
        <div
            class="lqd-ext-chatbot-window-conversation-message-time"
            @if ($is_editor) x-show="activeChatbot.show_date_and_time" @endif
        >
            @lang('20.08.2024 / 15:29:21')
        </div>
    </div>
</div>
