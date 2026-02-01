@for ($i = 0; $i < 6; $i++)
    <div class="lqd-ext-chatbot-window-conversations-list-item">
        <figure class="lqd-ext-chatbot-window-conversations-list-item-fig">
            <img
                :src="() => activeChatbot.avatar ? `${window.location.origin}/${activeChatbot.avatar}` : ''"
                width="27"
                height="27"
            >
        </figure>
        <div class="lqd-ext-chatbot-window-conversations-list-item-info">
            <p class="lqd-ext-chatbot-window-conversations-list-item-info-name">
                MagicAI Bot
            </p>
            <p class="lqd-ext-chatbot-window-conversations-list-item-info-last-message">
                {{ [
                    'I noted some aspects of the platform that need...',
                    'The chatbot interface is very user-friendly and...',
                    'I had a great experience using the chatbot for...',
                    'The response time of the chatbot is impressive...',
                    'I found the chatbot to be very helpful in...',
                    'The design of the chatbot is sleek and modern...',
                ][rand(0, 5)] }}
            </p>
        </div>
        <div class="lqd-ext-chatbot-window-conversations-list-item-time">
            {{ now()->subMinutes(rand(120, 14400))->format('M d') }}
        </div>
        <a
            class="lqd-ext-chatbot-window-conversations-list-item-link"
            data-id="0"
            href="#"
            title="{{ __('View Messages') }}"
            @click.prevent="openConversation(0, false)"
        >
        </a>
    </div>
@endfor
