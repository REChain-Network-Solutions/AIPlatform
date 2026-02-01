@if (!$chatbot['active'])
    <p>
        @lang('This chatbot is not active.')
    </p>
@else
    @include('chatbot::frontend-ui.frontend-ui', [
        'is_editor' => false,
        'is_iframe' => true,
        'session' => $session,
        'chatbot' => $chatbot,
        'conversations' => $conversations,
        'routes' => [
            'index' => route('api.v2.chatbot.index', [$chatbot->getAttribute('uuid'), $session]),
            'getSession' => route('api.v2.chatbot.index.session', [$chatbot->getAttribute('uuid'), $session]),
            'conversations' => route('api.v2.chatbot.conversion.store', [$chatbot->getAttribute('uuid'), $session]),
        ],
    ])
@endif
