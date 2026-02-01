{{-- This is the frontend ui but with some changes for the editor --}}

@push('css')
    <style>
        .lqd-chatbot-preview .lqd-ext-chatbot-window {
            width: var(--lqd-ext-chat-window-w);
            height: var(--lqd-ext-chat-window-h);
        }
    </style>
@endpush

<div class="lqd-chatbot-preview sticky bottom-8">
    @include('chatbot::frontend-ui.frontend-ui', [
        'is_editor' => true,
        'is_iframe' => false,
    ])
</div>
