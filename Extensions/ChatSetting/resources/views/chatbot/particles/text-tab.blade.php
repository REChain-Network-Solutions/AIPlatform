<div
        class="hidden"
        :class="{ 'hidden': activeTab !== 'text' }"
>
    <form
            class="flex flex-col gap-5"
            id="add-form"
            action="{{ route('dashboard.user.chat-setting.chatbot.text', $item) }}"
    >
        @csrf

        <x-form-step
                step="1"
                label="{{ __('Add Text') }}"
        />

        <input
                id="text_id"
                type="hidden"
                name="text_id"
                value=""
        >

        <x-forms.input
                id="title"
                name="title"
                size="lg"
                placeholder="{{ __('Type your title here') }}"
                label="{{ __('Title') }}"
        />

        <x-forms.input
                id="content_text"
                name="text"
                placeholder="{{ __('Type your text here') }}"
                label="{{ __('Text') }}"
                rows="6"
                type="textarea"
                size="lg"
        />

        <x-button
                class="w-full"
                data-submit="addtext"
                data-form="#add-form"
                data-list="#text-list"
                variant="success"
                type="button"
        >
            <x-tabler-plus class="size-4" />
            @lang('Add')
        </x-button>

    </form>

    <form
            class="mt-10 flex flex-col gap-5"
            id="form-train-text"
            method="post"
            action="{{ route('dashboard.user.chat-setting.chatbot.training', $item->id) }}"
    >
        @php
            $texts = $data->where('type', 'text');
        @endphp

        <input
                type="hidden"
                name="type"
                value="text"
        >
        <x-form-step
                id="text-list-alert"
                step="2"
                label="{{ __('Text list') }}"
                @class(['text-list-alert', 'hidden' => !$texts->count()])
        />

        <div
                class="text-list list-common space-y-4"
                id="text-list"
        >
            @include('chat-setting::chatbot.particles.text.list', ['items' => $texts])
        </div>
    </form>
</div>
