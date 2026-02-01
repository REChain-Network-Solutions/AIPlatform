<div
    class="col-start-1 col-end-1 row-start-1 row-end-1"
    x-show="activeTab === 'qa'"
    x-transition.opacity.150ms
>
    <form
        class="flex flex-col gap-5"
        action="{{ route('dashboard.chatbot.train.qa') }}"
        @submit.prevent="ev => addText(ev, 'qa')"
    >
        <x-form-step
            step="1"
            label="{{ __('Add Q/A') }}"
        />

        <x-forms.input
            name="question"
            size="lg"
            placeholder="{{ __('Type your title here') }}"
            label="{{ __('Title') }}"
            ::value="editingItem.title"
        />

        <x-forms.input
            name="answer"
            placeholder="{{ __('Type your text here') }}"
            label="{{ __('Text') }}"
            rows="6"
            type="textarea"
            size="lg"
            ::value="editingItem.content"
        />

        <x-button
            class="group w-full"
            variant="success"
            type="submit"
            ::disabled="fetching"
        >
            <x-tabler-plus
                class="size-4"
                x-show="!editingItem.id"
            />
            <span x-text="editingItem.id ? '{{ __('Update') }}' : '{{ __('Add') }}'"></span>
        </x-button>
    </form>

    <form
        class="mt-10 flex flex-col gap-5"
        action="{{ route('dashboard.chatbot.train.generate.embedding') }}"
        @submit.prevent="trainEmbeddings"
    >
        <x-form-step
            step="2"
            label="{{ __('Manage Content') }}"
            x-show="embeddings.filter(e => e.type === 'qa').length"
        >
            <button
                class="group ms-auto text-2xs font-semibold"
                type="button"
                @click="toggleSelectAll"
            >
                <span class="group-[&.has-selected]:hidden">
                    @lang('Select All')
                </span>
                <span class="hidden group-[&.has-selected]:block">
                    @lang('Deselect All')
                </span>
            </button>
        </x-form-step>

        <div class="space-y-4">
            <template x-for="embedding in embeddings.filter(e => e.type === 'qa')">
                <div
                    class="flex items-center justify-between rounded-lg border p-1.5"
                    :key="'text-item-' + embedding.id"
                >
                    <x-forms.input
                        class:container="grow"
                        class:custom-wrap="size-7"
                        class:label="text-foreground"
                        data-type="qa"
                        ::id="'train-qa-item-' + embedding.id"
                        name="embedding-item"
                        type="checkbox"
                        ::value="embedding.id"
                        ::checked="embedding.status === 'Trained'"
                        x-init="$el.closest('label').setAttribute('for', 'train-qa-item-' + embedding.id);"
                        custom
                    >
                        <x-slot:label>
                            <span x-text="embedding.title"></span>
                        </x-slot:label>
                    </x-forms.input>

                    <div class="flex items-center justify-between gap-1">
                        <x-badge
                            class="whitespace-nowrap text-2xs hover:translate-y-0 hover:shadow-none"
                            ::class="{ '!bg-green-500/15': embedding.status === 'Trained', '!text-green-500': embedding.status === 'Trained' }"
                        >
                            <span x-text="embedding.status"></span>
                        </x-badge>
                        {{-- <x-button
                            class="size-7 inline-flex items-center justify-center"
                            variant="outline"
                            size="none"
                            @click.prevent="setEditingItem(embedding.id)"
                        >
                            <x-tabler-pencil class="size-5" />
                            <span class="sr-only">{{ __('Edit') }}</span>
                        </x-button> --}}
                        <x-button
                            class="inline-flex items-center justify-center text-red-600"
                            variant="link"
                            size="none"
                            @click.prevent="deleteEmbedding(embedding.id)"
                            ::disabled="fetching"
                        >
                            <x-tabler-circle-minus
                                class="size-7"
                                stroke-width="1.5"
                            />
                            <span class="sr-only">
                                {{ __('Delete') }}
                            </span>
                        </x-button>
                    </div>
                </div>
            </template>

            <x-button
                class="!mt-8 w-full"
                size="lg"
                type="submit"
                x-show="embeddings.filter(e => e.type === 'qa').length"
                ::disabled="fetching"
            >
                @lang('Train GPT')
            </x-button>
        </div>
    </form>
</div>
