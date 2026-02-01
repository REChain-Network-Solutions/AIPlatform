<div
    class="col-start-1 col-end-1 row-start-1 row-end-1"
    x-show="activeTab === 'pdf'"
    x-transition.opacity.150ms
>
    <form
        class="mb-4 flex w-full flex-col gap-5"
        @submit.prevent="uploadFile"
        action="{{ route('dashboard.chatbot.train.file') }}"
    >
        @csrf

        <x-form-step
            step="1"
            label="{{ __('Add File') }}"
        />
        <label
            class="group flex min-h-56 w-full cursor-pointer flex-col items-center justify-center rounded-lg border border-dashed border-foreground/10 bg-background text-center text-[12px] transition-colors hover:bg-background/80"
            for="train-file"
        >
            <div class="flex flex-col items-center justify-center py-6">
                <x-tabler-circle-plus
                    class="mb-3.5 size-11"
                    stroke-width="1"
                    x-show="!uploading"
                />
                <x-tabler-refresh
                    class="mb-3.5 size-11 animate-spin"
                    stroke-width="1"
                    x-show="uploading"
                />

                <p
                    class="mb-1 font-semibold"
                    data-original-text="{{ __('UPLOAD PDF, XLSX, CSV') }}"
                    x-ref="fileName"
                >
                    {{ __('UPLOAD PDF, XLSX, CSV') }}
                </p>

                <p
                    class="mb-0"
                    x-text="!uploading ? '{{ __('Upload a File (Max: 25Mb)') }}' : '{{ __('UPLOADING...') }}'"
                >
                    {{ __('Upload a File (Max: 25Mb)') }}
                </p>
            </div>

            <input
                class="hidden"
                id="train-file"
                name="file"
                type="file"
                @change="if ($event.target?.files[0]) $refs.fileName.innerText = $event.target.files[0].name"
            />
        </label>

        <x-button
            class="group w-full"
            variant="success"
            type="submit"
            ::disabled="uploading || fetching"
        >
            <x-tabler-plus class="size-4" />
            {{ 'Upload' }}
        </x-button>
    </form>

    <form
        class="mt-14"
        action="{{ route('dashboard.chatbot.train.generate.embedding') }}"
        @submit.prevent="trainEmbeddings"
    >
        @csrf

        <x-form-step
            class="mb-6"
            step="2"
            label="{{ __('Select Pages') }}"
            x-show="embeddings.filter(e => e.type === 'file').length"
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
            <template x-for="embedding in embeddings.filter(e => e.type === 'file')">
                <div
                    class="flex items-center justify-between rounded-lg border p-1.5"
                    :key="'file-item-' + embedding.id"
                >
                    <x-forms.input
                        class:container="grow"
                        class:custom-wrap="size-7"
                        class:label="text-foreground"
                        data-type="file"
                        ::id="'train-file-item-' + embedding.id"
                        name="embedding-item"
                        type="checkbox"
                        ::value="embedding.id"
                        ::checked="embedding.status === 'Trained'"
                        x-init="$el.closest('label').setAttribute('for', 'train-file-item-' + embedding.id);"
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
                x-show="embeddings.filter(e => e.type === 'file').length"
                ::disabled="fetching"
            >
                @lang('Train GPT')
            </x-button>
        </div>
    </form>
</div>
