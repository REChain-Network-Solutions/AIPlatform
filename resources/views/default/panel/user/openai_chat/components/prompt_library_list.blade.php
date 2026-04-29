<div
    class="grid grid-cols-1 gap-4 md:grid-cols-2"
    id="lqd-prompt-list"
>
    @forelse ($promptData as $prompt)
        <x-card
            class="bg-transparent hover:-translate-y-0.5 hover:shadow-xl hover:shadow-black/5"
            data-title="{{ str()->lower($prompt->title) }}"
            data-prompt="{{ str()->lower($prompt->prompt) }}"
            data-favorite="{{ $favData->pluck('item_id')->contains($prompt->id) ? 'true' : 'false' }}"
            variant="outline"
            size="sm"
            x-init
            x-show="(searchPromptStr === '' && promptFilter === 'all') || ($el.getAttribute('data-title').includes(searchPromptStr) || $el.getAttribute('data-prompt').includes(searchPromptStr)) && ( promptFilter === 'all' || (promptFilter === 'favorite' &&  $el.getAttribute('data-favorite') === 'true') )"
            @favorite-toggled="if($event.detail.id == {{ $prompt->id }}) { $el.setAttribute('data-favorite', $event.detail.isFavorite) }"
        >
            <div class="mb-3 flex gap-3">
                <h4 class="mb-0 mt-2 w-full text-lg font-semibold">
                    {{ $prompt->title }}
                </h4>

                <x-favorite-button
                    class="ms-auto shrink-0"
                    id="{{ $prompt->id }}"
                    is-favorite="{{ $favData->pluck('item_id')->contains($prompt->id) }}"
                    update-url="/dashboard/user/openai/chat/update-prompt"
                    @click="$el.closest('.lqd-card').setAttribute('data-favorite',  $el.closest('.lqd-card').getAttribute('data-favorite') === 'true' ? 'false' : 'true')"
                />
                <x-trash-button
                    class="shrink-0 hover:bg-red-500 hover:text-white"
                    id="{{ $prompt->id }}"
                    delete-url="/dashboard/user/openai/chat/delete-prompt"
                />
            </div>
            <p class="text-2xs font-normal">
                {{ $prompt->prompt }}
            </p>
            <a
                class="absolute inset-0 rounded-2xl"
                href="#"
                @click.prevent="setPrompt($el.closest('.lqd-card').getAttribute('data-prompt')); promptLibraryShow = false; focusOnPrompt()"
            >
                <span class="sr-only">{{ __('Add the template') }}</span>
            </a>
        </x-card>
    @empty
        <h4>{{ __('No Prompts, Please input new one') }}</h4>
    @endforelse
</div>
