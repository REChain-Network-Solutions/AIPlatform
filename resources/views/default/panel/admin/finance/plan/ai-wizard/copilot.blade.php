<div class="min-w-0 lg:sticky lg:top-0 lg:self-start">
    <x-card
        class="flex h-[min(70vh,640px)] w-full flex-col"
        class:body="flex min-h-0 grow flex-col"
        size="none"
    >
        <x-slot:head>
            <div class="flex items-center gap-2">
                <span class="grid size-8 place-items-center rounded-full bg-primary/10 text-primary">
                    <x-tabler-message-chatbot class="size-5" />
                </span>
                <div>
                    <h3 class="m-0 text-sm font-semibold">{{ __('CoPilot') }}</h3>
                    <p class="m-0 text-2xs text-foreground/60">{{ __('Chat to build your plan') }}</p>
                </div>
            </div>
        </x-slot:head>

        <div
            id="ai-wizard-chat-scroll"
            class="flex grow flex-col gap-3 overflow-y-auto p-4"
        >
            <div
                class="rounded-card bg-amber-500/10 p-3 text-xs text-amber-600 dark:text-amber-300"
                x-show="isDemo"
            >
                {{ __('This feature is disabled in demo mode.') }}
            </div>

            <div class="rounded-card bg-surface p-3 text-xs text-foreground/70">
                {{ __('Hi! Tell me about the plan you want — for example:') }}
                <span class="mt-1 block italic">
                    {{ __('"A $29 monthly Pro plan with a 7-day trial and generous credits"') }}
                </span>
            </div>

            <template x-for="(message, index) in messages" :key="index">
                <div
                    class="flex max-w-[90%] flex-col gap-1.5 rounded-card p-3 text-xs [&.assistant]:self-start [&.assistant]:bg-surface [&.user]:self-end [&.user]:bg-primary/10"
                    :class="message.role"
                    x-show="message.role === 'user' || displayContent(message).trim().length > 0"
                >
                    <template x-if="message.role === 'assistant'">
                        <span
                            class="[&_a]:text-primary [&_a]:underline [&_li+li]:mt-1 [&_ol]:m-0 [&_ol]:list-decimal [&_ol]:ps-4 [&_p+p]:mt-2 [&_p]:m-0 [&_ul]:m-0 [&_ul]:list-disc [&_ul]:ps-4"
                            x-html="renderMarkdown(displayContent(message))"
                        ></span>
                    </template>
                    <template x-if="message.role === 'user'">
                        <span
                            class="whitespace-pre-wrap"
                            x-text="message.content"
                        ></span>
                    </template>
                    <span
                        class="inline-flex items-center gap-1 self-start text-2xs font-medium text-emerald-600 dark:text-emerald-400"
                        x-show="message.updated"
                    >
                        <x-tabler-check class="size-3.5" />
                        {{ __('Draft updated') }}
                    </span>
                </div>
            </template>

            <template x-if="streaming && !streamHasContent">
                <div class="self-start">
                    <x-shimmar />
                </div>
            </template>
        </div>

        <x-slot:foot>
            <div class="flex flex-col gap-2 p-3">
                <div
                    class="flex flex-wrap gap-1.5"
                    x-show="messages.length === 0"
                >
                    <template x-for="prompt in quickPrompts" :key="prompt">
                        <button
                            type="button"
                            class="rounded-full border border-card-border px-2.5 py-1 text-2xs text-foreground/70 transition-colors hover:border-primary hover:text-primary"
                            x-text="prompt"
                            @click="chatInput = prompt; sendMessage()"
                        ></button>
                    </template>
                </div>
                <form
                    class="flex items-center gap-2"
                    @submit.prevent="sendMessage()"
                >
                    <x-forms.input
                        id="ai-wizard-chat-input"
                        containerClass="grow"
                        placeholder="{{ __('Ask the CoPilot...') }}"
                        x-model="chatInput"
                        ::disabled="streaming"
                    />
                    <x-button
                        class="size-9 shrink-0"
                        size="none"
                        variant="primary"
                        type="submit"
                        title="{{ __('Send') }}"
                        ::disabled="streaming || !chatInput.trim()"
                    >
                        <x-tabler-send class="size-4" />
                    </x-button>
                </form>
            </div>
        </x-slot:foot>
    </x-card>
</div>
