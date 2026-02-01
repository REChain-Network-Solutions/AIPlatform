<template x-for="(message, index) in chatsList.find(c => c.id == activeChat)?.histories">
    <div
        class="lqd-ext-chatbot-history-message flex gap-2"
        data-sender="message.role"
        :class="{ 'flex-row-reverse': message.role === 'assistant' }"
        :key="'message-' + index"
    >
        <div class="lqd-ext-chatbot-history-message-content-wrap space-y-2 lg:max-w-[50%]">
            <div
                class="lqd-ext-chatbot-history-message-content rounded-xl p-3"
                :class="{
                    'bg-heading-foreground/5 text-heading-foreground': message.role === 'user',
                    'bg-secondary text-secondary-foreground ms-auto dark:bg-zinc-700 dark:text-primary-foreground': message.role === 'assistant'
                }"
            >
                <pre
                    class="prose m-0 w-full whitespace-normal font-body text-sm text-current dark:prose-invert"
                    :class="{ 'text-heading-foreground': message.role === 'user', 'text-black': message.role === 'assistant' }"
                    x-html="getFormattedString(message.message)"
                ></pre>
            </div>
            <div class="lqd-ext-chatbot-history-message-time text-[11px] opacity-50">
                <p
                    class="m-0"
                    :class="{ 'text-end': message.role === 'assistant' }"
                    x-text="new Date(message.created_at).toLocaleString()"
                ></p>
            </div>
        </div>
    </div>
</template>
