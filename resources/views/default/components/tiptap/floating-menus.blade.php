{{-- Floating menus --}}
<div
    class="tiptap-floating-menu relative z-10 flex gap-px rounded-lg bg-surface-background p-0.5 shadow-lg shadow-black/5"
    x-cloak
>
    {{-- h1 button --}}

    <button
        class="inline-gird size-8 place-items-center rounded-md bg-transparent leading-none transition hover:bg-foreground/5 [&.is-active]:bg-primary [&.is-active]:text-primary-foreground"
        @click.prevent="$store.tiptapEditor.setHeading({level: 1})"
        title="{{ __('H1') }}"
    >
        <x-tabler-h-1 class="size-4" />
    </button>

    {{-- h1 button --}}
    <button
        class="inline-gird size-8 place-items-center rounded-md bg-transparent leading-none transition hover:bg-foreground/5 [&.is-active]:bg-primary [&.is-active]:text-primary-foreground"
        @click.prevent="$store.tiptapEditor.setHeading({level: 2})"
        title="{{ __('H2') }}"
    >
        <x-tabler-h-2 class="size-4" />
    </button>

    {{-- ordered list --}}
    <button
        class="inline-gird size-8 place-items-center rounded-md bg-transparent leading-none transition hover:bg-foreground/5 [&.is-active]:bg-primary [&.is-active]:text-primary-foreground"
        @click.prevent="$store.tiptapEditor.toggleOrderedList()"
        title="{{ __('Numeric List') }}"
        :class="$store.tiptapEditor.isActive('orderedList', $store.tiptapEditor._updated_at) ? 'is-active' : ''"
    >
        <x-tabler-list-numbers class="size-4" />
    </button>

    {{-- bullet list --}}
    <button
        class="inline-gird size-8 place-items-center rounded-md bg-transparent leading-none transition hover:bg-foreground/5 [&.is-active]:bg-primary [&.is-active]:text-primary-foreground"
        @click.prevent="$store.tiptapEditor.toggleBulletList()"
        title="{{ __('Bullet List') }}"
        :class="$store.tiptapEditor.isActive('bulletList', $store.tiptapEditor._updated_at) ? 'is-active' : ''"
    >
        <x-tabler-list class="size-4" />
    </button>

    {{-- code block --}}
    <button
        class="inline-gird size-8 place-items-center rounded-md bg-transparent leading-none transition hover:bg-foreground/5 [&.is-active]:bg-primary [&.is-active]:text-primary-foreground"
        @click.prevent="$store.tiptapEditor.setCodeBlock()"
        title="{{ __('Code Block') }}"
        :disabled="$store.tiptapEditor.isActive('codeBlock', $store.tiptapEditor._updated_at)"
    >
        <x-tabler-source-code class="size-4" />
    </button>

    {{-- blockquote button --}}
    <button
        class="inline-gird size-8 place-items-center rounded-md bg-transparent leading-none transition hover:bg-foreground/5 [&.is-active]:bg-primary [&.is-active]:text-primary-foreground"
        @click.prevent="$store.tiptapEditor.toggleBlockquote()"
        title="{{ __('Blockquote') }}"
        :class="$store.tiptapEditor.isActive('blockquote', $store.tiptapEditor._updated_at) ? 'is-active' : ''"
    >
        <x-tabler-blockquote class="size-4" />
    </button>
</div>
