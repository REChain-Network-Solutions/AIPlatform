{{-- Bubble menus --}}
<div
    class="tiptap-bubble-menu relative z-10 flex gap-px rounded-lg bg-surface-background p-0.5 shadow-lg shadow-black/5"
    x-cloak
>
    {{-- bold button --}}
    <button
        class="inline-gird size-8 place-items-center rounded-md bg-transparent leading-none transition hover:bg-foreground/5 [&.is-active]:bg-primary [&.is-active]:text-primary-foreground"
        @click.prevent="$store.tiptapEditor.toggleBold()"
        title="{{ __('Bold') }}"
        :class="$store.tiptapEditor.isActive('bold', $store.tiptapEditor._updated_at) ? 'is-active' : ''"
    >
        <x-tabler-bold class="size-4" />
    </button>

    {{-- italic button --}}
    <button
        class="inline-gird size-8 place-items-center rounded-md bg-transparent leading-none transition hover:bg-foreground/5 [&.is-active]:bg-primary [&.is-active]:text-primary-foreground"
        @click.prevent="$store.tiptapEditor.toggleItalic()"
        title="{{ __('Italic') }}"
        :class="$store.tiptapEditor.isActive('italic', $store.tiptapEditor._updated_at) ? 'is-active' : ''"
    >
        <x-tabler-italic class="size-4" />
    </button>

    {{-- strike button --}}
    <button
        class="inline-gird size-8 place-items-center rounded-md bg-transparent leading-none transition hover:bg-foreground/5 [&.is-active]:bg-primary [&.is-active]:text-primary-foreground"
        @click.prevent="$store.tiptapEditor.toggleStrike()"
        title="{{ __('Strike / Line through') }}"
        :class="$store.tiptapEditor.isActive('strike', $store.tiptapEditor._updated_at) ? 'is-active' : ''"
    >
        <x-tabler-strikethrough class="size-4" />
    </button>

    {{-- underline button --}}
    <button
        class="inline-gird size-8 place-items-center rounded-md bg-transparent leading-none transition hover:bg-foreground/5 [&.is-active]:bg-primary [&.is-active]:text-primary-foreground"
        @click.prevent="$store.tiptapEditor.toggleUnderline()"
        title="{{ __('Underline') }}"
        :class="$store.tiptapEditor.isActive('underline', $store.tiptapEditor._updated_at) ? 'is-active' : ''"
    >
        <x-tabler-underline class="size-4" />
    </button>

    {{-- link button --}}
    <button
        class="inline-gird size-8 place-items-center rounded-md bg-transparent leading-none transition hover:bg-foreground/5 [&.is-active]:bg-primary [&.is-active]:text-primary-foreground"
        @click.prevent="$store.tiptapEditor.isActive('link', $store.tiptapEditor._updated_at) ? $store.tiptapEditor.unsetLink() : $store.tiptapEditor.setLink()"
        title="{{ __('Link') }}"
        :class="$store.tiptapEditor.isActive('link', $store.tiptapEditor._updated_at) ? 'is-active' : ''"
    >
        <x-tabler-link
            class="size-4"
            x-show="!$store.tiptapEditor.isActive('link', $store.tiptapEditor._updated_at)"
        />
        <x-tabler-link-off
            class="size-4"
            x-show="$store.tiptapEditor.isActive('link', $store.tiptapEditor._updated_at)"
        />
    </button>

    {{-- code button --}}
    <button
        class="inline-gird size-8 place-items-center rounded-md bg-transparent leading-none transition hover:bg-foreground/5 [&.is-active]:bg-primary [&.is-active]:text-primary-foreground"
        @click.prevent="$store.tiptapEditor.toggleCode()"
        title="{{ __('Inline Code') }}"
        :class="$store.tiptapEditor.isActive('code', $store.tiptapEditor._updated_at) ? 'is-active' : ''"
    >
        <x-tabler-code class="size-4" />
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
        @click.prevent="$store.tiptapEditor.toggleCodeBlock()"
        title="{{ __('Code Block') }}"
        :class="$store.tiptapEditor.isActive('codeBlock', $store.tiptapEditor._updated_at) ? 'is-active' : ''"
    >
        <x-tabler-source-code class="size-4" />
    </button>

    {{-- heading --}}
    <x-dropdown.dropdown :teleport="false">
        <x-slot:trigger
            class="inline-gird size-8 place-items-center rounded-md bg-transparent leading-none transition hover:bg-foreground/5 [&.is-active]:bg-primary [&.is-active]:text-primary-foreground"
            title="{{ __('Headings') }}"
        >
            <x-tabler-heading class="size-4" />
        </x-slot:trigger>
        <x-slot:dropdown>
            <div class="flex flex-col">
                <button
                    class="w-full rounded-md px-2 py-1 text-start leading-none transition hover:bg-foreground/5 [&.is-active]:bg-primary [&.is-active]:text-primary-foreground"
                    @click.prevent="$store.tiptapEditor.toggleHeading({level: 1})"
                    :class="$store.tiptapEditor.isActive('heading', { level: 1 }) ? 'is-active' : ''"
                >
                    <h1 class="mb-0 text-current">H1</h1>
                </button>
                <button
                    class="w-full rounded-md px-2 py-1 text-start leading-none transition hover:bg-foreground/5 [&.is-active]:bg-primary [&.is-active]:text-primary-foreground"
                    @click.prevent="$store.tiptapEditor.toggleHeading({level: 2})"
                    :class="$store.tiptapEditor.isActive('heading', { level: 2 }) ? 'is-active' : ''"
                >
                    <h2 class="mb-0 text-current">H2</h2>
                </button>
                <button
                    class="w-full rounded-md px-2 py-1 text-start leading-none transition hover:bg-foreground/5 [&.is-active]:bg-primary [&.is-active]:text-primary-foreground"
                    @click.prevent="$store.tiptapEditor.toggleHeading({level: 3})"
                    :class="$store.tiptapEditor.isActive('heading', { level: 3 }) ? 'is-active' : ''"
                >
                    <h3 class="mb-0 text-current">H3</h3>
                </button>
                <button
                    class="w-full rounded-md px-2 py-1 text-start leading-none transition hover:bg-foreground/5 [&.is-active]:bg-primary [&.is-active]:text-primary-foreground"
                    @click.prevent="$store.tiptapEditor.toggleHeading({level: 4})"
                    :class="$store.tiptapEditor.isActive('heading', { level: 4 }) ? 'is-active' : ''"
                >
                    <h4 class="mb-0 text-current">H4</h4>
                </button>
                <button
                    class="w-full rounded-md px-2 py-1 text-start leading-none transition hover:bg-foreground/5 [&.is-active]:bg-primary [&.is-active]:text-primary-foreground"
                    @click.prevent="$store.tiptapEditor.toggleHeading({level: 5})"
                    :class="$store.tiptapEditor.isActive('heading', { level: 5 }) ? 'is-active' : ''"
                >
                    <h5 class="mb-0 text-current">H5</h5>
                </button>
                <button
                    class="w-full rounded-md px-2 py-1 text-start leading-none transition hover:bg-foreground/5 [&.is-active]:bg-primary [&.is-active]:text-primary-foreground"
                    @click.prevent="$store.tiptapEditor.toggleHeading({level: 6})"
                    :class="$store.tiptapEditor.isActive('heading', { level: 6 }) ? 'is-active' : ''"
                >
                    <h6 class="mb-0 text-current">H6</h6>
                </button>
            </div>
        </x-slot:dropdown>
    </x-dropdown.dropdown>
</div>
