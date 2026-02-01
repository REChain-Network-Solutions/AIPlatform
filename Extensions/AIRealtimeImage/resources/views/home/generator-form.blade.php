<div class="lqd-realtime-image-form-wrap pb-6 pt-16">
    <h1 class="mb-11 text-center lg:text-5xl lg:tracking-tight [&_span]:opacity-50">
        {!! __('Create <span>AI Images in Real-time</span>') !!}
    </h1>

    <form
        class="lqd-realtime-image-form relative flex min-h-16 w-full flex-wrap rounded-full bg-background/60 font-medium shadow-lg shadow-black/5 sm:text-2xs lg:flex-nowrap"
        action="#"
        @submit.prevent="changePrompt"
    >
        @csrf

        <x-forms.input
            class:container="flex border-e border-heading-foreground/5 max-w-36 max-sm:w-full min-h-16"
            class="h-auto w-full appearance-none overflow-hidden text-ellipsis whitespace-nowrap rounded-s-full border-none bg-transparent py-0 pe-5 ps-8 text-heading-foreground/90"
            id="image_style"
            type="select"
            name="image_style"
            x-model="imageStyle"
            @change="onPromptInput"
        >
            @foreach ($image_styles as $key => $label)
                <option value="{{ $key }}">
                    {{ $label }}
                </option>
            @endforeach
        </x-forms.input>

        <div class="flex min-h-16 grow ps-5">
            <x-button
                class="size-8 shrink-0 justify-center self-center text-center hover:translate-y-0 hover:scale-105 hover:shadow-none"
                title="{{ __('Browse pre-defined prompts') }}"
                variant="none"
                size="none"
                @click.prevent="togglePromptLibraryShow()"
            >
                <svg
                    width="19"
                    height="20"
                    viewBox="0 0 19 20"
                    fill="currentColor"
                    xmlns="http://www.w3.org/2000/svg"
                >
                    <path
                        d="M1 16.0212C1.1795 15.9071 1.37075 15.8109 1.57375 15.7327C1.77675 15.6546 1.99575 15.6155 2.23075 15.6155H3.3845V3H2.23075C1.88208 3 1.58975 3.1215 1.35375 3.3645C1.11792 3.60733 1 3.89608 1 4.23075V16.0212ZM2.23075 20C1.61108 20 1.08442 19.7868 0.65075 19.3605C0.216917 18.9343 0 18.4167 0 17.8078V4.23075C0 3.61108 0.216917 3.08442 0.65075 2.65075C1.08442 2.21692 1.61108 2 2.23075 2H9.5V3H4.3845V15.6155H11.6155V11.5H12.6155V16.6155H2.23075C1.89608 16.6155 1.60733 16.7294 1.3645 16.9573C1.1215 17.1851 1 17.4674 1 17.8043C1 18.1411 1.1215 18.4246 1.3645 18.6548C1.60733 18.8849 1.89608 19 2.23075 19H15V10.5H16V20H2.23075ZM13.5 10.5C13.5 9.106 13.9848 7.92417 14.9545 6.9545C15.9242 5.98483 17.106 5.5 18.5 5.5C17.106 5.5 15.9242 5.01517 14.9545 4.0455C13.9848 3.07583 13.5 1.894 13.5 0.5C13.5 1.894 13.0152 3.07583 12.0455 4.0455C11.0758 5.01517 9.894 5.5 8.5 5.5C9.894 5.5 11.0758 5.98483 12.0455 6.9545C13.0152 7.92417 13.5 9.106 13.5 10.5Z"
                    />
                </svg>
            </x-button>

            <x-forms.input
                class:container="grow flex"
                class="h-auto rounded-none border-none bg-transparent px-2 placeholder:text-heading-foreground/90 focus:ring-0 max-sm:text-[11px]"
                placeholder="{{ __('Describe your idea or select a pre-defined prompt') }}"
                name="prompt"
                x-model="prompt"
                x-ref="prompt"
                @input="onPromptInput"
            />
        </div>

        <input
            type="hidden"
            name="image_style"
            value=""
        />

        <div class="ms-auto flex min-h-16 items-center gap-6 max-sm:w-full max-sm:flex-col-reverse max-sm:flex-wrap sm:pe-8">
            <x-button
                class="w-full items-center justify-center bg-gradient-to-br from-[#CA99E7] via-[#4FACBE] to-[#8364FF] p-3 disabled:text-primary-foreground disabled:opacity-50 sm:size-10 sm:p-0"
                type="submit"
                size="none"
                title="{{ __('Generate Image') }}"
                ::disabled="busy"
            >
                <span class="sm:hidden">
                    {{ __('Generate') }}
                </span>
                <span class="inline-grid place-content-center sm:size-full">
                    <svg
                        class="col-start-1 col-end-1 row-start-1 row-end-1"
                        width="14"
                        height="14"
                        viewBox="0 0 14 14"
                        xmlns="http://www.w3.org/2000/svg"
                        fill="currentColor"
                        x-show="!busy"
                    >
                        <path
                            fill-rule="evenodd"
                            clip-rule="evenodd"
                            d="M2.85637 0.654433C3.04651 -0.214165 4.28328 -0.219571 4.481 0.647332L4.49082 0.690566C4.49756 0.720292 4.50384 0.748056 4.51058 0.77677C4.75709 1.82667 5.60676 2.6289 6.67036 2.81393C7.57548 2.9714 7.57548 4.27074 6.67036 4.42821C5.60109 4.61423 4.74804 5.42404 4.50669 6.4822L4.481 6.59481C4.28328 7.46172 3.04651 7.4563 2.85637 6.58771L2.83522 6.49108C2.6027 5.42891 1.75069 4.61298 0.679451 4.42661C-0.223876 4.26946 -0.223883 2.97268 0.679451 2.81553C1.74696 2.62981 2.59677 1.8189 2.83276 0.762152L2.84848 0.690608L2.85637 0.654433ZM10.6352 3.61651C10.8194 3.53901 11.0172 3.49908 11.217 3.49908C11.4168 3.49908 11.6146 3.53901 11.7988 3.61651C11.9823 3.69375 12.1486 3.80675 12.288 3.94894L12.2895 3.9504L13.5557 5.22658C13.6955 5.36569 13.8066 5.53101 13.8824 5.71309C13.9586 5.89588 13.9978 6.09194 13.9978 6.28996C13.9978 6.48798 13.9586 6.68404 13.8824 6.86682C13.8065 7.04896 13.6955 7.21433 13.5556 7.35347L13.5541 7.35496L7.35665 13.5924C7.2733 13.6763 7.16293 13.7279 7.04513 13.7381L4.04513 13.9981C3.89803 14.0109 3.75281 13.958 3.6484 13.8536C3.544 13.7491 3.49108 13.6039 3.50382 13.4568L3.76382 10.4568C3.77403 10.339 3.82566 10.2287 3.90954 10.1453L10.1473 3.94757C10.2864 3.80603 10.4522 3.6935 10.6352 3.61651Z"
                        />
                    </svg>
                    <x-tabler-refresh
                        class="col-start-1 col-end-1 row-start-1 row-end-1 size-5 animate-spin"
                        x-cloak
                        x-show="busy"
                    />
                </span>
            </x-button>
        </div>
    </form>
</div>

@include('panel.user.openai_chat.components.prompt_library_modal')
