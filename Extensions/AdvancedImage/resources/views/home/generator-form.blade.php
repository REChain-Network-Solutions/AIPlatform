<div class="lqd-adv-img-editor-form-wrap pb-6 pt-16">
    <h1 class="mb-11 text-center">
        @lang('Create Stunning <span class="opacity-50">AI Images</span>')
    </h1>

    <form
        class="lqd-adv-img-editor-form relative flex min-h-16 w-full flex-wrap rounded-full bg-background/60 font-medium shadow-lg shadow-black/5 sm:text-2xs lg:flex-nowrap"
        id="submitForm"
        method="POST"
        action="{{ route('dashboard.user.openai.output') }}"
        enctype="multipart/form-data"
        x-data="{
            generator: '{{ $generators[0]['value'] }}',
            pickerModalOpen: false,
            editingImage: {},
            handleDragOver(event) {
                this.$refs.dropArea.classList.add('drag-over');
            },
            handleDragLeave(event) {
                this.$refs.dropArea.classList.remove('drag-over');
            },
            handleDrop(event) {
                const { dataTransfer } = event;
                const file = dataTransfer.files[0];
                if (!file) return;
                this.handleFile(file);
                this.$refs.dropArea.classList.remove('drag-over');
            },
            handleFileChange(event) {
                const file = event.target.files[0];
        
                if (!file) return;
        
                this.handleFile(file);
            },
            handleFile(file) {
                this.editingImage = {
                    output: URL.createObjectURL(file),
                    name: file.name
                };
        
                this.pickerModalOpen = false;
                this.$refs.promptInput?.focus();
        
                const formData = new FormData();
        
                formData.append('image', file);
        
                this.$refs.promptInput.disabled = true;
                this.$refs.promptInput.placeholder = '{{ __('Analyzing image... Please wait...') }}';
        
                fetch('{{ route('dashboard.user.generate.prompt') }}', {
                        method: 'POST',
                        headers: {
                            'X-CSRF-TOKEN': '{{ csrf_token() }}'
                        },
                        body: formData
                    })
                    .then(response => response.json())
                    .then(data => {
                        if (data.status === 'success') {
                            this.prompt = data.prompt;
                        } else {
                            toastr.error(data.prompt);
                        }
                        this.$refs.promptInput.disabled = false;
                        this.$refs.promptInput.placeholder = '{{ __('Describe your idea or select a pre-defined prompt') }}';
                    })
                    .catch(error => {
                        this.$refs.promptInput.disabled = false;
                        this.$refs.promptInput.placeholder = '{{ __('Describe your idea or select a pre-defined prompt') }}';
                    });
        
                if (this.generator === 'stable_diffusion') return;
                this.generator = 'stable_diffusion';
        
                toastr.info('{{ __('Image generator changed to Stable Diffusion') }}');
            }
        }"
    >
        @csrf
        <input
            id="image_ratio"
            hidden
            name="image_ratio"
        >
        <x-forms.input
            class:container="flex border-e border-heading-foreground/5 min-w-32 max-sm:w-full min-h-16"
            class="h-auto appearance-none rounded-s-full border-none bg-transparent py-0 pe-5 ps-8 text-heading-foreground/90"
            id="image_generator"
            type="select"
            name="image_generator"
            x-model="generator"
        >
            @foreach ($generators as $generator)
                <option
                    class="bg-background"
                    value="{{ $generator['value'] }}"
                >{{ $generator['label'] }}</option>
            @endforeach
        </x-forms.input>

        <div class="flex min-h-16 grow ps-5">
            <x-button
                class="size-8 shrink-0 justify-center self-center text-center hover:translate-y-0 hover:scale-105 hover:shadow-none"
                title="{{ __('Browse pre-defined prompts') }}"
                variant="none"
                size="none"
                @click.prevent="setRandomPrompt()"
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

            <img
                class="me-1 aspect-square size-11 self-center rounded-lg object-cover object-center shadow-sm"
                x-cloak
                :src="editingImage?.output"
                x-show="editingImage?.output && generator === 'stable_diffusion'"
                x-transition
            >
            <span
                class="template-chip me-1 hidden items-center gap-2 self-center rounded bg-primary px-2 py-1 text-[12px] font-semibold text-primary-foreground lg:whitespace-nowrap"
                x-show="selectedTemplate"
                :class="{ 'hidden': !selectedTemplate, 'inline-flex': selectedTemplate }"
            >
                <span
                    class="template-chip-title"
                    x-text="selectedTemplate"
                ></span>
                <button
                    class="template-chip-remove inline-grid size-5 place-content-center rounded-full transition-all hover:bg-background hover:text-heading-foreground"
                    type="button"
                    @click.prevent="selectedTemplate = ''"
                >
                    <x-tabler-x class="size-3.5" />
                </button>
            </span>

            <x-forms.input
                class:container="grow flex"
                class="h-auto rounded-none border-none bg-transparent px-2 placeholder:text-heading-foreground/90 focus:ring-0 max-sm:text-[11px]"
                id="description"
                name="description"
                placeholder="{{ __('Describe your idea or select a pre-defined prompt') }}"
                x-model="prompt"
                x-ref="promptInput"
            />
        </div>

        <input
            id="template_description"
            type="hidden"
            name="template_description"
            value=""
            x-model="selectedTemplateDescription"
        />

        <input
            id="prompt_description"
            type="hidden"
            name="prompt_description"
            value=""
            x-model="selectedPromptDescription"
        />

        <input
            type="hidden"
            name="post_type"
            value="ai_image_generator"
        />

        <input
            type="hidden"
            name="image_mood"
            value=""
        />

        <input
            type="hidden"
            name="image_style"
            value=""
        />

        <input
            type="hidden"
            name="quality"
            value="standard"
        />

        <input
            type="hidden"
            name="image_lighting"
            value=""
        />

        <input
            type="hidden"
            name="size"
            value="1024x1024"
        />

        <input
            id="image_resolution"
            type="hidden"
            name="image_resolution"
            value="{{ $settings_two->stablediffusion_default_model === \App\Domains\Entity\Enums\EntityEnum::STABLE_DIFFUSION_XL_1024_V_1_0->value ? '640x1536' : '896x512' }}"
        />

        <input
            id="type"
            type="hidden"
            name="type"
            value="text-to-image"
        />

        <input
            type="hidden"
            name="image_number_of_images"
            value="1"
        />

        <input
            type="hidden"
            name="negative_prompt"
            value=""
        />

        <input
            id="model"
            type="hidden"
            name="model"
            value=""
        />

        <input
            id="description_flux_pro"
            type="hidden"
            name="description_flux_pro"
            value=""
        />

        <input
            id="description_ideogram"
            type="hidden"
            name="description_ideogram"
            value=""
        />

        <input
            id="stable_description"
            type="hidden"
            name="stable_description"
            value=""
        />

        <input
            type="hidden"
            name="style_preset"
            value=""
        />

        <input
            type="hidden"
            name="sampler"
            value=""
        />

        <input
            type="hidden"
            name="clip_guidance_preset"
            value=""
        />

        <input
            type="hidden"
            name="openai_id"
            value="{{ $openai->id }}"
        />
        <div class="ms-auto flex min-h-16 items-center gap-6 max-sm:w-full max-sm:flex-col-reverse max-sm:flex-wrap sm:pe-8">
            @if (in_array('stable_diffusion', array_column($generators, 'value')))
                <div
                    class="group/btn-wrap relative cursor-pointer"
                    tabindex="-1"
                    @keydown.escape.window="pickerModalOpen = false"
                >
                    <x-button
                        class="gap-3 lowercase text-heading-foreground/90 underline group-hover/btn-wrap:underline-offset-2 group-focus/btn-wrap:underline-offset-2"
                        variant="link"
                        type="button"
                        tabindex="-1"
                        @click.prevent="pickerModalOpen = true"
                    >
                        <x-tabler-photo-plus
                            class="size-7 transition-transform group-hover/btn-wrap:scale-105 group-focus/btn-wrap:scale-105"
                            stroke-width="1.5"
                        />
                        <span>
                            @lang('or upload an image')
                        </span>
                    </x-button>

                    <div
                        class="pointer-events-none invisible fixed inset-0 z-10 flex flex-col items-center justify-center py-[calc(var(--header-h)+1rem)] opacity-0 transition-all"
                        :class="{ 'opacity-0 invisible pointer-events-none': !
                            pickerModalOpen, 'opacity-100 visible pointer-events-auto': pickerModalOpen }"
                    >
                        <div
                            class="absolute inset-0 z-0 bg-black/25 dark:bg-black/50"
                            @click="pickerModalOpen = false"
                        ></div>
                        <div
                            class="group/drop-area relative z-2 m-auto flex h-[min(570px,90vh)] w-[min(735px,90vw)] scale-95 flex-col justify-center rounded-3xl border-2 border-dashed border-heading-foreground/20 bg-background/70 py-5 backdrop-blur-xl backdrop-saturate-[125%] transition-all dark:border-heading-foreground/10 dark:bg-background/90 [&.drag-over]:border-heading-foreground"
                            :class="{ 'scale-100': pickerModalOpen, 'scale-95': !pickerModalOpen }"
                            @dragover.prevent="handleDragOver"
                            @dragleave.prevent="handleDragLeave"
                            @drop.prevent="handleDrop"
                            x-ref="dropArea"
                        >
                            <x-button
                                class="absolute end-6 top-6 z-10 size-[34px]"
                                variant="outline"
                                hover-variant="danger"
                                size="none"
                                @click.prevent="pickerModalOpen = false"
                            >
                                <x-tabler-x class="size-4" />
                            </x-button>
                            <div
                                class="mx-auto flex w-[min(100%,400px)] flex-col items-center justify-center gap-4 text-center">
                                <div class="grid place-content-center">
                                    <div
                                        class="col-start-1 col-end-1 row-start-1 row-end-1"
                                        {{-- x-show="!editingImage.output"
                                        x-transition --}}
                                    >
                                        <div class="mx-auto mb-4 inline-grid w-12 place-content-center">
                                            <svg
                                                class="col-start-1 col-end-1 row-start-1 row-end-1 h-auto w-full text-heading-foreground/20 transition-all group-[&.drag-over]/drop-area:scale-50 group-[&.drag-over]/drop-area:opacity-0"
                                                width="48"
                                                height="49"
                                                viewBox="0 0 48 49"
                                                fill="currentColor"
                                                xmlns="http://www.w3.org/2000/svg"
                                            >
                                                <path
                                                    d="M40.9355 41.3123C36.2903 45.9574 30.6452 48.28 24 48.28C17.3548 48.28 11.6774 45.9574 6.96774 41.3123C2.32258 36.6026 0 30.9252 0 24.28C0 17.6348 2.32258 11.9897 6.96774 7.34451C11.6774 2.63484 17.3548 0.279999 24 0.279999C30.6452 0.279999 36.2903 2.63484 40.9355 7.34451C45.6452 11.9897 48 17.6348 48 24.28C48 30.9252 45.6452 36.6026 40.9355 41.3123ZM37.6452 10.6348C33.9032 6.82839 29.3548 4.92516 24 4.92516C18.6452 4.92516 14.0645 6.82839 10.2581 10.6348C6.51613 14.3768 4.64516 18.9252 4.64516 24.28C4.64516 29.6348 6.51613 34.2155 10.2581 38.0219C14.0645 41.7639 18.6452 43.6348 24 43.6348C29.3548 43.6348 33.9032 41.7639 37.6452 38.0219C41.4516 34.2155 43.3548 29.6348 43.3548 24.28C43.3548 18.9252 41.4516 14.3768 37.6452 10.6348ZM25.9355 36.6671H22.0645C21.2903 36.6671 20.9032 36.28 20.9032 35.5058V27.28C20.9032 25.6231 19.5601 24.28 17.9032 24.28H14.4194C13.9032 24.28 13.5484 24.0542 13.3548 23.6026C13.1613 23.0865 13.2258 22.6671 13.5484 22.3445L23.2258 12.6671C23.7419 12.151 24.2581 12.151 24.7742 12.6671L34.4516 22.3445C34.7742 22.6671 34.8387 23.0865 34.6452 23.6026C34.4516 24.0542 34.0968 24.28 33.5806 24.28H30.0968C28.4399 24.28 27.0968 25.6231 27.0968 27.28V35.5058C27.0968 36.28 26.7097 36.6671 25.9355 36.6671Z"
                                                />
                                            </svg>
                                            <svg
                                                class="col-start-1 col-end-1 row-start-1 row-end-1 h-auto w-full scale-50 text-heading-foreground opacity-0 transition-all group-[&.drag-over]/drop-area:scale-100 group-[&.drag-over]/drop-area:opacity-100"
                                                xmlns="http://www.w3.org/2000/svg"
                                                viewBox="0 0 24 24"
                                                fill="none"
                                                stroke="currentColor"
                                                stroke-linecap="round"
                                                stroke-linejoin="round"
                                                width="24"
                                                height="24"
                                                stroke-width="1.5"
                                            >
                                                <path d="M19 11v-2a2 2 0 0 0 -2 -2h-8a2 2 0 0 0 -2 2v8a2 2 0 0 0 2 2h2">
                                                </path>
                                                <path d="M13 13l9 3l-4 2l-2 4l-3 -9"></path>
                                                <path d="M3 3l0 .01"></path>
                                                <path d="M7 3l0 .01"></path>
                                                <path d="M11 3l0 .01"></path>
                                                <path d="M15 3l0 .01"></path>
                                                <path d="M3 7l0 .01"></path>
                                                <path d="M3 11l0 .01"></path>
                                                <path d="M3 15l0 .01"></path>
                                            </svg>
                                        </div>
                                        <h4 class="text-base">
                                            @lang('Drag and Drop an Image')
                                        </h4>
                                    </div>
                                    {{-- <img
                                        class="col-start-1 col-end-1 row-start-1 row-end-1 w-60 rounded-xl"
                                        :src="editingImage?.output"
                                    > --}}
                                </div>
                                <div
                                    class="mx-auto flex w-3/4 items-center gap-7 text-2xs font-medium text-heading-foreground"
                                    {{-- x-show="!editingImage?.output" --}}
                                >
                                    <span class="inline-flex h-px grow bg-heading-foreground/5"></span>
                                    @lang('or')
                                    <span class="inline-flex h-px grow bg-heading-foreground/5"></span>
                                </div>
                                <span
                                    class="inline-flex rounded-full border px-8 py-3 text-sm font-medium text-heading-foreground transition-all group-hover/drop-area:bg-heading-foreground group-hover/drop-area:text-heading-background"
                                    {{-- x-show="!editingImage?.output" --}}
                                >
                                    @lang('Browse Files')
                                </span>
                                <p
                                    class="text-3xs font-medium opacity-60"
                                    x-text="editingImage?.name || '{{ __('PNG or JPG (Max: 25Mb') }}'"
                                ></p>

                                <input
                                    class="absolute inset-0 z-5 cursor-pointer opacity-0"
                                    id="image_src"
                                    name="image_src"
                                    type="file"
                                    accept="image/*"
                                    x-ref="formFileInput"
                                    @input="handleFileChange"
                                >
                            </div>
                        </div>
                    </div>
                </div>
            @endif

            <x-button
                class="size-10 bg-gradient-to-b from-[#CA99E7] via-[#4FACBE] to-[#8364FF] max-sm:w-full"
                type="submit"
                size="none"
                title="{{ __('Generate Image') }}"
            >
                <svg
                    width="14"
                    height="14"
                    viewBox="0 0 14 14"
                    xmlns="http://www.w3.org/2000/svg"
                    fill="currentColor"
                >
                    <path
                        fill-rule="evenodd"
                        clip-rule="evenodd"
                        d="M2.85637 0.654433C3.04651 -0.214165 4.28328 -0.219571 4.481 0.647332L4.49082 0.690566C4.49756 0.720292 4.50384 0.748056 4.51058 0.77677C4.75709 1.82667 5.60676 2.6289 6.67036 2.81393C7.57548 2.9714 7.57548 4.27074 6.67036 4.42821C5.60109 4.61423 4.74804 5.42404 4.50669 6.4822L4.481 6.59481C4.28328 7.46172 3.04651 7.4563 2.85637 6.58771L2.83522 6.49108C2.6027 5.42891 1.75069 4.61298 0.679451 4.42661C-0.223876 4.26946 -0.223883 2.97268 0.679451 2.81553C1.74696 2.62981 2.59677 1.8189 2.83276 0.762152L2.84848 0.690608L2.85637 0.654433ZM10.6352 3.61651C10.8194 3.53901 11.0172 3.49908 11.217 3.49908C11.4168 3.49908 11.6146 3.53901 11.7988 3.61651C11.9823 3.69375 12.1486 3.80675 12.288 3.94894L12.2895 3.9504L13.5557 5.22658C13.6955 5.36569 13.8066 5.53101 13.8824 5.71309C13.9586 5.89588 13.9978 6.09194 13.9978 6.28996C13.9978 6.48798 13.9586 6.68404 13.8824 6.86682C13.8065 7.04896 13.6955 7.21433 13.5556 7.35347L13.5541 7.35496L7.35665 13.5924C7.2733 13.6763 7.16293 13.7279 7.04513 13.7381L4.04513 13.9981C3.89803 14.0109 3.75281 13.958 3.6484 13.8536C3.544 13.7491 3.49108 13.6039 3.50382 13.4568L3.76382 10.4568C3.77403 10.339 3.82566 10.2287 3.90954 10.1453L10.1473 3.94757C10.2864 3.80603 10.4522 3.6935 10.6352 3.61651Z"
                    />
                </svg>
            </x-button>
        </div>
    </form>
</div>

@push('script')
    <script>
        $('#submitForm').on('submit', function(e) {
            e.preventDefault();
            Alpine.store('appLoadingIndicator').show();

            const imageGeneratorValue = $('#image_generator').val();
            const descriptionInput = $('#description');
            const templateInput = $('#template_description');
            const promptInput = $('#prompt_description');
            const imageGenerator = $('#image_generator');
            const fileInput = $('#image_src')[0];
            const image = fileInput?.files[0];

            if (templateInput.val()) {
                descriptionInput.val(`${templateInput.val()} content: ${descriptionInput.val()}`);
                if (imageGeneratorValue === 'stable_diffusion') {
                    $('#stable_description').val(descriptionInput.val());
                } else if (imageGeneratorValue === 'openai') {
                    descriptionInput.attr('name', 'description');
                } else if (imageGeneratorValue === 'ideogram') {
                    $('#image_resolution').val("1x1");
                    $('#description_ideogram').val(descriptionInput.val());
                    $('#model').val("ideogram-v2");
                } else {
                    $('#image_resolution').val("1x1");
                    $('#description_flux_pro').val(descriptionInput.val());
                    $('#model').val(imageGeneratorValue);
                }
            } else if (promptInput.val()) {
                descriptionInput.val(`${promptInput.val()} content: ${descriptionInput.val()}`);
                if (imageGeneratorValue === 'stable_diffusion') {
                    $('#stable_description').val(descriptionInput.val());
                } else if (imageGeneratorValue === 'openai') {
                    descriptionInput.attr('name', 'description');
                } else if (imageGeneratorValue === 'ideogram') {
                    $('#image_resolution').val("1x1");
                    $('#description_ideogram').val(descriptionInput.val());
                    $('#model').val("ideogram-v2");
                } else {
                    $('#image_resolution').val("1x1");
                    $('#description_flux_pro').val(descriptionInput.val());
                    $('#model').val(imageGeneratorValue);
                }
            } else {
                if (image) {
                    $('#image_resolution').val("640x1536");
                    $('#type').val("image-to-image");
                    imageGenerator.val("stable_diffusion");
                    $('#stable_description').val(descriptionInput.val());
                } else {
                    if (imageGeneratorValue === 'stable_diffusion') {
                        $('#stable_description').val(descriptionInput.val());
                    } else if (imageGeneratorValue === 'openai') {
                        descriptionInput.attr('name', 'description');
                    } else if (imageGeneratorValue === 'ideogram') {
                        $('#image_resolution').val("1x1");
                        $('#description_ideogram').val(descriptionInput.val());
                        $('#model').val("ideogram-v2");
                    } else {
                        $('#image_resolution').val("1x1");
                        $('#description_flux_pro').val(descriptionInput.val());
                        $('#model').val(imageGeneratorValue);
                    }
                }
            }

            let formData = new FormData(this);
            $.ajax({
                type: "POST",
                url: "{{ route('dashboard.user.openai.output') }}",
                data: formData,
                contentType: false,
                processData: false,
                success: function(data) {
                    handleResponse(data);
                },
                error: function(data) {
                    handleResponse(data);
                },
                complete: function() {
                    Alpine.store('appLoadingIndicator').hide();
                }
            });

        });

        function handleResponse(data) {
            if (data.status === 'success') {
                toastr.success("Image Generated Successfully");
                setTimeout(function() {
                    location.reload();
                }, 1000);
            } else {
                if (data.responseJSON?.errors) {
                    $.each(data.responseJSON.errors, function(index, value) {
                        toastr.error(value);
                    });
                } else {
                    toastr.error(data.responseJSON?.message ?? data.message);
                }
            }
        }
    </script>
@endpush
