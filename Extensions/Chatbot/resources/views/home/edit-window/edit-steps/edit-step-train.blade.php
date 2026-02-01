{{-- Editing Step 3 - Train --}}
<div
    class="col-start-1 col-end-1 row-start-1 row-end-1 transition-all"
    data-step="3"
    x-data="externalChatbotTraining"
    x-show="editingStep === 3"
    x-transition:enter-start="opacity-0 -translate-x-3"
    x-transition:enter-end="opacity-100 translate-x-0"
    x-transition:leave-start="opacity-100 translate-x-0"
    x-transition:leave-end="opacity-0 translate-x-3"
>
    <h2 class="mb-3.5">
        @lang('Chatbot Training')
    </h2>
    <p class="mb-3 text-xs/5 opacity-60 lg:max-w-[360px]">
        @lang('This step is optional but highly recommended to personalize your chatbot experience.')
    </p>
    <x-button
        class="w-full text-2xs"
        variant="outline"
        x-show="editingStep !== 1"
        size="lg"
        @click.prevent="setEditingStep('>')"
        type="button"
    >
        @lang('Skip')
        <x-tabler-chevron-right class="size-4" />
    </x-button>

    @php
        $tabs = [
            'website' => ['label' => __('Website')],
            'pdf' => ['label' => __('PDF')],
            'text' => ['label' => __('Text')],
            'qa' => ['label' => __('Q&A')],
        ];
    @endphp
    <div class="lqd-ext-chatbot-training mt-16 flex flex-col justify-center gap-9">
        <ul class="flex w-full flex-wrap justify-between gap-3 rounded-3xl bg-foreground/5 p-1 text-xs font-medium sm:flex-nowrap sm:rounded-full">
            @foreach ($tabs as $key => $tab)
                <li class="grow">
                    <button
                        @class([
                            'px-6 py-2.5 grow leading-tight rounded-full transition-all hover:bg-background/80 [&.lqd-is-active]:bg-background [&.lqd-is-active]:shadow-[0_2px_12px_hsl(0_0%_0%/10%)]',
                            'lqd-is-active' => $loop->first,
                        ])
                        @click="setActiveTab('{{ $key }}')"
                        :class="{ 'lqd-is-active': activeTab === '{{ $key }}' }"
                        :disabled="fetching"
                    >
                        @lang(Str::ucfirst($tab['label']))
                    </button>
                </li>
            @endforeach
        </ul>
        <div class="lqd-ext-chatbot-training-content grid">
            @include('chatbot::home.training.training-tabs.training-tab-website')
            @include('chatbot::home.training.training-tabs.training-tab-pdf')
            @include('chatbot::home.training.training-tabs.training-tab-text')
            @include('chatbot::home.training.training-tabs.training-tab-qa')
        </div>
    </div>

    <x-button
        class="mt-4 w-full"
        variant="secondary"
        @click.prevent="setEditingStep('>')"
        size="lg"
        type="button"
    >
        @lang('Next')
    </x-button>
</div>

@push('script')
    <script>
        (() => {
            document.addEventListener('alpine:init', () => {
                Alpine.data('externalChatbotTraining', () => ({
                    activeTab: 'website',
                    fetching: false,
                    uploading: false,
                    embeddings: [],
                    editingItem: {},
                    init() {
                        this.$data.externalChatbotTraining = this;
                    },
                    setActiveTab(tab) {
                        if (tab === this.activeTab) return;

                        this.activeTab = tab;
                    },
                    toggleSelectAll(event) {
                        const btn = event.currentTarget;
                        const checkboxes = document.getElementsByName('embedding-item');
                        const relevantCheckboxes = Array.from(checkboxes).filter(el => el.getAttribute('data-type') === this.activeTab);
                        const allChecked = relevantCheckboxes.every(el => el.checked);

                        relevantCheckboxes.forEach(el => el.checked = !allChecked);

                        if (relevantCheckboxes.some(el => el.checked)) {
                            btn.classList.add('has-selected');
                        } else {
                            btn.classList.remove('has-selected');
                        }
                    },
                    async fetchEmbeddings() {
                        if (!this.activeChatbot || this.fetching) return;

                        this.fetching = true;

                        const res = await fetch(`{{ route('dashboard.chatbot.train.data') }}?id=${this.activeChatbot.id}`);

                        if (!res.ok) {
                            this.fetching = false;
                            toastr.error('{{ $app_is_demo ? __('This feature is disabled in Demo version.') : __('Failed to fetch embeddings') }}');
                            return;
                        }

                        const data = await res.json();
                        const embeddings = data.data;

                        if (!embeddings) {
                            this.fetching = false;
                            toastr.error('{{ __('Failed to fetch embeddings') }}');
                            return;
                        }

                        this.embeddings = embeddings;

                        this.fetching = false;
                    },
                    async trainEmbeddings(event) {
                        if (this.fetching) return;

                        this.fetching = true;

                        const form = event.target;
                        const checkboxes = form.elements['embedding-item'];
                        const embdeddingData = {
                            id: this.activeChatbot.id,
                            data: (checkboxes.length ? Array.from(checkboxes) : [checkboxes]).filter(el => el.checked).map(el => el.value),
                        };

                        const res = await fetch(form.action, {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                                'Accept': 'application/json',
                                'X-CSRF-TOKEN': '{{ csrf_token() }}',
                            },
                            body: JSON.stringify(embdeddingData),
                        });

                        if (!res.ok) {
                            this.fetching = false;
                            const errorData = await res.json();
                            toastr.error(errorData.message || '{{ __('Failed to train embeddings') }}');
                            return;
                        }

                        const data = await res.json();
                        const embeddings = data.data;

                        if (!embeddings) {
                            this.fetching = false;
                            toastr.error('{{ __('Failed to train embeddings') }}');
                            return;
                        }

                        this.embeddings = this.embeddings.map(embedding => {
                            const newEmbedding = embeddings.find(e => e.id === embedding.id);

                            if (newEmbedding) {
                                return newEmbedding;
                            }

                            return embedding;
                        });

                        toastr.success('{{ __('Training done successfully') }}');

                        this.fetching = false;
                    },
                    async deleteEmbedding(embeddingId) {
                        if (!embeddingId || this.fetching) return;

                        this.fetching = true;

                        if (!confirm(`{{ __('Are you sure you want to delete this embedding?') }}`)) {
                            this.fetching = false;
                            return;
                        }

                        const res = await fetch('{{ route('dashboard.chatbot.train.delete') }}', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                                'Accept': 'application/json',
                                'X-CSRF-TOKEN': '{{ csrf_token() }}',
                            },
                            body: JSON.stringify({
                                id: this.activeChatbot.id,
                                data: [embeddingId],
                            }),
                        });

                        if (!res.ok) {
                            this.fetching = false;
                            toastr.error('{{ $app_is_demo ? __('This feature is disabled in Demo version.') : __('Failed to delete embedding') }}');
                            return;
                        }

                        const data = await res.json();

                        if (data.status === 200 && data.message) {
                            toastr.success(data.message);
                        }

                        this.embeddings = this.embeddings.filter(embedding => embedding.id !== embeddingId);

                        this.fetching = false;
                    },
                    setEditingItem(id) {
                        this.editingItem = this.embeddings.find(embedding => embedding.id === id) || {};
                    },

                    // Start Train URL Tab
                    async addUrl(event) {
                        if (this.fetching) return;

                        this.fetching = true;

                        const form = event.target;
                        const formData = new FormData(form);

                        formData.append('id', this.activeChatbot.id);

                        const res = await fetch(form.action, {
                            method: 'POST',
                            headers: {
                                'X-CSRF-TOKEN': '{{ csrf_token() }}',
                                'Accept': 'application/json',
                            },
                            body: formData,
                        });

                        if (!res.ok) {
                            this.fetching = false;
                            console.log('adasd');
                            toastr.error('{{ $app_is_demo ? __('This feature is disabled in Demo version.') : __('Failed to fetch urls') }}');
                            return;
                        }

                        const data = await res.json();
                        const websites = data.data;

                        if (!websites) {
                            this.fetching = false;
                            toastr.error('{{ __('Failed to fetch urls') }}');
                            return;
                        }

                        this.embeddings = this.embeddings
                            .filter(embedding => embedding.type !== 'website')
                            .concat(websites);

                        this.fetching = false;
                    },
                    // End Train URL Tab

                    // Start Train File Tab
                    async uploadFile(event) {
                        this.uploading = this.fetching = true;

                        const form = event.target;
                        const formData = new FormData(form);

                        if (!form.elements.file?.files?.length) {
                            this.uploading = this.fetching = false;
                            toastr.error('{{ __('Please select a file') }}');
                            return;
                        }

                        formData.append('id', this.activeChatbot.id);

                        const res = await fetch(form.action, {
                            method: 'POST',
                            headers: {
                                'X-CSRF-TOKEN': '{{ csrf_token() }}',
                                'Accept': 'application/json',
                            },
                            body: formData,
                        });

                        if (!res.ok) {
                            this.uploading = this.fetching = false;
                            toastr.error('{{ $app_is_demo ? __('This feature is disabled in Demo version.') : __('Failed to upload file') }}');
                            return;
                        }

                        const data = await res.json();
                        const files = data.data;

                        if (!files) {
                            this.uploading = this.fetching = false;
                            toastr.error('{{ __('Failed to upload file') }}');
                            return;
                        }

                        this.embeddings = this.embeddings
                            .filter(embedding => embedding.type !== 'file')
                            .concat(files);

                        this.$refs.fileName.innerText = this.$refs.fileName.getAttribute('data-original-text');
                        this.$refs.fileName.value = null;
                        this.$refs.fileName.files = new DataTransfer().files;

                        this.uploading = this.fetching = false;
                    },
                    // End Train File Tab

                    // Start Train Text & QA Tab
                    async addText(event, textOrQA = 'text') {
                        this.fetching = true;

                        const form = event.target;
                        const formData = new FormData(form);
                        const title = formData.get(textOrQA === 'text' ? 'title' : 'question');
                        const content = formData.get(textOrQA === 'text' ? 'content' : 'answer');

                        if (!title || !content || !title.trim() || !content.trim()) {
                            toastr.error('{{ __('Please fill in the title and content') }}');
                            return;
                        }

                        formData.append('id', this.activeChatbot.id);

                        const res = await fetch(form.action, {
                            method: 'POST',
                            headers: {
                                'X-CSRF-TOKEN': '{{ csrf_token() }}',
                                'Accept': 'application/json',
                            },
                            body: formData,
                        });

                        if (!res.ok) {
                            this.fetching = false;
                            toastr.error('{{ $app_is_demo ? __('This feature is disabled in Demo version.') : __('Failed to add the new item') }}');
                            return;
                        }

                        const data = await res.json();
                        const items = data.data;

                        if (!items) {
                            this.fetching = false;
                            toastr.error('{{ __('Failed to add the new item') }}');
                            return;
                        }

                        this.embeddings = this.embeddings
                            .filter(embedding => embedding.type !== textOrQA)
                            .concat(items);

                        event.target.reset();
                        event.target.elements[textOrQA === 'text' ? 'title' : 'question'].focus();

                        this.fetching = false;
                    },
                    // End Train Text Tab

                }));
            });
        })();
    </script>
@endpush
