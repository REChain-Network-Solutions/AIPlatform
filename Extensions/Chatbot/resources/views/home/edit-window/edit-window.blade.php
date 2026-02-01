{{-- Edit Window --}}
<template x-teleport="body">
    <div
        class="lqd-chatbot-edit-window fixed bottom-0 end-0 start-0 top-0 z-[100] overflow-y-auto bg-background lg:start-[--navbar-width]"
        x-show="activeChatbot.id"
        :class="{ active: activeChatbot }"
    >
        {{-- Edit Window Header --}}
        <div
            class="lqd-chatbot-edit-window-header sticky top-0 z-2 border-b bg-background/60 backdrop-blur-lg backdrop-saturate-150">
            <div class="flex flex-wrap items-center justify-between gap-4 px-3 py-3 lg:h-[--header-height] lg:px-12">
                {{-- Header Actions --}}
                <div class="flex flex-wrap items-center gap-3">
                    <x-button
                        class="text-2xs text-heading-foreground/50 hover:text-heading-foreground"
                        @click.prevent="setActiveChatbot(null)"
                        variant="link"
                    >
                        <x-tabler-chevron-left class="size-4" />
                        @lang('Close Customizer')
                    </x-button>

                    <span class="opacity-10">
                        |
                    </span>

                    <span class="text-heading-foreground/50">
                        @lang('Editing'):
                        <span
                            class="text-heading-foreground"
                            x-text="activeChatbot.title"
                        ></span>
                    </span>
                </div>

                {{-- Header Steps --}}
                <div class="lqd-steps hidden flex-col gap-1 lg:flex">
                    <div class="lqd-steps-steps flex items-center justify-between gap-1 lg:gap-3">
                        @foreach (\App\Extensions\Chatbot\System\Enums\StepEnum::toArray() as $step)
                            @continue(!\App\Extensions\Chatbot\System\Helpers\ChatbotHelper::existChannels() && $step === 'channel')
                            <button
                                class="lqd-step group/step flex gap-3 rounded p-2 text-3xs font-semibold capitalize text-heading-foreground transition-colors hover:bg-heading-foreground/5 disabled:pointer-events-none disabled:opacity-50 lg:min-w-32"
                                type="button"
                                @click.prevent="setEditingStep({{ $loop->index + 1 }})"
                                :disabled="submittingData"
                            >
                                <span
                                    class="inline-grid size-[21px] place-items-center rounded-md border border-heading-foreground/10 transition-colors group-hover/step:border-heading-foreground group-hover/step:bg-heading-foreground group-hover/step:text-heading-background"
                                >
                                    <span
                                        class="col-start-1 col-end-1 row-start-1 row-end-1"
                                        x-show="editingStep <= {{ $loop->index + 1 }}"
                                        x-transition
                                    >
                                        {{ $loop->index + 1 }}
                                    </span>
                                    <svg
                                        class="col-start-1 col-end-1 row-start-1 row-end-1"
                                        x-show="editingStep > {{ $loop->index + 1 }}"
                                        x-transition
                                        width="9"
                                        height="7"
                                        viewBox="0 0 9 7"
                                        fill="currentColor"
                                        xmlns="http://www.w3.org/2000/svg"
                                    >
                                        <path
                                            d="M3.14724 7L0 3.68191L0.78681 2.85239L3.14724 5.34096L8.21319 0L9 0.829522L3.14724 7Z"
                                        />
                                    </svg>
                                </span>
                                @lang($step)
                            </button>
                        @endforeach
                    </div>
                    <div
                        class="lqd-step-progress relative h-[3px] w-full overflow-hidden rounded-lg bg-heading-foreground/5">
                        <div
                            class="lqd-step-progress-bar absolute start-0 top-0 h-full w-0 rounded-full bg-gradient-to-r from-gradient-from to-gradient-to transition-all"
                            :style="{
                                width: editingStep *
                                    {{ \App\Helpers\Classes\MarketplaceHelper::isRegistered('chatbot-multi-channel') ? 20 : 25 }} +
                                    '%'
                            }"
                        ></div>
                    </div>
                </div>
            </div>
        </div>

        <div class="lqd-chatbot-edit-window-content py-8 lg:py-0">
            <div class="container flex flex-wrap justify-between gap-y-5">
                {{-- Options Container --}}
                <div
                    class="lqd-chatbot-edit-window-options grid w-full lg:min-h-[calc(100vh-var(--header-height))] lg:w-[430px] lg:py-16">
                    <input
                        type="hidden"
                        name="id"
                        x-model="activeChatbot.id"
                    >
                    @include('chatbot::home.edit-window.edit-steps.edit-step-configure')
                    @include('chatbot::home.edit-window.edit-steps.edit-step-customize', [
                        'avatars',
                        $avatars,
                    ])
                    @include('chatbot::home.edit-window.edit-steps.edit-step-train')
                    @include('chatbot::home.edit-window.edit-steps.edit-step-embed')
					@if(\App\Extensions\Chatbot\System\Helpers\ChatbotHelper::existChannels())
						@include('chatbot::home.edit-window.edit-steps.edit-step-channel')
					@endif
                    <div
                        class="col-start-1 col-end-2 row-start-2 row-end-2 mt-10 flex flex-col gap-3"
                        :class="{ 'invisible': editingStep > 2 }"
                        x-transition
                    >
                        <x-button
                            class="w-full"
                            variant="secondary"
                            ::class="{ 'invisible opacity-0': editingStep > 2 }"
                            @click.prevent="setEditingStep('>')"
                            size="lg"
                            type="button"
                            ::disabled="submittingData"
                        >
                            @lang('Next')
                        </x-button>
                        <x-button
                            class="w-full"
                            variant="outline"
                            ::class="{ 'invisible opacity-0': editingStep <= 1 }"
                            @click.prevent="setEditingStep('<')"
                            size="lg"
                            type="button"
                            ::disabled="submittingData"
                        >
                            @lang('Back')
                        </x-button>
                    </div>
                </div>

                {{-- Preview Container --}}
                <div class="hidden w-full lg:grid lg:w-1/2 lg:py-16">
                    <div
                        class="col-start-1 col-end-1 row-start-1 row-end-1 flex w-full items-center justify-center rounded-3xl bg-heading-foreground/5 p-5 transition-all lg:py-10"
                        {{-- x-show="editingStep !== 3"
                        x-transition:enter-start="opacity-0 -translate-x-3"
                        x-transition:enter-end="opacity-100 translate-x-0"
                        x-transition:leave-start="opacity-100 translate-x-0"
                        x-transition:leave-end="opacity-0 translate-x-3" --}}
                    >
                        @include('chatbot::home.frontend-ui-preview')
                    </div>
                </div>
            </div>
        </div>
    </div>
</template>

@push('script')
    <script>
        (() => {
            document.addEventListener('alpine:init', () => {
                Alpine.data('customAvatar', () => ({
                    customAvatarPicker: {
                        ['x-ref']: 'customAvatarPicker',
                        ['@change']: 'onPickerChange'
                    },
                    async onPickerChange() {
                        const avatarsList = document.querySelector('.lqd-chatbot-avatar-list');
                        const file = this.$refs.customAvatarPicker.files[0];

                        if (!file) {
                            return toastr.error('{{ __('Please select a file!') }}');
                        }

                        const formData = new FormData();
                        formData.append('avatar', file);

                        toastr.info('{{ __('Uploading Avatar...') }}');

                        await fetch('{{ route('dashboard.chatbot.upload.avatar') }}', {
                                method: 'POST',
                                body: formData,
                            })
                            .then(response => {
                                if (!response.ok) {

                                    toastr.clear();
                                    throw new Error('Uploading Failed!');
                                }
                                return response.json();
                            })
                            .then(data => {
                                const avatar = data.data;
                                const lastAvatarItem = avatarsList.querySelector(
                                    '.lqd-chatbot-avatar-list-item:last-child');
                                const newAvatarItem = lastAvatarItem.cloneNode(true);

                                const newInput = newAvatarItem.querySelector('input');
                                const newLabel = newAvatarItem.querySelector('label');
                                const newImg = newLabel.querySelector('img');

                                newInput.id = `avatar-${avatar.id}`;
                                newInput.value = avatar.avatar;
                                newLabel.setAttribute('for', `avatar-${avatar.id}`);
                                newImg.src = avatar.avatar_url;

                                avatarsList.appendChild(newAvatarItem);

                                toastr.clear();

                                toastr.success('{{ __('Uploaded Successfully...') }}');
                            })
                            .catch(error => {
                                toastr.error(error);
                            });
                    }
                }));

                Alpine.data('customColorPicker', () => ({
                    colorValue: '',
                    customColorTrigger: {
                        ['x-ref']: 'customColorTrigger',
                        ['@click']() {
                            this.$refs.customColorColorInput.click();
                        }
                    },
                    customColorRadioInput: {
                        ['x-ref']: 'customColorRadioInput',
                    },
                    customColorColorInput: {
                        ['x-ref']: 'customColorColorInput',
                        ['@input']() {
                            this.colorValue = this.$refs.customColorColorInput.value;
                            this.$refs.customColorOutput.style.backgroundColor = this.colorValue;
                            this.$refs.customColorRadioInput.value = this.colorValue;
                        }
                    },
                    customColorOutput: {
                        ['x-ref']: 'customColorOutput'
                    },
                }));

                Alpine.data('logoPreviewHandler', () => ({
                    uploadedLogo: null,
                    logoPicker: {
                        ['x-ref']: 'logoPicker',
                        ['@change']: 'onPickerChange',
                    },
                    onPickerChange() {
                        const file = this.$refs.logoPicker.files[0];
                        const reader = new FileReader();

                        reader.onload = (e) => {
                            this.uploadedLogo = e.target.result;
                        };

                        reader.readAsDataURL(file);
                    },
                }));
            });
        })();
    </script>
@endpush
