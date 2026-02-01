@extends('panel.layout.settings')
@section('title', __('AI Persona'))
@section('titlebar_subtitle',
    __('Create studio-quality videos with AI Persona and voiceovers in 130+ languages.
    It’s as easy as making a slide deck.'))
@section('settings')
    <form
        class="flex flex-col gap-5"
        id="synthesia_form"
        action="{{ route('dashboard.user.ai-persona.store') }}"
        method="POST"
    >
        @csrf
        <x-form-step
            step="1"
            label="{{ __('Character') }}"
        />

        <div
            class="relative"
            x-data="{ open: false, selectedAvatar: null }"
        >
            <label
                class="mb-2 block text-2xs font-medium text-label"
                for="avatar"
            >
                {{ __('Avatar Name') }}
            </label>
            <div
                class="flex min-h-11 cursor-pointer items-center rounded-input border px-4 py-2 text-input-foreground sm:text-2xs"
                @click="open = !open"
            >
                <template x-if="selectedAvatar">
                    <div class="flex w-full items-center">
                        <img
                            class="mr-2 h-10 w-auto"
                            :src="selectedAvatar.preview_image_url"
                            alt=""
                        >
                        <span x-text="selectedAvatar.avatar_name"></span>
                        <x-tabler-chevron-down class="ms-auto size-4 text-label" />
                    </div>
                </template>
                <template x-if="!selectedAvatar">
                    <div class="flex w-full items-center">
                        <span>{{ __('Select an avatar') }}</span>
                        <x-tabler-chevron-down class="ms-auto size-4 text-label" />
                    </div>
                </template>
            </div>
            <div
                class="absolute z-10 mt-1 max-h-60 w-full overflow-y-auto border bg-background"
                x-show="open"
                @click.away="open = false"
            >
                <div class="grid grid-cols-2 gap-4 p-2">
                    @foreach ($avatars as $avatar)
                        <div
                            class="flex cursor-pointer flex-col items-center p-2 hover:bg-heading-foreground/5"
                            @click="selectedAvatar = { avatar_id: '{{ $avatar['avatar_id'] }}', avatar_name: '{{ $avatar['avatar_name'] }}', preview_image_url: '{{ $avatar['preview_image_url'] }}' }; open = false"
                        >
                            <img
                                class="h-24 w-24 rounded-full border object-cover shadow-md"
                                src="{{ $avatar['preview_image_url'] }}"
                                alt="{{ $avatar['avatar_name'] }}"
                            >
                            <div class="mt-2 text-center">
                                <span>{{ $avatar['avatar_name'] }}</span>
                            </div>
                        </div>
                    @endforeach

                </div>
            </div>
            <input
                type="hidden"
                required
                name="avatar_id"
                :value="selectedAvatar ? selectedAvatar.avatar_id : ''"
            >
        </div>

        <x-forms.input
            id="avatar_style"
            size="lg"
            type="select"
            label="{{ __('Avatar Style') }}"
            name="avatar_style"
            required
        >
            <option value="normal">{{ __('Normal') }}</option>
            <option value="circle">{{ __('Circle') }}</option>
            <option value="closeUp">{{ __('CloseUp') }}</option>
        </x-forms.input>

        <x-forms.input
            id="matting"
            size="lg"
            type="select"
            label="{{ __('Matting') }}"
            name="matting"
            required
        >
            <option value="true">{{ __('True') }}</option>
            <option value="false">{{ __('False') }}</option>
        </x-forms.input>

        <x-forms.input
            id="caption"
            size="lg"
            type="select"
            label="{{ __('Caption') }}"
            name="caption"
            required
        >
            <option value="false">{{ __('False') }}</option>
            <option value="true">{{ __('True') }}</option>
        </x-forms.input>

        <x-form-step
            step="2"
            label="{{ __('Voice') }}"
        >
        </x-form-step>

        <div
            class="voice-select-container"
            x-data="voicePreview()"
        >
            <div class="flex items-end gap-4">
                <x-forms.input
                    id="voice-select"
                    type="select"
                    container-class="w-full"
                    x-model="selectedAudio"
                    @change="selectedVoiceId = $event.target.options[$event.target.selectedIndex].dataset.voiceId"
                    name="voice_audio"
                    required
                    label="{{ __('Select a Voice') }}"
                >
                    <option value="">{{ __('Select a voice') }}</option>
                    @foreach ($voices as $voice)
                        <option
                            data-voice-id="{{ $voice['voice_id'] }}"
                            value="{{ $voice['preview_audio'] }}"
                        >
                            {{ $voice['name'] }} ({{ $voice['language'] }} - {{ ucfirst($voice['gender']) }})
                        </option>
                    @endforeach
                </x-forms.input>

                <x-button
                    class="preview-speech group size-10 shrink-0"
                    variant="ghost-shadow"
                    size="none"
                    type="button"
                    title="{{ __('Preview') }}"
                    @click="playPreview"
                >
                    <x-tabler-volume class="size-4 group-[.loading]:hidden" />
                    <x-tabler-refresh class="lqd-icon-loader hidden size-4 animate-spin group-[.loading]:block" />
                </x-button>
            </div>

            <!-- Hidden input for voice_id -->
            <input
                type="hidden"
                name="voice_id"
                x-model="selectedVoiceId"
            >
        </div>

        <x-forms.input
            id="input_text"
            label="{{ __('Input Text') }}"
            placeholder="{{ __('Input for text to voice.') }}"
            type="textarea"
            rows="3"
            name="input_text"
            required
        ></x-forms.input>

        @if ($app_is_demo)
            <x-button
                type="button"
                onclick="return toastr.info('This feature is disabled in Demo version.');"
            >
                {{ __('Generate Video') }}
            </x-button>
        @else
            <x-button
                id="synthesia_btn"
                type="submit"
                form="synthesia_form"
            >
                {{ __('Generate Video') }}
            </x-button>
        @endif

    </form>
    <div
        class="spinner-border text-primary"
        id="preloader"
        role="status"
        style="display: none;width: 3rem; height: 3rem;"
    >
        <span class="sr-only">Loading...</span>
    </div>
@endsection
<style>
    #preloader {
        display: none;
        position: fixed;
        top: 50%;
        left: 50%;
        z-index: 1000;
    }
</style>
@push('script')
    <script>
        document.getElementById('synthesia_form').addEventListener('submit', function(event) {
            document.getElementById('preloader').style.display = 'block';
        });
    </script>
    <script>
        function voicePreview() {
            return {
                selectedAudio: null,
                selectedVoiceId: null, // Seçilen sesin voice_id'si
                isLoading: false,
                audioPlayer: null,

                playPreview() {
                    if (!this.selectedAudio) {
                        alert("{{ __('Please select a voice to preview.') }}");
                        return;
                    }

                    if (this.audioPlayer) {
                        this.audioPlayer.pause();
                        this.audioPlayer.currentTime = 0;
                    }

                    this.isLoading = true;
                    this.audioPlayer = new Audio(this.selectedAudio);
                    this.audioPlayer.play()
                        .then(() => {
                            this.isLoading = false;
                        })
                        .catch((error) => {
                            console.error("Audio playback error:", error);
                            alert("{{ __('Failed to play the selected audio.') }}");
                            this.isLoading = false;
                        });

                    this.audioPlayer.addEventListener('ended', () => {
                        this.isLoading = false;
                    });
                }
            };
        }
    </script>
@endpush
