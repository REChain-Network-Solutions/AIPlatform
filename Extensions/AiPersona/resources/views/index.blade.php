@extends('panel.layout.app', ['disable_tblr' => true])
@section('title', __('AI Persona'))
@section('titlebar_subtitle', __('Create studio-quality videos with AI Persona and voiceovers in 130+ languages. It’s as easy as making a slide deck.'))
@section('titlebar_actions')
    <x-button href="{{ LaravelLocalization::localizeUrl(route('dashboard.user.ai-persona.create')) }}">
        <x-tabler-plus class="size-4" />
        {{ __('New Video') }}
    </x-button>
@endsection

@section('content')
    <div class="py-10">
        <x-card
                class="lqd-ai-avatar-card relative mb-12 overflow-hidden bg-[#F4E3FD] shadow-[0_2px_2px_hsla(0,0%,0%,0.1)] dark:bg-primary/5 lg:before:absolute lg:before:end-0 lg:before:top-0 lg:before:z-0 lg:before:h-full lg:before:w-4/12 lg:before:bg-gradient-to-r lg:before:from-transparent lg:before:to-[#9A6EE3] dark:lg:before:to-primary/20"
                class:body="flex flex-wrap justify-between gap-y-6"
                variant="solid"
                size="none"
        >
            <div class="w-full self-center p-10 lg:w-6/12 lg:p-14">
                <h3 class="mb-8 leading-6">
                    @lang('Create engaging videos with AI Persona and voiceovers in 130+ languages.')
                </h3>
                <div class="flex flex-wrap items-center gap-2">
                    <x-button href="{{ LaravelLocalization::localizeUrl(route('dashboard.user.ai-persona.create')) }}">
                        <x-tabler-plus class="size-4" />
                        @lang('Create New Video')
                    </x-button>
                    <x-button
                            class="bg-heading-background text-heading-foreground hover:bg-primary hover:text-primary-foreground"
                            href="#videos-container"
                    >
                        @lang('Browse Videos')
                    </x-button>
                </div>
            </div>
            <div class="flex w-full self-end lg:w-6/12 lg:justify-end lg:pe-12">
                <figure>
                    <img
                            width="295"
                            height="218"
                            src="{{ custom_theme_url('/assets/img/misc/ai-avatar.png') }}"
                            alt="@lang('AI Persona')"
                    >
                </figure>
            </div>
        </x-card>

        <div
                class="lqd-ai-videos-wrap"
                id="lqd-ai-videos-wrap"
        >
            <svg
                    width="0"
                    height="0"
            >
                <defs>
                    <linearGradient
                            id="loader-spinner-gradient"
                            x1="0.667969"
                            y1="6.10667"
                            x2="23.0413"
                            y2="25.84"
                            gradientUnits="userSpaceOnUse"
                    >
                        <stop stop-color="#82E2F4" />
                        <stop
                                offset="0.502"
                                stop-color="#8A8AED"
                        />
                        <stop
                                offset="1"
                                stop-color="#6977DE"
                        />
                    </linearGradient>
                </defs>
            </svg>

            @if (filled($list))
                <h3 class="mb-8">
                    @lang('My Videos')
                </h3>
            @else
                <h2 class="col-span-full flex items-center justify-center">
                    @lang('No videos found.')
                </h2>
            @endif

            <div id="videos-container">
                @include('ai-persona::videos-list', ['list' => $list])
            </div>

        </div>
    </div>
@endsection

<script>
    function checkVideoStatus() {
        fetch('{!! route('dashboard.user.ai-persona.check', ['ids' => $inProgress]) !!}')
            .then(response => response.json())
            .then(data => {
                for (const [id, item] of Object.entries(data.data)) {
                    let videoElement = document.getElementById(item.divId);
                    if (videoElement) {
                        videoElement.innerHTML = item.html;
                    }
                }
            })
            .catch(error => console.error('Error:', error));
    }

    document.addEventListener('DOMContentLoaded', function () {
        setInterval(checkVideoStatus, 5000);
    });
</script>



