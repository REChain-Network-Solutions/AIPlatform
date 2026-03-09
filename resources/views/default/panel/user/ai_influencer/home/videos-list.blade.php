{{-- Videos List --}}
<div
    class="py-14"
    x-data="exportedVideoData"
>
    <h2 class="mb-9">
        @lang('My Videos')
    </h2>

    <svg
        width="0"
        height="0"
        xmlns="http://www.w3.org/2000/svg"
    >
        <defs>
            <linearGradient
                id="icon-gradient"
                x1="0.546875"
                y1="3.69866"
                x2="12.7738"
                y2="14.7613"
                gradientUnits="userSpaceOnUse"
            >
                <stop stop-color="hsl(var(--gradient-from))" />
                <stop
                    offset="0.502"
                    stop-color="hsl(var(--gradient-via))"
                />
                <stop
                    offset="1"
                    stop-color="hsl(var(--gradient-to))"
                />
            </linearGradient>
        </defs>
    </svg>

    <div class="grid grid-cols-1 gap-5 md:grid-cols-2 lg:grid-cols-3">
        <template x-for="video in exportedVideos">
            <template x-if="checkExtensionsStatus(video)">
                <div class="flex flex-col overflow-hidden rounded-xl border">
                    <div class="flex h-full w-full justify-center">
                        <template x-if="video.status == '{{ \App\Enums\AiInfluencer\VideoStatusEnum::COMPLETED->value }}'">
                            <div class="flex h-[213px] w-full justify-center bg-foreground/80">
                                <video
                                    class="h-full object-cover"
                                    x-show="video.videoDuration"
                                    :id="'generated-video-item-' + video.id"
                                    :src="video.video_url"
                                    loop
                                    @loadedmetadata="video.videoDuration = $event.target.duration"
                                ></video>
                                <div
                                    class="flex h-full w-full flex-col items-center justify-center gap-2.5"
                                    x-show="!video.videoDuration"
                                >
                                    <x-tabler-loader-2
                                        class="text-gradient size-6 animate-spin"
                                        stroke="url(#icon-gradient)"
                                    />
                                    <span
                                        class="bg-gradient-to-r from-gradient-from via-gradient-via to-gradient-to bg-clip-text text-sm font-semibold text-transparent">{{ __('Loading') }}</span>
                                </div>
                            </div>
                        </template>
                        <template x-if="video.status == '{{ \App\Enums\AiInfluencer\VideoStatusEnum::IN_PROGRESS->value }}'">
                            <div
                                class="flex h-full min-h-80 w-full flex-col items-center justify-center gap-2.5 bg-foreground/80"
                                x-init="fetchPendingVideo(video)"
                            >
                                <x-tabler-loader-2
                                    class="text-gradient size-6 animate-spin"
                                    stroke="url(#icon-gradient)"
                                />
                                <span
                                    class="bg-gradient-to-r from-gradient-from via-gradient-via to-gradient-to bg-clip-text text-sm font-semibold text-transparent"
                                    x-text="video.progress ? `In Progress ${video.progress}%` : 'In Progress' "
                                ></span>
                            </div>
                        </template>
                    </div>
                    <template x-if="video.status == '{{ \App\Enums\AiInfluencer\VideoStatusEnum::COMPLETED->value }}'">
                        <div class="flex w-full flex-shrink-0 flex-col items-center justify-center gap-2 py-5">
                            <div class="flex max-w-56 flex-col gap-2">
                                <span
                                    class="text-center text-2xs font-medium text-heading-foreground"
                                    x-text="getDurationByString(video.videoDuration)"
                                ></span>
                                <span
                                    class="text-center text-sm font-semibold text-heading-foreground"
                                    x-text="video.title"
                                ></span>
                                <span
                                    class="text-center text-2xs font-medium text-foreground"
                                    x-text="'Created ' + getCreatedTimeByString(Math.floor((new Date() - new Date(video.created_at)) / 1000))"
                                >
                                </span>
                            </div>
                            <div class="flex items-center gap-3">
                                <span
                                    class="flex size-9 cursor-pointer items-center justify-center"
                                    :class="!video.videoDuration ? 'pointer-events-none' : ''"
                                    @click="playVideo(video)"
                                >
                                    <x-tabler-player-play
                                        class="size-6"
                                        x-show="!video.playing"
                                    />
                                    <x-tabler-player-pause
                                        class="size-6"
                                        x-show="video.playing"
                                    />
                                </span>
                                <x-button
                                    class="download size-9 bg-background text-foreground hover:bg-background hover:bg-emerald-400 hover:text-white"
                                    size="none"
                                    ::href="video.video_url"
                                    download
                                >
                                    <x-tabler-circle-chevron-down class="size-6" />
                                </x-button>
                                <span
                                    class="flex size-9 cursor-pointer items-center justify-center"
                                    @click.prevent="deleteGeneratedVideo(video.id)"
                                >
                                    <x-tabler-xbox-x class="size-6" />
                                </span>
                            </div>
                        </div>
                    </template>
                </div>
            </template>
        </template>
    </div>

    <x-modal
        class:close-btn="player-modal-close"
        title="{{ __('Play a Video') }}"
    >
        <x-slot:trigger
            custom
        >
            <button
                class="hidden"
                id="ai-influencer-player-modal-trigger"
                @click.prevent="modalOpen=true"
            ></button>
        </x-slot:trigger>
        <x-slot:modal
            id="ai-influencer-player-modal"
        >
            <div
                class="flex h-[80vh] items-center justify-center"
                x-init="$watch('modalOpen', value => { if (value == false) { outsideClickHandler() } })"
            >
                <video
                    class="player-modal-video h-full object-cover"
                    src=""
                    controls
                ></video>
            </div>
        </x-slot:modal>
    </x-modal>
</div>

@push('script')
    <script>
        document.addEventListener('alpine:init', () => {
            Alpine.data('exportedVideoData', () => ({
                playingVideo: null,
                // exported videos
                exportedVideos: @json($exportedVideos) || [],
                extensionStatus: {
                    urlToVideo: {{ \App\Helpers\Classes\MarketplaceHelper::isRegistered('url-to-video') ? 'true' : 'false' }},
                    aiViralClip: {{ \App\Helpers\Classes\MarketplaceHelper::isRegistered('ai-viral-clips') ? 'true' : 'false' }},
                    influencerAvatar: {{ \App\Helpers\Classes\MarketplaceHelper::isRegistered('influencer-avatar') ? 'true' : 'false' }},
                },
                // video play modal
                videoPlayModal: {
                    /**@type {HTMLElement} */
                    video: null,
                    /**@type {HTMLElement} */
                    modalCloseBtn: null,
                    /**@type {HTMLElement} */
                    modalTriggerBtn: null,
                    init() {
                        Alpine.nextTick(() => {
                            const modal = document.getElementById('ai-influencer-player-modal');

                            this.video = modal.querySelector('.player-modal-video');
                            this.modalCloseBtn = document.querySelector('.player-modal-close');
                            this.modalTriggerBtn = document.getElementById('ai-influencer-player-modal-trigger');
                        });
                    }
                },
                init() {
                    Alpine.store('exportedVideoData', this);

                    // video play modal init
                    this.videoPlayModal.init();
                },
                /**
                 * ---------------------------------------
                 * Add Event Listeners
                 * ---------------------------------------
                 */
                // play the video
                playVideo(video) {
                    if (this.playingVideo) {
                        if (this.playingVideo.id != video.id && this.playingVideo.playing) {
                            this._changeVideoStatus(this.playingVideo);
                        } else {
                            this.playingVideo = video;
                        }
                    } else {
                        this.playingVideo = video;
                    }

                    this._changeVideoStatus(video);
                },
                // change vidoe status whether paly or pause
                _changeVideoStatus(video) {
                    video.playing = !video.playing;

                    if (video.playing) {
                        const videoEl = document.getElementById('generated-video-item-' + video.id);

                        this.videoPlayModal.modalTriggerBtn.click();
                        this.videoPlayModal.video.src = videoEl.src;
                        this.videoPlayModal.video?.play();
                    } else {
                        this.videoPlayModal.video?.pause();
                        this.videoPlayModal.video.currentTime = 0;
                    }
                },
                // get duration
                getDurationByString(duration) {
                    duration = parseInt(duration || 0);

                    const mins = Math.floor(duration / 60);
                    const secs = Math.floor(duration % 60);

                    return (mins > 9 ? mins : `0${mins}`) + ':' + (secs > 9 ? secs : `0${secs}`)
                },
                // get diff time between create the video
                getCreatedTimeByString(diff) {
                    return diff < 60 ? " {{ __('Just now') }}" :
                        diff < 3600 ? (Math.floor(diff / 60) === 1 ?
                            "1 {{ __('minute ago') }}" : Math.floor(diff / 60) +
                            " {{ __('minutes ago') }}") :
                        diff < 86400 ? (Math.floor(diff / 3600) === 1 ?
                            "1 {{ __('hour ago') }}" : Math.floor(diff / 3600) +
                            " {{ __('hours ago') }}") :
                        Math.floor(diff / 86400) === 1 ? "1 {{ __('day ago') }}" : Math.floor(
                            diff / 86400) + " {{ __('days ago') }}"
                },
                // click outside of area
                outsideClickHandler() {
                    console.log(this.playingVideo);
                    if (this.playingVideo) {
                        if (this.playingVideo.playing) {
                            this._changeVideoStatus(this.playingVideo);
                        }
                        this.playingVideo = null;
                    }
                },
                // delete the video
                async deleteGeneratedVideo(videoId) {
                    if (!window.confirm(
                            '{{ __('Are you sure you want to delete this video?') }}')) {
                        return;
                    }

                    try {
                        Alpine.store('appLoadingIndicator').show();

                        const res = await fetch(
                            '{{ route('dashboard.user.ai-influencer.delete-exported-video') }}', {
                                method: 'delete',
                                headers: {
                                    'Content-Type': 'application/json',
                                    'Accept': 'application/json'
                                },
                                body: JSON.stringify({
                                    'video_id': videoId
                                })
                            });

                        const resData = await res.json();

                        if (!res.ok || resData.status == 'error') {
                            throw new Error(resData?.message ||
                                '{{ __('Unexpected issue happen') }}')
                        }

                        const index = this.exportedVideos.findIndex(item =>
                            item.id == videoId);
                        if (index != -1) {
                            this.exportedVideos.splice(index, 1);
                        }
                        toastr.success('{{ __('Successfully removed video') }}')
                    } catch (error) {
                        toastr.error(error?.message || error);
                        console.error(error);
                    }
                    Alpine.store('appLoadingIndicator').hide();
                },
                // add new inprogress video
                addNewInProgressVideo(video) {
                    Array.isArray(this.exportedVideos) ? this.exportedVideos.unshift(
                        video) : this.exportedVideos = [video];
                },
                // fetch pending videos
                fetchPendingVideo(video) {
                    if (video.used_ai_tool == 'fal-ai') {
                        this.checkVideoGenerationStatusFalAI(video.task_id);
                    } else if (video.used_ai_tool == 'creatify') {
                        this.checkVideoGenerationStatusCreatify(video)
                    } else if (video.used_ai_tool == 'topview') {
                        this.getExportVideoResultTopview(video);
                    } else if (video.used_ai_tool == 'klap') {
                        this.getExportResultKlap(video);
                    }
                },
                // check if extensions are available
                checkExtensionsStatus(video) {
                    if (video.status ===
                        '{{ \App\Enums\AiInfluencer\VideoStatusEnum::COMPLETED->value }}') {
                        return true;
                    }

                    if (video.used_ai_tool == 'fal-ai') {
                        return this.extensionStatus.influencerAvatar;
                    } else if (video.used_ai_tool == 'creatify' || video.used_ai_tool == 'topview') {
                        return this.extensionStatus.urlToVideo;
                    } else if (video.used_ai_tool == 'klap' || video.used_ai_tool == 'vizard') {
                        return this.extensionStatus.aiViralClip;
                    }

                    return false;
                },
                // check status for fal ai video result
                checkVideoGenerationStatusFalAI(requestId) {
                    const ms = (delay) => new Promise(resolve => setTimeout(resolve, delay));

                    $.ajax({
                        type: 'get',
                        url: '/dashboard/user/influencer-avatar/check-video-status' +
                            `/${requestId}`,
                        success: async (data) => {
                            if (data.resData.status == 'COMPLETED') {
                                this.getFinalResultFalAI(requestId);
                            } else {
                                await ms(2000);
                                this.checkVideoGenerationStatusFalAI(requestId);
                            }
                        },
                        error: (error) => {
                            toastr.error(error?.responseJSON?.message ||
                                '{{ __('Unexpected issue happen while generate video') }}'
                            );
                            console.error(error);
                        }
                    });
                },
                // get final result for fal ai video
                getFinalResultFalAI(requestId) {
                    $.ajax({
                        type: 'get',
                        url: '/dashboard/user/influencer-avatar/get-final-result' +
                            `/${requestId}`,
                        success: (data) => {
                            const videoIndex = this.exportedVideos.findIndex((video) =>
                                video
                                .task_id == requestId);

                            if (data.status == 'success') {
                                if (videoIndex != -1) {
                                    this.exportedVideos[videoIndex] = data.resData;
                                }
                            } else {
                                this.exportedVideos.splice(videoIndex, 1);
                                console.error(data.message ||
                                    '{{ __('Unexpected issue happen') }}');
                            }
                        },
                        error: (error) => {
                            toastr.error(error?.responseJSON?.message ||
                                '{{ __('Unexpected issue happen while generate video') }}'
                            );
                            console.error(error);
                        }
                    });
                },
                // check status for creatify video result
                checkVideoGenerationStatusCreatify(video) {
                    const ms = (delay) => new Promise(resolve => setTimeout(resolve, delay));

                    $.ajax({
                        type: 'get',
                        url: `/creatify/get-video-result?id=${video.task_id}`,
                        success: async (data) => {
                            if (data.resData.status == 'done') {
                                this.storeFinalGeneratedVideoCreatify(data.resData);
                            } else if (data.resData.status == 'pending' || data.resData
                                .status == 'in_queue' || data.resData.status == 'running'
                            ) {
                                video.progress = Math.floor(data.resData.progress * 100);
                                await ms(1000);
                                this.checkVideoGenerationStatusCreatify(video);
                            } else {
                                this.storeFinalGeneratedVideoCreatify(data.resData);
                                toastr.error(
                                    'Unexpected issue happen while generate final video'
                                );
                            }
                        },
                        error: (error) => {
                            toastr.error(error?.responseJSON?.message ||
                                '{{ __('Unexpected issue happen while generate video') }}'
                            );
                            console.error(error);
                        }
                    });
                },
                // store final generated video for creatify
                async storeFinalGeneratedVideoCreatify(params) {
                    const data = {
                        status: params.status,
                        title: params.name || 'Video',
                        video_url: params.video_output || '',
                        task_id: params.id
                    }

                    try {
                        const res = await fetch(
                            '/dashboard/user/url-to-video/store-creatify-final-video', {
                                headers: {
                                    'Accept': 'application/json',
                                    'Content-Type': 'application/json'
                                },
                                method: 'POST',
                                body: JSON.stringify(data)
                            });

                        if (!res.ok) {
                            throw new Error('Error happen while store final result');
                        }

                        const resData = await res.json();

                        if (resData.status == 'error') {
                            throw new Error('Error happen while store final result');
                        }

                        const videoIndex = this.exportedVideos.findIndex((video) =>
                            video.task_id == params.id);

                        if (resData.status == 'success') {
                            if (videoIndex != -1) {
                                if (params.status == 'done') {
                                    this.exportedVideos[videoIndex] = resData.resData;
                                } else {
                                    this.exportedVideos.splice(videoIndex, 1);
                                }
                            }
                        } else {
                            this.exportedVideos.splice(videoIndex, 1);
                            throw new Error(resData.message ||
                                '{{ __('Unexpected issue happen') }}');
                        }
                    } catch (error) {
                        console.error(error);
                    }
                },
                // fetch topview export video result
                async getExportVideoResultTopview(video) {
                    const ms = (time) => new Promise((resolve) => setTimeout(resolve, time));

                    const taskId = video.task_id.split(',')[0];
                    const scriptId = video.task_id.split(',')[1];
                    while (true) {
                        try {
                            const res = await fetch(
                                `/topview/avatar-marketing-video/query-task?taskId=${taskId}`, {
                                    method: 'get',
                                    headers: {
                                        'Accept': 'application/json'
                                    }
                                })

                            if (!res.ok) {
                                throw new Error(res.message || 'Unexpected issue happen');
                            }

                            const resData = await res.json();
                            if (resData.status == 'error') {
                                throw new Error(resData.message || 'Unexpected issue happen');
                            }

                            const exportedVideo = resData.resData;
                            if (exportedVideo.status == 'success') {
                                const selectedVideo = exportedVideo.exportVideos.find(video => video
                                    .scriptId == scriptId);
                                if (selectedVideo.status == 'success') {
                                    selectedVideo.task_id = video.task_id;
                                    selectedVideo.status = 'success';

                                    this.storeFinalGeneratedVideoTopview(selectedVideo);
                                    break;
                                } else {
                                    await ms(2000);
                                }
                            } else if (exportedVideo.errorMsg == null || exportedVideo.errorMsg ==
                                '') {
                                await ms(2000);
                            } else {
                                exportedVideo.task_id = video.task_id;
                                exportedVideo.status = 'error';

                                this.storeFinalGeneratedVideoTopview(exportedVideo);

                                throw new Error(previewResult.errorMsg ||
                                    '{{ __('Unexpected issue happen while export the video') }}'
                                );
                            }
                        } catch (error) {
                            toastr.error(error?.message || error);
                            console.error(error);
                            break;
                        }
                    }
                },
                // store final generated video for topview
                async storeFinalGeneratedVideoTopview(params) {
                    const data = {
                        status: params.status,
                        title: params.title || 'Video',
                        video_url: params.videoUrl || '',
                        task_id: params.task_id
                    }

                    try {
                        const res = await fetch(
                            '/dashboard/user/url-to-video/store-topview-final-video', {
                                headers: {
                                    'Accept': 'application/json',
                                    'Content-Type': 'application/json'
                                },
                                method: 'POST',
                                body: JSON.stringify(data)
                            });

                        if (!res.ok) {
                            throw new Error('Error happen while store final result');
                        }

                        const resData = await res.json();

                        if (resData.status == 'error') {
                            throw new Error('Error happen while store final result');
                        }

                        const videoIndex = this.exportedVideos.findIndex((video) =>
                            video.task_id == params.task_id);

                        if (resData.status == 'success') {
                            if (videoIndex != -1) {
                                if (params.status == 'success') {
                                    this.exportedVideos[videoIndex] = resData.resData;
                                } else {
                                    this.exportedVideos.splice(videoIndex, 1);
                                }
                            }
                        } else {
                            this.exportedVideos.splice(videoIndex, 1);
                            throw new Error(resData.message ||
                                '{{ __('Unexpected issue happen') }}');
                        }
                    } catch (error) {
                        console.error(error);
                    }
                },
                // fetch klap export result
                async getExportResultKlap(video) {
                    const ms = (time) => new Promise((resolve) => setTimeout(resolve, time));

                    const ids = video.task_id.split(',');

                    const export_id = ids[0];
                    const folderId = ids[1];
                    const projectId = ids[2];

                    while (true) {
                        try {
                            const res = await fetch(
                                `/ai-viral-clips/export-video-status?export_id=${export_id}&folderId=${folderId}&projectId=${projectId}`, {
                                    method: 'get',
                                    headers: {
                                        'Accept': 'application/json'
                                    }
                                })

                            if (!res.ok) {
                                throw new Error(res.message || 'Unexpected issue happen');
                            }

                            const resData = await res.json();
                            if (resData.status == 'error') {
                                throw new Error(resData.message || 'Unexpected issue happen');
                            }

                            const exportedVideo = resData.resData;
                            if (exportedVideo.status == 'processing') {
                                await ms(2000);
                            } else if (exportedVideo.status == 'error') {
                                exportedVideo.task_id = video.task_id;
                                exportedVideo.status = 'error';

                                this.storeFinalResultKlap(exportedVideo);

                                throw new Error(previewResult.errorMsg ||
                                    '{{ __('Unexpected issue happen while export the video') }}'
                                );
                            } else {
                                const finalResult = exportedVideo;
                                finalResult.task_id = video.task_id;
                                selectedVideo.status = 'success';

                                this.storeFinalResultKlap(selectedVideo);
                            }
                        } catch (error) {
                            toastr.error(error?.message || error);
                            console.error(error);
                            break;
                        }
                    }
                },
                // store final generated video for klap
                async storeFinalResultKlap(params) {
                    const data = {
                        status: params.status,
                        title: params.name || 'Video',
                        video_url: params.src_url || '',
                        task_id: params.task_id
                    }

                    try {
                        const res = await fetch(
                            '/ai-viral-clips/store-final-result-klap', {
                                headers: {
                                    'Accept': 'application/json',
                                    'Content-Type': 'application/json'
                                },
                                method: 'POST',
                                body: JSON.stringify(data)
                            });

                        if (!res.ok) {
                            throw new Error('Error happen while store final result');
                        }

                        const resData = await res.json();

                        if (resData.status == 'error') {
                            throw new Error('Error happen while store final result');
                        }

                        const videoIndex = this.exportedVideos.findIndex((video) =>
                            video.task_id == params.task_id);

                        if (resData.status == 'success') {
                            if (videoIndex != -1) {
                                if (params.status == 'success') {
                                    this.exportedVideos[videoIndex] = resData.resData;
                                } else {
                                    this.exportedVideos.splice(videoIndex, 1);
                                }
                            }
                        } else {
                            this.exportedVideos.splice(videoIndex, 1);
                            throw new Error(resData.message ||
                                '{{ __('Unexpected issue happen') }}');
                        }
                    } catch (error) {
                        console.error(error);
                    }
                }
            }))
        });
    </script>
@endpush
