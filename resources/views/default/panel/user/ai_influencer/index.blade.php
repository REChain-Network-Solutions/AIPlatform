@extends('panel.layout.app', ['disable_tblr' => true])
@section('title', __('AI Influencer'))
@section('titlebar_subtitle')
    {{ __('Promote your products with engaging ads and videos.') }}
@endsection
@section('titlebar_actions')
    @if (\App\Helpers\Classes\MarketplaceHelper::isRegistered('url-to-video'))
        <x-button
            href="#"
            x-data
            @click.prevent="$store.aiUrlToVideoData.toggleUrlToVideoWindow()"
        >
            <x-tabler-plus class="size-4" />
            @lang('Create Video')
        </x-button>
    @endif
@endsection

@section('content')
    <div class="py-10">
        <div
            class="lqd-external-chatbot-edit"
            x-data="aiInflucencerData"
        >
            @include('panel.user.ai_influencer.home.actions-grid')
            @include('panel.user.ai_influencer.home.videos-list')
        </div>
    </div>
@endsection

@push('script')
    <script>
        document.addEventListener('alpine:init', () => {
            Alpine.data('aiInflucencerData', () => ({
                init() {
                    Alpine.store('aiInflucencerData', this);
                },
                // when open or close the window, change some neccessary css.
                toggleWindow(open = true) {
                    const topNoticeBar = document.querySelector('.top-notice-bar');
                    const navbar = document.querySelector('.lqd-navbar');
                    const pageContentWrap = document.querySelector('.lqd-page-content-wrap');

                    document.documentElement.style.overflow = open ? 'hidden' : '';

                    if (window.innerWidth >= 992) {

                        if (navbar) {
                            navbar.style.position = open ? 'fixed' : '';
                        }

                        if (pageContentWrap) {
                            pageContentWrap.style.paddingInlineStart = open ?
                                'var(--navbar-width)' : '';
                        }

                        if (topNoticeBar) {
                            topNoticeBar.style.visibility = open ? 'hidden' : '';
                        }
                    }
                }
            }))
        });
    </script>
@endpush
