<div class="flex flex-nowrap items-center justify-around gap-11 max-xl:flex-col">
    @includeWhen(
        \App\Helpers\Classes\MarketplaceHelper::isRegistered('url-to-video'),
        'url-to-video::home.actions-grid')
    @includeWhen(
        \App\Helpers\Classes\MarketplaceHelper::isRegistered('influencer-avatar'),
        'influencer-avatar::home.actions-grid')
    @includeWhen(
        \App\Helpers\Classes\MarketplaceHelper::isRegistered('ai-viral-clips'),
        'ai-viral-clips::home.actions-grid')
</div>
