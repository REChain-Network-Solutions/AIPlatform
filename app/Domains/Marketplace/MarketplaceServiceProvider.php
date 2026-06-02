<?php

declare(strict_types=1);

namespace App\Domains\Marketplace;

use App\Domains\Marketplace\Repositories\Contracts\ExtensionRepositoryInterface;
use App\Domains\Marketplace\Repositories\ExtensionRepository;
use App\Domains\Marketplace\Services\ExtensionInstallService;
use App\Domains\Marketplace\Services\ExtensionUninstallService;
use App\Extensions\AdvancedImage\System\AdvancedImageServiceProvider;
use App\Extensions\Affilate\System\AffilateServiceProvider;
use App\Extensions\AiAvatar\System\AiAvatarServiceProvider;
use App\Extensions\AiCaptions\System\AiCaptionsServiceProvider;
use App\Extensions\AIChatPro\System\AIChatProServiceProvider;
use App\Extensions\AIChatProDeepResearch\System\AIChatProDeepResearchServiceProvider;
use App\Extensions\AiChatProEntityHighlight\System\AiChatProEntityHighlightServiceProvider;
use App\Extensions\AIChatProFileChat\System\AIChatProFileChatServiceProvider;
use App\Extensions\AIChatProFolders\System\AIChatProFoldersServiceProvider;
use App\Extensions\AiChatProHighlightToAsk\System\AiChatProHighlightToAskServiceProvider;
use App\Extensions\AiChatProImageChat\System\AiChatProImageChatServiceProvider;
use App\Extensions\AIChatProMemory\System\AIChatProMemoryServiceProvider;
use App\Extensions\AIChatProSkills\System\AIChatProSkillsServiceProvider;
use App\Extensions\AiChatProSmartImage\System\AiChatProSmartImageServiceProvider;
use App\Extensions\AIImagePro\System\AIImageProServiceProvider;
use App\Extensions\AiMusic\System\AiMusicServiceProvider;
use App\Extensions\AiMusicPro\System\AiMusicProServiceProvider;
use App\Extensions\AiPersona\System\AiPersonaServiceProvider;
use App\Extensions\AIPhotoshoot\System\AIPhotoshootServiceProvider;
use App\Extensions\AIPlagiarism\System\AIPlagiarismServiceProvider;
use App\Extensions\AiPresentation\System\AiPresentationServiceProvider;
use App\Extensions\AIRealtimeImage\System\AIRealtimeImageServiceProvider;
use App\Extensions\AISocialMedia\System\AISocialMediaServiceProvider;
use App\Extensions\AiVideoPro\System\AiVideoProServiceProvider;
use App\Extensions\AIVideoToVideo\System\AIVideoToVideoServiceProvider;
use App\Extensions\AiViralClips\System\AiViralClipsServiceProvider;
use App\Extensions\AIVoiceIsolator\System\AIVoiceIsolatorServiceProvider;
use App\Extensions\AIWebChat\System\AIWebChatServiceProvider;
use App\Extensions\AIWriterTemplates\System\AIWriterTemplateServiceProvider;
use App\Extensions\Announcement\System\AnnouncementServiceProvider;
use App\Extensions\AzureOpenai\System\AzureOpenaiServiceProvider;
use App\Extensions\AzureTTS\System\AzureTTSServiceProvider;
use App\Extensions\BlogPilot\System\BlogPilotServiceProvider;
use App\Extensions\Canvas\System\CanvasServiceProvider;
use App\Extensions\Chatbot\System\ChatbotServiceProvider;
use App\Extensions\ChatbotAgent\System\ChatbotAgentServiceProvider;
use App\Extensions\ChatbotBooking\System\ChatbotBookingServiceProvider;
use App\Extensions\ChatbotCustomerTag\System\ChatbotCustomerTagServiceProvider;
use App\Extensions\ChatbotEcommerce\System\ChatbotEcommerceServiceProvider;
use App\Extensions\ChatbotInstagram\System\ChatbotInstagramServiceProvider;
use App\Extensions\ChatbotMessenger\System\ChatbotMessengerServiceProvider;
use App\Extensions\ChatbotReview\System\ChatbotReviewServiceProvider;
use App\Extensions\ChatbotTelegram\System\ChatbotTelegramServiceProvider;
use App\Extensions\ChatbotVoice\System\ChatbotVoiceServiceProvider;
use App\Extensions\ChatbotVoiceCall\System\ChatbotVoiceCallServiceProvider;
use App\Extensions\ChatbotWhatsapp\System\ChatbotWhatsappServiceProvider;
use App\Extensions\ChatProTempChat\System\ChatProTempChatServiceProvider;
use App\Extensions\ChatSetting\System\ChatSettingServiceProvider;
use App\Extensions\ChatShare\System\ChatShareServiceProvider;
use App\Extensions\CheckoutRegistration\System\RegistrationServiceProvider;
use App\Extensions\Cloudflare\System\CloudflareServiceProvider;
use App\Extensions\ContentManager\System\ContentManagerServiceProvider;
use App\Extensions\CreativeSuite\System\CreativeSuiteServiceProvider;
use App\Extensions\CreativeSuiteAITemplate\System\CreativeSuiteAITemplateServiceProvider;
use App\Extensions\CreativeSuiteAnnotations\System\CreativeSuiteAnnotationsServiceProvider;
use App\Extensions\Cryptomus\System\CryptomusServiceProvider;
use App\Extensions\DemoExtension\System\DemoExtensionServiceProvider;
use App\Extensions\DiscountManager\System\DiscountManagerServiceProvider;
use App\Extensions\ElevenLabsVoiceChat\System\ElevenLabsVoiceChatServiceProvider;
use App\Extensions\FashionStudio\System\FashionStudioServiceProvider;
use App\Extensions\FluxPro\System\FluxProServiceProvider;
use App\Extensions\FocusMode\System\FocusModeServiceProvider;
use App\Extensions\FooterMenu\System\FooterMenuServiceProvider;
use App\Extensions\Hubspot\System\HubspotServiceProvider;
use App\Extensions\Ideogram\System\IdeogramServiceProvider;
use App\Extensions\InfluencerAvatar\System\InfluencerAvatarServiceProvider;
use App\Extensions\LiveCustomizer\System\LiveCustomizerServiceProvider;
use App\Extensions\Mailchimp\System\MailchimpServiceProvider;
use App\Extensions\Maintenance\System\MaintenanceServiceProvider;
use App\Extensions\MarketingBot\System\MarketingBotServiceProvider;
use App\Extensions\MegaMenu\System\MegaMenuServiceProvider;
use App\Extensions\Menu\System\MenuServiceProvider;
use App\Extensions\Midjourney\System\MidjourneyServiceProvider;
use App\Extensions\Migration\System\MigrationServiceProvider;
use App\Extensions\ModelCouncil\System\ModelCouncilServiceProvider;
use App\Extensions\MultiModel\System\MultiModelServiceProvider;
use App\Extensions\NanoBanana\System\NanoBananaServiceProvider;
use App\Extensions\Newsletter\System\NewsletterServiceProvider;
use App\Extensions\Onboarding\System\OnboardingServiceProvider;
use App\Extensions\OnboardingPro\System\OnboardingProServiceProvider;
use App\Extensions\OpenAIRealtimeChat\System\OpenAIRealtimeChatServiceProvider;
use App\Extensions\OpenRouter\System\OpenRouterServiceProvider;
use App\Extensions\Perplexity\System\PerplexityServiceProvider;
use App\Extensions\PhotoStudio\System\PhotoStudioServiceProvider;
use App\Extensions\ProductPhotography\System\ProductPhotographyServiceProvider;
use App\Extensions\SeeDreamV4\System\SeeDreamV4ServiceProvider;
use App\Extensions\SEOTool\System\SEOToolServiceProvider;
use App\Extensions\SocialMedia\System\SocialMediaServiceProvider;
use App\Extensions\SocialMediaAgent\System\SocialMediaAgentServiceProvider;
use App\Extensions\SocialMediaAutomation\System\SocialMediaAutomationServiceProvider;
use App\Extensions\SpeechifyTTS\System\SpeechifyServiceProvider;
use App\Extensions\UGCCreator\System\UGCCreatorServiceProvider;
use App\Extensions\UGCFactory\System\UGCFactoryServiceProvider;
use App\Extensions\UrlToVideo\System\UrlToVideoServiceProvider;
use App\Extensions\VideoDubbing\System\VideoDubbingServiceProvider;
use App\Extensions\VideoEditor\System\VideoEditorServiceProvider;
use App\Extensions\Wordpress\System\WordpressServiceProvider;
use App\Extensions\Xero\System\XeroServiceProvider;
use Illuminate\Routing\Router;
use Illuminate\Support\Facades\Route;
use Illuminate\Support\ServiceProvider;

class MarketplaceServiceProvider extends ServiceProvider
{
    /**
     * The service provider bindings.
     */
    public array $bindings = [
        ExtensionRepositoryInterface::class        => ExtensionRepository::class,
    ];

    /**
     * The service provider bindings.
     *
     * @note Do not remove namespaces from the beginning of the class.
     */
    public static array $extensionProviders = [
        'chatbot'                       => ChatbotServiceProvider::class,
        'focus-mode'                    => FocusModeServiceProvider::class,
        'newsletter'                    => NewsletterServiceProvider::class,
        'photo-studio'                  => PhotoStudioServiceProvider::class,
        'ai-product-shot'               => ProductPhotographyServiceProvider::class,
        'ai-avatar'                     => AiAvatarServiceProvider::class,
        'ai-persona'                    => AiPersonaServiceProvider::class,
        'ai-music'                      => AiMusicServiceProvider::class,
        'ai-video-pro'                  => AiVideoProServiceProvider::class,
        'seo-tool'                      => SEOToolServiceProvider::class,
        'ai-social-media'               => AISocialMediaServiceProvider::class,
        'webchat'                       => AIWebChatServiceProvider::class,
        'onboarding'                    => OnboardingServiceProvider::class,
        'flux-pro'                      => FluxProServiceProvider::class,
        'chat-share'                    => ChatShareServiceProvider::class,
        'voice-isolator'                => AIVoiceIsolatorServiceProvider::class,
        'chat-setting'                  => ChatSettingServiceProvider::class,
        'hubspot'                       => HubspotServiceProvider::class,
        'menu'                          => MenuServiceProvider::class,
        'azure-tts'                     => AzureTTSServiceProvider::class,
        'plagiarism'                    => AIPlagiarismServiceProvider::class,
        'cloudflare-r2'                 => CloudflareServiceProvider::class,
        'wordpress'                     => WordpressServiceProvider::class,
        'cryptomus'                     => CryptomusServiceProvider::class,
        'affilate'                      => AffilateServiceProvider::class,
        'mailchimp-newsletter'          => MailchimpServiceProvider::class,
        'ai-writer-templates'           => AIWriterTemplateServiceProvider::class,
        'maintenance'                   => MaintenanceServiceProvider::class,
        'open-router'                   => OpenRouterServiceProvider::class,
        'advanced-image'                => AdvancedImageServiceProvider::class,
        'mega-menu'                     => MegaMenuServiceProvider::class,
        'onboarding-pro'                => OnboardingProServiceProvider::class,
        'ideogram'                      => IdeogramServiceProvider::class,
        'perplexity'                    => PerplexityServiceProvider::class,
        'checkout-registration'         => RegistrationServiceProvider::class,
        'openai-realtime-chat'          => OpenAIRealtimeChatServiceProvider::class,
        'ai-video-to-video'             => AIVideoToVideoServiceProvider::class,
        'midjourney'                    => MidjourneyServiceProvider::class,
        'social-media'                  => SocialMediaServiceProvider::class,
        'social-media-agent'            => SocialMediaAgentServiceProvider::class,
        'social-media-automation'       => SocialMediaAutomationServiceProvider::class,
        'blogpilot'                     => BlogPilotServiceProvider::class,
        'chatbot-agent'                 => ChatbotAgentServiceProvider::class,
        'chatbot-booking'               => ChatbotBookingServiceProvider::class,
        'chatbot-ecommerce'             => ChatbotEcommerceServiceProvider::class,
        'chatbot-customer-tag'          => ChatbotCustomerTagServiceProvider::class,
        'chatbot-review'                => ChatbotReviewServiceProvider::class,
        'xero'                          => XeroServiceProvider::class,
        'speechify-tts'                 => SpeechifyServiceProvider::class,
        'ai-chat-pro'                   => AIChatProServiceProvider::class,
        'announcement'                  => AnnouncementServiceProvider::class,
        'ai-realtime-image'             => AIRealtimeImageServiceProvider::class,
        'azure-openai'                  => AzureOpenaiServiceProvider::class,
        'chatbot-voice'                 => ChatbotVoiceServiceProvider::class,
        'chatbot-voice-call'            => ChatbotVoiceCallServiceProvider::class,
        'chatbot-telegram'              => ChatbotTelegramServiceProvider::class,
        'chatbot-whatsapp'              => ChatbotWhatsappServiceProvider::class,
        'chatbot-messenger'             => ChatbotMessengerServiceProvider::class,
        'chatbot-instagram'             => ChatbotInstagramServiceProvider::class,
        'marketing-bot'                 => MarketingBotServiceProvider::class,
        'migration'                     => MigrationServiceProvider::class,
        'live-customizer'               => LiveCustomizerServiceProvider::class,
        'elevenlabs-voice-chat'         => ElevenLabsVoiceChatServiceProvider::class,
        'creative-suite'                => CreativeSuiteServiceProvider::class,
        'creative-suite-annotations'    => CreativeSuiteAnnotationsServiceProvider::class,
        'url-to-video'                  => UrlToVideoServiceProvider::class,
        'ai-viral-clips'                => AiViralClipsServiceProvider::class,
        'influencer-avatar'             => InfluencerAvatarServiceProvider::class,
        'content-manager'               => ContentManagerServiceProvider::class,
        'canvas'                        => CanvasServiceProvider::class,
        'discount-manager'              => DiscountManagerServiceProvider::class,
        'footer-menu'                   => FooterMenuServiceProvider::class,
        'chat-pro-temp-chat'            => ChatProTempChatServiceProvider::class,
        'demo-extension'                => DemoExtensionServiceProvider::class,
        'multi-model'                   => MultiModelServiceProvider::class,
        'model-council'                 => ModelCouncilServiceProvider::class,
        'ai-chat-pro-skills'            => AIChatProSkillsServiceProvider::class,
        'ai-chat-pro-deep-research'     => AIChatProDeepResearchServiceProvider::class,
        'nano-banana'                   => NanoBananaServiceProvider::class,
        'ai-chat-pro-file-chat'         => AIChatProFileChatServiceProvider::class,
        'ai-music-pro'                  => AiMusicProServiceProvider::class,
        'see-dream-v4'                  => SeeDreamV4ServiceProvider::class,
        'ai-presentation'               => AiPresentationServiceProvider::class,
        'ai-image-pro'                  => AIImageProServiceProvider::class,
        'ai-chat-pro-image-chat'        => AiChatProImageChatServiceProvider::class,
        'ai-chat-pro-folders'           => AIChatProFoldersServiceProvider::class,
        'ai-chat-pro-memory'            => AIChatProMemoryServiceProvider::class,
        'ai-chat-pro-smart-image'       => AiChatProSmartImageServiceProvider::class,
        'fashion-studio'                => FashionStudioServiceProvider::class,
        'creative-suite-ai-template'    => CreativeSuiteAITemplateServiceProvider::class,
        'ai-photoshoot'                 => AIPhotoshootServiceProvider::class,
        'ai-chat-pro-entity-highlight'  => AiChatProEntityHighlightServiceProvider::class,
        'ai-chat-pro-highlight-to-ask'  => AiChatProHighlightToAskServiceProvider::class,
        'video-editor'					             => VideoEditorServiceProvider::class,
        'ugc-factory'                   => UGCFactoryServiceProvider::class,
        'ugc-creator'                   => UGCCreatorServiceProvider::class,
        'video-dubbing'                 => VideoDubbingServiceProvider::class,
        'ai-captions'                   => AiCaptionsServiceProvider::class,
    ];

    public function register(): void
    {
        $this->extensionProviderRegister();
    }

    public function boot(): void
    {
        $this->registerRoutes();
    }

    private function registerRoutes(): void
    {
        $this->router()
            ->group([
                'middleware' => ['web', 'auth'],
            ], function (Router $route) {
                $route->get('dashboard/marketplace/extension/{slug}/install', function (string $slug) {
                    return $this
                        ->app
                        ->make(ExtensionInstallService::class)
                        ->install($slug);
                })->name('marketplace.extension.install');

                $route->get('dashboard/marketplace/extension/{slug}/uninstall', function (string $slug) {
                    return $this
                        ->app
                        ->make(ExtensionUninstallService::class)
                        ->uninstall($slug);
                })->name('marketplace.extension.uninstall');
            });
    }

    private function router(): Router|Route
    {
        return $this->app['router'];
    }

    public function extensionProviderRegister(): void
    {
        foreach (static::$extensionProviders as $provider) {
            if (class_exists($provider)) {
                $this->app->register($provider);
            }
        }
    }

    public static function uninstallExtension(string $slug): void
    {
        if (isset(self::$extensionProviders[$slug])) {

            $provider = self::$extensionProviders[$slug];

            if (method_exists($provider, 'uninstall')) {
                $provider::uninstall();
            }
        }
    }

    public static function getExtensionProviders(): array
    {
        return static::$extensionProviders;
    }
}
