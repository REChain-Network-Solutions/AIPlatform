<?php

namespace App\Http\Middleware;

use App\Enums\AccessType;
use App\Helpers\Classes\Helper;
use App\Models\OpenAIGenerator;
use Closure;
use Illuminate\Http\Request;
use Symfony\Component\HttpFoundation\Response;

class CheckTemplateTypeAndPlan
{
    public function handle(Request $request, Closure $next): Response
    {
        // if admin then allow everything
        if ($request->user()->isAdmin()) {
            return $next($request);
        }

        // if user then start checks
        if ($this->performUserChecks($request)) {
            return $next($request);
        }

        return to_route('dashboard.user.payment.subscription')->with(['message' => trans('If you want to use premium service, update your plan.'), 'type' => 'error']);
    }

    private function performUserChecks(Request $request): bool
    {
        $user = $request->user();
        if ($user?->getAttribute('team_manager_id')) {
            $user = $user?->getAttribute('teamManager');
        }

        // get the slug from the route
        $slug = $request->route('slug');

        // some routes have not any slug, so we need to add slug to them manually
        $routesDoesNotHaveAnySlug = [
            'dashboard.user.generator.index'             => 'ai_editor',
            'dashboard.user.openai.list'                 => 'ai_writer',
            'dashboard.user.openai.chat.chat'            => 'ai_chat_all',
            'dashboard.user.openai.chat.list'            => 'ai_chat_all',
            'dashboard.user.openai.plagiarism.index'     => 'ai_plagiarism',
            'dashboard.user.openai.detectaicontent.index'=> 'ai_content_detect',
            'dashboard.user.openai.articlewizard.new'    => 'ai_article_wizard_generator',
            'dashboard.user.openai.webchat.workbook'     => 'ai_webchat',
            'dashboard.user.openai.rewriter'             => 'ai_rewriter',
            'dashboard.user.seo.index'                   => 'seo',
            'dashboard.user.brand.index'                 => 'brand_voice',
            'dashboard.support.list'                     => 'support',
            'dashboard.user.photo-studio.index'          => 'photo_studio_extension',
            'dashboard.user.automation.index'            => 'ai_social_media_extension',
            'dashboard.user.automation.list'             => 'ai_social_media_extension',
            'dashboard.chatbot.index'                    => 'ext_chat_bot',
            'dashboard.user.social-media.index'          => 'ext_social_media_dropdown',
            'dashboard.user.social-media.campaign.index' => 'ext_social_media_dropdown',
            'dashboard.user.social-media.platforms'      => 'ext_social_media_dropdown',
            'dashboard.user.social-media.post.index'     => 'ext_social_media_dropdown',
            'dashboard.user.social-media.calendar'       => 'ext_social_media_dropdown',
            'dashboard.user.ai-music-pro.index'          => 'ext_ai_music_pro',
            'dashboard.user.ai-presentation.index'       => 'ai_presentation',
        ];

        // Wildcard route prefixes mapped to slugs
        $wildcardRoutes = [
            'dashboard.user.fashion-studio.' => 'ext_fashion_studio_dropdown',
            'dashboard.user.ai-image-pro.'   => 'ai_image_pro',
            'ai-image-pro.'                  => 'ai_image_pro',
            'ai-chat-image.'                 => 'ai_chat_pro_image_chat',
        ];

        $slugsToSkip = ['ai_realtime_voice_chat'];
        $routeName = $request->route()?->getName();

        if (! in_array($slug, $slugsToSkip, true)) {
            if (array_key_exists($routeName, $routesDoesNotHaveAnySlug)) {
                $slug = $routesDoesNotHaveAnySlug[$routeName];
            } else {
                // Check wildcard route prefixes
                foreach ($wildcardRoutes as $prefix => $wildcardSlug) {
                    if ($routeName && str_starts_with($routeName, $prefix)) {
                        $slug = $wildcardSlug;

                        break;
                    }
                }
            }
        }
        // get openai record if the slug exist
        $openAi = OpenAIGenerator::query()
            ->where('slug', $slug)
            ->where('active', 1)
            ->first();

        $plan = $user?->relationPlan;
        if (! $plan) {
            // if no plan then its free user, can access all templates and features with credits they have except the premium
            if (in_array($slug, Helper::setting('free_open_ai_items'), true)) {
                return true;
            }

            if (! $openAi) {
                return false;
            }

            $isPremium = $openAi->getAttribute('premium');
            $isAccessTypePremium = $openAi->getAttribute('access_type') === AccessType::PREMIUM->value;

            return ! ($isPremium === 1 || $isAccessTypePremium);
        }
        // now even if slug exist in route, openai table does not contain all slugs
        $slugsNotInOpenAiGenerator = ['ai_image_pro', 'ai_chat_pro_image_chat', 'ext_fashion_studio_dropdown', 'ai_chat_all', 'ai_editor', 'ai_writer', 'ai_social_media_extension', 'ext_chat_bot', 'brand_voice', 'photo_studio_extension', 'ext_ai_music_pro', 'ai_presentation'];
        // if openai record exist or slug is in the list of slugs that are not in openai generator
        if ($openAi || in_array($slug, $slugsNotInOpenAiGenerator, true)) {
            $setting = $this->settingSlug($slug);
            if ($setting['status']) {
                $setting = Helper::setting($setting['setting']);
                // abort if the feature is disabled, we don't need to return a message
                if ($setting === 0) {
                    abort(404);
                }
            }

            if ($plan->checkOpenAiItem($slug)) {
                return true;
            }
        }

        return false;
    }

    public function settingSlug($slug): array
    {
        $data = [
            'ai_article_wizard_generator' => 'feature_ai_article_wizard',
            'ai_writer'                   => 'feature_ai_writer',
            'ai_rewriter'                 => 'feature_ai_rewriter',
            'ai_chat_image'               => 'feature_ai_chat_image',
            'ai_image_generator'          => 'feature_ai_image',
            'ai_code_generator'           => 'feature_ai_code',
            'ai_speech_to_text'           => 'feature_ai_speech_to_text',
            'ai_voiceover'                => 'feature_ai_voiceover',
            'ai_vision'                   => 'feature_ai_vision',
            'ai_pdf'                      => 'feature_ai_pdf',
            'ai_youtube'                  => 'feature_ai_youtube',
            'ai_rss'                      => 'feature_ai_youtube',
        ];

        if (array_key_exists($slug, $data)) {
            return [
                'status'  => true,
                'setting' => $data[$slug],
            ];
        }

        return [
            'status' => false,
        ];
    }
}
