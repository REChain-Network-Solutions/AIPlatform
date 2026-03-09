<?php

namespace App\Helpers\Classes;

use App\Models\OpenAIGenerator;
use App\Models\Plan;
use Illuminate\Support\Facades\Auth;

class MenuHelper
{
    public $userPlan = null;

    public $generators = null;

    public $freeOpenAiItems = null;

    public $user = null;

    public array $routesDoesNotHaveAnySlug = [
        'ai_editor',
        'ai_writer',
        'ai_chat_all',
        'ai_chat_all',
        'ai_plagiarism',
        'ai_content_detect',
        'ai_article_wizard_generator',
        'ai_webchat',
        'ai_rewriter',
        'seo',
        'brand_voice',
        'support',
        'photo_studio_extension',
        'ai_social_media_extension',
        'ai_social_media_extension',
        'ext_chat_bot',
    ];

    public array $slugsNotInOpenAiGenerator = [
        'ai_chat_all',
        'ai_editor',
        'ai_writer',
        'ai_social_media_extension',
        'ext_chat_bot',
        'brand_voice',
        'photo_studio_extension',
        'ai_vision',
        'ai_product_shot',
        'ai_image_generator',
        'ai_video',
        'seo_tool_extension',
        'ai_voiceover',
        'ai_pdf',
        'ai_speech_to_text',
        'ai_rewriter',
        'ai_code_generator',
        'ai_youtube',
        'ai_chat_image',
        'ai_rss',
        'ai_voiceover_clone',
        'ai_web_chat_extension',
        'ai_realtime_voice_chat',
        'ai_detector_extension',
        'ai_plagiarism_extension',
        'ai_article_wizard',
        'ai_voice_isolator',
        'ext_voice_chatbot',
        'ext_social_media_dropdown',
        'ai_influencer',
        'creative_suite',
        'url_to_video',
        'viral_clips',
        'influencer_avatar',
        'brand_voice',
        'support',
    ];

    public function __construct()
    {
        $this->generators = OpenAIGenerator::query()
            ->whereIn('slug', $this->routesDoesNotHaveAnySlug)
            ->where('active', 1)
            ->get();

        $this->freeOpenAiItems = Helper::setting('free_open_ai_items');

        $this->user = Auth::user();

        if ($this->user?->getAttribute('team_manager_id')) {
            $this->user = $this->user?->getAttribute('teamManager');
        }

        $this->userPlan = $this->user->relationPlan;

    }

    public function checker(array $menu)
    {
        $slug = $menu['key'];

        $data = $this->routesDoesNotHaveAnySlug + $this->slugsNotInOpenAiGenerator + $this->freeOpenAiItems;

        if (Auth::user()?->isAdmin()) {
            return true;
        }

        $plan = $this?->userPlan;

        $menuIsAdmin = data_get($menu, 'is_admin', false);

        if (! $plan && $menuIsAdmin === false && $menu['type'] === 'item' && $menu['key'] !== 'dashboard') {
            return false;
        }

        if (! in_array($slug, $data)) {
            return true;
        }

        $openAi = $this->generators
            ->where('slug', $slug)
            ->first();

        if (! $plan) {
            // if no plan then its free user, can access all templates and features with credits they have except the premium
            if (in_array($slug, $this->freeOpenAiItems, true)) {
                return true;
            }

            return $openAi?->getAttribute('premium') !== 1;
        }

        if ($openAi || in_array($slug, $this->slugsNotInOpenAiGenerator, true)) {
            $setting = $this->settingSlug($slug);
            if ($setting['status']) {
                $setting = Helper::setting($setting['setting']);
                // abort if the feature is disabled, we don't need to return a message
                if ($setting === 0) {
                    return false;
                }
            }

            if ($plan->checkOpenAiItem($slug)) {
                return true;
            }
        }
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
