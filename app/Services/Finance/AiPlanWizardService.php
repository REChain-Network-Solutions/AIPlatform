<?php

declare(strict_types=1);

namespace App\Services\Finance;

use App\Domains\Engine\Enums\EngineEnum;
use App\Domains\Entity\Enums\EntityEnum;
use App\Enums\AITokenType;
use App\Enums\Plan\FrequencyEnum;
use App\Enums\Plan\TypeEnum;
use App\Helpers\Classes\ApiHelper;
use App\Helpers\Classes\Helper;
use App\Http\Controllers\Finance\PaymentProcessController;
use App\Models\Plan;
use App\Services\Ai\AiCompletionService;
use App\Services\Common\MenuService;
use Illuminate\Support\Arr;
use Illuminate\Support\Str;
use OpenAI\Laravel\Facades\OpenAI;
use Symfony\Component\HttpFoundation\StreamedResponse;
use Throwable;

class AiPlanWizardService
{
    public const UPDATES_DELIMITER = '<<<PLAN_UPDATES>>>';

    /**
     * @var array<string, float>
     */
    public const CREDIT_TIERS = [
        'starter'    => 0.5,
        'standard'   => 1.0,
        'pro'        => 2.0,
        'enterprise' => 5.0,
    ];

    /**
     * Fields the CoPilot and the wizard form are allowed to set on a plan draft.
     *
     * @var array<int, string>
     */
    public const ALLOWED_FIELDS = [
        'type', 'name', 'description', 'price', 'frequency', 'trial_days',
        'active', 'is_featured', 'features', 'credit_system_type',
        'shared_credits_amount', 'credit_tier', 'credit_limits', 'plan_ai_tools', 'plan_features',
    ];

    /**
     * User-facing model categories a per-category credit limit can target.
     *
     * @var array<int, string>
     */
    public const CREDIT_CATEGORIES = ['word', 'image', 'video', 'audio', 'presentation'];

    /**
     * Category map for plan AI tool keys so the CoPilot can resolve themed
     * requests ("only enable image tools") without guessing from labels.
     * Keys missing here fall back to the "other" category.
     *
     * @var array<string, string>
     */
    public const TOOL_CATEGORIES = [
        'ai_writer'                    => 'word',
        'ai_rewriter'                  => 'word',
        'ai_editor'                    => 'word',
        'ai_article_wizard'            => 'word',
        'seo_tool_extension'           => 'word',
        'ai_code_generator'            => 'word',
        'ai_rss'                       => 'word',
        'ai_detector_extension'        => 'word',
        'ai_plagiarism_extension'      => 'word',
        'ai_social_media_extension'    => 'word',
        'ext_social_media_dropdown'    => 'word',
        'ai_image_generator'           => 'image',
        'ai_image_pro'                 => 'image',
        'ai_product_shot'              => 'image',
        'photo_studio_extension'       => 'image',
        'ext_ai_photo_studio_dropdown' => 'image',
        'ext_fashion_studio_dropdown'  => 'image',
        'ai_chat_image'                => 'image',
        'ai_chat_pro_image_chat'       => 'image',
        'ai_chat_pro_smart_image'      => 'image',
        'ai_vision'                    => 'image',
        'creative_suite'               => 'image',
        'creative_suite_annotations'   => 'image',
        'ai_video'                     => 'video',
        'video_dubbing'                => 'video',
        'ai_captions'                  => 'video',
        'url_to_video'                 => 'video',
        'viral_clips'                  => 'video',
        'ai_youtube'                   => 'video',
        'ai_influencer'                => 'video',
        'influencer_avatar'            => 'video',
        'ugc_factory'                  => 'video',
        'ugc_creator'                  => 'video',
        'video_editor'                 => 'video',
        'ai_voiceover'                 => 'audio',
        'ai_voiceover_clone'           => 'audio',
        'ai_speech_to_text'            => 'audio',
        'ai_voice_isolator'            => 'audio',
        'ext_ai_music_pro'             => 'audio',
        'ai_chat_all'                  => 'chat',
        'ext_chat_bot'                 => 'chat',
        'ext_voice_chatbot'            => 'chat',
        'ai_pdf'                       => 'chat',
        'ai_web_chat_extension'        => 'chat',
        'ai_realtime_voice_chat'       => 'chat',
        'ai_chat_pro_skills'           => 'chat',
        'deep_research'                => 'chat',
        'ai_chat_pro_entity_highlight' => 'chat',
        'ai_chat_pro_highlight_to_ask' => 'chat',
        'ai_agent'                     => 'chat',
        'ext_voice_call'               => 'chat',
        'ext_phone_call_agent'         => 'chat',
        'ai_presentation'              => 'presentation',
    ];

    /**
     * Display order of tool categories inside the CoPilot system prompt.
     *
     * @var array<int, string>
     */
    public const TOOL_CATEGORY_ORDER = ['word', 'image', 'video', 'audio', 'chat', 'presentation', 'other'];

    public const DEFAULT_BASE_CREDIT = 100.0;

    public function __construct(protected AiCompletionService $completionService) {}

    /**
     * @return array{plans: array<int, array<string, mixed>>, tools: array<int, array<string, mixed>>, features: array<int, array<string, mixed>>, currency: string}
     */
    public function buildContextSummary(): array
    {
        $plans = Plan::query()
            ->orderBy('price')
            ->get(['id', 'name', 'type', 'frequency', 'price', 'currency', 'trial_days', 'credit_system_type', 'shared_credits_amount', 'is_featured', 'active'])
            ->map(fn (Plan $plan): array => [
                'name'               => $plan->name,
                'type'               => $plan->type,
                'frequency'          => $plan->frequency,
                'price'              => (float) $plan->price,
                'currency'           => $plan->currency,
                'trial_days'         => (int) $plan->trial_days,
                'credit_system_type' => $plan->credit_system_type,
                'is_featured'        => (bool) $plan->is_featured,
                'active'             => (bool) $plan->active,
            ])
            ->values()
            ->toArray();

        try {
            $currencySymbol = currency()->symbol ?? '$';
        } catch (Throwable) {
            $currencySymbol = '$';
        }

        return [
            'plans'    => $plans,
            'tools'    => collect(MenuService::planAiToolsMenu())->map(fn (array $item): array => Arr::only($item, ['key', 'label']))->values()->toArray(),
            'features' => collect(MenuService::planFeatureMenu())->map(fn (array $item): array => Arr::only($item, ['key', 'label']))->values()->toArray(),
            'currency' => $currencySymbol,
        ];
    }

    /**
     * AI-suggested plan concepts based on the existing pricing data. Falls back
     * to heuristic presets when the AI response is unavailable or malformed.
     *
     * @return array<int, array<string, mixed>>
     */
    public function suggestPresets(): array
    {
        $summary = $this->buildContextSummary();

        try {
            $response = $this->completionService->complete(
                $this->presetSystemPrompt(),
                json_encode(Arr::only($summary, ['plans']), JSON_PRETTY_PRINT) ?: '[]'
            );

            $presets = json_decode($this->extractJson($response), true);

            if (is_array($presets) && $presets !== []) {
                return collect($presets)
                    ->filter(fn ($preset) => is_array($preset) && ! empty($preset['name']))
                    ->take(3)
                    ->map(fn (array $preset): array => $this->sanitizePreset($preset))
                    ->values()
                    ->toArray();
            }
        } catch (Throwable $e) {
            report($e);
        }

        return $this->fallbackPresets($summary['plans']);
    }

    /**
     * Scale per-model credit limits by a tier multiplier. The base values come
     * from the best existing separated-credit plan; a flat default is used on
     * fresh installs where no plan has credits assigned yet. Models whose
     * category has an explicit limit receive that exact credit instead.
     *
     * @param  array<string, mixed>  $categoryLimits
     *
     * @return array<string, array<string, array{credit: float, isUnlimited: bool}>>
     */
    public function distributeCredits(float $multiplier, array $categoryLimits = []): array
    {
        $limits = $this->baseCreditMap();
        $categoryLimits = $this->sanitizeCategoryLimits($categoryLimits);

        foreach ($limits as $engine => $models) {
            foreach ($models as $model => $limit) {
                $category = self::creditCategoryFor($model);

                $limits[$engine][$model]['credit'] = $category !== null && array_key_exists($category, $categoryLimits)
                    ? $categoryLimits[$category]
                    : round(((float) ($limit['credit'] ?? 0)) * $multiplier, 2);
            }
        }

        return $limits;
    }

    /**
     * Resolve the user-facing credit category (word, image, video, audio,
     * presentation) for a plan-limit model slug.
     */
    public static function creditCategoryFor(string $modelSlug): ?string
    {
        $model = EntityEnum::tryFrom(str_replace('__', '.', $modelSlug));

        if ($model === null) {
            return null;
        }

        if (in_array($model, [EntityEnum::MUSIC_01, EntityEnum::ELEVENLABS_AI_MUSIC, EntityEnum::LYRIA_3_CLIP, EntityEnum::LYRIA_3_PRO], true)) {
            return 'audio';
        }

        try {
            $tokenType = $model->tokenType();
        } catch (Throwable) {
            return null;
        }

        return match ($tokenType) {
            AITokenType::WORD, AITokenType::VISION, AITokenType::PLAGIARISM => 'word',
            AITokenType::IMAGE => 'image',
            AITokenType::IMAGE_TO_VIDEO, AITokenType::TEXT_TO_VIDEO, AITokenType::VIDEO_TO_VIDEO, AITokenType::SECOND, AITokenType::MINUTE => 'video',
            AITokenType::TEXT_TO_SPEECH, AITokenType::SPEECH_TO_TEXT, AITokenType::CHARACTER => 'audio',
            AITokenType::PRESENTATION => 'presentation',
        };
    }

    /**
     * Resolve the tool category for a plan AI tool key.
     */
    public static function toolCategoryFor(string $toolKey): string
    {
        return self::TOOL_CATEGORIES[$toolKey] ?? 'other';
    }

    /**
     * @param  array<string, mixed>  $categoryLimits
     *
     * @return array<string, float>
     */
    private function sanitizeCategoryLimits(array $categoryLimits): array
    {
        $sanitized = [];

        foreach (self::CREDIT_CATEGORIES as $category) {
            $value = $categoryLimits[$category] ?? null;

            if (is_numeric($value) && (float) $value >= 0) {
                $sanitized[$category] = round((float) $value, 2);
            }
        }

        return $sanitized;
    }

    /**
     * @return array<string, array<string, array{credit: float, isUnlimited: bool}>>
     */
    private function baseCreditMap(): array
    {
        $reference = Plan::query()
            ->where('type', TypeEnum::SUBSCRIPTION->value)
            ->where('credit_system_type', 'separated')
            ->orderByDesc('active')
            ->orderByDesc('price')
            ->get(['id', 'type', 'price', 'active', 'credit_system_type', 'ai_models'])
            ->first(fn (Plan $plan): bool => $this->hasAnyCredit((array) $plan->ai_models));

        $limits = EngineEnum::getNestedPlanLimits();

        if (! $reference) {
            foreach ($limits as $engine => $models) {
                foreach (array_keys($models) as $model) {
                    $limits[$engine][$model]['credit'] = self::DEFAULT_BASE_CREDIT;
                }
            }

            return $limits;
        }

        foreach ((array) $reference->ai_models as $engine => $models) {
            foreach ((array) $models as $model => $limit) {
                if (isset($limits[$engine][$model])) {
                    $limits[$engine][$model] = [
                        'credit'      => (float) ($limit['credit'] ?? 0),
                        'isUnlimited' => (bool) ($limit['isUnlimited'] ?? false),
                    ];
                }
            }
        }

        return $limits;
    }

    /**
     * @param  array<string, mixed>  $aiModels
     */
    private function hasAnyCredit(array $aiModels): bool
    {
        foreach ($aiModels as $models) {
            foreach ((array) $models as $limit) {
                if (((float) ($limit['credit'] ?? 0)) > 0 || ! empty($limit['isUnlimited'])) {
                    return true;
                }
            }
        }

        return false;
    }

    public static function tierMultiplier(float|int|string|null $tier): float
    {
        if (is_numeric($tier)) {
            return max(0.0, (float) $tier);
        }

        return self::CREDIT_TIERS[$tier] ?? 1.0;
    }

    /**
     * @param  array<string, mixed>  $attributes
     */
    public function store(array $attributes): Plan
    {
        $attributes = Arr::only($attributes, self::ALLOWED_FIELDS);

        $type = Arr::pull($attributes, 'type', TypeEnum::SUBSCRIPTION->value);
        $creditTier = Arr::pull($attributes, 'credit_tier');
        $creditLimits = Arr::pull($attributes, 'credit_limits');
        $planAiTools = Arr::pull($attributes, 'plan_ai_tools');
        $planFeatures = Arr::pull($attributes, 'plan_features');

        $plan = $type === TypeEnum::TOKEN_PACK->value
            ? Plan::createFreshTokenPackPlan()
            : Plan::createFreshPlan();

        $plan->fill($attributes);

        if (is_array($planAiTools)) {
            $plan->plan_ai_tools = array_merge((array) $plan->plan_ai_tools, array_map(boolval(...), $planAiTools));
        }

        if (is_array($planFeatures)) {
            $plan->plan_features = array_merge((array) $plan->plan_features, array_map(boolval(...), $planFeatures));
        }

        if ($plan->isSharedCreditPlan()) {
            $plan->shared_credits_amount = (float) ($attributes['shared_credits_amount'] ?? 0);
        } else {
            $plan->shared_credits_amount = 0;
            $plan->ai_models = $this->distributeCredits(
                self::tierMultiplier($creditTier),
                is_array($creditLimits) ? $creditLimits : []
            );
        }

        if (is_null($plan->hidden_url)) {
            $plan->hidden_url = Helper::parseUrl(config('app.url'), 'plan/private', $plan->type, Str::random(20));
        }

        $plan->save();

        PaymentProcessController::saveGatewayProducts($plan);
        app(MenuService::class)->regenerate();
        Plan::forgetCache();

        return $plan;
    }

    /**
     * @param  array<int, array{role: string, content: string}>  $messages
     * @param  array<string, mixed>  $draft
     */
    public function streamChat(array $messages, array $draft, ?EngineEnum $engine = null): StreamedResponse
    {
        $systemPrompt = $this->buildSystemPrompt($draft);
        $engine ??= Helper::defaultEngine();

        return response()->stream(function () use ($systemPrompt, $messages, $engine): void {
            if ($engine === EngineEnum::OPEN_AI) {
                $this->streamViaOpenAi($systemPrompt, $messages);
            } else {
                $this->streamViaFallback($systemPrompt, $messages, $engine);
            }

            echo 'data: [DONE]' . PHP_EOL . PHP_EOL;
            $this->flushOutput();
        }, 200, [
            'Cache-Control'     => 'no-cache',
            'X-Accel-Buffering' => 'no',
            'Content-Type'      => 'text/event-stream',
        ]);
    }

    /**
     * @param  array<string, mixed>  $draft
     */
    public function buildSystemPrompt(array $draft): string
    {
        $summary = $this->buildContextSummary();

        $toolsByCategory = collect($summary['tools'])
            ->groupBy(fn (array $item): string => self::toolCategoryFor($item['key']));
        $toolGroups = collect(self::TOOL_CATEGORY_ORDER)
            ->filter(fn (string $category): bool => $toolsByCategory->has($category))
            ->map(fn (string $category): string => '  ' . $category . ': ' . $toolsByCategory[$category]
                ->map(fn (array $item): string => $item['key'] . ' ("' . $item['label'] . '")')
                ->implode(', '))
            ->implode("\n");
        $featureKeys = implode(', ', array_map(
            fn (array $item): string => $item['key'] . ' ("' . $item['label'] . '")',
            $summary['features']
        ));
        $frequencies = implode(', ', array_column(FrequencyEnum::cases(), 'value'));
        $tiers = implode(', ', array_keys(self::CREDIT_TIERS));
        $categories = implode(', ', self::CREDIT_CATEGORIES);
        $delimiter = self::UPDATES_DELIMITER;

        $existingPlans = json_encode($summary['plans']) ?: '[]';
        $currentDraft = json_encode(Arr::only($draft, self::ALLOWED_FIELDS)) ?: '{}';

        return <<<PROMPT
You are a pricing plan CoPilot inside a SaaS admin panel. You help the admin create a pricing plan through a step-by-step wizard while chatting.

Existing plans (JSON): {$existingPlans}

Current plan draft (JSON): {$currentDraft}

You may update the draft by appending a single line at the very END of your reply:
{$delimiter}{"field": value, ...}

Allowed fields and values:
- type: "subscription" or "prepaid" (prepaid = one-time token pack)
- name: string
- description: string (short marketing description)
- price: number (>= 0)
- frequency: one of [{$frequencies}] (subscription only)
- trial_days: integer >= 0 (subscription only)
- active: boolean
- is_featured: boolean
- features: string, comma-separated marketing bullet points
- credit_system_type: "separated" or "shared"
- shared_credits_amount: number (only when credit_system_type is "shared")
- credit_tier: one of [{$tiers}] or a numeric multiplier (only when credit_system_type is "separated"; scales default per-model credits)
- credit_limits: object of {category: number}; valid categories: [{$categories}]; only when credit_system_type is "separated". Sets the EXACT credit amount for every model in that category. Categories you omit keep the credit_tier scaling.
- plan_ai_tools: object of {toolKey: boolean}. Valid tool keys with labels, grouped by category:
{$toolGroups}
- plan_features: object of {featureKey: boolean}; valid keys with labels: {$featureKeys}
- plan_ai_tools_only: array of tool keys — enables ONLY the listed tools and disables every other tool
- plan_features_only: array of feature keys — enables ONLY the listed features and disables every other feature

Rules:
- You CANNOT perform actions later. The ONLY way the form changes is the {$delimiter} line at the very end of your CURRENT reply. If the user describes a plan or requests any change, you MUST append that line in this same reply. Never say "I'll set up" or "I will" — apply the change now and confirm in past tense ("Done — set...").
- Reply conversationally and briefly first, then the {$delimiter} line. Skip it only when the user asks a pure question with no change requested.
- Never show internal keys (like ai_video, ext_chat_bot) in your reply text — always use the human labels ("AI Video", "AI Chat Bots"). Keys belong only inside the {$delimiter} JSON.
- Format replies with simple Markdown: **bold** for values, "-" bullet lists for summaries. No headings, no tables, no code blocks.
- Toggle updates MERGE: keys you omit in plan_ai_tools/plan_features keep their current value. When the user wants an exclusive subset ("only image models", "just the video tools"), use plan_ai_tools_only / plan_features_only instead — include EXACTLY the keys listed under the requested category above. Never include keys from other categories, and never guess from label wording. Add keys from a different category (or "other") only when the user names them explicitly.
- When the user gives explicit credit amounts per model type (e.g. "10000 credits for word models, 100 for image models"), set credit_system_type to "separated" and use credit_limits — do NOT try to express it with credit_tier. credit_limits also MERGES: omitted categories keep their current value.
- The user NEVER sees the {$delimiter} line or the JSON — it is applied to the form silently. Never mention it, never say "here is the update", "as follows", "below" or similar. Instead state the changes in plain words, e.g. "Done — set the price to $29/month, named it Pro and marked it featured."
- Only include fields you are changing. Never invent keys outside the list above.
- When you create or reshape a plan, ALWAYS fill "features" (required field) with 3-6 short marketing bullet points matching the enabled tools, and "description" if empty.
- Use the existing plans to give pricing advice (gaps, positioning, naming).
- The JSON after {$delimiter} must be valid single-line JSON. No markdown fences around it.

Example — user says "build a video subscription plan for 4.99 per month, 1 day trial". Correct reply (plan_ai_tools_only carries every available key from the video category):
Done — created **Video Plan**: a **\$4.99/month** subscription with a **1-day trial** and only the video-related tools enabled.
{$delimiter}{"type":"subscription","name":"Video Plan","description":"Video creation tools for creators.","price":4.99,"frequency":"monthly","trial_days":1,"features":"AI Video Generator, Video Dubbing, AI Captions, URL to Video, Viral Clips","plan_ai_tools_only":["ai_video","video_dubbing","ai_captions","url_to_video","viral_clips","ai_youtube","ai_influencer","influencer_avatar"]}

Example — user says "only enable image models". Correct reply (plan_ai_tools_only carries every available key from the image category — never SEO, writing or chat tools):
Done — enabled **only the image tools**; every other tool is now disabled.
{$delimiter}{"features":"AI Image Generator, AI Image Pro, AI Photoshoot, Product Photography, Creative Suite","plan_ai_tools_only":["ai_image_generator","ai_image_pro","ai_product_shot","photo_studio_extension","ai_chat_image","ai_vision","creative_suite"]}

Example — user says "build a \$99 plan, 30 day trial, 10000 credits for word models and 100 for image models". Correct reply:
Done — created a **\$99/month** plan with a **30-day trial**, **10,000 credits** for word models and **100 credits** for image models.
{$delimiter}{"type":"subscription","name":"Premium","description":"High-volume AI content creation.","price":99,"frequency":"monthly","trial_days":30,"features":"AI Writer, AI Images, AI Chat, Templates","credit_system_type":"separated","credit_limits":{"word":10000,"image":100}}
PROMPT;
    }

    /**
     * @param  array<int, array{role: string, content: string}>  $messages
     */
    private function streamViaOpenAi(string $systemPrompt, array $messages): void
    {
        ApiHelper::setOpenAiKey();

        $stream = OpenAI::chat()->createStreamed([
            'model'    => Helper::defaultWordModel()->value,
            'messages' => array_merge(
                [['role' => 'system', 'content' => $systemPrompt]],
                $messages
            ),
        ]);

        foreach ($stream as $response) {
            $content = $response->choices[0]->delta->content ?? '';

            if ($content === '') {
                continue;
            }

            echo 'data: ' . json_encode(['content' => $content]) . PHP_EOL . PHP_EOL;
            $this->flushOutput();
        }
    }

    /**
     * @param  array<int, array{role: string, content: string}>  $messages
     */
    private function streamViaFallback(string $systemPrompt, array $messages, EngineEnum $engine): void
    {
        $conversation = collect($messages)
            ->map(fn (array $message): string => Str::ucfirst($message['role']) . ': ' . $message['content'])
            ->implode("\n\n");

        try {
            $reply = $this->completionService->complete($systemPrompt, $conversation, $engine);
        } catch (Throwable $e) {
            report($e);
            $reply = __('Sorry, I could not reach the AI service. Please try again.');
        }

        echo 'data: ' . json_encode(['content' => $reply]) . PHP_EOL . PHP_EOL;
        $this->flushOutput();
    }

    private function flushOutput(): void
    {
        if (app()->runningUnitTests()) {
            return;
        }

        if (ob_get_level() > 0) {
            @ob_flush();
        }

        flush();
    }

    private function presetSystemPrompt(): string
    {
        $frequencies = implode(', ', array_column(FrequencyEnum::cases(), 'value'));
        $tiers = implode(', ', array_keys(self::CREDIT_TIERS));

        return <<<PROMPT
You are a SaaS pricing strategist. The user message contains the existing pricing plans of an AI content platform as JSON.

Analyze the line-up (price gaps, missing tiers, positioning) and suggest up to 3 NEW plan concepts that complement it.

Respond with ONLY a JSON array (no markdown, no prose). Each item:
{"name": string, "reason": string (one short sentence why this plan fills a gap), "type": "subscription"|"prepaid", "price": number, "frequency": one of [{$frequencies}], "trial_days": integer, "credit_system_type": "separated"|"shared", "shared_credits_amount": number, "credit_tier": one of [{$tiers}], "description": string, "features": string (comma-separated marketing bullets)}
PROMPT;
    }

    /**
     * @param  array<string, mixed>  $preset
     *
     * @return array<string, mixed>
     */
    private function sanitizePreset(array $preset): array
    {
        $frequency = in_array($preset['frequency'] ?? null, array_column(FrequencyEnum::cases(), 'value'), true)
            ? $preset['frequency']
            : FrequencyEnum::MONTHLY->value;

        return [
            'name'                  => (string) $preset['name'],
            'reason'                => (string) ($preset['reason'] ?? ''),
            'type'                  => ($preset['type'] ?? null) === TypeEnum::TOKEN_PACK->value ? TypeEnum::TOKEN_PACK->value : TypeEnum::SUBSCRIPTION->value,
            'price'                 => max(0, (float) ($preset['price'] ?? 0)),
            'frequency'             => $frequency,
            'trial_days'            => max(0, (int) ($preset['trial_days'] ?? 0)),
            'credit_system_type'    => ($preset['credit_system_type'] ?? null) === 'shared' ? 'shared' : 'separated',
            'shared_credits_amount' => max(0, (float) ($preset['shared_credits_amount'] ?? 0)),
            'credit_tier'           => array_key_exists($preset['credit_tier'] ?? '', self::CREDIT_TIERS) ? $preset['credit_tier'] : 'standard',
            'description'           => (string) ($preset['description'] ?? ''),
            'features'              => (string) ($preset['features'] ?? ''),
        ];
    }

    /**
     * @param  array<int, array<string, mixed>>  $existingPlans
     *
     * @return array<int, array<string, mixed>>
     */
    private function fallbackPresets(array $existingPlans): array
    {
        $prices = collect($existingPlans)
            ->where('type', TypeEnum::SUBSCRIPTION->value)
            ->pluck('price');

        $maxPrice = (float) ($prices->max() ?? 0);

        $presets = [
            [
                'name'        => __('Starter'),
                'reason'      => __('An affordable entry point for new users.'),
                'price'       => $maxPrice > 0 ? round(max(5, $maxPrice * 0.3)) : 9,
                'trial_days'  => 7,
                'credit_tier' => 'starter',
                'description' => __('Everything you need to get started with AI content.'),
            ],
            [
                'name'        => __('Pro'),
                'reason'      => __('A mid-tier plan for growing teams.'),
                'price'       => $maxPrice > 0 ? round($maxPrice * 1.5) : 29,
                'trial_days'  => 7,
                'credit_tier' => 'pro',
                'description' => __('Advanced AI tools with generous credits for professionals.'),
            ],
            [
                'name'        => __('Enterprise'),
                'reason'      => __('A premium tier for heavy usage.'),
                'price'       => $maxPrice > 0 ? round($maxPrice * 3) : 99,
                'trial_days'  => 14,
                'credit_tier' => 'enterprise',
                'description' => __('Maximum credits and every feature unlocked.'),
            ],
        ];

        return array_map(fn (array $preset): array => $this->sanitizePreset(array_merge([
            'type'                  => TypeEnum::SUBSCRIPTION->value,
            'frequency'             => FrequencyEnum::MONTHLY->value,
            'credit_system_type'    => 'separated',
            'shared_credits_amount' => 0,
            'features'              => __('AI Writer, AI Images, AI Chat, Templates'),
        ], $preset)), $presets);
    }

    private function extractJson(string $response): string
    {
        $response = trim($response);
        $response = preg_replace('/^```(?:json)?|```$/m', '', $response) ?? $response;

        $start = strpos($response, '[');
        $end = strrpos($response, ']');

        if ($start !== false && $end !== false && $end > $start) {
            return substr($response, $start, $end - $start + 1);
        }

        return trim($response);
    }
}
