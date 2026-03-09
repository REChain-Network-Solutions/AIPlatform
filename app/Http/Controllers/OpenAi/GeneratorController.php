<?php

namespace App\Http\Controllers\OpenAi;

use App\Domains\Engine\Enums\EngineEnum;
use App\Domains\Entity\Enums\EntityEnum;
use App\Domains\Entity\Facades\Entity as EntityFacade;
use App\Domains\Entity\Models\Entity;
use App\Extensions\AiChatProImageChat\System\Services\AIChatImageService;
use App\Extensions\SocialMedia\System\Models\SocialMediaPostDailyMetric;
use App\Extensions\SocialMediaAgent\System\Models\SocialMediaAgent;
use App\Helpers\Classes\ApiHelper;
use App\Helpers\Classes\Helper;
use App\Helpers\Classes\MarketplaceHelper;
use App\Helpers\Classes\PlanHelper;
use App\Http\Controllers\Controller;
use App\Models\Company;
use App\Models\OpenAIGenerator;
use App\Models\OpenaiGeneratorFilter;
use App\Models\PdfData;
use App\Models\Product;
use App\Models\Setting;
use App\Models\SettingTwo;
use App\Models\Usage;
use App\Models\UserOpenai;
use App\Models\UserOpenaiChat;
use App\Models\UserOpenaiChatMessage;
use App\Services\Stream\StreamService;
use App\Services\VectorService;
use Exception;
use GuzzleHttp\Client;
use GuzzleHttp\Exception\GuzzleException;
use Illuminate\Database\Eloquent\Builder;
use Illuminate\Http\Request;
use Illuminate\Support\Carbon;
use Illuminate\Support\Collection;
use Illuminate\Support\Facades\Auth;
use Illuminate\Support\Facades\Http;
use Illuminate\Support\Facades\Log;
use Illuminate\Support\Str;
use Illuminate\View\View;
use JsonException;
use Random\RandomException;
use Symfony\Component\HttpFoundation\StreamedResponse;
use Throwable;

class GeneratorController extends Controller
{
    protected bool $realtimeCreditsFailed = false;

    protected $settings;

    protected $settings_two;

    public StreamService $streamService;

    public function __construct()
    {
        $this->settings = Setting::getCache();
        $this->settings_two = SettingTwo::getCache();
        $this->middleware(function (Request $request, $next) {
            ApiHelper::setOpenAiKey($this->settings);

            return $next($request);
        });
        $this->streamService = new StreamService($this->settings, $this->settings_two);
    }

    public function realtime(): View
    {
        return view('panel.admin.openai.realtime.index');
    }

    /**
     * @throws GuzzleException
     */
    public function buildStreamedOutput(Request $request): ?StreamedResponse
    {
        $template_type = $request->get('template_type', 'chatbot');

        // If the template type is chat, then we will build a chat streamed output or other ai template streamed output
        return match ($template_type) {
            'chatbot', 'vision', 'chatPro', 'chatPro-image', 'socialMediaAgent' => $this->buildChatStreamedOutput($request),
            default         => $this->buildOtherStreamedOutput($request),
        };
    }

    /**
     * @throws GuzzleException
     */
    public function buildChatStreamedOutput(Request $request): ?StreamedResponse
    {
        $chatParams = $this->extractChatParameters($request);
        if (empty($chatParams)) {
            return response()->stream(function () {
                echo "event: message\n";
                echo 'data: 0' . "\n\n";

                echo "event: data\n";
                echo 'data: ' . __('Chat not found. Please refresh and try again.');
                echo "\n\n";
                flush();
                echo "event: stop\n";
                echo 'data: [DONE]';
                echo "\n\n";
                flush();
            }, 200, [
                'Cache-Control'     => 'no-cache',
                'X-Accel-Buffering' => 'no',
                'Connection'        => 'keep-alive',
                'Content-Type'      => 'text/event-stream',
            ]);
        }

        $user = Auth::user();

        if (
            ! empty($request->shared_message_uuid) &&
            MarketplaceHelper::isRegistered('multi-model') &&
            ((bool) setting('ai_chat_pro_multi_model_feature', '1'))
        ) {
            $chatParams['shared_message_uuid'] = $request->get('shared_message_uuid');
        }

        $chat_bot = $this->determineChatBot($chatParams['chatbot_front_model']);
        $default_ai_engine = $this->determineAiEngine($chat_bot, $chatParams['chatbot_front_model']);

        $message = $this->createChatMessage($user, $chatParams);
        $history = $this->buildChatHistory($chatParams, $message->user_openai_chat_id);
        $isFileSearch = setting('openai_file_search', 0) && ! empty($chatParams['chat']->openai_vector_id);

        if ($this->realtimeCreditsFailed) {
            return response()->stream(function () use ($message) {
                echo "event: message\n";
                echo 'data: ' . $message->id . "\n\n";

                echo "event: data\n";
                echo 'data: ' . __('Insufficient credits for realtime search. Please check your credits and try again.');
                echo "\n\n";
                flush();
                echo "event: stop\n";
                echo 'data: [DONE]';
                echo "\n\n";
                flush();
            }, 200, [
                'Cache-Control'     => 'no-cache',
                'X-Accel-Buffering' => 'no',
                'Connection'        => 'keep-alive',
                'Content-Type'      => 'text/event-stream',
            ]);
        }

        return $this->streamService->ChatStream(
            $chat_bot,
            $history,
            $message,
            $chatParams,
            $default_ai_engine,
            fileChat: $isFileSearch
        );
    }

    private function extractChatParameters(Request $request): array
    {
        $chat_id = $request->get('chat_id');
        $chat = UserOpenaiChat::with('category')->find($chat_id);

        if (! $chat) {
            return [];
        }

        $data = [
            'prompt'              => $request->get('prompt'),
            'realtime'            => $request->get('realtime', false),
            'chat_brand_voice'    => $request->get('chat_brand_voice'),
            'brand_voice_prod'    => $request->get('brand_voice_prod'),
            'chat_id'             => $chat_id,
            'chat_type'           => $request->get('template_type'),
            'agent_id'            => $request->integer('chat_open_ai_agent_id') ?: null,
            'images'              => $request->get('images', null),
            'pdfname'             => $request->get('pdfname', null),
            'pdfpath'             => $request->get('pdfpath', null),
            'assistant'           => $request->get('assistant', null),
            'chatbot_front_model' => $request->get('chatbot_front_model', null),
            'chat'                => $chat,
            'openRouter'          => $this->determineOpenRouter($request->get('chatbot_front_model', null)),
            'contain_images'      => false, // Will be determined later
        ];

        if ($request->get('template_type') === 'chatPro-image' && MarketplaceHelper::isRegistered('ai-chat-pro-image-chat')) {
            $imageParams = AIChatImageService::extractImageChatParameters($request);
            $data += $imageParams;

            // Build contextual prompt for image editing when edit_tab is present
            if (! empty($imageParams['edit_tab'])) {
                $data['prompt'] = AIChatImageService::buildEditPrompt(
                    (string) ($data['prompt'] ?? ''),
                    $imageParams['edit_tab'],
                    $imageParams['edit_mode'] ?? 'text',
                    $imageParams['edit_has_highlights'] ?? false,
                );
            }

            // For reimagine: if main prompt is empty but reimagine_prompt is set, use reimagine_prompt
            if (empty($data['prompt']) && ! empty($imageParams['reimagine_prompt'])) {
                $data['prompt'] = $imageParams['reimagine_prompt'];
            }
        }

        return $data;
    }

    private function determineChatBot(?string $chatbot_front_model): string
    {
        $default_ai_engine = setting('default_ai_engine', EngineEnum::OPEN_AI->value);

        if ($default_ai_engine === EngineEnum::OPEN_AI->value) {
            $chat_bot = $this->settings?->openai_default_model ?: EntityEnum::GPT_4_O->value;
        } elseif ($default_ai_engine === EngineEnum::GEMINI->value) {
            $chat_bot = setting('gemini_default_model', 'gemini-1.5-pro-latest');
        } elseif ($default_ai_engine === EngineEnum::ANTHROPIC->value) {
            $chat_bot = setting('anthropic_default_model', EntityEnum::CLAUDE_3_OPUS->value);
        } elseif ($default_ai_engine === EngineEnum::DEEP_SEEK->value) {
            $chat_bot = setting('deepseek_default_model', EntityEnum::DEEPSEEK_CHAT->value);
        } elseif ($default_ai_engine === EngineEnum::X_AI->value) {
            $chat_bot = setting('xai_default_model', EntityEnum::GROK_2_1212->value);
        } else {
            $chat_bot = $this->settings?->openai_default_model ?: EntityEnum::GPT_4_O->value;
        }

        $chat_bot_model = PlanHelper::userPlanAiModel();
        if ($chat_bot_model && empty($chatbot_front_model)) {
            $default_ai_engine_new = Entity::query()
                ->where('key', $chat_bot)
                ->first()
                ?->getAttribute('default_ai_engine');
            if ($default_ai_engine_new) {
                $chat_bot = $chat_bot_model;
            }
        }

        if (! empty($chatbot_front_model)) {
            $engine = Entity::query()
                ->where('key', $chatbot_front_model)
                ->first();

            if ($engine) {
                $chat_bot = $chatbot_front_model;
            }
        }

        return $chat_bot;
    }

    private function determineAiEngine(string $chat_bot, ?string $chatbot_front_model): string
    {
        $default_ai_engine = setting('default_ai_engine', EngineEnum::OPEN_AI->value);

        if (! empty($chatbot_front_model)) {
            $engine = Entity::query()
                ->where('key', $chatbot_front_model)
                ->first();

            if ($engine) {
                $engineValue = $engine->engine;
                if ($engineValue instanceof EngineEnum) {
                    return $engineValue->value;
                }

                return $engineValue;
            }
        }

        return $default_ai_engine;
    }

    private function determineOpenRouter(?string $chatbot_front_model): ?string
    {
        if (! empty($chatbot_front_model) &&
            (int) setting('open_router_status') === 1 &&
            EntityEnum::fromSlug($chatbot_front_model)->engine() === EngineEnum::OPEN_ROUTER) {
            return $chatbot_front_model;
        }

        return null;
    }

    private function createChatMessage($user, array $chatParams): UserOpenaiChatMessage
    {
        $attributes = [
            'user_id'             => $user?->id,
            'user_openai_chat_id' => $chatParams['chat_id'],
            'input'               => $chatParams['prompt'],
            'response'            => null,
            'realtime'            => $chatParams['realtime'] ?? 0,
            'output'              => __("(If you encounter this message, please attempt to send your message again. If the error persists beyond multiple attempts, please don't hesitate to contact us for assistance!)"),
            'hash'                => Str::random(256),
            'credits'             => 0,
            'words'               => 0,
            'images'              => is_array($chatParams['images']) ? implode(',', $chatParams['images']) : $chatParams['images'],
            'pdfName'             => $chatParams['pdfname'],
            'pdfPath'             => $chatParams['pdfpath'],
        ];

        if (! empty($chatParams['shared_message_uuid'])) {
            $attributes['model_slug'] = $chatParams['chatbot_front_model'];
            $attributes['shared_uuid'] = $chatParams['shared_message_uuid'];
        }

        return UserOpenaiChatMessage::create($attributes);
    }

    private function buildChatHistory(array &$chatParams, int $chat_id): array
    {
        $chat = $chatParams['chat'];
        $category = $chat->category;
        $systemRole = EntityEnum::fromSlug($this->determineChatBot($chatParams['chatbot_front_model']))->isBetaEntity() ? 'system' : 'user';

        $history = $this->initializeHistory($category, $systemRole);

        if (($chatParams['chat_type'] ?? null) === 'socialMediaAgent') {
            $history[] = [
                'role'    => $systemRole,
                'content' => 'You are the AI Social Media Agent. Help with strategy, analysis, scheduling, and general questions conversationally. Only call the `generate_social_post` function when the user clearly asks you to draft, write, or generate a new social media post. For all other requests, answer normally and do not use any tool.',
            ];

            $history = $this->appendSocialMediaAgentContext($history, $chatParams, $systemRole);
        }
        $history = $this->addFileOrInstructionsToHistory($history, $category, $chat_id, $chatParams['prompt'], $systemRole);
        $history = $this->addPreviousMessagesToHistory($history, $chat, $chatParams['assistant']);
        $history = $this->checkBrandVoice($chatParams['chat_brand_voice'], $chatParams['brand_voice_prod'], $history);

        return $this->addCurrentPromptToHistory($history, $chatParams, $systemRole);
    }

    private function appendSocialMediaAgentContext(array $history, array $chatParams, string $systemRole): array
    {
        $agent = $this->resolveChatAgent($chatParams['agent_id'] ?? null);

        if (! $agent) {
            return $history;
        }

        $platforms = $agent->platforms();

        if ($profile = $this->formatSocialMediaAgentProfile($agent, $platforms)) {
            $history[] = [
                'role'    => $systemRole,
                'content' => $profile,
            ];
        }

        if ($metrics = $this->buildSocialMediaAgentMetricsSummary($agent, $platforms)) {
            $history[] = [
                'role'    => $systemRole,
                'content' => $metrics,
            ];
        }

        return $this->addCurrentPromptToHistory($history, $chatParams, $systemRole);
    }

    private function resolveChatAgent(?int $agentId): ?SocialMediaAgent
    {
        $userId = Auth::id();

        if (! $userId) {
            return null;
        }

        $query = SocialMediaAgent::query()
            ->where('user_id', $userId)
            ->orderBy('id');

        if ($agentId) {
            $agent = (clone $query)->whereKey($agentId)->first();
            if ($agent) {
                return $agent;
            }
        }

        return $query->first();
    }

    private function formatSocialMediaAgentProfile(SocialMediaAgent $agent, Collection $platforms): string
    {
        $platformNames = $platforms
            ->pluck('platform')
            ->filter()
            ->map(fn ($name) => Str::headline((string) $name))
            ->values()
            ->all();

        $brandSummary = $agent->branding_description
            ?: $agent->site_description
            ?: null;

        $scheduleParts = [];
        if ($agent->daily_post_count) {
            $scheduleParts[] = $agent->daily_post_count . ' posts/day';
        }
        if (! empty($agent->schedule_days)) {
            $scheduleParts[] = 'on ' . $this->formatList($agent->schedule_days, 'set days', true);
        }
        if (! empty($agent->schedule_times)) {
            $scheduleParts[] = 'around ' . $this->formatList($agent->schedule_times, 'set times');
        }

        $lines = array_filter([
            'SOCIAL MEDIA AGENT PROFILE',
            'Agent: ' . $agent->name,
            $brandSummary ? ('Brand summary: ' . Str::limit(trim((string) $brandSummary), 400)) : null,
            'Platforms: ' . (! empty($platformNames) ? implode(', ', $platformNames) : 'No connected platforms yet'),
            'Target audiences: ' . $this->formatList($agent->target_audience ?: null),
            'Content pillars: ' . $this->formatList($agent->categories ?: null, 'Not specified', true),
            'Post formats: ' . $this->formatList($agent->post_types ?: null, 'Not specified', true),
            'CTA templates: ' . $this->formatList($agent->cta_templates ?: null, 'Not provided'),
            'Primary goals: ' . $this->formatList($agent->goals ?: null, 'Not specified'),
            'Tone & voice: ' . ($agent->tone ? Str::headline($agent->tone) : 'Not specified'),
            'Language: ' . ($agent->language ?: 'Not specified'),
            'Preferred length: ' . ($agent->approximate_words ? $agent->approximate_words . ' words' : 'Flexible'),
            'Hashtag allowance: ' . ($agent->hashtag_count ? $agent->hashtag_count . ' per post' : 'Flexible'),
            'Asset expectations: ' . ($agent->has_image ? 'Posts should include visuals' : 'Copy-only posts are acceptable'),
            'Posting cadence: ' . (! empty($scheduleParts) ? implode(' ', $scheduleParts) : 'No schedule defined'),
            $agent->average_impressions ? ('Avg impressions/post: ' . $this->formatNumber($agent->average_impressions)) : null,
            $agent->average_engagement ? ('Avg engagement actions/post: ' . $this->formatNumber($agent->average_engagement)) : null,
        ]);

        return implode("\n", $lines);
    }

    private function buildSocialMediaAgentMetricsSummary(SocialMediaAgent $agent, Collection $platforms): ?string
    {
        $endDate = Carbon::now()->endOfDay();
        $startDate = $endDate->copy()->subDays(29)->startOfDay();

        $records = SocialMediaPostDailyMetric::query()
            ->where('agent_id', $agent->id)
            ->whereBetween('date', [$startDate->toDateString(), $endDate->toDateString()])
            ->selectRaw('platform, social_media_platform_id, COUNT(DISTINCT social_media_post_id) as post_count, SUM(view_count) as views, SUM(like_count) as likes, SUM(comment_count) as comments, SUM(share_count) as shares')
            ->groupBy('platform', 'social_media_platform_id')
            ->orderByDesc('views')
            ->get();

        $windowLabel = sprintf('%s – %s', $startDate->toDateString(), $endDate->toDateString());

        if ($records->isEmpty()) {
            return "30-Day Performance Snapshot ({$windowLabel})\nNo performance metrics were captured for this agent over the last 30 days. If a user requests analytics, be transparent that data is not yet available and focus on actionable next steps.";
        }

        $totals = [
            'posts'    => (int) $records->sum('post_count'),
            'views'    => (int) $records->sum('views'),
            'likes'    => (int) $records->sum('likes'),
            'comments' => (int) $records->sum('comments'),
            'shares'   => (int) $records->sum('shares'),
        ];

        $engagementActions = $totals['likes'] + $totals['comments'] + $totals['shares'];
        $overallRate = $totals['views'] > 0
            ? round(($engagementActions / $totals['views']) * 100, 2)
            : 0;

        $platformLookup = $platforms->keyBy('id');

        $platformLines = $records->map(function ($record) use ($platformLookup) {
            $platformName = $record->platform
                ?? optional($platformLookup->get($record->social_media_platform_id))->platform
                ?? ('Platform ' . ($record->social_media_platform_id ?? '?'));

            $engagementCount = (int) $record->likes + (int) $record->comments + (int) $record->shares;
            $rate = ($record->views ?? 0) > 0
                ? round(($engagementCount / $record->views) * 100, 2)
                : 0;

            return sprintf(
                '- %s: %d posts, %s views, %s engagements (%s likes / %s comments / %s shares), %s%% engagement rate',
                Str::headline($platformName),
                (int) $record->post_count,
                $this->formatNumber((float) $record->views),
                $this->formatNumber($engagementCount),
                $this->formatNumber((float) $record->likes),
                $this->formatNumber((float) $record->comments),
                $this->formatNumber((float) $record->shares),
                $this->formatNumber($rate, 2)
            );
        })->all();

        $summaryLines = [
            "30-Day Performance Snapshot ({$windowLabel})",
            sprintf(
                'Totals: %d posts, %s views, %s engagement actions (likes + comments + shares), %s%% overall engagement rate.',
                $totals['posts'],
                $this->formatNumber($totals['views']),
                $this->formatNumber($engagementActions),
                $this->formatNumber($overallRate, 2)
            ),
            'Platform breakdown:',
            ...$platformLines,
            'Base any performance reports on these stats. When estimating future outcomes, clearly describe the assumptions you are making.',
        ];

        return implode("\n", $summaryLines);
    }

    private function formatList($value, string $fallback = 'Not provided', bool $humanize = false): string
    {
        if (empty($value)) {
            return $fallback;
        }

        $items = is_array($value) ? $value : [$value];

        $formatted = array_filter(array_map(function ($item) use ($humanize) {
            if (is_array($item)) {
                $item = $item['label']
                    ?? $item['value']
                    ?? $item['name']
                    ?? implode(' ', array_filter($item));
            }

            $item = trim((string) $item);

            if ($item === '') {
                return null;
            }

            return $humanize ? Str::headline($item) : $item;
        }, $items));

        return empty($formatted) ? $fallback : implode(', ', $formatted);
    }

    private function formatNumber(float|int|null $value, int $decimals = 0): string
    {
        if ($value === null) {
            return $decimals > 0
                ? number_format(0, $decimals, '.', ',')
                : '0';
        }

        return number_format((float) $value, $decimals, '.', ',');
    }

    private function initializeHistory($category, string $systemRole): array
    {
        if ($category->chat_completions) {
            $chat_completions = json_decode($category->chat_completions, true);
            $history = [];
            foreach ($chat_completions as $item) {
                $history[] = [
                    'role'    => $item['role'],
                    'content' => $item['content'] ?? '',
                ];
            }

            return $history;
        }

        return [['role' => $systemRole, 'content' => __('You are a helpful assistant.')]];
    }

    private function addFileOrInstructionsToHistory(array $history, $category, int $chat_id, string $prompt, string $systemRole): array
    {
        $isFileSearch = setting('openai_file_search', 0) && $category->openai_vector_id !== null;

        if (! $isFileSearch && ($category->chatbot_id || PdfData::where('chat_id', $chat_id)->exists())) {
            try {
                $extra_prompt = (new VectorService)->getMostSimilarText($prompt, $chat_id, 2, $category->chatbot_id);
                if ($extra_prompt) {
                    if ($category->slug === 'ai_webchat') {
                        $history[] = [
                            'role'    => $systemRole,
                            'content' => "You are a Web Page Analyzer assistant. When referring to content from a specific website or link, please include a brief summary or context of the content. If users inquire about the content or purpose of the website/link, provide assistance professionally without explicitly mentioning the content. Website/link content: \n$extra_prompt",
                        ];
                    } else {
                        $history[] = [
                            'role'    => $systemRole,
                            'content' => "You are a File Analyzer assistant. When referring to content from a specific file, please include a brief summary or context of the content. If users inquire about the content or purpose of the file, provide assistance professionally without explicitly mentioning the content. File content: \n$extra_prompt",
                        ];
                    }
                }
            } catch (Throwable $th) {
                // Handle error silently
                Log::error('Error fetching similar text for chat history: ' . $th->getMessage());
            }
        } else {
            // Get instructions with user override support
            $instructions = $this->getInstructions($category);

            if ($instructions) {
                $history[] = ['role' => $systemRole, 'content' => $instructions];
            }
        }

        return $history;
    }

    /**
     * Get instructions for current user/guest
     * Priority: User-specific > Admin default
     */
    private function getInstructions($category): ?string
    {
        $categoryId = $category->id;

        if (MarketplaceHelper::isRegistered('ai-chat-pro-memory')) {
            if (Auth::check()) {
                // Check for user-specific instructions first
                $userInstructions = \App\Extensions\AIChatProMemory\System\Models\UserChatInstruction::getForUser(
                    Auth::id(),
                    $categoryId
                );

                if ($userInstructions) {
                    return $category->instructions ? $category->instructions . "\n\n" . $userInstructions : $userInstructions;
                }
            } else {
                // Check for guest instructions by IP
                $ipAddress = request()?->header('CF-Connecting-IP') ?? request()?->ip();
                $guestInstructions = \App\Extensions\AIChatProMemory\System\Models\UserChatInstruction::getForGuest(
                    $ipAddress,
                    $categoryId
                );

                if ($guestInstructions) {
                    return $category->instructions ? $category->instructions . "\n\n" . $guestInstructions : $guestInstructions;
                }
            }
        }

        // Fall back to admin's default instructions
        return $category->instructions;
    }

    private function addPreviousMessagesToHistory(array $history, $chat, $assistant): array
    {
        $lastThreeMessageQuery = $chat->messages()
            ->whereNotNull('input')
            ->orderBy('created_at', 'desc')
            ->take(4)
            ->get()
            ->reverse();

        $contain_images = $this->checkIfHistoryContainsImages($lastThreeMessageQuery);
        $count = count($lastThreeMessageQuery);

        if ($count > 1) {
            foreach ($lastThreeMessageQuery as $threeMessage) {
                if ($contain_images) {
                    $history[] = [
                        'role'    => 'user',
                        'content' => array_merge(
                            [
                                [
                                    'type' => 'input_text',
                                    'text' => $threeMessage->input,
                                ],
                            ],
                            $this->processMessageImages($threeMessage->images, $assistant)
                        ),
                    ];
                } else {
                    $history[] = ['role' => 'user', 'content' => $threeMessage->input ?? ''];
                }

                if ($threeMessage->output !== null && $threeMessage->output !== '') {
                    $history[] = ['role' => 'assistant', 'content' => $threeMessage->output];
                }
            }
        }

        return $history;
    }

    /**
     * @throws GuzzleException
     * @throws JsonException
     */
    private function addCurrentPromptToHistory(array $history, array &$chatParams, string $systemRole): array
    {
        if (empty($chatParams['images']) && $chatParams['chat']->category->slug !== 'ai_vision') {
            if ($chatParams['realtime']) {
                $history[] = ['role' => 'user', 'content' => $this->getRealtimeEngine($chatParams) ?? ''];
            } else {
                $history[] = ['role' => 'user', 'content' => $chatParams['prompt'] ?? ''];
            }
        } else {
            $history = $this->addVisionPromptToHistory($history, $chatParams, $systemRole);
            $chatParams['contain_images'] = true;
        }

        return $history;
    }

    private function addVisionPromptToHistory(array $history, array $chatParams, string $systemRole): array
    {
        if ($chatParams['chat_type'] === 'vision') {
            $history[] = [
                'role'    => $systemRole,
                'content' => 'You will now play a character and respond as that character (You will never break character). Your name is Vision AI. Must not introduce by yourself as well as greetings. Help also with asked questions based on previous responses and images if exists.',
            ];
        }

        $images = explode(',', $chatParams['images']);
        $history[] = [
            'role'    => 'user',
            'content' => array_merge(
                [
                    [
                        'type' => 'input_text',
                        'text' => $chatParams['prompt'],
                    ],
                ],
                $this->processImages($images, $chatParams['assistant'])
            ),
        ];

        return $history;
    }

    /**
     * @throws GuzzleException
     * @throws JsonException
     */
    private function getRealtimeEngine(array $chatParams): string
    {
        if (! is_null($this->settings_two->serper_api_key) && setting('default_realtime', 'serper') === 'serper') {
            return $this->getRealtimePrompt($chatParams['prompt']);
        }

        if (setting('default_realtime') == 'perplexity' && ! is_null(setting('perplexity_key'))) {
            return $this->realtimePromptPerplexity($chatParams['prompt']);
        }

        return $chatParams['prompt'];
    }

    private function processMessageImages($images, $assistant): array
    {
        return collect($images)->map(static function ($item) use ($assistant) {
            $images = explode(',', $item);
            $imageResults = [];
            if (! empty($images)) {
                foreach ($images as $image) {
                    if (Str::startsWith($image, 'http')) {
                        $imageData = file_get_contents($image);
                    } else {
                        $img = ltrim($image, '/');
                        if (! file_exists($img)) {
                            continue;
                        }
                        $imageData = file_get_contents($img);
                    }
                    $base64Image = base64_encode($imageData);

                    if ($assistant !== null) {
                        $imageResults[] = [
                            'type'      => 'input_image',
                            'image_url' => $image,
                        ];
                    } else {
                        $imageResults[] = [
                            'type'      => 'input_image',
                            'image_url' => 'data:image/png;base64,' . $base64Image,
                        ];
                    }
                }
            }

            return $imageResults;
        })->reject(fn ($value) => empty($value))->flatten(1)->toArray();
    }

    private function processImages(array $images, $assistant): array
    {
        return collect($images)->map(static function ($item) use ($assistant) {
            if (! empty($item)) {
                if (Str::startsWith($item, 'http')) {
                    $imageData = file_get_contents($item);
                } else {
                    $imageData = file_get_contents(substr($item, 1));
                }
                $base64Image = base64_encode($imageData);

                if ($assistant !== null) {
                    return [
                        'type'      => 'input_image',
                        'image_url' => $item,
                    ];
                }

                return [
                    'type'      => 'input_image',
                    'image_url' => 'data:image/png;base64,' . $base64Image,
                ];
            }

            return null;
        })->reject(null)->toArray();
    }

    private function checkBrandVoice($chat_brand_voice, $brand_voice_prod, $history)
    {
        if (! empty($chat_brand_voice) && ! empty($brand_voice_prod)) {
            // check if there is a company input included in the request
            $company = Company::find($chat_brand_voice);
            $product = Product::find($brand_voice_prod);
            if ($company && $product) {
                $type = $product->type === 0 ? 'Service' : 'Product';
                $prompt = "Focus on my company and {$type}'s information: \n";
                // Company information
                if ($company->name) {
                    $prompt .= "The company's name is {$company->name}. ";
                }
                // explode industry
                $industry = explode(',', $company->industry);
                $count = count($industry);
                if ($count > 0) {
                    $prompt .= 'The company is in the ';
                    foreach ($industry as $index => $ind) {
                        $prompt .= $ind;
                        if ($index < $count - 1) {
                            $prompt .= ' and ';
                        }
                    }
                }
                if ($company->website) {
                    $prompt .= ". The company's website is {$company->website}. ";
                }
                if ($company->target_audience) {
                    $prompt .= "The company's target audience is: {$company->target_audience}. ";
                }
                if ($company->specific_instructions) {
                    $prompt .= "The company's specific instructions are: {$company->specific_instructions}. ";
                }
                if ($company->tagline) {
                    $prompt .= "The company's tagline is {$company->tagline}. ";
                }
                if ($company->description) {
                    $prompt .= "The company's description is {$company->description}. ";
                }
                if ($product) {
                    if ($product->key_features) {
                        $prompt .= "The {$product->type}'s key features are {$product->key_features}. ";
                    }

                    if ($product->name) {
                        $prompt .= "The {$product->type}'s name is {$product->name}. \n";
                    }
                }
                $prompt .= "\n";
                $history[] = ['role' => 'user', 'content' => $prompt];

                return $history;
            }
        }

        return $history;
    }

    // ai writer template and etc.
    public function buildOtherStreamedOutput(Request $request): StreamedResponse
    {
        $default_ai_engine = setting('default_ai_engine', EngineEnum::OPEN_AI->value);

        if ($default_ai_engine === EngineEnum::OPEN_AI->value) {
            $chatBot = ! $this->settings?->openai_default_model ? EntityEnum::GPT_3_5_TURBO->value : $this->settings?->openai_default_model;
        } elseif ($default_ai_engine === EngineEnum::GEMINI->value) {
            $chatBot = setting('gemini_default_model', EntityEnum::GEMINI_1_5_FLASH->value);
        } elseif ($default_ai_engine === EngineEnum::ANTHROPIC->value) {
            $chatBot = setting('anthropic_default_model', EntityEnum::CLAUDE_3_OPUS->value);
        } elseif ($default_ai_engine === EngineEnum::DEEP_SEEK->value) {
            $chatBot = setting('deepseek_default_model', EntityEnum::DEEPSEEK_CHAT->value);
        } elseif ($default_ai_engine === EngineEnum::X_AI->value) {
            $chatBot = setting('xai_default_model', EntityEnum::GROK_2_1212->value);
        } else {
            $chatBot = ! $this->settings?->openai_default_model ? EntityEnum::GPT_3_5_TURBO->value : $this->settings?->openai_default_model;
        }

        if ($chat_bot_model = PlanHelper::userPlanAiModel()) {

            $default_ai_engine_new = Entity::query()
                ->where('key', $chatBot)
                ->first()
                ?->getAttribute('default_ai_engine');

            if ($default_ai_engine_new) {
                $chatBot = $chat_bot_model;
                $default_ai_engine = $default_ai_engine_new;
            }
        }

        $chatbot_front_model = $request->get('chatbot_front_model', null);

        if (! empty($chatbot_front_model)) {
            $oldChatbot = $chatBot;

            $engine = Entity::query()
                ->where('key', $chatbot_front_model)
                ->first();

            if ($engine) {
                $default_ai_engine = $engine->engine;
                $chatBot = $chatbot_front_model;
                if ($default_ai_engine instanceof EngineEnum) {
                    $default_ai_engine = $default_ai_engine->value;
                } else {
                    $chatBot = $oldChatbot;
                }
            }
        }

        return $this->streamService->OtherStream($request, $chatBot, $default_ai_engine);
    }

    // reduce tokens when the stream is interrupted
    public function reduceTokensWhenIntterruptStream(Request $request, $type): void
    {
        $this->streamService->reduceTokensWhenIntterruptStream($request, $type);
    }

    private function getRealtimePrompt($realtimePrompt): StreamedResponse|string
    {
        $driver = EntityFacade::driver(EntityEnum::SERPER);

        if (! $driver->hasCreditBalance()) {
            $this->realtimeCreditsFailed = true;

            return $realtimePrompt;
        }

        $client = new Client;
        $headers = [
            'X-API-KEY'    => $this->settings_two->serper_api_key,
            'Content-Type' => 'application/json',
        ];
        $body = [
            'q' => $realtimePrompt,
        ];
        $response = $client->post('https://google.serper.dev/search', [
            'headers' => $headers,
            'json'    => $body,
        ]);
        $toGPT = $response->getBody()->getContents();

        try {
            $toGPT = json_decode($toGPT, false, 512, JSON_THROW_ON_ERROR);
        } catch (Throwable $th) {
        }

        $driver->input($realtimePrompt)->calculateCredit()->decreaseCredit();
        Usage::getSingle()->updateWordCounts($driver->calculate());

        $searchRes = json_encode($toGPT, JSON_THROW_ON_ERROR);

        if (empty($searchRes)) {
            return $realtimePrompt;
        }

        return 'Prompt: ' . $realtimePrompt .
            '\n\nWeb search json results: '
            . $searchRes .
            '\n\nInstructions: Based on the Prompt generate a proper response with help of Web search results(if the Web search results in the same context). Only if the prompt require links: (make curated list of links and descriptions using only the <a target="_blank">, write links with using <a target="_blank"> with mrgin Top of <a> tag is 5px and start order as number and write link first and then write description). Must not write links if its not necessary. Must not mention anything about the prompt text.';
    }

    public function realtimePromptPerplexity($realtimePrompt): string
    {
        $url = 'https://api.perplexity.ai/chat/completions';
        $token = setting('perplexity_key');

        $payload = [
            'model'    => 'sonar',
            'messages' => [
                [
                    'role'    => 'user',
                    'content' => $realtimePrompt,
                ],
            ],
        ];

        try {
            $response = Http::withToken($token)
                ->withHeaders([
                    'Content-Type' => 'application/json',
                ])
                ->post($url, $payload);
            if ($response->successful()) {
                $data = $response->json();
                $response = $data['choices'][0]['message']['content'];

                return 'Prompt: ' . $realtimePrompt .
                    '\n\nWeb search results: '
                    . $response .
                    '\n\nInstructions: Based on the Prompt generate a proper response with help of Web search results(if the Web search results in the same context). Only if the prompt require links: (make curated list of links and descriptions using only the <a target="_blank">, write links with using <a target="_blank"> with mrgin Top of <a> tag is 5px and start order as number and write link first and then write description). Must not write links if its not necessary. Must not mention anything about the prompt text.';

            }

            return $realtimePrompt;
        } catch (Exception $e) {
            return $realtimePrompt;
        }
    }

    private function checkIfHistoryContainsImages($lastThreeMessages): bool
    {
        if (! is_iterable($lastThreeMessages)) {
            return false;
        }

        return collect($lastThreeMessages)->contains(static function ($message) {
            return ! empty($message->images);
        });
    }

    /**
     * @throws RandomException
     */
    public function index($workbook_slug = null)
    {
        abort_if(Helper::setting('feature_ai_advanced_editor') === 0, 404);
        $apiUrl = base64_encode('https://api.openai.com/v1/chat/completions');
        $settings_two = SettingTwo::getCache();
        if ($settings_two->openai_default_stream_server === 'backend') {
            $apikeyPart1 = base64_encode(random_int(1, 100));
            $apikeyPart2 = base64_encode(random_int(1, 100));
            $apikeyPart3 = base64_encode(random_int(1, 100));
        } else {
            $apiKey = ApiHelper::setOpenAiKey();
            $len = strlen($apiKey);
            $len = max($len, 6);
            $parts[] = substr($apiKey, 0, $l[] = random_int(1, $len - 5));
            $parts[] = substr($apiKey, $l[0], $l[] = random_int(1, $len - $l[0] - 3));
            $parts[] = substr($apiKey, array_sum($l));
            $apikeyPart1 = base64_encode($parts[0]);
            $apikeyPart2 = base64_encode($parts[1]);
            $apikeyPart3 = base64_encode($parts[2]);
        }
        if ($workbook_slug) {
            $workbook = UserOpenai::where('slug', $workbook_slug)->where('user_id', auth()->user()->id)->first();
        } else {
            $workbook = null;
        }

        return view('panel.user.generator.index', [
            'list' => OpenAIGenerator::query()
                ->where('active', true)
                ->get(),
            'filters' => OpenaiGeneratorFilter::query()
                ->where(function ($query) {
                    $query->where('user_id', auth()->id())
                        ->orWhereNull('user_id');
                })
                ->get(),
            'apikeyPart1' => $apikeyPart1,
            'apikeyPart2' => $apikeyPart2,
            'apikeyPart3' => $apikeyPart3,
            'apiUrl'      => $apiUrl,
            'workbook'    => $workbook,
            'models'      => Entity::planModels(),
        ]);
    }

    public function generator(Request $request, $slug): void {}

    public function generatorOptions(Request $request, $slug): string
    {
        $openai = OpenAIGenerator::query()
            ->where('slug', $slug)
            ->firstOrFail();
        $apiUrl = base64_encode('https://api.openai.com/v1/chat/completions');
        $settings_two = SettingTwo::getCache();
        if ($settings_two->openai_default_stream_server === 'backend') {
            $apikeyPart1 = base64_encode(random_int(1, 100));
            $apikeyPart2 = base64_encode(random_int(1, 100));
            $apikeyPart3 = base64_encode(random_int(1, 100));
        } else {
            $apiKey = ApiHelper::setOpenAiKey();
            $len = strlen($apiKey);
            $len = max($len, 6);
            $parts[] = substr($apiKey, 0, $l[] = random_int(1, $len - 5));
            $parts[] = substr($apiKey, $l[0], $l[] = random_int(1, $len - $l[0] - 3));
            $parts[] = substr($apiKey, array_sum($l));
            $apikeyPart1 = base64_encode($parts[0]);
            $apikeyPart2 = base64_encode($parts[1]);
            $apikeyPart3 = base64_encode($parts[2]);
        }

        $apiSearch = base64_encode('https://google.serper.dev/search');
        $auth = $request->user();

        $models = Entity::planModels();

        return view(
            'panel.user.generator.components.generator-options',
            compact(
                'slug',
                'openai',
                'apiSearch',
                'apikeyPart1',
                'apikeyPart2',
                'apikeyPart3',
                'apiUrl',
                'auth',
                'models'
            )
        )->render();
    }

    protected function openai(Request $request): Builder
    {
        $team = $request->user()->getAttribute('team');

        $myCreatedTeam = $request->user()->getAttribute('myCreatedTeam');

        return UserOpenai::query()
            ->where(function (Builder $query) use ($team, $myCreatedTeam) {
                $query->where('user_id', auth()->id())
                    ->when($team || $myCreatedTeam, function ($query) use ($team, $myCreatedTeam) {
                        if ($team && $team?->is_shared) {
                            $query->orWhere('team_id', $team->id);
                        }
                        if ($myCreatedTeam) {
                            $query->orWhere('team_id', $myCreatedTeam->id);
                        }
                    });
            });
    }
}
