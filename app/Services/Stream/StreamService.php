<?php

namespace App\Services\Stream;

use App\Domains\Engine\Enums\EngineEnum;
use App\Domains\Engine\Services\AnthropicService;
use App\Domains\Engine\Services\GeminiService;
use App\Domains\Entity\BaseDriver;
use App\Domains\Entity\Enums\EntityEnum;
use App\Domains\Entity\Facades\Entity;
use App\Enums\BedrockEngine;
use App\Extensions\AIChatPro\System\Services\AiChatProService;
use App\Extensions\AiChatProEntityHighlight\System\Services\EntityHighlightService;
use App\Extensions\AIChatProFileChat\System\Services\AIFileChatService;
use App\Extensions\AiChatProImageChat\System\Services\AIChatImageService;
use App\Extensions\AIChatProSkills\System\Services\SkillToolService;
use App\Extensions\AiChatProSmartImage\System\Services\SmartImageService;
use App\Extensions\AzureOpenai\System\Services\AzureOpenaiService;
use App\Extensions\OpenRouter\System\Services\RouterAiService;
use App\Extensions\SocialMediaAgent\System\Services\Chat\SocialMediaAgentChatService;
use App\Helpers\Classes\ApiHelper;
use App\Helpers\Classes\Helper;
use App\Helpers\Classes\MarketplaceHelper;
use App\Helpers\Classes\OpenAiParamHelper;
use App\Models\Setting;
use App\Models\SettingTwo;
use App\Models\Usage;
use App\Models\UserOpenai;
use App\Models\UserOpenaiChat;
use App\Models\UserOpenaiChatMessage;
use App\Services\Assistant\AssistantService;
use App\Services\Bedrock\BedrockRuntimeService;
use Exception;
use GuzzleHttp\Client;
use GuzzleHttp\Exception\GuzzleException;
use Illuminate\Http\Request;
use Illuminate\Support\Arr;
use Illuminate\Support\Collection;
use Illuminate\Support\Facades\Auth;
use Illuminate\Support\Facades\Log;
use Illuminate\Support\Facades\Session;
use Illuminate\Support\Str;
use JsonException;
use OpenAI as OpenAIMain;
use OpenAI\Laravel\Facades\OpenAI;
use OpenAI\Responses\Responses\Output\OutputFunctionToolCall;
use OpenAI\Responses\Responses\Output\OutputMessage;
use stdClass;
use Symfony\Component\HttpFoundation\StreamedResponse;
use Throwable;

class StreamService
{
    public bool $guest = false;

    private string $tempChatSessionKey = 'temp_chat_history_';

    public bool $tempChatActive = false;

    public bool $isFirstMessage = false;

    public bool $shouldGenerateSuggestions = false;

    private bool $entityHighlightsEnabled = false;

    private bool $isCouncilSubRequest = false;

    private bool $entityBlockSuppressed = false;

    private bool $titleEmitted = false;

    private string $titleBuffer = '';

    private string $suggestionsBuffer = '';

    private Collection $autoSkills;

    private array $usedSkills = [];

    public function __construct(
        Setting $setting,
        SettingTwo $settingTwo,
    ) {
        $this->autoSkills = collect();
        match (setting('default_ai_engine', EngineEnum::OPEN_AI->value)) {
            EngineEnum::ANTHROPIC->value => ApiHelper::setAnthropicKey($setting),
            EngineEnum::GEMINI->value    => ApiHelper::setGeminiKey($setting),
            EngineEnum::X_AI->value      => ApiHelper::setXAiKey($setting),
            default                      => ApiHelper::setOpenAiKey($setting),
        };
    }

    public function prepareStreamEnvironment(): void
    {
        if (function_exists('ini_set')) {
            @ini_set('output_buffering', 'off');
            @ini_set('zlib.output_compression', '0');
            @ini_set('implicit_flush', '1');
        }

        if (function_exists('ob_implicit_flush')) {
            ob_implicit_flush(true);
        }

        $minLevel = config('octane.enabled') ? 1 : 0;
        while (ob_get_level() > $minLevel) {
            if (! @ob_end_flush()) {
                break;
            }
        }
    }

    public function safeFlush(): void
    {
        if (function_exists('ob_flush')) {
            @ob_flush();
        }
        if (function_exists('flush')) {
            @flush();
        }
    }

    /**
     * Emit the skills_used SSE event if any skills were used.
     */
    private bool $skillsEmitted = false;

    private function emitUsedSkills(): void
    {
        if (empty($this->usedSkills) || $this->skillsEmitted) {
            return;
        }

        $this->skillsEmitted = true;

        echo PHP_EOL;
        echo "event: skills_used\n";
        echo 'data: ' . json_encode(['skills' => $this->usedSkills]);
        echo "\n\n";
        $this->safeFlush();
    }

    /**
     * Emit a stream data chunk, filtering out title/suggestions JSON.
     * Entity-highlight blocks are streamed through and stripped by the frontend.
     */
    private function emitStreamChunk(string $messageFix): void
    {
        if (! $this->titleEmitted && $this->isFirstMessage) {
            $this->titleBuffer .= $messageFix;

            if (preg_match('/\{"title"\s*:\s*"[^"]*"[^}]*\}/', $this->titleBuffer)) {
                $this->titleEmitted = true;
                $clean = preg_replace('/^\s*(```[\w]*\s*(<br\s*\/?>|\s)*)?\{"title"\s*:\s*"[^"]*"[^}]*\}\s*(```\s*)?(<br\s*\/?>|\s)*/i', '', $this->titleBuffer);
                $this->titleBuffer = '';
                if ($clean === '') {
                    return;
                }
                $messageFix = $clean;
            } else {
                return;
            }
        }

        // Entity block tail suppression: once triggered, swallow ALL remaining chunks.
        // The entity block is always the LAST thing before [DONE], so no exit conditions needed.
        if ($this->entityBlockSuppressed) {
            return;
        }

        if ($this->shouldGenerateSuggestions || $this->entityHighlightsEnabled) {
            $this->suggestionsBuffer .= $messageFix;

            // Detect :::meta (or legacy :::entity-highlights) marker — suppress all remaining output
            if ($this->entityHighlightsEnabled && str_contains($this->suggestionsBuffer, ':::') && preg_match('/:::\s*(?:meta|entity[- ]?highlights?)/i', $this->suggestionsBuffer)) {
                $clean = (string) preg_replace('/\s*(<br\s*\/?>|\n)*\s*:::\s*(?:meta|entity[- ]?highlights?)[\s\S]*$/si', '', $this->suggestionsBuffer);
                $this->entityBlockSuppressed = true;
                $this->suggestionsBuffer = '';
                if ($clean === '') {
                    return;
                }
                $messageFix = $clean;
            }
            // Early detection: model dropped ':::' prefix but emitted a bare `meta`
            // marker followed by the entity JSON opener (`{` ... `"entities"`). Detect
            // as soon as the START of the block is visible, strip from `meta` onward,
            // and suppress all remaining chunks — same lifecycle as the `:::meta` path.
            // Without this, chunks that split mid-JSON (e.g. ending at `"suggestions`)
            // fall through to the generic suggestions branches, which only strip the
            // closing JSON and leave `meta<br/>{"entities":[...]}` in the output.
            elseif ($this->entityHighlightsEnabled
                && preg_match('/(?:^|\s|<br\s*\/?>|>)meta\b\s*(?:<br\s*\/?>|\n)*\s*\{[\s\S]*?"entities"\s*:/si', $this->suggestionsBuffer)) {
                $clean = (string) preg_replace('/\s*(<br\s*\/?>|\n)*\s*(?:\*{3,}|-{3,}|_{3,})?\s*(<br\s*\/?>|\n)*\s*\bmeta\b\s*(?:<br\s*\/?>|\n)*\s*\{[\s\S]*$/si', '', $this->suggestionsBuffer);
                $this->entityBlockSuppressed = true;
                $this->suggestionsBuffer = '';
                if ($clean === '') {
                    return;
                }
                $messageFix = $clean;
            }
            // Fallback: bare `meta` tail that closes with `"suggestions":...}` shape
            // without a preceding `{...entities` opener (legacy/partial variant).
            elseif ($this->entityHighlightsEnabled
                && preg_match('/(?:^|\s|<br\s*\/?>|>)meta\b\s*(?:<br\s*\/?>|\n)*\s*[\[\{\],}][\s\S]*?"suggestions"\s*:[\s\S]*?\}/si', $this->suggestionsBuffer)) {
                $clean = (string) preg_replace('/\s*(<br\s*\/?>|\n)*\s*\bmeta\b\s*(?:<br\s*\/?>|\n)*\s*[\[\{\],}][\s\S]*$/si', '', $this->suggestionsBuffer);
                $this->entityBlockSuppressed = true;
                $this->suggestionsBuffer = '';
                if ($clean === '') {
                    return;
                }
                $messageFix = $clean;
            }
            // Check if buffer contains a complete suggestions JSON block ending with } — strip it
            elseif ($this->shouldGenerateSuggestions && preg_match('/\{\s*"[^"]+"/s', $this->suggestionsBuffer) && str_contains($this->suggestionsBuffer, '}')) {
                $clean = (string) preg_replace('/\s*(<br\s*\/?>|\n)*\s*(?:\*{3,}|-{3,}|_{3,})?\s*(<br\s*\/?>|\n)*\s*,*\s*(?:\bmeta\b\s*(?:<br\s*\/?>|\n)*\s*)?(```[\w]*\s*(<br\s*\/?>|\s)*)?\[?\s*\{[\s\S]*\}\s*\]?\s*,*\s*(```\s*)?(<br\s*\/?>|\s)*$/si', '', $this->suggestionsBuffer);
                $this->suggestionsBuffer = '';
                if ($clean === '') {
                    return;
                }
                $messageFix = $clean;
            }
            // Fallback: `"suggestions":` key appears without a leading `{` (malformed tail)
            // and the buffer is closed by `}`. Strip the tail from the key onward.
            elseif ($this->shouldGenerateSuggestions && preg_match('/["\']suggestions["\']\s*:/i', $this->suggestionsBuffer) && str_contains($this->suggestionsBuffer, '}')) {
                $clean = (string) preg_replace('/\s*(<br\s*\/?>|\n)*\s*[\[\{,]?\s*["\']?suggestions["\']?\s*:[\s\S]*$/si', '', $this->suggestionsBuffer);
                $this->suggestionsBuffer = '';
                if ($clean === '') {
                    return;
                }
                $messageFix = $clean;
            }
            // Check if buffer ends with partial patterns worth waiting for — keep buffering
            elseif (preg_match('/:::\s*$/s', $this->suggestionsBuffer)
                || ($this->entityHighlightsEnabled && preg_match('/^\s*(?:<br\s*\/?>|\n)*\s*\bmeta\b/si', $this->suggestionsBuffer) && ! str_contains($this->suggestionsBuffer, '}'))
                || ($this->shouldGenerateSuggestions && preg_match('/\{\s*"?\s*$/s', $this->suggestionsBuffer))
                || ($this->shouldGenerateSuggestions && str_contains($this->suggestionsBuffer, '{"') && ! str_contains($this->suggestionsBuffer, '}'))
                || ($this->shouldGenerateSuggestions && preg_match('/["\']sugg[a-z]*$/i', $this->suggestionsBuffer))
                // Markdown horizontal-rule separator (***, ---, ___) at end of buffer —
                // the model sometimes emits this just before the suggestions/meta JSON tail.
                // Wait one more chunk to see whether JSON follows; if not, the separator
                // gets flushed as legitimate content on the next non-matching chunk.
                || (($this->shouldGenerateSuggestions || $this->entityHighlightsEnabled)
                    && preg_match('/(?:^|<br\s*\/?>|\n|\s)(?:\*{3,}|-{3,}|_{3,})\s*(?:<br\s*\/?>|\n|\s)*$/s', $this->suggestionsBuffer))) {
                return;
            }
            // No pattern detected — flush buffer as normal content
            else {
                $messageFix = $this->suggestionsBuffer;
                $this->suggestionsBuffer = '';
            }
        }

        echo PHP_EOL;
        echo "event: data\n";
        echo 'data: ' . $messageFix;
        echo "\n\n";
        $this->safeFlush();
    }

    private function resetStreamChunkState(): void
    {
        $this->titleEmitted = false;
        $this->titleBuffer = '';
    }

    public function createDriver(EntityEnum $model): ?BaseDriver
    {
        if ($this->guest) {
            return Entity::driverForGuest($model);
        }

        return Entity::driver($model);
    }

    /**
     * @throws Exception
     * @throws GuzzleException
     */
    public function ChatStream(string $chat_bot, $history, $main_message, $chatParams, $ai_engine = null, $fileChat = false, $tempChatActive = false): ?StreamedResponse
    {
        $chat_type = $chatParams['chat_type'];
        $contain_images = $chatParams['contain_images'];
        $assistant = $chatParams['assistant'];
        $openRouter = $chatParams['openRouter'];

        // Initialize skill tracking
        $this->autoSkills = collect($chatParams['auto_skills'] ?? []);
        $this->usedSkills = $chatParams['used_skills'] ?? [];

        $this->tempChatActive = $tempChatActive;

        // If temp chat is active, merge with session history
        if ($this->tempChatActive && $main_message->user_openai_chat_id) {
            $tempHistory = $this->getTempChatHistory($main_message->user_openai_chat_id);

            // Merge temp history with current history, avoiding duplicates
            if (! empty($tempHistory)) {
                // Keep system message at the beginning if it exists
                $systemMessages = array_filter($history, fn ($msg) => $msg['role'] === 'system');
                $nonSystemHistory = array_filter($history, fn ($msg) => $msg['role'] !== 'system');

                // Combine: system messages + temp history + current non-system messages
                $history = array_merge($systemMessages, $tempHistory, $nonSystemHistory);

                // Remove duplicates based on content and role
                $history = $this->removeDuplicateMessages($history);
            }

            // Store the current user message immediately for context
            $this->addToTempHistory($main_message->user_openai_chat_id, [
                'role'    => 'user',
                'content' => $main_message->input,
            ]);
        }

        if (! $ai_engine) {
            $ai_engine = setting('default_ai_engine', EngineEnum::OPEN_AI->value);
        }

        $this->isFirstMessage = ! empty($chatParams['is_first_message']) && ! $this->tempChatActive && ! $this->guest;

        if ($this->isFirstMessage) {
            $history = $this->injectTitleInstruction($history);
        }

        $isCouncilSubRequest = (bool) request()?->input('council_sub_request', false);
        $this->isCouncilSubRequest = $isCouncilSubRequest;

        $this->shouldGenerateSuggestions = $chat_type === 'chatPro'
            && MarketplaceHelper::isRegistered('ai-chat-pro')
            && ! $this->tempChatActive
            && ! $isCouncilSubRequest;

        $entityHighlightActive = $chat_type === 'chatPro'
            && ! $isCouncilSubRequest
            && MarketplaceHelper::isRegistered('ai-chat-pro-entity-highlight')
            && EntityHighlightService::isEnabled();

        // When the entity highlight extension is active, suggestions are emitted
        // inside the same :::meta block — skip the standalone suggestions prompt to
        // avoid two competing "tail blocks" that the model often resolves by dropping one.
        if ($this->shouldGenerateSuggestions && ! $entityHighlightActive) {
            $history = $this->injectSuggestionsInstruction($history);
        }

        // Inject smart image system prompt when feature is enabled (OpenAI and Gemini)
        if ($chat_type === 'chatPro'
            && ! $isCouncilSubRequest
            && in_array($ai_engine, [EngineEnum::OPEN_AI->value, EngineEnum::GEMINI->value], true)
            && MarketplaceHelper::isRegistered('ai-chat-pro-smart-image')
            && SmartImageService::isEnabled()) {
            $history = $this->injectSmartImageInstruction($history);
        }

        // Inject entity highlight system prompt when feature is enabled. When suggestions
        // would also be active, fold them into the same metadata block.
        if ($entityHighlightActive) {
            $history = $this->injectEntityHighlightInstruction($history, $this->shouldGenerateSuggestions);
            $this->entityHighlightsEnabled = true;
        }

        $this->resetStreamChunkState();

        if ($chat_bot === EntityEnum::AZURE_OPENAI->slug() && MarketplaceHelper::isRegistered('azure-openai')) {
            return AzureOpenaiService::azureOpenaiStream($chat_bot, $history, $main_message, $chat_type, $contain_images);
        }

        // Pre-build skill function tools early so they can be merged with extension tools
        $earlySkillTools = [];
        if ($this->autoSkills->isNotEmpty() && class_exists(SkillToolService::class)) {
            if ($ai_engine === EngineEnum::OPEN_AI->value) {
                $earlySkillTools = SkillToolService::openAiTools($this->autoSkills);
            }
        }

        if ($chat_type === 'chatPro' && MarketplaceHelper::isRegistered('ai-chat-pro')) {
            if (! auth()->check()) {
                $this->guest = true;
            }

            $pass = false;
            if (MarketplaceHelper::isRegistered('ai-chat-pro-file-chat') && ((int) setting('chatpro_file_chat_allowed', 1) === 1)) {
                $service = new AIFileChatService(
                    request()?->input('pdfpath'),
                    request()?->input('chat_id')
                );

                $fileChat = $service->validateAndAnalyzeFile();
                if ($fileChat) {
                    $contain_images = false;
                    $pass = true;
                }
            }

            $hasImageGeneration = ! $isCouncilSubRequest
                && (bool) setting('ai_chat_pro_image_generation_feature', '0');
            $hasSmartImage = ! $isCouncilSubRequest
                && MarketplaceHelper::isRegistered('ai-chat-pro-smart-image')
                && SmartImageService::isEnabled();

            if (! $pass && $ai_engine === EngineEnum::OPEN_AI->value && ($hasImageGeneration || $hasSmartImage)) {
                $mergedTools = array_merge(AiChatProService::tools(), $earlySkillTools);

                return $this->openaiChatStream($chat_bot, $history, $main_message, $chat_type, $contain_images, tools: $mergedTools);
            }
        }

        if ($chat_type === 'chatPro-image' && MarketplaceHelper::isRegistered('ai-chat-pro-image-chat')) {
            return AIChatImageService::chatImageStream($chat_bot, $history, $main_message, $chatParams);
        }

        if ($chat_type === 'socialMediaAgent' && MarketplaceHelper::isRegistered('social-media-agent')) {
            $mergedTools = array_merge(SocialMediaAgentChatService::tools(), $earlySkillTools);

            return $this->openaiChatStream($chat_bot, $history, $main_message, $chat_type, $contain_images, tools: $mergedTools);
        }

        if ($fileChat) {
            return $this->openaiFileChat($chat_bot, $history, $main_message, $chat_type, $contain_images);
        }

        if (! is_null($assistant)) {
            return $this->assistantStream($chat_bot, $history, $main_message, $assistant);
        }

        if (! is_null($openRouter) && setting('open_router_status') == 1) {
            return $this->openRouterChatStream($chat_bot, $history, $main_message, $contain_images, $openRouter);
        }

        // Build skill tools for auto-use skills if any exist
        $skillTools = [];
        if ($this->autoSkills->isNotEmpty() && class_exists(SkillToolService::class)) {
            $skillTools = match ($ai_engine) {
                EngineEnum::OPEN_AI->value   => SkillToolService::openAiTools($this->autoSkills),
                EngineEnum::ANTHROPIC->value => SkillToolService::anthropicTools($this->autoSkills),
                EngineEnum::GEMINI->value    => SkillToolService::geminiTools($this->autoSkills),
                default                      => [],
            };
        }

        return match ($ai_engine) {
            EngineEnum::OPEN_AI->value   => $this->openaiChatStream($chat_bot, $history, $main_message, $chat_type, $contain_images, tools: ! empty($skillTools) ? $skillTools : []),
            EngineEnum::ANTHROPIC->value => $this->anthropicChatStream($chat_bot, $history, $main_message, $chat_type, $contain_images),
            EngineEnum::GEMINI->value    => $this->geminiChatStream($chat_bot, $history, $main_message, $chat_type, $contain_images),
            EngineEnum::DEEP_SEEK->value => $this->deepseekChatStream($chat_bot, $history, $main_message, $contain_images),
            EngineEnum::X_AI->value      => $this->xAiChatStream($chat_bot, $history, $main_message, $chat_type, $contain_images),
            default                      => throw new Exception('Invalid AI Engine'),
        };
    }

    private function openRouterChatStream($chat_bot, $history, $main_message, $contain_images, $openRouter)
    {
        $total_used_tokens = 0;
        $output = '';
        $responsedText = '';

        if ($contain_images) {
            $driver = $this->createDriver(EntityEnum::GPT_5_MINI);
        } else {
            $driver = $this->createDriver(EntityEnum::fromSlug($openRouter));
        }

        $this->prepareStreamEnvironment();

        return response()->stream(function () use ($openRouter, $driver, $chat_bot, $history, &$total_used_tokens, &$output, &$responsedText, $main_message, $contain_images) {

            $chat_id = $main_message->user_openai_chat_id;
            $chat = UserOpenaiChat::whereId($chat_id)->first();

            echo "event: message\n";
            echo 'data: ' . $main_message->id . "\n\n";

            if (! $driver->hasCreditBalance()) {
                echo PHP_EOL;
                echo "event: data\n";
                echo 'data: ' . __('You have no credits left. Please buy more credits to continue.');
                echo "\n\n";
                $this->safeFlush();
                echo "event: stop\n";
                echo 'data: [DONE]';
                echo "\n\n";
                $this->safeFlush();

                return null;
            }

            if (! $contain_images) {
                $historyMessages = array_filter($history, function ($item) {
                    return $item['role'] != 'system';
                });

                $service = new RouterAiService;
                $response = $service->response(last($historyMessages)['content'], $openRouter);

                foreach (explode("\n", $response) as $line) {
                    if (str_starts_with($line, 'data:')) {
                        $data = trim(substr($line, 5));
                        if ($data === '[DONE]') {
                            break;
                        }

                        $json = json_decode($data, true);

                        if (isset($json['choices'][0]['delta']['content'])) {
                            $content = $json['choices'][0]['delta']['content'];

                            if (! empty($content)) {
                                $output .= $content;
                                $responsedText .= $content;
                                $total_used_tokens += str_word_count($content);

                                $content = str_replace(["\r\n", "\r", "\n"], '<br/>', $content);
                                $this->emitStreamChunk($content);

                                if (connection_aborted()) {
                                    break;
                                }
                            }
                        }
                    }
                }
            } else {
                ApiHelper::setOpenAiKey();
                $chat_bot = EntityEnum::GPT_5_MINI->value;
                $stream = OpenAI::responses()->createStreamed([
                    'model'             => $chat_bot,
                    'input'             => $history,
                    'max_output_tokens' => 2000,
                    'temperature'       => 1.0,
                    'stream'            => true,
                ]);
                foreach ($stream as $response) {
                    if (($response->event === 'response.output_text.delta') && isset($response->response->delta)) {
                        $text = $response->response->delta;
                        $messageFix = str_replace(["\r\n", "\r", "\n"], '<br/>', $text);
                        $output .= $messageFix;
                        $responsedText .= $text;
                        $total_used_tokens += countWords($text);
                        if (connection_aborted()) {
                            break;
                        }
                        $this->emitStreamChunk($messageFix);
                    }
                }
            }

            $this->saveStreamResponse($main_message, $chat, $responsedText, $output, $total_used_tokens, $driver);
        }, 200, [
            'Cache-Control'     => 'no-cache',
            'X-Accel-Buffering' => 'no',
            'Connection'        => 'keep-alive',
            'Content-Type'      => 'text/event-stream',
        ]);
    }

    public function fixMessageHistory(array $history): array
    {
        $fixedHistory = [];
        $firstMessage = null;
        foreach ($history as $message) {
            if ($firstMessage === null) {
                $firstMessage = $message;
            } else {
                if ($firstMessage['role'] === $message['role']) {
                    if (is_array($message['content'])) {
                        $firstMessage['content'] = $message['content'];
                    } else {
                        $firstMessage['content'] .= ' ' . $message['content'];
                    }
                } else {
                    // Add the current message to the fixed history
                    $fixedHistory[] = $firstMessage;
                    // Start a new message
                    $firstMessage = $message;
                }
            }
        }
        if ($firstMessage !== null) {
            $fixedHistory[] = $firstMessage;
        }

        return $fixedHistory;
    }

    private function deepseekChatStream($chat_bot, $history, $main_message, $contain_images): StreamedResponse
    {
        ini_set('max_execution_time', 440);
        set_time_limit(0);

        $history = $this->fixMessageHistory($history);
        $total_used_tokens = 0;
        $output = '';
        $responsedText = '';

        if ($contain_images) {
            $driver = $this->createDriver(EntityEnum::GPT_5_MINI);
        } else {
            $driver = $this->createDriver(EntityEnum::fromSlug($chat_bot));
        }
        $this->prepareStreamEnvironment();

        return response()->stream(
            function () use ($driver, $chat_bot, $history, $main_message, $contain_images, &$total_used_tokens, &$output, &$responsedText) {
                $chat_id = $main_message->user_openai_chat_id;
                $chat = UserOpenaiChat::whereId($chat_id)->first();
                echo "event: message\n";
                echo 'data: ' . $main_message->id . "\n\n";
                if (! $driver->hasCreditBalance()) {
                    echo PHP_EOL;
                    echo "event: data\n";
                    echo 'data: ' . __('You have no credits left. Please buy more credits to continue.');
                    echo "\n\n";
                    $this->safeFlush();
                    echo "event: stop\n";
                    echo 'data: [DONE]';
                    echo "\n\n";
                    $this->safeFlush();

                    return null;
                }
                if (! $contain_images) {
                    ini_set('max_execution_time', 3000);
                    set_time_limit(3000);
                    $client = new Client;
                    ApiHelper::setDeepseekKey();
                    $url = 'https://api.deepseek.com/chat/completions';
                    $apikey = config('deepseek.api_key');
                    $headers = [
                        'Content-Type'  => 'application/json',
                        'Accept'        => 'application/json',
                        'Authorization' => "Bearer $apikey",
                    ];

                    $body = [
                        'messages'          => $history,
                        'model'             => $chat_bot,
                        'max_tokens'        => (int) setting('deepseek_max_output_length', 200),
                        'response_format'   => [
                            'type' => 'text',
                        ],
                        'stop'           => null,
                        'stream'         => true,
                        'stream_options' => null,
                        'temperature'    => 1,
                        'top_p'          => 1,
                        'tools'          => null,
                        'tool_choice'    => 'none',
                        'logprobs'       => false,
                        'top_logprobs'   => null,
                    ];
                    $response = $client->post($url, [
                        'headers' => $headers,
                        'json'    => $body,
                    ]);
                    $bodyStream = $response->getBody();
                    $buffer = '';
                    $emptyLinesAdded = false;
                    while (! $bodyStream->eof()) {
                        $chunk = $bodyStream->read(1024);
                        $buffer .= $chunk;

                        while (($pos = strpos($buffer, "\n")) !== false) {
                            $line = substr($buffer, 0, $pos);
                            $buffer = substr($buffer, $pos + 1);

                            if (str_starts_with(trim($line), 'data: ')) {
                                $json = trim(substr($line, 5)); // Remove "data: "
                                if (! empty($json)) {
                                    $decoded = json_decode($json, true);
                                    if (json_last_error() === JSON_ERROR_NONE) {
                                        $delta = $decoded['choices'][0]['delta'] ?? [];

                                        // Handle reasoning content
                                        if (isset($delta['reasoning_content']) && $delta['reasoning_content'] !== null) {
                                            // Add start signal if this is the first reasoning content
                                            if (! isset($reasoningStarted)) {
                                                $reasoningStarted = true;
                                                $startSignal = '[START_REASONING]';
                                                $output .= $startSignal;
                                                $responsedText .= '[START_REASONING]';

                                                echo PHP_EOL;
                                                echo "event: data\n";
                                                echo 'data: ' . $startSignal;
                                                echo "\n\n";
                                                $this->safeFlush();
                                            }

                                            $messageFix = str_replace(["\r\n", "\r", "\n"], '<br/>', $delta['reasoning_content']);
                                            $output .= $messageFix;
                                            $responsedText .= $messageFix;
                                            // $total_used_tokens += countWords($messageFix); do we calculate reasoning content?

                                            echo PHP_EOL;
                                            echo "event: data\n";
                                            echo 'data: ' . $messageFix;
                                            echo "\n\n";
                                            $this->safeFlush();
                                        }

                                        // Handle regular content
                                        if (isset($delta['content']) && $delta['content'] !== null) {
                                            // Add end signal if we were in reasoning mode
                                            if (isset($reasoningStarted)) {
                                                $endSignal = '[END_REASONING]<br/><br/>';
                                                $output .= $endSignal;
                                                $responsedText .= "[END_REASONING]\n\n";

                                                echo PHP_EOL;
                                                echo "event: data\n";
                                                echo 'data: ' . $endSignal;
                                                echo "\n\n";
                                                $this->safeFlush();

                                                unset($reasoningStarted);
                                            }

                                            if (! $emptyLinesAdded) {
                                                echo "event: data\n";
                                                echo 'data: <br/><br/><br/>';
                                                echo "\n\n";
                                                $this->safeFlush();
                                                $emptyLinesAdded = true;
                                            }

                                            $messageFix = str_replace(["\r\n", "\r", "\n"], '<br/>', $delta['content']);
                                            $output .= $messageFix;
                                            $responsedText .= $messageFix;
                                            $total_used_tokens += countWords($messageFix);
                                            $this->emitStreamChunk($messageFix);
                                        }
                                    }
                                }
                            }
                        }

                        if (connection_aborted()) {
                            break;
                        }
                    }

                } else {
                    ApiHelper::setOpenAiKey();
                    $chat_bot = EntityEnum::GPT_5_MINI->value;
                    $stream = OpenAI::responses()->createStreamed([
                        'model'                    => $chat_bot,
                        'input'                    => $history,
                        'max_output_tokens'        => 2000,
                        'temperature'              => 1.0,
                        'stream'                   => true,
                    ]);
                    foreach ($stream as $response) {
                        if (($response->event === 'response.output_text.delta') && isset($response->response->delta)) {
                            $text = $response->response->delta;
                            $messageFix = str_replace(["\r\n", "\r", "\n"], '<br/>', $text);
                            $output .= $messageFix;
                            $responsedText .= $text;
                            $total_used_tokens += countWords($text);
                            if (connection_aborted()) {
                                break;
                            }
                            $this->emitStreamChunk($messageFix);
                        }
                    }
                }
                $this->saveStreamResponse($main_message, $chat, $responsedText, $output, $total_used_tokens, $driver);
            },
            200,
            [
                'Cache-Control'     => 'no-cache',
                'X-Accel-Buffering' => 'no',
                'Connection'        => 'keep-alive',
                'Content-Type'      => 'text/event-stream',
            ]
        );
    }

    private function deepseekOtherStream(Request $request, $chat_bot)
    {
        ini_set('max_execution_time', 440);
        set_time_limit(0);

        $total_used_tokens = 0;
        $output = '';
        $responsedText = '';

        $prompt = $request->get('prompt');
        $message_id = $request->get('message_id');
        $openai_id = $request->get('openai_id');
        $title = $request->get('title');

        $history[] = ['role' => 'user', 'content' => $prompt];

        $driver = $this->createDriver(EntityEnum::fromSlug($chat_bot));

        $this->prepareStreamEnvironment();

        return response()->stream(function () use (&$total_used_tokens, &$output, &$responsedText, $driver, $message_id, $title, $openai_id, $prompt, $history, $chat_bot) {

            $user = Auth::user();
            $entry = UserOpenai::firstOrCreate(
                [
                    'id' => $message_id,
                ],
                [
                    'user_id'   => $user->id,
                    'input'     => $prompt,
                    'hash'      => str()->random(256),
                    'team_id'   => $user->team_id,
                    'slug'      => str()->random(7) . str($user?->fullName())->slug() . '-workbook',
                    'openai_id' => $openai_id ?? 1,
                ]);

            echo "event: message\n";
            echo 'data: ' . $message_id . "\n\n";

            if (! $driver->hasCreditBalance()) {
                echo PHP_EOL;
                echo "event: data\n";
                echo 'data: ' . __('You have no credits left. Please buy more credits to continue.');
                echo "\n\n";
                $this->safeFlush();
                echo "event: stop\n";
                echo 'data: [DONE]';
                echo "\n\n";
                $this->safeFlush();

                return null;
            }

            $client = new Client;

            ApiHelper::setDeepseekKey();

            $url = 'https://api.deepseek.com/chat/completions';
            $apikey = config('deepseek.api_key');
            $headers = [
                'Content-Type'  => 'application/json',
                'Accept'        => 'application/json',
                'Authorization' => "Bearer $apikey",
            ];

            $body = [
                'messages'          => $history,
                'model'             => $chat_bot,
                'max_tokens'        => (int) setting('deepseek_max_output_length', 200),
                'response_format'   => [
                    'type' => 'text',
                ],
                'stop'           => null,
                'stream'         => true,
                'stream_options' => null,
                'temperature'    => 1,
                'top_p'          => 1,
                'tools'          => null,
                'tool_choice'    => 'none',
                'logprobs'       => false,
                'top_logprobs'   => null,
            ];

            $response = $client->post($url, [
                'headers' => $headers,
                'json'    => $body,
            ]);

            $bodyStream = $response->getBody();
            $buffer = '';
            $emptyLinesAdded = false;
            while (! $bodyStream->eof()) {
                $chunk = $bodyStream->read(1024);
                $buffer .= $chunk;

                while (($pos = strpos($buffer, "\n")) !== false) {
                    $line = substr($buffer, 0, $pos);
                    $buffer = substr($buffer, $pos + 1);

                    if (str_starts_with(trim($line), 'data: ')) {
                        $json = trim(substr($line, 5)); // Remove "data: "
                        if (! empty($json)) {
                            $decoded = json_decode($json, true);
                            if (json_last_error() === JSON_ERROR_NONE) {
                                $delta = $decoded['choices'][0]['delta'] ?? [];
                                if (isset($delta['reasoning_content']) && $delta['reasoning_content'] !== null) {
                                    $messageFix = str_replace(["\r\n", "\r", "\n"], '<br/>', $delta['reasoning_content']);
                                    $output .= $messageFix;
                                    $responsedText .= $messageFix;
                                    // $total_used_tokens += countWords($messageFix); do we calculate reasoning content?
                                    echo PHP_EOL;
                                    echo "event: data\n";
                                    echo 'data: ' . $messageFix;
                                    echo "\n\n";
                                    $this->safeFlush();
                                }

                                if (isset($delta['content']) && $delta['content'] !== null) {
                                    if (! $emptyLinesAdded) {
                                        echo "event: data\n";
                                        echo 'data: <br/><br/><br/>';
                                        echo "\n\n";
                                        $this->safeFlush();
                                        $emptyLinesAdded = true;
                                    }
                                    $messageFix = str_replace(["\r\n", "\r", "\n"], '<br/>', $delta['content']);

                                    $output .= $messageFix;
                                    $responsedText .= $messageFix;
                                    $total_used_tokens += countWords($messageFix);

                                    echo "event: data\n";
                                    echo 'data: ' . $messageFix;
                                    echo "\n\n";
                                    $this->safeFlush();
                                }
                            }
                        }
                    }
                }

                if (connection_aborted()) {
                    break;
                }
            }

            echo "event: stop\n";
            echo 'data: [DONE]';
            echo "\n\n";
            $this->safeFlush();

            $entry->update([
                'title'    => $title ?: null,
                'credits'  => $total_used_tokens,
                'words'    => $total_used_tokens,
                'response' => $responsedText,
                'output'   => $output,
            ]);

            $driver->input($responsedText)
                ->calculateCredit()
                ->decreaseCredit();
            Usage::getSingle()->updateWordCounts($driver->calculate());

        }, 200, [
            'Cache-Control'     => 'no-cache',
            'X-Accel-Buffering' => 'no',
            'Connection'        => 'keep-alive',
            'Content-Type'      => 'text/event-stream',
        ]);
    }

    /**
     * @throws GuzzleException
     * @throws JsonException
     * @throws Exception
     */
    public function assistantStream(string $chat_bot, $history, $main_message, $assistant): ?StreamedResponse
    {
        $chat = UserOpenaiChat::query()->where('id', $main_message->user_openai_chat_id)->first();
        $threadId = $chat?->thread_id;
        $driver = $this->createDriver(EntityEnum::fromSlug($chat_bot));

        $assistantService = new AssistantService;

        $tmp = $assistantService->createMessage($threadId, $history);

        return $assistantService->createRun($chat_bot, $assistant, $threadId, $main_message, $driver);
    }

    public function OtherStream(Request $request, string $chat_bot, $ai_engine = null): StreamedResponse
    {
        if (! $ai_engine) {
            $ai_engine = setting('default_ai_engine', EngineEnum::OPEN_AI->value);
        }

        if ($chat_bot === EntityEnum::AZURE_OPENAI->slug() && MarketplaceHelper::isRegistered('azure-openai')) {
            return AzureOpenaiService::azureOpenaiOtherStream($request, $chat_bot);
        }

        if (setting('open_router_status') == 1 && $request->open_router_model !== 'undefined' && ! empty($request->open_router_model)) {
            return $this->openRouterStream($request);
        }

        return match ($ai_engine) {
            EngineEnum::ANTHROPIC->value => $this->anthropicOtherStream($request, $chat_bot),
            EngineEnum::GEMINI->value    => $this->geminiOtherStream($request, $chat_bot),
            EngineEnum::DEEP_SEEK->value => $this->deepseekOtherStream($request, $chat_bot),
            EngineEnum::X_AI->value 	    => $this->xAiOtherStream($request, $chat_bot),
            default                      => $this->openaiOtherStream($request, $chat_bot),
        };
    }

    private function openRouterStream(Request $request)
    {
        $prompt = $request->get('prompt');
        $message_id = $request->get('message_id');
        $openai_id = $request->get('openai_id');
        $title = $request->get('title');
        $open_router_model = $request->get('open_router_model');
        $total_used_tokens = 0;
        $output = '';
        $responsedText = '';
        $driver = $this->createDriver(EntityEnum::fromSlug($open_router_model));

        $this->prepareStreamEnvironment();

        return response()->stream(function () use ($driver, &$total_used_tokens, &$output, &$responsedText, $message_id, $title, $openai_id, $prompt, $open_router_model) {
            $user = Auth::user();
            $entry = UserOpenai::find($message_id);
            if (! $entry) {
                $entry = new UserOpenai;
                $entry->user_id = $user?->id;
                $entry->input = $prompt;
                $entry->hash = str()->random(256);
                $entry->team_id = $user?->team_id;
                $entry->slug = str()->random(7) . str($user?->fullName())->slug() . '-workbook';
                $entry->openai_id = $openai_id ?? 1;
            }
            echo "event: message\n";
            echo 'data: ' . $message_id . "\n\n";

            if (! $driver->hasCreditBalance()) {
                echo PHP_EOL;
                echo "event: data\n";
                echo 'data: ' . __('You have no credits left. Please buy more credits to continue.');
                echo "\n\n";
                $this->safeFlush();
                echo "event: stop\n";
                echo 'data: [DONE]';
                echo "\n\n";
                $this->safeFlush();

                return null;
            }

            $service = new RouterAiService;
            $response = $service->response($entry->input, $open_router_model);

            foreach (explode("\n", $response) as $line) {
                if (str_starts_with($line, 'data:')) {
                    $data = trim(substr($line, 5));
                    if ($data === '[DONE]') {
                        break;
                    }

                    $json = json_decode($data, true);

                    if (isset($json['choices'][0]['delta']['content'])) {
                        $content = $json['choices'][0]['delta']['content'];

                        // Boş içerik varsa atla
                        if (! empty($content)) {
                            $output .= $content;
                            $responsedText .= $content;
                            $total_used_tokens += str_word_count($content);

                            $content = str_replace(["\r\n", "\r", "\n"], '<br/>', $content);

                            echo PHP_EOL;
                            echo "event: data\n";
                            echo 'data: ' . $content;
                            echo "\n\n";
                            $this->safeFlush();

                            if (connection_aborted()) {
                                break;
                            }
                        }
                    }
                }
            }

            echo "event: stop\n";
            echo 'data: [DONE]';
            echo "\n\n";
            $this->safeFlush();

            $this->saveOtherStreamResponse($entry, $title, $responsedText, $output, $total_used_tokens, $driver);
        }, 200, [
            'Cache-Control'     => 'no-cache',
            'X-Accel-Buffering' => 'no',
            'Connection'        => 'keep-alive',
            'Content-Type'      => 'text/event-stream',
        ]);
    }

    public function reduceTokensWhenIntterruptStream(Request $request, $type): void
    {
        $model = Helper::setting('openai_default_model') ?: EntityEnum::GPT_5_MINI->value;
        $streamed_text = $request->get('streamed_text');
        $streamed_text = preg_replace('/search_images\s*\(\s*\{[^}]*\}\s*\)/', '', $streamed_text);
        $message_id = $request->get('streamed_message_id');
        if ($streamed_text) {
            $total_used_tokens = countWords($streamed_text);
            $this->createDriver(EntityEnum::fromSlug($model))->input($streamed_text)->calculateCredit()->decreaseCredit();
            if (! empty($message_id)) {
                if ($type === 'writer') {
                    $entry = UserOpenai::find($message_id);
                    if ($entry) {
                        $entry->title = null;
                        $entry->credits = $total_used_tokens;
                        $entry->words = $total_used_tokens;
                        $entry->response = $streamed_text;
                        $entry->output = $streamed_text;
                        $entry->save();
                    }
                } else { // chat
                    $main_message = UserOpenaiChatMessage::find($message_id);
                    if ($main_message) {
                        $chat = UserOpenaiChat::find($main_message->user_openai_chat_id);
                        $main_message->response = $streamed_text;
                        $main_message->output = $streamed_text;
                        $main_message->credits = $total_used_tokens;
                        $main_message->words = $total_used_tokens;
                        $main_message->save();

                        if ($chat) {
                            $chat->total_credits += $total_used_tokens;
                            $chat->save();
                        }
                    }
                }
            }
        }
    }

    // X-AI Stream
    /**
     * @throws Exception
     */
    private function xAiChatStream(string $chat_bot, $history, $main_message, $chat_type, $contain_images): ?StreamedResponse
    {
        $total_used_tokens = 0;
        $output = '';
        $responsedText = '';

        if ($contain_images) {
            $driver = $this->createDriver(EntityEnum::GPT_5_MINI);
        } else {
            $driver = $this->createDriver(EntityEnum::fromSlug($chat_bot));
        }
        $this->prepareStreamEnvironment();

        return response()->stream(function () use ($driver, $history, &$total_used_tokens, &$output, &$responsedText, $main_message, $contain_images) {
            $chat_id = $main_message->user_openai_chat_id;
            $chat = UserOpenaiChat::whereId($chat_id)->first();

            echo "event: message\n";
            echo 'data: ' . $main_message->id . "\n\n";

            // Emit "Used X Skill" as the very first event
            if (! empty($this->usedSkills)) {
                $this->emitUsedSkills();
            }

            if (! $driver->hasCreditBalance()) {
                echo PHP_EOL;
                echo "event: data\n";
                echo 'data: ' . __('You have no credits left. Please buy more credits to continue.');
                echo "\n\n";
                $this->safeFlush();
                echo "event: stop\n";
                echo 'data: [DONE]';
                echo "\n\n";
                $this->safeFlush();

                return null;
            }

            $model = $driver->enum()->value;
            if ($contain_images) {
                $options = OpenAiParamHelper::sanitizeChatParams([
                    'model'                       => EntityEnum::GPT_5_MINI->value,
                    'messages'                    => $history,
                    'temperature'                 => 1.0,
                    'stream'                      => true,
                    'max_output_tokens'           => 2000,
                ]);
                $stream = OpenAI::chat()->createStreamed($options);
            } else {
                $api = ApiHelper::setXAiKey();

                try {
                    $cli = OpenAIMain::factory()->withBaseUri('https://api.x.ai/v1')
                        ->withHttpHeader('Authorization', 'Bearer ' . $api)
                        ->withApiKey($api)
                        ->make();
                    $stream = $cli->chat()->createStreamed([
                        'model'             => $model,
                        'messages'          => $history,
                        'stream'            => true,
                        'temperature'       => 1.0,
                    ]);
                } catch (Exception|Throwable $e) {
                    echo PHP_EOL;
                    echo "event: data\n";
                    echo 'data: ' . __('Something went wrong. Please try again later.');
                    echo "\n\n";
                    $this->safeFlush();
                    echo "event: stop\n";
                    echo 'data: [DONE]';
                    echo "\n\n";
                    $this->safeFlush();

                    return null;
                }
            }

            foreach ($stream as $response) {
                if (isset($response->choices[0]->delta->content)) {
                    $text = $response->choices[0]->delta->content;
                    $messageFix = str_replace(["\r\n", "\r", "\n"], '<br/>', $text);
                    $output .= $messageFix;
                    $responsedText .= $text;
                    $total_used_tokens += countWords($text);
                    if (connection_aborted()) {
                        break;
                    }
                    $this->emitStreamChunk($messageFix);
                }
            }
            $this->saveStreamResponse($main_message, $chat, $responsedText, $output, $total_used_tokens, $driver);
        }, 200, [
            'Cache-Control'     => 'no-cache',
            'X-Accel-Buffering' => 'no',
            'Connection'        => 'keep-alive',
            'Content-Type'      => 'text/event-stream',
        ]);
    }

    private function xAiOtherStream(Request $request, $chat_bot): ?StreamedResponse
    {
        $apiKey = ApiHelper::setXAiKey();
        $xai = OpenAIMain::factory()
            ->withApiKey($apiKey)
            ->withBaseUri('https://api.x.ai/v1')
            ->make();

        $prompt = $request->get('prompt');
        $message_id = $request->get('message_id');
        $openai_id = $request->get('openai_id');
        $title = $request->get('title');

        $history[] = ['role' => 'user', 'content' => $prompt];
        $total_used_tokens = 0;
        $output = '';
        $responsedText = '';
        $user = Auth::user();
        $driver = $this->createDriver(EntityEnum::fromSlug($chat_bot));

        $this->prepareStreamEnvironment();

        return response()->stream(function () use ($user, $driver, $history, &$total_used_tokens, &$output, &$responsedText, $message_id, $title, $openai_id, $prompt) {
            $entry = UserOpenai::find($message_id);
            if (! $entry) {
                $entry = new UserOpenai;
                $entry->user_id = $user->id;
                $entry->input = $prompt;
                $entry->hash = str()->random(256);
                $entry->team_id = $user->team_id;
                $entry->slug = str()->random(7) . str($user?->fullName())->slug() . '-workbook';
                $entry->openai_id = $openai_id ?? 1;
            }

            echo "event: message\n";
            echo 'data: ' . $message_id . "\n\n";

            if (! $driver->hasCreditBalance()) {
                echo PHP_EOL;
                echo "event: data\n";
                echo 'data: ' . __('You have no credits left. Please buy more credits to continue.');
                echo "\n\n";
                $this->safeFlush();
                echo "event: stop\n";
                echo 'data: [DONE]';
                echo "\n\n";
                $this->safeFlush();

                return null;
            }

            $api = ApiHelper::setXAiKey();

            try {
                $cli = OpenAIMain::factory()->withBaseUri('https://api.x.ai/v1')
                    ->withHttpHeader('Authorization', 'Bearer ' . $api)
                    ->withApiKey($api)
                    ->make();
                $stream = $cli->chat()->createStreamed([
                    'model'             => $driver->enum()->value,
                    'messages'          => $history,
                    'stream'            => true,
                    'temperature'       => 1.0,
                ]);
            } catch (Exception|Throwable $e) {
                echo PHP_EOL;
                echo "event: data\n";
                echo 'data: ' . __('Something went wrong. Please try again later.');
                echo "\n\n";
                $this->safeFlush();
                echo "event: stop\n";
                echo 'data: [DONE]';
                echo "\n\n";
                $this->safeFlush();

                return null;
            }

            foreach ($stream as $response) {
                if (isset($response->choices[0]->delta->content)) {
                    $text = $response->choices[0]->delta->content;
                    $messageFix = str_replace(["\r\n", "\r", "\n"], '<br/>', $text);
                    $output .= $messageFix;
                    $responsedText .= $text;
                    $total_used_tokens += countWords($text);
                    if (connection_aborted()) {
                        break;
                    }
                    echo PHP_EOL;
                    echo "event: data\n";
                    echo 'data: ' . $messageFix;
                    echo "\n\n";
                    $this->safeFlush();
                }
            }
            echo "event: stop\n";
            echo 'data: [DONE]';
            echo "\n\n";
            $this->safeFlush();

            $this->saveOtherStreamResponse($entry, $title, $responsedText, $output, $total_used_tokens, $driver);
        }, 200, [
            'Cache-Control'     => 'no-cache',
            'X-Accel-Buffering' => 'no',
            'Connection'        => 'keep-alive',
            'Content-Type'      => 'text/event-stream',
        ]);
    }

    // OpenAI Stream
    /**
     * Resolve the configured reasoning effort against the model's supported values,
     * downgrading to the closest accepted effort when the configured value isn't allowed
     * (e.g. 'none' on GPT-5 original or O-series, 'minimal' on O-series).
     */
    private function resolveReasoningEffort(EntityEnum $modelEnum): string
    {
        return $modelEnum->resolveReasoningEffort(
            setting('openai_reasoning_models_effort', 'low')
        );
    }

    /**
     * @throws Exception
     */
    private function openaiChatStream(string $chat_bot, $history, $main_message, $chat_type, $contain_images, ?array $tools = []): ?StreamedResponse
    {
        // When manual skills are already injected, remove auto-skill function tools — they're not needed
        if (! empty($this->usedSkills)) {
            $tools = array_filter($tools ?? [], fn ($t) => ! str_starts_with($t['name'] ?? '', 'use_skill_'));
            $tools = array_values($tools);
        }

        $total_used_tokens = 0;
        $output = '';
        $responsedText = '';

        if ($contain_images) {
            $driver = $this->createDriver(EntityEnum::GPT_5_MINI);
        } else {
            if ($tools && EntityEnum::fromSlug($chat_bot) === EntityEnum::GPT_5_CHAT) {
                $chat_bot = EntityEnum::GPT_5->slug();
            }

            $driver = $this->createDriver(EntityEnum::fromSlug($chat_bot));
        }
        $this->prepareStreamEnvironment();

        return response()->stream(function () use ($driver, $history, &$total_used_tokens, &$output, &$responsedText, $main_message, $contain_images, $tools, $chat_type) {

            $chat_id = $main_message->user_openai_chat_id;
            $chat = UserOpenaiChat::whereId($chat_id)->first();

            echo "event: message\n";
            echo 'data: ' . $main_message->id . "\n\n";

            // Emit "Used X Skill" as the very first event
            if (! empty($this->usedSkills)) {
                $this->emitUsedSkills();
            }

            if (! $driver->hasCreditBalance()) {
                echo PHP_EOL;
                echo "event: data\n";
                echo 'data: ' . __('You have no credits left. Please buy more credits to continue.');
                echo "\n\n";
                $this->safeFlush();
                echo "event: stop\n";
                echo 'data: [DONE]';
                echo "\n\n";
                $this->safeFlush();

                return null;
            }

            $model = $driver->enum()->value;
            $isSearchModel = in_array($model, [EntityEnum::GPT_4_O_MINI_SEARCH_PREVIEW->value, EntityEnum::GPT_4_O_SEARCH_PREVIEW->value], true);

            if ($isSearchModel) {
                // Use chat() endpoint for search models
                $options = [
                    'model'    => $model,
                    'messages' => $history,
                    'stream'   => true,
                ];

                if ($contain_images) {
                    $options['max_tokens'] = 2000;
                    $options['model'] = EntityEnum::GPT_5_MINI->value;
                }

                $options = OpenAiParamHelper::sanitizeChatParams($options);
                $stream = OpenAI::chat()->createStreamed($options);

                foreach ($stream as $response) {
                    if (connection_aborted()) {
                        break;
                    }

                    // Handle regular content
                    if (isset($response->choices[0]->delta->content)) {
                        $text = $response->choices[0]->delta->content;
                        $messageFix = str_replace(["\r\n", "\r", "\n"], '<br/>', $text);
                        $output .= $messageFix;
                        $responsedText .= $text;
                        $total_used_tokens += countWords($text);
                        $this->emitStreamChunk($messageFix);
                    }
                }
            } else {
                // Use responses() endpoint for non-search models
                $options = [
                    'model'  => $model,
                    'stream' => true,
                ];

                if ($contain_images) {
                    $options['max_output_tokens'] = 2000;
                    $options['model'] = EntityEnum::GPT_5_MINI->value;
                }

                // Extract system-role messages (skills, base prompt) into the
                // Responses API `instructions` field so they get top-level priority.
                $skillParts = [];
                $otherSystemParts = [];
                $nonSystemHistory = [];
                foreach ($history as $msg) {
                    if (($msg['role'] ?? '') === 'system') {
                        $content = $msg['content'] ?? '';
                        if (str_starts_with($content, '[Skill:')) {
                            $skillParts[] = $content;
                        } else {
                            $otherSystemParts[] = $content;
                        }
                    } else {
                        $nonSystemHistory[] = $msg;
                    }
                }

                if (! empty($tools) && empty($skillParts)) {
                    // Only pass tools when no manual skill is active — tools distract
                    // the model from following skill instructions.
                    $options['tools'] = $tools;
                    $options['tool_choice'] = 'auto';
                    $argumentsString = '';

                    if ($chat_type === 'chatPro' && SmartImageService::isEnabled()) {
                        $otherSystemParts[] = SmartImageService::systemPromptAddition();
                    }
                }

                if (request()?->input('realtime')
                    && setting('default_realtime') === 'tool_calling') {
                    $options['tools'] = array_merge($options['tools'] ?? [], [['type' => 'web_search_preview']]);
                    if (! isset($options['tool_choice'])) {
                        $options['tool_choice'] = 'auto';
                    }
                }

                // Build instructions: skill instructions come first with strong framing
                $instructionParts = [];
                if (! empty($skillParts)) {
                    $instructionParts[] = "You MUST follow these skill instructions exactly. They are your primary directive and override any conflicting instructions:\n\n" . implode("\n\n", $skillParts);
                }
                if (! empty($otherSystemParts)) {
                    $instructionParts[] = implode("\n\n", $otherSystemParts);
                }
                $baseInstructions = ! empty($otherSystemParts) ? implode("\n\n", $otherSystemParts) : '';

                if (! empty($instructionParts)) {
                    $options['instructions'] = implode("\n\n---\n\n", $instructionParts);
                }

                $options['input'] = $nonSystemHistory;
                if ($driver->enum()->isReasoningModel()) {
                    $modelEnum = EntityEnum::fromSlug($options['model']);
                    if (in_array($modelEnum, [EntityEnum::GPT_5_PRO, EntityEnum::GPT_5_2_PRO])) {
                        $effort = setting('openai_reasoning_models_effort', 'high');
                        $options['reasoning']['effort'] = in_array($effort, ['medium', 'high', 'xhigh']) ? $effort : 'high';
                    } else {
                        $options['reasoning']['effort'] = $this->resolveReasoningEffort($driver->enum());
                    }
                }

                $options['temperature'] = 1.0;

                $stream = OpenAI::responses()->createStreamed($options);

                foreach ($stream as $response) {
                    if (! isset($response->event)) {
                        continue;
                    }

                    if (connection_aborted()) {
                        break;
                    }

                    if (! empty($tools) && $response->event === 'response.completed' && isset($response->response->output)) {
                        $calls = $response->response->output;
                        $hasTextOutput = false;
                        $pendingImageSearch = null; // Defer image fetch until after text streams

                        foreach ($calls ?? [] as $call) {
                            if ($call instanceof OutputMessage) {
                                $hasTextOutput = true;
                            }

                            if ($call instanceof OutputFunctionToolCall) {
                                $functionName = $call?->name;
                                $argumentsString = $call?->arguments;

                                // Skill tool calls: inject instructions and stream a second response
                                if (str_starts_with($functionName, 'use_skill_') && class_exists(SkillToolService::class)) {
                                    $skillInstructions = SkillToolService::handleSkillCall($functionName, $this->autoSkills);
                                    $skillMeta = SkillToolService::getSkillMeta($functionName, $this->autoSkills);

                                    if ($skillMeta) {
                                        $this->usedSkills[] = $skillMeta;
                                        $this->emitUsedSkills();
                                    }

                                    if ($skillInstructions) {
                                        $cleanInstructions = rtrim($skillInstructions, " \t\n\r-");
                                        $history[] = ['role' => 'system', 'content' => '[Skill: ' . ($skillMeta['name'] ?? '') . "]\n{$cleanInstructions}"];

                                        // Remove title instruction — skill responses shouldn't include title JSON
                                        $history = array_values(array_filter($history, function ($msg) {
                                            return ! str_contains($msg['content'] ?? '', '{"title":"short descriptive title');
                                        }));
                                        $this->isFirstMessage = false;

                                        // Stream a second response with skill instructions injected
                                        // Build instructions: base chat instructions + skill instructions
                                        $skillFollowUpParts = [];
                                        if ($baseInstructions !== '') {
                                            $skillFollowUpParts[] = $baseInstructions;
                                        }
                                        $skillFollowUpParts[] = "You MUST follow these skill instructions exactly. They are your primary directive and override any conflicting instructions:\n\n[Skill: " . ($skillMeta['name'] ?? '') . "]\n{$cleanInstructions}";
                                        $skillFollowUpInstructions = implode("\n\n---\n\n", $skillFollowUpParts);

                                        // Separate non-system messages for input (system messages go in instructions)
                                        $skillNonSystemInput = array_values(array_filter($history, fn ($msg) => ($msg['role'] ?? '') !== 'system'));

                                        $skillOptions = [
                                            'model'        => $model,
                                            'stream'       => true,
                                            'instructions' => $skillFollowUpInstructions,
                                            'input'        => $skillNonSystemInput,
                                            'temperature'  => 1.0,
                                        ];
                                        if ($driver->enum()->isReasoningModel()) {
                                            if (in_array($driver->enum(), [EntityEnum::GPT_5_PRO, EntityEnum::GPT_5_2_PRO])) {
                                                $effort = setting('openai_reasoning_models_effort', 'high');
                                                $skillOptions['reasoning']['effort'] = in_array($effort, ['medium', 'high', 'xhigh']) ? $effort : 'high';
                                            } else {
                                                $skillOptions['reasoning']['effort'] = $this->resolveReasoningEffort($driver->enum());
                                            }
                                        }

                                        $skillStream = OpenAI::responses()->createStreamed($skillOptions);

                                        foreach ($skillStream as $skillResponse) {
                                            if (! isset($skillResponse->event)) {
                                                continue;
                                            }
                                            if (connection_aborted()) {
                                                break;
                                            }
                                            if (isset($skillResponse->response->delta) && $skillResponse->event === 'response.output_text.delta') {
                                                $text = $skillResponse->response->delta;
                                                $messageFix = str_replace(["\r\n", "\r", "\n"], '<br/>', $text);
                                                $output .= $messageFix;
                                                $responsedText .= $text;
                                                $total_used_tokens += countWords($text);
                                                $this->emitStreamChunk($messageFix);
                                            }
                                        }

                                        $hasTextOutput = true;
                                    }

                                    continue;
                                }

                                // For search_images: defer the actual fetch, just notify frontend
                                if ($functionName === 'search_images') {
                                    echo PHP_EOL;
                                    echo "event: function_call\n";
                                    echo 'data: search_images' . "\n\n";
                                    $this->safeFlush();

                                    // Save for later — fetch images AFTER text starts streaming
                                    $pendingImageSearch = [
                                        'functionName'    => $functionName,
                                        'argumentsString' => $argumentsString,
                                        'chat_type'       => $chat_type,
                                    ];
                                } else {
                                    // Non-image tool calls: execute immediately
                                    $functionResponse = null;
                                    if ($chat_type === 'chatPro') {
                                        $functionResponse = AiChatProService::callFunction($functionName, $argumentsString);
                                    } elseif ($chat_type === 'socialMediaAgent') {
                                        $functionResponse = SocialMediaAgentChatService::callFunction($functionName, $argumentsString);
                                    }

                                    if (isset($functionResponse)) {
                                        // When generating a social post, clear any AI conversational text
                                        // that was streamed before the tool call so only the post card shows.
                                        if ($functionName === 'generate_social_post') {
                                            $output = '';
                                            $responsedText = '';
                                            echo PHP_EOL;
                                            echo "event: clear_content\n";
                                            echo "data: \n\n";
                                            $this->safeFlush();
                                        }

                                        $output .= $functionResponse;
                                        echo PHP_EOL;
                                        echo "event: data\n";
                                        echo 'data: ' . $functionResponse;
                                        echo "\n\n";
                                        $this->safeFlush();
                                        $hasTextOutput = true;
                                    }
                                }
                            }
                        }

                        // Send image search query to frontend IMMEDIATELY for parallel async fetch
                        // Frontend fetches images independently while text streams below
                        if ($pendingImageSearch) {
                            $searchArgs = json_decode($pendingImageSearch['argumentsString'], true);
                            $searchQuery = $searchArgs['query'] ?? '';
                            if ($searchQuery !== '') {
                                echo PHP_EOL;
                                echo "event: smart_image_search\n";
                                echo 'data: ' . json_encode([
                                    'query'      => $searchQuery,
                                    'message_id' => $main_message->id ?? null,
                                ]) . "\n\n";
                                $this->safeFlush();
                            }
                        }

                        // Now start text streaming
                        if (! $hasTextOutput) {
                            $fallbackOptions = [
                                'model'        => $model,
                                'stream'       => true,
                                'input'        => $nonSystemHistory,
                                'temperature'  => 1.0,
                            ];
                            // Pass tools with tool_choice=none so the model knows tools exist
                            // but won't try to call them again. Without this, the model leaks
                            // tool-routing tokens (e.g. "to=search_images …") as visible text
                            // because the instructions still describe the tool but no tools
                            // are advertised on this request.
                            $fallbackOptions['tools'] = $tools;
                            $fallbackOptions['tool_choice'] = 'none';
                            if ($baseInstructions !== '') {
                                $fallbackOptions['instructions'] = $baseInstructions;
                            }
                            if ($driver->enum()->isReasoningModel()) {
                                if (in_array($driver->enum(), [EntityEnum::GPT_5_PRO, EntityEnum::GPT_5_2_PRO])) {
                                    $effort = setting('openai_reasoning_models_effort', 'high');
                                    $fallbackOptions['reasoning']['effort'] = in_array($effort, ['medium', 'high', 'xhigh']) ? $effort : 'high';
                                } else {
                                    $fallbackOptions['reasoning']['effort'] = $this->resolveReasoningEffort($driver->enum());
                                }
                            }

                            $fallbackStream = OpenAI::responses()->createStreamed($fallbackOptions);

                            foreach ($fallbackStream as $fallbackResponse) {
                                if (! isset($fallbackResponse->event)) {
                                    continue;
                                }

                                if (connection_aborted()) {
                                    break;
                                }

                                if (isset($fallbackResponse->response->delta) && $fallbackResponse->event === 'response.output_text.delta') {
                                    $text = $fallbackResponse->response->delta;
                                    $messageFix = str_replace(["\r\n", "\r", "\n"], '<br/>', $text);
                                    $output .= $messageFix;
                                    $responsedText .= $text;
                                    $total_used_tokens += countWords($text);
                                    $this->emitStreamChunk($messageFix);
                                }
                            }
                        }
                    }
                    if ((isset($response->response->delta) && $response->event === 'response.output_text.delta')) {
                        $text = $response->response->delta;
                        $messageFix = str_replace(["\r\n", "\r", "\n"], '<br/>', $text);
                        $output .= $messageFix;
                        $responsedText .= $text;
                        $total_used_tokens += countWords($text);
                        $this->emitStreamChunk($messageFix);
                    }
                }
            }

            $this->saveStreamResponse($main_message, $chat, $responsedText, $output, $total_used_tokens, $driver);
        }, 200, [
            'Cache-Control'     => 'no-cache',
            'X-Accel-Buffering' => 'no',
            'Connection'        => 'keep-alive',
            'Content-Type'      => 'text/event-stream',
        ]);
    }

    /**
     * @throws Exception
     */
    private function openaiFileChat(string $chat_bot, $history, $main_message, $chat_type, $contain_images): ?StreamedResponse
    {
        $total_used_tokens = 0;
        $output = '';
        $responsedText = '';

        if ($contain_images) {
            $driver = $this->createDriver(EntityEnum::GPT_5_MINI);
        } else {
            $driver = $this->createDriver(EntityEnum::fromSlug($chat_bot));
        }
        $this->prepareStreamEnvironment();

        return response()->stream(function () use ($driver, $history, &$total_used_tokens, &$output, &$responsedText, $main_message, $contain_images) {
            $chat_id = $main_message->user_openai_chat_id;
            $chat = UserOpenaiChat::whereId($chat_id)->first();

            echo "event: message\n";
            echo 'data: ' . $main_message->id . "\n\n";

            // Emit "Used X Skill" as the very first event
            if (! empty($this->usedSkills)) {
                $this->emitUsedSkills();
            }

            if (! $driver->hasCreditBalance()) {
                echo PHP_EOL;
                echo "event: data\n";
                echo 'data: ' . __('You have no credits left. Please buy more credits to continue.');
                echo "\n\n";
                $this->safeFlush();
                echo "event: stop\n";
                echo 'data: [DONE]';
                echo "\n\n";
                $this->safeFlush();

                return null;
            }

            $model = $driver->enum()->value;
            $options = [
                'model'             => $model,
                'input'             => $history,
                'stream'            => true,
            ];

            if (! in_array($model, [EntityEnum::GPT_4_O_MINI_SEARCH_PREVIEW->value, EntityEnum::GPT_4_O_SEARCH_PREVIEW->value], true)) {
                $options['temperature'] = 1.0;
            }

            if ($contain_images) {
                $options['max_output_tokens'] = 2000;
                $options['model'] = EntityEnum::GPT_5_MINI->value;
            } else {
                $vectorId = $chat?->openai_vector_id ?? '';
                if ($vectorId === '') {
                    echo PHP_EOL;
                    echo "event: data\n";
                    echo 'data: ' . __('File search is not ready. Please try attaching the document again.');
                    echo "\n\n";
                    $this->safeFlush();
                    echo "event: stop\n";
                    echo 'data: [DONE]';
                    echo "\n\n";
                    $this->safeFlush();

                    return null;
                }
                $options['tools'] = [
                    [
                        'type'             => 'file_search',
                        'vector_store_ids' => [$vectorId],
                        'max_num_results'  => 1,
                    ],
                ];
            }
            ApiHelper::setOpenAiKey();
            $stream = OpenAI::responses()->createStreamed($options);
            foreach ($stream as $response) {
                if (($response->event === 'response.output_text.delta') && isset($response->response->delta)) {
                    $text = $response->response->delta;
                    $messageFix = str_replace(["\r\n", "\r", "\n"], '<br/>', $text);
                    $output .= $messageFix;
                    $responsedText .= $text;
                    $total_used_tokens += countWords($text);
                    if (connection_aborted()) {
                        break;
                    }
                    echo PHP_EOL;
                    echo "event: data\n";
                    echo 'data: ' . $messageFix;
                    echo "\n\n";
                    $this->safeFlush();
                }
            }

            $this->saveStreamResponse($main_message, $chat, $responsedText, $output, $total_used_tokens, $driver);
        }, 200, [
            'Cache-Control'     => 'no-cache',
            'X-Accel-Buffering' => 'no',
            'Connection'        => 'keep-alive',
            'Content-Type'      => 'text/event-stream',
        ]);
    }

    private function openaiOtherStream(Request $request, $chat_bot): ?StreamedResponse
    {
        $prompt = $request->get('prompt');
        $message_id = $request->get('message_id');
        $openai_id = $request->get('openai_id');
        $title = $request->get('title');

        $history[] = ['role' => 'user', 'content' => $prompt];
        $total_used_tokens = 0;
        $output = '';
        $responsedText = '';
        $user = Auth::user();
        $driver = $this->createDriver(EntityEnum::fromSlug($chat_bot));

        $this->prepareStreamEnvironment();

        return response()->stream(function () use ($user, $driver, $history, &$total_used_tokens, &$output, &$responsedText, $message_id, $title, $openai_id, $prompt) {
            $entry = UserOpenai::find($message_id);
            if (! $entry) {
                $entry = new UserOpenai;
                $entry->user_id = $user->id;
                $entry->input = $prompt;
                $entry->hash = str()->random(256);
                $entry->team_id = $user->team_id;
                $entry->slug = str()->random(7) . str($user?->fullName())->slug() . '-workbook';
                $entry->openai_id = $openai_id ?? 1;
            }

            echo "event: message\n";
            echo 'data: ' . $message_id . "\n\n";

            if (! $driver->hasCreditBalance()) {
                echo PHP_EOL;
                echo "event: data\n";
                echo 'data: ' . __('You have no credits left. Please buy more credits to continue.');
                echo "\n\n";
                $this->safeFlush();
                echo "event: stop\n";
                echo 'data: [DONE]';
                echo "\n\n";
                $this->safeFlush();

                return null;
            }

            $reasoningOptions = [];
            if ($driver->enum()->isReasoningModel()) {
                if (in_array($driver->enum(), [EntityEnum::GPT_5_PRO, EntityEnum::GPT_5_2_PRO])) {
                    $effort = setting('openai_reasoning_models_effort', 'high');
                    $reasoningOptions['reasoning']['effort'] = in_array($effort, ['medium', 'high', 'xhigh']) ? $effort : 'high';
                } else {
                    $reasoningOptions['reasoning']['effort'] = $this->resolveReasoningEffort($driver->enum());
                }
            }

            $stream = OpenAI::responses()->createStreamed([
                'model'             => $driver->enum()->value,
                'input'             => $history,
                ...$reasoningOptions,
                ...(! in_array($driver->enum()->value, [EntityEnum::GPT_4_O_MINI_SEARCH_PREVIEW->value, EntityEnum::GPT_4_O_SEARCH_PREVIEW->value], true) ? [
                    'temperature' => 1.0,
                ] : []),
                'stream'            => true,
            ]);

            foreach ($stream as $response) {
                if (($response->event === 'response.output_text.delta') && isset($response->response->delta)) {
                    $text = $response->response->delta;
                    $messageFix = str_replace(["\r\n", "\r", "\n"], '<br/>', $text);
                    $output .= $messageFix;
                    $responsedText .= $text;
                    $total_used_tokens += countWords($text);
                    if (connection_aborted()) {
                        break;
                    }
                    echo PHP_EOL;
                    echo "event: data\n";
                    echo 'data: ' . $messageFix;
                    echo "\n\n";
                    $this->safeFlush();
                }
            }
            echo "event: stop\n";
            echo 'data: [DONE]';
            echo "\n\n";
            $this->safeFlush();

            $this->saveOtherStreamResponse($entry, $title, $responsedText, $output, $total_used_tokens, $driver);
        }, 200, [
            'Cache-Control'     => 'no-cache',
            'X-Accel-Buffering' => 'no',
            'Connection'        => 'keep-alive',
            'Content-Type'      => 'text/event-stream',
        ]);
    }

    // AnthropicService Stream
    private function anthropicChatStream(string $chat_bot, $history, $main_message, $chat_type, $contain_images): ?StreamedResponse
    {
        $total_used_tokens = 0;
        $output = '';
        $responsedText = '';
        $client = app(AnthropicService::class);
        $driver = $this->createDriver(EntityEnum::fromSlug($chat_bot));

        $this->prepareStreamEnvironment();

        return response()->stream(function () use ($driver, $client, $history, &$total_used_tokens, &$output, &$responsedText, $main_message, $contain_images) {
            $chat_id = $main_message->user_openai_chat_id;
            $chat = UserOpenaiChat::whereId($chat_id)->first();

            echo "event: message\n";
            echo 'data: ' . $main_message->id . "\n\n";

            if (! $driver->hasCreditBalance()) {
                echo PHP_EOL;
                echo "event: data\n";
                echo 'data: ' . __('You have no credits left. Please buy more credits to continue.');
                echo "\n\n";
                $this->safeFlush();
                echo "event: stop\n";
                echo 'data: [DONE]';
                echo "\n\n";
                $this->safeFlush();

                return null;
            }

            if (! $contain_images) {
                $historyMessages = array_filter($history, function ($item) {
                    return $item['role'] !== 'system';
                });
                $systemParts = array_map(
                    fn ($item) => $item['content'] ?? '',
                    array_filter($history, fn ($item) => ($item['role'] ?? '') === 'system')
                );
                $system = implode("\n\n", array_filter($systemParts)) ?: null;

                if (setting('anthropic_default_model') === BedrockEngine::BEDROCK->value) {
                    $bedrockService = new BedrockRuntimeService([
                        'region'      => config('filesystems.disks.s3.region'),
                        'version'     => 'latest',
                        'credentials' => [
                            'key'    => config('filesystems.disks.s3.key'),
                            'secret' => config('filesystems.disks.s3.secret'),
                        ],
                    ]);
                    $responseBody = $bedrockService->invokeClaude($main_message->input);
                    $driver = $this->createDriver(EntityEnum::CLAUDE_2_1);
                    if (! $driver->hasCreditBalance()) {
                        echo PHP_EOL;
                        echo "event: data\n";
                        echo 'data: ' . __('You have no credits left. Please buy more credits to continue.');
                        echo "\n\n";
                        $this->safeFlush();
                        echo "event: stop\n";
                        echo 'data: [DONE]';
                        echo "\n\n";
                        $this->safeFlush();

                        return null;
                    }

                    if ($responseBody) {
                        $response = $this->anthropicBedrockResponse($responseBody);
                        $output = $response['output'];
                        $responsedText = $response['responsedText'];
                        $total_used_tokens = $response['total_used_tokens'];
                    }
                } else {
                    // Add skill tools if available (skip when manual skills are already injected)
                    $skillTools = [];
                    if (empty($this->usedSkills) && $this->autoSkills->isNotEmpty() && class_exists(SkillToolService::class)) {
                        $skillTools = SkillToolService::anthropicTools($this->autoSkills);
                    }

                    // Anthropic native web_search server tool when realtime + tool_calling.
                    // Uses the broad-compat 20250305 version (works on Claude 3+). Skip on
                    // Claude 2.x where the tool would 4xx.
                    if (request()?->input('realtime')
                        && setting('default_realtime') === 'tool_calling'
                        && ! str_starts_with($driver->enum()->value, 'claude-2')) {
                        $skillTools[] = [
                            'type' => 'web_search_20250305',
                            'name' => 'web_search',
                        ];
                    }

                    $data = $client->setStream(true)
                        ->setSystem($system)
                        ->setTools($skillTools)
                        ->setMessages(array_values($historyMessages))
                        ->stream()
                        ->body();

                    $toolUseBlock = null;
                    $toolUseInput = '';

                    foreach (explode("\n", $data) as $chunk) {
                        if (strlen($chunk) < 6) {
                            continue;
                        }
                        if (! Str::contains($chunk, 'data: ')) {
                            continue;
                        }
                        $chunk = str_replace('data: {', '{', $chunk);
                        $jsonData = json_decode($chunk, false, 512, JSON_THROW_ON_ERROR);

                        // Detect tool_use blocks
                        if (isset($jsonData->type) && $jsonData->type === 'content_block_start' && isset($jsonData->content_block->type) && $jsonData->content_block->type === 'tool_use') {
                            $toolUseBlock = $jsonData->content_block;
                            $toolUseInput = '';
                        } elseif (isset($jsonData->type) && $jsonData->type === 'content_block_delta' && isset($jsonData->delta->type) && $jsonData->delta->type === 'input_json_delta') {
                            $toolUseInput .= $jsonData->delta->partial_json ?? '';
                        } elseif (isset($jsonData->type) && $jsonData->type === 'content_block_stop' && $toolUseBlock !== null) {
                            // Tool use complete — handle skill call
                            $functionName = $toolUseBlock->name ?? '';
                            if (str_starts_with($functionName, 'use_skill_') && class_exists(SkillToolService::class)) {
                                $skillInstructions = SkillToolService::handleSkillCall($functionName, $this->autoSkills);
                                $skillMeta = SkillToolService::getSkillMeta($functionName, $this->autoSkills);
                                if ($skillMeta) {
                                    $this->usedSkills[] = $skillMeta;
                                }

                                if ($skillInstructions) {
                                    $this->emitUsedSkills();

                                    // Build follow-up with tool result

                                    $followUpMessages = array_values($historyMessages);
                                    $followUpMessages[] = [
                                        'role'    => 'assistant',
                                        'content' => [
                                            ['type' => 'tool_use', 'id' => $toolUseBlock->id, 'name' => $functionName, 'input' => json_decode($toolUseInput ?: '{}', true) ?: new stdClass],
                                        ],
                                    ];
                                    $followUpMessages[] = [
                                        'role'    => 'user',
                                        'content' => [
                                            ['type' => 'tool_result', 'tool_use_id' => $toolUseBlock->id, 'content' => $skillInstructions],
                                        ],
                                    ];

                                    $followUpData = $client->setStream(true)
                                        ->setSystem($system)
                                        ->setTools([])
                                        ->setMessages($followUpMessages)
                                        ->stream()
                                        ->body();

                                    foreach (explode("\n", $followUpData) as $followChunk) {
                                        if (strlen($followChunk) < 6) {
                                            continue;
                                        }
                                        if (! Str::contains($followChunk, 'data: ')) {
                                            continue;
                                        }
                                        $followChunk = str_replace('data: {', '{', $followChunk);
                                        $followJson = json_decode($followChunk, false, 512, JSON_THROW_ON_ERROR);
                                        if (isset($followJson->delta->text)) {
                                            $message = $followJson->delta->text;
                                            $messageFix = str_replace(["\r\n", "\r", "\n"], '<br/>', $message);
                                            $output .= $messageFix;
                                            $responsedText .= $message;
                                            $total_used_tokens += countWords($message);
                                            $this->emitStreamChunk($messageFix);
                                        }
                                        if (connection_aborted()) {
                                            break;
                                        }
                                    }
                                }
                            }
                            $toolUseBlock = null;
                        } elseif (isset($jsonData->delta->text)) {
                            $message = $jsonData->delta->text;
                            $messageFix = str_replace(["\r\n", "\r", "\n"], '<br/>', $message);
                            $output .= $messageFix;
                            $responsedText .= $message;
                            $total_used_tokens += countWords($message);
                            $this->emitStreamChunk($messageFix);
                        }

                        if (connection_aborted()) {
                            break;
                        }
                    }
                }
            } else {
                ApiHelper::setOpenAiKey();
                $driver = $this->createDriver(EntityEnum::GPT_5_MINI);
                $stream = OpenAI::responses()->createStreamed([
                    'model'                    => $driver->enum()->value,
                    'input'                    => $history,
                    'max_output_tokens'        => 2000,
                    'temperature'              => 1.0,
                    'stream'                   => true,
                ]);
                foreach ($stream as $response) {
                    if (($response->event === 'response.output_text.delta') && isset($response->response->delta)) {
                        $text = $response->response->delta;
                        $messageFix = str_replace(["\r\n", "\r", "\n"], '<br/>', $text);
                        $output .= $messageFix;
                        $responsedText .= $text;
                        $total_used_tokens += countWords($text);
                        if (connection_aborted()) {
                            break;
                        }
                        $this->emitStreamChunk($messageFix);
                    }
                }
            }

            $this->saveStreamResponse($main_message, $chat, $responsedText, $output, $total_used_tokens, $driver);
        }, 200, [
            'Cache-Control'     => 'no-cache',
            'X-Accel-Buffering' => 'no',
            'Connection'        => 'keep-alive',
            'Content-Type'      => 'text/event-stream',
        ]);
    }

    private function anthropicOtherStream(Request $request, $chat_bot): StreamedResponse
    {
        $prompt = $request->get('prompt');
        $message_id = $request->get('message_id');
        $openai_id = $request->get('openai_id');
        $title = $request->get('title');
        $driver = $this->createDriver(EntityEnum::fromSlug($chat_bot));
        $history[] = ['role' => 'user', 'content' => $prompt];
        $total_used_tokens = 0;
        $output = '';
        $responsedText = '';

        $this->prepareStreamEnvironment();

        return response()->stream(function () use ($driver, $history, &$total_used_tokens, &$output, &$responsedText, $message_id, $title, $openai_id, $prompt) {
            echo "event: message\n";
            echo 'data: ' . $message_id . "\n\n";

            if (! $driver->hasCreditBalance()) {
                echo PHP_EOL;
                echo "event: data\n";
                echo 'data: ' . __('You have no credits left. Please buy more credits to continue.');
                echo "\n\n";
                $this->safeFlush();
                echo "event: stop\n";
                echo 'data: [DONE]';
                echo "\n\n";
                $this->safeFlush();

                return null;
            }

            $user = Auth::user();
            $entry = UserOpenai::find($message_id);
            if (is_null($entry)) {
                $entry = new UserOpenai;
                $entry->user_id = $user?->id;
                $entry->input = $prompt;
                $entry->hash = str()->random(256);
                $entry->team_id = $user?->team_id;
                $entry->slug = str()->random(7) . str($user?->fullName())->slug() . '-workbook';
                $entry->openai_id = $openai_id ?? 1;
            }

            $client = app(AnthropicService::class);
            $historyMessages = array_filter($history, function ($item) {
                return $item['role'] !== 'system';
            });
            $system = Arr::first(array_filter($history, function ($item) {
                return $item['role'] === 'system';
            }));

            $system = data_get($system, 'content');
            if (setting('anthropic_default_model') === BedrockEngine::BEDROCK->value) {
                $bedrockService = new BedrockRuntimeService([
                    'region'      => config('filesystems.disks.s3.region'),
                    'version'     => 'latest',
                    'credentials' => [
                        'key'    => config('filesystems.disks.s3.key'),
                        'secret' => config('filesystems.disks.s3.secret'),
                    ],
                ]);
                $driver = $this->createDriver(EntityEnum::CLAUDE_2_1);
                if (! $driver->hasCreditBalance()) {
                    echo PHP_EOL;
                    echo "event: data\n";
                    echo 'data: ' . __('You have no credits left. Please buy more credits to continue.');
                    echo "\n\n";
                    $this->safeFlush();
                    echo "event: stop\n";
                    echo 'data: [DONE]';
                    echo "\n\n";
                    $this->safeFlush();

                    return null;
                }
                $responseBody = $bedrockService->invokeClaude($entry->input);
                if ($responseBody) {
                    $response = self::anthropicBedrockResponse($responseBody);
                    $output = $response['output'];
                    $responsedText = $response['responsedText'];
                    $total_used_tokens = $response['total_used_tokens'];
                    echo "event: stop\n";
                    echo 'data: [DONE]';
                    echo "\n\n";
                    $this->safeFlush();
                }

            } else {
                $data = $client->setStream(true)
                    ->setSystem($system)
                    ->setMessages(array_values($historyMessages))
                    ->stream()
                    ->body();
                foreach (explode("\n", $data) as $chunk) {
                    if (strlen($chunk) < 6) {
                        continue;
                    }
                    if (! Str::contains($chunk, 'data: ')) {
                        continue;
                    }
                    $chunk = str_replace('data: {', '{', $chunk);
                    if (isset(json_decode($chunk, false, 512, JSON_THROW_ON_ERROR)->delta->text)) {
                        $message = json_decode($chunk, false, 512, JSON_THROW_ON_ERROR)->delta->text;
                        $messageFix = str_replace(["\r\n", "\r", "\n"], '<br/>', $message);
                        $output .= $messageFix;
                        $responsedText .= $message;
                        $total_used_tokens += countWords($message);

                        echo PHP_EOL;
                        echo "event: data\n";
                        echo 'data: ' . $messageFix;
                        echo "\n\n";
                        $this->safeFlush();
                    }
                    if (connection_aborted()) {
                        break;
                    }
                }
                echo "event: stop\n";
                echo 'data: [DONE]';
                echo "\n\n";
                $this->safeFlush();

            }

            $this->saveOtherStreamResponse($entry, $title, $responsedText, $output, $total_used_tokens, $driver);
        }, 200, [
            'Cache-Control'     => 'no-cache',
            'X-Accel-Buffering' => 'no',
            'Connection'        => 'keep-alive',
            'Content-Type'      => 'text/event-stream',
        ]);
    }

    // GeminiService Stream
    private function geminiChatStream(string $chat_bot, $history, $main_message, $chat_type, $contain_images): StreamedResponse
    {
        $total_used_tokens = 0;
        $output = '';
        $responsedText = '';
        $newhistory = convertHistoryToGemini($history);
        $driver = $this->createDriver(EntityEnum::fromSlug($chat_bot));

        if ($contain_images) {
            // I will improve later
            $newhistory = $this->getLastMessageAndImage($newhistory);
            if (count($newhistory['parts']) === 1) {
                $newhistory['parts'][0] = [
                    'text' => $newhistory['parts'][0]['text'],
                ];

                $contain_images = false;
            }

            $newhistory = [$newhistory];
        }
        $this->prepareStreamEnvironment();

        return response()->stream(function () use ($driver, $newhistory, &$total_used_tokens, &$output, &$responsedText, $main_message, $contain_images, $chat_type) {

            $chat_id = $main_message->user_openai_chat_id;
            $chat = UserOpenaiChat::whereId($chat_id)->first();
            echo "event: message\n";
            echo 'data: ' . $main_message->id . "\n\n";

            if ($contain_images) {
                $driver = $this->createDriver(EntityEnum::GEMINI_1_5_FLASH);
            }

            if (! $driver->hasCreditBalance()) {
                echo PHP_EOL;
                echo "event: data\n";
                echo 'data: ' . __('You have no credits left. Please buy more credits to continue.');
                echo "\n\n";
                $this->safeFlush();
                echo "event: stop\n";
                echo 'data: [DONE]';
                echo "\n\n";
                $this->safeFlush();

                return null;
            }

            $client = app(GeminiService::class);

            // Add skill tools if available (skip when manual skills are already injected)
            $geminiSkillTools = [];
            if (empty($this->usedSkills) && $this->autoSkills->isNotEmpty() && class_exists(SkillToolService::class)) {
                $geminiSkillTools = SkillToolService::geminiTools($this->autoSkills);
            }

            // Add smart image tool if enabled (only for chatPro, skip for council sub-requests)
            if ($chat_type === 'chatPro'
                && ! $this->isCouncilSubRequest
                && MarketplaceHelper::isRegistered('ai-chat-pro-smart-image')
                && SmartImageService::isEnabled()) {
                $geminiSkillTools[] = SmartImageService::geminiToolDefinition();
            }

            // Gemini native Google Search grounding when realtime + tool_calling mode is on.
            // Gemini 3.x can combine grounding with function declarations; older Gemini
            // versions cannot, so for them we drop function tools while grounding is active.
            if (request()?->input('realtime')
                && setting('default_realtime') === 'tool_calling') {
                $modelValue = $driver->enum()->value;
                $isGemini3Plus = str_starts_with($modelValue, 'gemini-3');
                $isGemini15 = str_starts_with($modelValue, 'gemini-1.5');

                $groundingTool = $isGemini15
                    ? ['google_search_retrieval' => new stdClass]
                    : ['google_search' => new stdClass];

                if ($isGemini3Plus) {
                    $geminiSkillTools[] = $groundingTool;
                } else {
                    $geminiSkillTools = [$groundingTool];
                }
            }

            try {
                $response = $client
                    ->setHistory($newhistory)
                    ->setTools($geminiSkillTools)
                    ->streamGenerateContent($driver->enum()->value);
            } catch (Throwable $e) {
                Log::error('[StreamDebug] Gemini connection failed', ['error' => $e->getMessage()]);

                echo PHP_EOL;
                echo "event: data\n";
                echo 'data: ' . __('An error occurred while connecting to the AI service. Please try again.');
                echo "\n\n";
                $this->safeFlush();
                echo "event: stop\n";
                echo 'data: [DONE]';
                echo "\n\n";
                $this->safeFlush();

                return;
            }

            while (! $response->getBody()->eof()) {
                $line = trim($client->readLine($response->getBody()));

                if ($line === '' || $line === '[' || $line === ']' || $line === ',') {
                    continue;
                }

                try {
                    $decodedLine = json_decode($line, true, 512, JSON_THROW_ON_ERROR);
                } catch (JsonException $e) {
                    if (str_starts_with(trim($line), '{')) {
                        Log::error('JSON decoding error: ' . $e->getMessage());
                        Log::error('Offending line: ' . $line);
                    }

                    continue;
                }

                if (isset($decodedLine['error'])) {
                    $errorMessage = $decodedLine['error']['message'] ?? 'Unknown error occurred.';
                    $formattedMessage = '⚠️ ' . $errorMessage;

                    echo PHP_EOL;
                    echo "event: data\n";
                    echo 'data: ' . $formattedMessage;
                    echo "\n\n";
                    $this->safeFlush();

                    break;
                }

                if (! isset($decodedLine['candidates'])) {
                    continue;
                }

                foreach ($decodedLine['candidates'] as $candidate) {
                    $parts = $candidate['content']['parts'] ?? [];
                    foreach ($parts as $part) {
                        // Handle function calls from Gemini
                        if (isset($part['functionCall'])) {
                            $functionName = $part['functionCall']['name'] ?? '';
                            $functionId = $part['functionCall']['id'] ?? '';

                            // Handle search_images — emit SSE for frontend async fetch, then do follow-up for text
                            if ($functionName === 'search_images') {
                                $searchArgs = $part['functionCall']['args'] ?? [];
                                $searchQuery = $searchArgs['query'] ?? '';

                                if ($searchQuery !== '') {
                                    // Emit SSE events for frontend
                                    echo PHP_EOL;
                                    echo "event: function_call\n";
                                    echo 'data: search_images' . "\n\n";
                                    $this->safeFlush();

                                    echo PHP_EOL;
                                    echo "event: smart_image_search\n";
                                    echo 'data: ' . json_encode([
                                        'query'      => $searchQuery,
                                        'message_id' => $main_message->id ?? null,
                                    ]) . "\n\n";
                                    $this->safeFlush();
                                }

                                // Build follow-up history with function response so Gemini generates text
                                // Ensure args is an object (not array) for Gemini API compatibility
                                $imgFunctionCallData = $part['functionCall'];
                                $imgFunctionCallData['args'] = ! empty($imgFunctionCallData['args']) && is_array($imgFunctionCallData['args'])
                                    ? (object) $imgFunctionCallData['args']
                                    : new stdClass;
                                // Preserve the entire part (includes thoughtSignature required by Gemini 3)
                                $modelPart = ['functionCall' => $imgFunctionCallData];
                                if (isset($part['thoughtSignature'])) {
                                    $modelPart['thoughtSignature'] = $part['thoughtSignature'];
                                }

                                $followUpHistory = $newhistory;
                                $followUpHistory[] = [
                                    'role'  => 'model',
                                    'parts' => [$modelPart],
                                ];
                                $followUpHistory[] = [
                                    'role'  => 'user',
                                    'parts' => [['functionResponse' => [
                                        'name'     => $functionName,
                                        'id'       => $functionId,
                                        'response' => ['status' => 'success', 'message' => 'Images are being loaded by the frontend. Now write your text response about the topic.'],
                                    ]]],
                                ];

                                try {
                                    $followUpClient = app(GeminiService::class);
                                    $followUpResponse = $followUpClient
                                        ->setHistory($followUpHistory)
                                        ->setTools([])
                                        ->streamGenerateContent($driver->enum()->value);

                                    while (! $followUpResponse->getBody()->eof()) {
                                        $followLine = trim($followUpClient->readLine($followUpResponse->getBody()));
                                        if ($followLine === '' || $followLine === '[' || $followLine === ']' || $followLine === ',') {
                                            continue;
                                        }

                                        try {
                                            $followDecoded = json_decode($followLine, true, 512, JSON_THROW_ON_ERROR);
                                        } catch (JsonException) {
                                            continue;
                                        }

                                        if (isset($followDecoded['error'])) {
                                            break;
                                        }

                                        if (isset($followDecoded['candidates'])) {
                                            foreach ($followDecoded['candidates'] as $followCandidate) {
                                                $followText = $followCandidate['content']['parts'][0]['text'] ?? '';
                                                if ($followText !== '') {
                                                    $messageFix = str_replace(["\r\n", "\r", "\n"], '<br/>', $followText);
                                                    $output .= $messageFix;
                                                    $responsedText .= $followText;
                                                    $total_used_tokens += countWords($followText);
                                                    $this->emitStreamChunk($messageFix);
                                                }
                                            }
                                        }
                                        if (connection_aborted()) {
                                            break;
                                        }
                                    }
                                } catch (Throwable) {
                                    // Follow-up failed — images still load, text is skipped
                                }

                                continue;
                            }

                            if (str_starts_with($functionName, 'use_skill_') && class_exists(SkillToolService::class)) {
                                $skillInstructions = SkillToolService::handleSkillCall($functionName, $this->autoSkills);
                                $skillMeta = SkillToolService::getSkillMeta($functionName, $this->autoSkills);
                                if ($skillMeta) {
                                    $this->usedSkills[] = $skillMeta;
                                }

                                if ($skillInstructions) {
                                    $this->emitUsedSkills();

                                    // Build follow-up history with function response
                                    // Preserve thoughtSignature required by Gemini 3

                                    // Ensure args is an object (not array) for Gemini API compatibility
                                    $functionCallData = $part['functionCall'];
                                    $functionCallData['args'] = ! empty($functionCallData['args']) && is_array($functionCallData['args'])
                                        ? (object) $functionCallData['args']
                                        : new stdClass;
                                    $skillModelPart = ['functionCall' => $functionCallData];
                                    if (isset($part['thoughtSignature'])) {
                                        $skillModelPart['thoughtSignature'] = $part['thoughtSignature'];
                                    }

                                    $followUpHistory = $newhistory;
                                    $followUpHistory[] = [
                                        'role'  => 'model',
                                        'parts' => [$skillModelPart],
                                    ];
                                    $followUpHistory[] = [
                                        'role'  => 'user',
                                        'parts' => [['functionResponse' => [
                                            'name'     => $functionName,
                                            'id'       => $functionId,
                                            'response' => ['instructions' => $skillInstructions],
                                        ]]],
                                    ];

                                    $followUpClient = app(GeminiService::class);
                                    $followUpResponse = $followUpClient
                                        ->setHistory($followUpHistory)
                                        ->setTools([])
                                        ->streamGenerateContent($driver->enum()->value);

                                    while (! $followUpResponse->getBody()->eof()) {
                                        $followLine = trim($followUpClient->readLine($followUpResponse->getBody()));
                                        if ($followLine === '' || $followLine === '[' || $followLine === ']' || $followLine === ',') {
                                            continue;
                                        }

                                        try {
                                            $followDecoded = json_decode($followLine, true, 512, JSON_THROW_ON_ERROR);
                                        } catch (JsonException) {
                                            continue;
                                        }

                                        if (isset($followDecoded['error'])) {
                                            break;
                                        }

                                        if (isset($followDecoded['candidates'])) {
                                            foreach ($followDecoded['candidates'] as $followCandidate) {
                                                $followParts = $followCandidate['content']['parts'] ?? [];
                                                foreach ($followParts as $followPart) {
                                                    $followText = $followPart['text'] ?? '';
                                                    if ($followText !== '') {
                                                        $messageFix = str_replace(["\r\n", "\r", "\n"], '<br/>', $followText);
                                                        $output .= $messageFix;
                                                        $responsedText .= $followText;
                                                        $total_used_tokens += countWords($followText);
                                                        $this->emitStreamChunk($messageFix);
                                                    }
                                                }
                                            }
                                        }

                                        if (connection_aborted()) {
                                            break;
                                        }
                                    }
                                }
                            }

                            continue;
                        }

                        // Handle regular text
                        $text = $part['text'] ?? '';
                        if ($text === '') {
                            continue;
                        }
                        $messageFix = str_replace(["\r\n", "\r", "\n"], '<br/>', $text);
                        $output .= $messageFix;
                        $responsedText .= $text;
                        $total_used_tokens += countWords($text);
                        $this->emitStreamChunk($messageFix);
                    }

                    if (connection_aborted()) {
                        break;
                    }
                }
            }

            $this->saveStreamResponse($main_message, $chat, $responsedText, $output, $total_used_tokens, $driver);
        }, 200, [
            'Cache-Control'     => 'no-cache',
            'X-Accel-Buffering' => 'no',
            'Connection'        => 'keep-alive',
            'Content-Type'      => 'text/event-stream',
        ]);
    }

    public function getLastMessageAndImage($newhistory)
    {
        return Arr::last($newhistory);
    }

    private function geminiOtherStream(Request $request, string $chat_bot): StreamedResponse
    {
        $driver = $this->createDriver(EntityEnum::fromSlug($chat_bot));
        $prompt = $request->get('prompt');
        $message_id = $request->get('message_id');
        $openai_id = $request->get('openai_id');
        $title = $request->get('title');

        $history[] = [
            'parts' => [
                [
                    'text' => $prompt,
                ],
            ],
            'role' => 'user',
        ];

        $total_used_tokens = 0;
        $output = '';
        $responsedText = '';
        $this->prepareStreamEnvironment();

        return response()->stream(function () use ($driver, $history, &$total_used_tokens, &$output, &$responsedText, $message_id, $title, $openai_id, $prompt) {
            $user = Auth::user();
            $entry = UserOpenai::find($message_id);
            if (is_null($entry)) {
                $entry = new UserOpenai;
                $entry->user_id = $user->id;
                $entry->input = $prompt;
                $entry->hash = str()->random(256);
                $entry->team_id = $user->team_id;
                $entry->slug = str()->random(7) . str($user?->fullName())->slug() . '-workbook';
                $entry->openai_id = $openai_id ?? 1;
            }

            echo "event: message\n";
            echo 'data: ' . $message_id . "\n\n";

            $client = app(GeminiService::class);
            $response = $client
                ->setHistory($history)
                ->streamGenerateContent($driver->enum()->value);

            while (! $response->getBody()->eof()) {

                $line = trim($client->readLine($response->getBody()));

                if ($line === '' || $line === '[' || $line === ']' || $line === ',') {
                    continue;
                }

                try {
                    $decodedLine = json_decode($line, true, 512, JSON_THROW_ON_ERROR);

                    if ($decodedLine === null || ! isset($decodedLine['candidates'])) {
                        continue;
                    }
                } catch (JsonException $e) {
                    if (str_starts_with($line, '{')) {
                        Log::error('JSON decoding error: ' . $e->getMessage());
                        Log::error('Offending line: ' . $line);
                    }

                    continue;
                }
                if ($decodedLine === null || ! isset($decodedLine['candidates'])) {
                    continue;
                }

                foreach ($decodedLine['candidates'] as $candidate) {
                    $text = $candidate['content']['parts'][0]['text'];
                    $messageFix = str_replace(["\r\n", "\r", "\n"], '<br/>', $text);
                    $output .= $messageFix;
                    $responsedText .= $text;
                    $total_used_tokens += countWords($text);
                    if (connection_aborted()) {
                        break;
                    }
                    echo PHP_EOL;
                    echo "event: data\n";
                    echo 'data: ' . $messageFix;
                    echo "\n\n";
                    $this->safeFlush();
                }
            }

            echo "event: stop\n";
            echo 'data: [DONE]';
            echo "\n\n";
            $this->safeFlush();

            $this->saveOtherStreamResponse($entry, $title, $responsedText, $output, $total_used_tokens, $driver);
        }, 200, [
            'Cache-Control'     => 'no-cache',
            'X-Accel-Buffering' => 'no',
            'Connection'        => 'keep-alive',
            'Content-Type'      => 'text/event-stream',
        ]);
    }

    private function anthropicBedrockResponse($responseBody): array
    {
        $completion = $responseBody['completion'];
        $parts = explode(':', $completion, 2);
        if (isset($parts[1])) {
            $completion = trim($parts[1]);
        }

        $words = explode(' ', $completion);
        $output = $completion;
        $responsedText = $completion;
        $total_used_tokens = count($words);
        foreach ($words as $word) {
            $messageFix = str_replace(["\r\n", "\r", "\n"], '<br/>', $word) . ' ';

            echo PHP_EOL;
            echo "event: data\n";
            echo 'data: ' . $messageFix;
            echo "\n\n";
            $this->safeFlush();

            if (connection_aborted()) {
                break;
            }
        }

        return [
            'output'            => $output,
            'responsedText'     => $responsedText,
            'total_used_tokens' => $total_used_tokens,
        ];
    }

    /**
     * @return array{0: string, 1: array<string, mixed>}
     */
    private function injectTitleInstruction(array $history): array
    {
        $instruction = 'RESPONSE FORMAT — your reply has TWO parts:'
            . "\n" . '1. FIRST LINE: one raw JSON object on a single line: {"title":"short descriptive title based on the user message"}'
            . "\n" . '2. BLANK LINE, then your FULL ANSWER to the user. The answer is the main content of your reply and MUST ALWAYS be written — it is never optional and never empty.'
            . "\n"
            . "\n" . 'The title JSON is metadata only; it does NOT replace the answer. Even for simple, casual, or ambiguous messages, you must still write a normal helpful answer after the title line.'
            . "\n" . 'Do NOT wrap the JSON in code fences, backticks, or markdown. Do not wrap the answer in JSON. Only the first line is raw JSON.';

        foreach ($history as $index => $message) {
            if (in_array($message['role'], ['system', 'user']) && is_string($message['content'] ?? null)) {
                $history[$index]['content'] = $message['content'] . "\n\n" . $instruction;

                break;
            }
        }

        return $history;
    }

    /**
     * @return array{0: ?string, 1: string}
     */
    private function parseTitleFromBuffer(string $buffer): array
    {
        $cleaned = preg_replace('/^```[\w]*\s*\n?/', '', trim($buffer));
        $cleaned = preg_replace('/\n```\s*$/', '', $cleaned);
        $cleaned = trim($cleaned);

        if (preg_match('/^\s*(\{"title"\s*:\s*"[^"]*"[^}]*\})/', $cleaned, $matches)) {
            $decoded = json_decode($matches[1], true);
            if (json_last_error() === JSON_ERROR_NONE && ! empty($decoded['title']) && is_string($decoded['title'])) {
                $remainder = trim(substr($cleaned, strlen($matches[0])));

                return [$decoded['title'], $remainder];
            }
        }

        return [null, $buffer];
    }

    private function emitChatTitle(?UserOpenaiChat $chat, string $title): void
    {
        if (! $chat || $title === '') {
            return;
        }

        $chat->title = $title;
        $chat->save();

        echo PHP_EOL;
        echo "event: title\n";
        echo 'data: ' . json_encode(['chat_id' => $chat->id, 'title' => $title]);
        echo "\n\n";
        $this->safeFlush();
    }

    private function injectSmartImageInstruction(array $history): array
    {
        $instruction = SmartImageService::systemPromptAddition();

        foreach ($history as $index => $message) {
            if (in_array($message['role'], ['system', 'user']) && is_string($message['content'] ?? null)) {
                $history[$index]['content'] = $instruction . "\n\n" . $message['content'];

                break;
            }
        }

        return $history;
    }

    private function injectEntityHighlightInstruction(array $history, bool $withSuggestions = false): array
    {
        $instruction = EntityHighlightService::systemPromptAddition($withSuggestions);

        foreach ($history as $index => $message) {
            if (in_array($message['role'], ['system', 'user']) && is_string($message['content'] ?? null)) {
                $history[$index]['content'] = $instruction . "\n\n" . $message['content'];

                break;
            }
        }

        return $history;
    }

    /**
     * Parse, persist, and strip entity highlight annotations from the stream output.
     */
    private function persistEntityHighlights(string $output, int $messageId, int $userId): string
    {
        if (! MarketplaceHelper::isRegistered('ai-chat-pro-entity-highlight')) {
            return $output;
        }

        try {
            $entities = EntityHighlightService::parseAnnotations($output);

            if (! empty($entities)) {
                EntityHighlightService::saveForMessage(
                    $userId,
                    $messageId,
                    $entities
                );

                // Emit entity highlights as SSE event so frontend can apply them
                echo PHP_EOL;
                echo "event: entity_highlights\n";
                echo 'data: ' . json_encode($entities);
                echo "\n\n";
                $this->safeFlush();
            }
        } catch (Throwable $e) {
            // Don't let entity highlight failures break stream completion
        }

        try {
            return EntityHighlightService::stripAnnotationBlock($output);
        } catch (Throwable $e) {
            return $output;
        }
    }

    private function injectSuggestionsInstruction(array $history): array
    {
        $instruction = 'FOLLOW-UP SUGGESTIONS METADATA:'
            . "\n" . 'Your PRIMARY task is to write a full, helpful answer to the user. The answer is the main content and MUST ALWAYS be written — never skip it, never leave it empty, never replace it with the JSON block below.'
            . "\n"
            . "\n" . 'AFTER you have finished writing your complete answer, append a blank line then ONE raw JSON object on a single line:'
            . "\n" . '{"suggestions":["2-6 word item","2-6 word item","2-6 word item","2-6 word item"]}'
            . "\n" . 'The 4 items should be diverse, actionable follow-up prompts related to the topic.'
            . "\n" . 'If the final sentences contains any suggestions, make sure to include them as well.'
            . "\n"
            . "\n" . 'STRICT FORMAT RULES:'
            . "\n" . '- The JSON block comes LAST, after the full answer. It is metadata, not a replacement for the answer.'
            . "\n" . '- The JSON MUST start with {"suggestions": and end with }.'
            . "\n" . '- The line immediately before the JSON must be a BLANK LINE — nothing else.'
            . "\n" . '- Do NOT write ANY preamble, heading, label, or introductory text before the JSON. Forbidden examples (never output these): "Suggestions for next steps:", "Here are some suggestions:", "Follow-ups:", "Next steps:", "You might also try:", "Related:", or any similar phrase.'
            . "\n" . '- Do NOT output a bare array [...], bullet points, or any other format — only the raw JSON object.'
            . "\n" . '- Do NOT wrap in code fences, backticks.'
            . "\n" . '- Do NOT mention, reference, announce, or acknowledge the suggestions/JSON/metadata anywhere in your visible answer. The answer must read as if the JSON does not exist — the JSON is handled invisibly by the system.'
            . "\n" . '- Do NOT mention the title JSON either. Never write phrases like "Title:", "Here is the title", or refer to it in any way.'
            . "\n"
            . "\n" . 'Skip ONLY the JSON block (still write the answer!) if the last user message is trivial (greetings, ok, thanks, yes/no, bye, short reactions).';

        foreach ($history as $index => $message) {
            if (in_array($message['role'], ['system', 'user']) && is_string($message['content'] ?? null)) {
                $history[$index]['content'] = $message['content'] . "\n\n" . $instruction;

                break;
            }
        }

        return $history;
    }

    /**
     * Strip suggestions JSON block from saved output/response so it doesn't appear on reload or copy.
     */
    private function stripSuggestionsJson(string $text): string
    {
        // Handle both raw newlines and <br/> line separators — standard format {"suggestions":[...]}
        $text = preg_replace('/\s*(?:<br\s*\/?>|\n)*\s*\{[\s\n]*(?:<br\s*\/?>|\n)*\s*"suggestions"\s*:\s*\[[\s\S]*?\]\s*(?:<br\s*\/?>|\n)*\s*\}\s*$/i', '', $text);

        // Also strip mid-text occurrences (in case entity block comes after)
        $text = preg_replace('/\s*(?:<br\s*\/?>|\n)*\s*\{[\s\n]*(?:<br\s*\/?>|\n)*\s*"suggestions"\s*:\s*\[[\s\S]*?\]\s*(?:<br\s*\/?>|\n)*\s*\}/i', '', $text);

        // Strip malformed suggestions JSON — model sometimes outputs {"item1","item2"} without "suggestions": key
        $text = preg_replace('/\s*(?:<br\s*\/?>|\n)*\s*\{\s*"[^"]{2,60}"\s*,\s*"[^"]{2,60}"[\s\S]*?\}\s*$/i', '', $text);

        // Strip malformed entity-meta tail where the model dropped the ':::' prefix and
        // wrote a bare `meta], "suggestions":[...]}` fragment at the end of the response.
        // Only matches the full malformed shape (meta + JSON terminator + "suggestions":
        // key + closing }) so legitimate uses of the word "meta" in chat are preserved.
        $text = preg_replace('/\s*(?:<br\s*\/?>|\n)*\s*(?:\*{3,}|-{3,}|_{3,})?\s*(?:<br\s*\/?>|\n)*\s*\bmeta\b\s*(?:<br\s*\/?>|\n)*\s*[\[\{\],}][\s\S]*?"suggestions"\s*:\s*\[[\s\S]*?\]\s*\}?\s*$/i', '', $text);

        // Strip a trailing markdown horizontal rule (***, ---, ___) that the model
        // sometimes emits as a visual separator before the (now-stripped) JSON tail.
        $text = preg_replace('/\s*(?:<br\s*\/?>|\n)*\s*(?:\*{3,}|-{3,}|_{3,})\s*(?:<br\s*\/?>|\n)*\s*$/', '', $text);

        // Strip a bare "suggestions":[...]} tail (no leading {) when present at end.
        $text = preg_replace('/\s*(?:<br\s*\/?>|\n)*\s*[\[\{,]?\s*"suggestions"\s*:\s*\[[\s\S]*?\]\s*\}?\s*$/i', '', $text);

        // Strip bare "entity-highlights" text leaked from AI
        $text = preg_replace('/\s*(?:<br\s*\/?>|\n)*\s*entity[- ]?highlights?\s*$/i', '', $text);

        return rtrim($text);
    }

    /**
     * @return array{0: ?array<string, mixed>, 1: string}
     */
    private function parseSuggestionsFromBuffer(string $buffer): array
    {
        $cleaned = preg_replace('/\s*```\s*$/', '', rtrim($buffer));

        if (preg_match('/(\{"suggestions"\s*:\s*\[.*\]\s*\})\s*$/s', $cleaned, $matches)) {
            $decoded = json_decode($matches[1], true);
            if (
                json_last_error() === JSON_ERROR_NONE &&
                ! empty($decoded['suggestions']) &&
                is_array($decoded['suggestions'])
            ) {
                $beforeJson = rtrim(substr($cleaned, 0, strrpos($cleaned, $matches[1])));

                return [$decoded, $beforeJson];
            }
        }

        return [null, $buffer];
    }

    private function emitSuggestions(array $payload, $main_message): void
    {
        echo PHP_EOL;
        echo "event: suggestions\n";
        echo 'data: ' . json_encode($payload);
        echo "\n\n";
        $this->safeFlush();

        $main_message->suggestions_response = $payload;
        $main_message->save();
    }

    public function saveStreamResponse($main_message, $chat, $responsedText, $output, $total_used_tokens, $driver): void
    {
        if ($this->isFirstMessage && $chat) {
            $bufferedContent = $this->titleBuffer;
            $this->titleBuffer = '';

            // Try to extract a title JSON from the full response text
            $fullText = ($bufferedContent !== '' ? strip_tags(str_replace('<br/>', "\n", $bufferedContent)) : '') . $responsedText;
            [$title, $cleanText] = $this->parseTitleFromBuffer($fullText);

            if ($title !== null) {
                $responsedText = $cleanText;
                $output = (string) preg_replace('/^\s*(```[\w]*\s*(<br\s*\/?>|\s)*)?\{"title"\s*:\s*"[^"]*"[^}]*\}\s*(```\s*)?(<br\s*\/?>|\s)*/i', '', $output);
                $this->emitChatTitle($chat, $title);

                // When the streaming title regex (line 152) never matched (e.g. title JSON
                // split across chunks), titleBuffer holds the entire response. The title was
                // only recovered here at end-of-stream — so the answer body was never emitted
                // as a data event and the UI bubble is empty until reload. Flush the cleaned
                // remainder now. Strip meta/suggestions tails first so we don't leak JSON.
                if ($bufferedContent !== '' && ! $this->entityBlockSuppressed) {
                    $remainder = (string) preg_replace('/^\s*(```[\w]*\s*(<br\s*\/?>|\s)*)?\{"title"\s*:\s*"[^"]*"[^}]*\}\s*(```\s*)?(<br\s*\/?>|\s)*/i', '', $bufferedContent);
                    $remainder = $this->stripSuggestionsJson($remainder);
                    $remainder = (string) preg_replace('/\s*(<br\s*\/?>|\n)*\s*:::\s*(meta|entity[- ]?highlights?)[\s\S]*$/i', '', $remainder);
                    $remainder = rtrim($remainder);
                    if ($remainder !== '') {
                        echo PHP_EOL;
                        echo "event: data\n";
                        echo 'data: ' . $remainder;
                        echo "\n\n";
                        $this->safeFlush();
                    }
                }
            } elseif ($bufferedContent !== '') {
                // No title found — flush the buffered content as data so it appears in the bubble
                echo PHP_EOL;
                echo "event: data\n";
                echo 'data: ' . $bufferedContent;
                echo "\n\n";
                $this->safeFlush();
            }

            $this->isFirstMessage = false;
        }

        // If still in entity block suppression, discard the buffer entirely (it's all entity/suggestions data)
        if ($this->entityBlockSuppressed) {
            $this->suggestionsBuffer = '';
            $this->entityBlockSuppressed = false;
        }

        // Flush any remaining suggestions buffer content (strip headings/JSON/entity blocks, emit clean text)
        if ($this->suggestionsBuffer !== '') {
            $buffered = (string) preg_replace('/\s*(<br\s*\/?>)*\s*(?:\*{3,}|-{3,}|_{3,})?\s*(<br\s*\/?>)*\s*,*\s*(?:\bmeta\b\s*(<br\s*\/?>|\n)*\s*)?(```[\w]*\s*(<br\s*\/?>|\s)*)?\[?\s*\{[\s\S]*\}\s*\]?\s*,*\s*(```\s*)?(<br\s*\/?>|\s)*$/si', '', $this->suggestionsBuffer);
            // Strip trailing markdown horizontal rule with nothing after it.
            $buffered = (string) preg_replace('/\s*(<br\s*\/?>|\n)*\s*(?:\*{3,}|-{3,}|_{3,})\s*(<br\s*\/?>|\n)*\s*$/', '', $buffered);
            // Strip :::meta / :::entity-highlights blocks or bare label text
            $buffered = (string) preg_replace('/\s*(<br\s*\/?>|\n)*\s*:::\s*(meta|entity[- ]?highlights?)[\s\S]*$/i', '', $buffered);
            $buffered = (string) preg_replace('/\s*(<br\s*\/?>|\n)*\s*:::?\s*(<br\s*\/?>|\n)*\s*(entity[- ]?highlights?\s*(<br\s*\/?>|\n)*\s*)?(\[?\{[\s\S]*"text"\s*:[\s\S]*)?$/i', '', $buffered);
            $buffered = (string) preg_replace('/\s*(<br\s*\/?>|\n)*\s*entity[- ]?highlights?\s*$/i', '', $buffered);
            $buffered = rtrim($buffered);
            if ($buffered !== '') {
                echo PHP_EOL;
                echo "event: data\n";
                echo 'data: ' . $buffered;
                echo "\n\n";
                $this->safeFlush();
            }
            $this->suggestionsBuffer = '';
        }

        if ($this->shouldGenerateSuggestions) {
            $suggestionsPayload = null;

            // When entity highlight is active, suggestions come from inside the :::meta block.
            if ($this->entityHighlightsEnabled) {
                $metaSuggestions = EntityHighlightService::parseSuggestions($output);
                if (is_array($metaSuggestions) && ! empty($metaSuggestions)) {
                    $suggestionsPayload = ['suggestions' => $metaSuggestions];
                }
            }

            // Fallback: standalone {"suggestions":[...]} JSON at the end of the response.
            if ($suggestionsPayload === null) {
                [$parsed, $cleanText] = $this->parseSuggestionsFromBuffer($responsedText);
                if ($parsed !== null) {
                    $responsedText = $cleanText;
                    $output = (string) preg_replace('/\s*(<br\s*\/?>)*\s*(```[\w]*\s*(<br\s*\/?>|\s)*)?\{"suggestions"\s*:\s*\[.*\]\s*\}\s*(```\s*)?(<br\s*\/?>|\s)*$/si', '', $output);
                    $suggestionsPayload = $parsed;
                }
            }

            if ($suggestionsPayload !== null) {
                $this->emitSuggestions($suggestionsPayload, $main_message);
            }

            $this->shouldGenerateSuggestions = false;
        }

        // Emit used skills event (for manual skills that didn't go through function calling)
        $this->emitUsedSkills();

        // Persist and emit entity highlights BEFORE [DONE] so frontend receives them
        if (! $this->tempChatActive && ! $this->isCouncilSubRequest && MarketplaceHelper::isRegistered('ai-chat-pro-entity-highlight')) {
            $output = $this->persistEntityHighlights($output, (int) $main_message->id, (int) $main_message->user_id);
            $responsedText = EntityHighlightService::stripAnnotationBlock($responsedText);
            $this->entityHighlightsEnabled = false;
        }

        echo "event: stop\n";
        echo 'data: [DONE]';
        echo "\n\n";
        $this->safeFlush();

        if ($this->tempChatActive) {
            if ($chat) {
                $this->addToTempHistory($chat->id, [
                    'role'    => 'assistant',
                    'content' => $responsedText,
                ]);

                $chat->messages()->delete();
            }

            return;
        }

        // Strip suggestions JSON from output before saving
        $output = $this->stripSuggestionsJson($output);
        $responsedText = $this->stripSuggestionsJson($responsedText);

        // Strip leaked function-call text (e.g. search_images({"query":"..."})) so it doesn't pollute chat history
        $output = preg_replace('/search_images\s*\(\s*\{[^}]*\}\s*\)/', '', $output);
        $responsedText = preg_replace('/search_images\s*\(\s*\{[^}]*\}\s*\)/', '', $responsedText);

        // Persist smart images to database if present in output
        $this->persistSmartImages($main_message, $output);

        $main_message->response = $responsedText;
        $main_message->output = $output;
        $main_message->credits = $total_used_tokens;
        $main_message->words = $total_used_tokens;

        if (! empty($this->usedSkills)) {
            $main_message->used_skills = $this->usedSkills;
        }

        $main_message->save();

        if ($chat) {
            $chat->total_credits += $total_used_tokens;
            $chat->save();
        }

        $driver?->input($responsedText)->calculateCredit()->decreaseCredit();
        Usage::getSingle()->updateWordCounts($driver?->calculate());
    }

    /**
     * Extract and persist smart images from the stream output to the database.
     */
    private function persistSmartImages($main_message, string $output): void
    {
        if (! MarketplaceHelper::isRegistered('ai-chat-pro-smart-image')) {
            return;
        }

        if (preg_match('/:::smart-images\s*\n(.*?)\n\s*:::/s', $output, $matches)) {
            try {
                $images = json_decode($matches[1], true, 512, JSON_THROW_ON_ERROR);
                if (! empty($images) && is_array($images)) {
                    SmartImageService::saveForMessage(
                        $main_message->id,
                        $main_message->user_id,
                        $images[0]['title'] ?? 'image search',
                        $images
                    );
                }
            } catch (Throwable $e) {
                // Silently fail — images will still display from the output
            }
        }
    }

    private function saveOtherStreamResponse($entry, $title, $responsedText, $output, $total_used_tokens, $driver): void
    {
        $entry->title = $title ?: null;
        $entry->credits = $total_used_tokens;
        $entry->words = $total_used_tokens;
        $entry->response = $responsedText;
        $entry->output = $output;
        $entry->save();

        $driver->input($responsedText)->calculateCredit()->decreaseCredit();
        Usage::getSingle()->updateWordCounts($driver->calculate());
    }

    /**
     * Get temporary chat history from session
     */
    private function getTempChatHistory(string $chatId): array
    {
        if (! $this->tempChatActive) {
            return [];
        }

        if (! auth()->check()) {
            return [];
        }

        $sessionKey = auth()->user()->id . '_' . $this->tempChatSessionKey . $chatId;

        return Session::get($sessionKey, []);
    }

    /**
     * Store temporary chat history in session
     */
    private function storeTempChatHistory(string $chatId, array $history): void
    {
        if (! $this->tempChatActive) {

            return;
        }

        if (! auth()->check()) {
            return;
        }

        // Limit history to last 20 messages to prevent session bloat
        $limitedHistory = array_slice($history, -20);

        $sessionKey = auth()->user()->id . '_' . $this->tempChatSessionKey . $chatId;

        Session::put($sessionKey, $limitedHistory);
    }

    /**
     * Add message to temporary chat history
     */
    private function addToTempHistory(string $chatId, array $message): void
    {
        if (! $this->tempChatActive) {
            return;
        }

        if (! auth()->check()) {
            return;
        }

        $history = $this->getTempChatHistory($chatId);
        $history[] = $message;

        $this->storeTempChatHistory($chatId, $history);
    }

    /**
     * Remove duplicate messages from history
     */
    private function removeDuplicateMessages(array $history): array
    {
        $seen = [];
        $filtered = [];

        foreach ($history as $message) {
            $key = $message['role'] . '|' . (is_array($message['content']) ? json_encode($message['content']) : $message['content']);

            if (! in_array($key, $seen)) {
                $seen[] = $key;
                $filtered[] = $message;
            }
        }

        return $filtered;
    }

    /**
     * Clear temporary chat history
     */
    public function clearTempChatHistory(bool $deleteConversations = true): void
    {
        if (! auth()->check()) {
            return;
        }

        $userId = auth()->user()->id;
        $sessionKeyPrefix = $userId . '_' . $this->tempChatSessionKey;

        // Get all session keys
        $allSessionKeys = array_keys(Session::all());

        // Filter keys that match our temp chat pattern
        $tempChatKeys = array_filter($allSessionKeys, static function ($key) use ($sessionKeyPrefix) {
            return str_starts_with($key, $sessionKeyPrefix);
        });

        $deletedConversations = [];
        $errors = [];

        // Remove each temp chat session and optionally delete conversations
        foreach ($tempChatKeys as $key) {
            try {
                // Extract chatId from the session key
                $chatId = $this->extractChatIdFromSessionKey($key, $sessionKeyPrefix);

                if ($chatId && $deleteConversations) {
                    // Delete the conversation from database
                    $deleted = $this->deleteConversation($chatId);
                    if ($deleted) {
                        $deletedConversations[] = $chatId;
                    }
                }

                // Remove from session
                Session::forget($key);
            } catch (Exception $e) {
                $errors[] = "Error with session {$key}: " . $e->getMessage();
            }
        }

        if (! empty($errors)) {
            Log::warning('Errors during temp chat cleanup: ' . json_encode($errors));
        }
    }

    /**
     * Extract chatId from session key
     */
    private function extractChatIdFromSessionKey(string $sessionKey, string $prefix): ?string
    {
        // Remove the prefix to get just the chatId
        if (str_starts_with($sessionKey, $prefix)) {
            $chatId = substr($sessionKey, strlen($prefix));

            // Validate that it's a valid chat ID (numeric or UUID format)
            if (is_numeric($chatId)) {
                return $chatId;
            }
        }

        return null;
    }

    /**
     * Delete conversation from database
     */
    private function deleteConversation(string $chatId): bool
    {
        try {
            $chat = UserOpenaiChat::find($chatId);
            if (! $chat) {
                return false;
            }

            // Delete the chat itself
            $chat?->delete();

            return true;
        } catch (Exception $e) {
            Log::error("Error deleting conversation {$chatId}: " . $e->getMessage());

            return false;
        }
    }
}
