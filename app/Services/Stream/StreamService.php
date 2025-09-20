<?php

namespace App\Services\Stream;

use App\Domains\Engine\Enums\EngineEnum;
use App\Domains\Engine\Services\AnthropicService;
use App\Domains\Engine\Services\GeminiService;
use App\Domains\Entity\BaseDriver;
use App\Domains\Entity\Enums\EntityEnum;
use App\Domains\Entity\Facades\Entity;
use App\Enums\BedrockEngine;
use App\Extensions\OpenRouter\System\Services\RouterAiService;
use App\Helpers\Classes\ApiHelper;
use App\Helpers\Classes\Helper;
use App\Helpers\Classes\MarketplaceHelper;
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
use Illuminate\Support\Facades\Auth;
use Illuminate\Support\Facades\Log;
use Illuminate\Support\Facades\Session;
use Illuminate\Support\Str;
use JsonException;
use OpenAI as OpenAIMain;
use OpenAI\Laravel\Facades\OpenAI;
use Symfony\Component\HttpFoundation\StreamedResponse;
use Throwable;

class StreamService
{
    public bool $guest = false;

    private string $tempChatSessionKey = 'temp_chat_history_';

    public bool $tempChatActive = false;

    public function __construct(
        Setting $setting,
        SettingTwo $settingTwo,
    ) {
        match (setting('default_ai_engine', EngineEnum::OPEN_AI->value)) {
            EngineEnum::ANTHROPIC->value => ApiHelper::setAnthropicKey($setting),
            EngineEnum::GEMINI->value    => ApiHelper::setGeminiKey($setting),
            EngineEnum::X_AI->value      => ApiHelper::setXAiKey($setting),
            default                      => ApiHelper::setOpenAiKey($setting),
        };
    }

    private function prepareStreamEnvironment(): void
    {
        if (function_exists('ini_set')) {
            @ini_set('output_buffering', 'off');
            @ini_set('zlib.output_compression', '0');
            @ini_set('implicit_flush', '1');
        }

        if (function_exists('ob_implicit_flush')) {
            ob_implicit_flush(true);
        }

        while (ob_get_level() > 0) {
            if (! @ob_end_flush()) {
                break;
            }
        }
    }

    private function safeFlush(): void
    {
        if (function_exists('ob_flush')) {
            @ob_flush();
        }
        if (function_exists('flush')) {
            @flush();
        }
    }

    private function createDriver(EntityEnum $model): ?BaseDriver
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
    public function ChatStream(string $chat_bot, $history, $main_message, $chat_type, $contain_images, $ai_engine = null, $assistant = null, $openRouter = null, $fileChat = false, $tempChatActive = false): ?StreamedResponse
    {
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

        if ($chat_bot === EntityEnum::AZURE_OPENAI->slug() && MarketplaceHelper::isRegistered('azure-openai')) {
            return \App\Extensions\AzureOpenai\System\Services\AzureOpenaiService::azureOpenaiStream($chat_bot, $history, $main_message, $chat_type, $contain_images);
        }

        if ($chat_type === 'chatPro' && MarketplaceHelper::isRegistered('ai-chat-pro')) {
            if (! auth()->check()) {
                $this->guest = true;
            }

            if ($ai_engine === EngineEnum::OPEN_AI->value && setting('ai_chat_pro_image_generation_feature', '0')) {
                return $this->openaiChatStream($chat_bot, $history, $main_message, $chat_type, $contain_images, tools: \App\Extensions\AIChatPro\System\Services\AiChatProService::tools());
            }
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

        return match ($ai_engine) {
            EngineEnum::OPEN_AI->value   => $this->openaiChatStream($chat_bot, $history, $main_message, $chat_type, $contain_images),
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
            $driver = $this->createDriver(EntityEnum::GPT_4_O);
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
            } else {
                ApiHelper::setOpenAiKey();
                $chat_bot = 'gpt-4o';
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
                        echo PHP_EOL;
                        echo "event: data\n";
                        echo 'data: ' . $messageFix;
                        echo "\n\n";
                        $this->safeFlush();
                    }
                }
            }

            echo "event: stop\n";
            echo 'data: [DONE]';
            echo "\n\n";
            $this->safeFlush();

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
            $driver = $this->createDriver(EntityEnum::GPT_4_O);
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

                } else {
                    ApiHelper::setOpenAiKey();
                    $chat_bot = 'gpt-4o';
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
                            echo PHP_EOL;
                            echo "event: data\n";
                            echo 'data: ' . $messageFix;
                            echo "\n\n";
                            $this->safeFlush();
                        }
                    }
                }
                echo "event: stop\n";
                echo 'data: [DONE]';
                echo "\n\n";
                $this->safeFlush();

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
            return \App\Extensions\AzureOpenai\System\Services\AzureOpenaiService::azureOpenaiOtherStream($request, $chat_bot);
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
        $model = Helper::setting('openai_default_model') ?: EntityEnum::GPT_4_O->value;
        $streamed_text = $request->get('streamed_text');
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
            $driver = $this->createDriver(EntityEnum::GPT_4_O);
        } else {
            $driver = $this->createDriver(EntityEnum::fromSlug($chat_bot));
        }
        $this->prepareStreamEnvironment();

        return response()->stream(function () use ($driver, $history, &$total_used_tokens, &$output, &$responsedText, $main_message, $contain_images) {
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

            $model = $driver->enum()->value;
            if ($contain_images) {
                $options = [
                    'model'                       => EntityEnum::GPT_4_O->value,
                    'messages'                    => $history,
                    'temperature'                 => 1.0,
                    'stream'                      => true,
                    'max_output_tokens'           => 2000,
                ];
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
     * @throws Exception
     */
    private function openaiChatStream(string $chat_bot, $history, $main_message, $chat_type, $contain_images, ?array $tools = []): ?StreamedResponse
    {
        // @todo: in beta entites: EntityEnum::fromSlug($chat_bot)->isBetaEntity() then output without stream, stream not working
        $total_used_tokens = 0;
        $output = '';
        $responsedText = '';

        if ($contain_images) {
            $driver = $this->createDriver(EntityEnum::GPT_4_O);
        } else {
            if ($tools && EntityEnum::fromSlug($chat_bot) === EntityEnum::GPT_5_CHAT) {
                $chat_bot = EntityEnum::GPT_5->slug();
            }

            $driver = $this->createDriver(EntityEnum::fromSlug($chat_bot));
        }
        $this->prepareStreamEnvironment();

        return response()->stream(function () use ($driver, $history, &$total_used_tokens, &$output, &$responsedText, $main_message, $contain_images, $tools) {
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

            $model = $driver->enum()->value;
            $options = [
                'model'             => $model,
                'stream'            => true,
            ];

            if (! in_array($model, [EntityEnum::GPT_4_O_MINI_SEARCH_PREVIEW->value, EntityEnum::GPT_4_O_SEARCH_PREVIEW->value], true)) {
                $options['temperature'] = 1.0;
            }

            if ($contain_images) {
                $options['max_output_tokens'] = 2000;
                $options['model'] = EntityEnum::GPT_4_O;
            }

            if (! empty($tools)) {
                $options['tools'] = $tools;
                $argumentsString = '';
            }

            $options['input'] = $history;
            if ($driver->enum()->isReasoningModel()) {
                $options['reasoning']['effort'] = setting('openai_reasoning_models_effort', 'low');
            }
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
                    foreach ($calls ?? [] as $call) {
                        if ($call instanceof \OpenAI\Responses\Responses\Output\OutputFunctionToolCall) {
                            $functionName = $call?->name;
                            $argumentsString = $call?->arguments;
                            // we can send event to display image loader
                            // if (! empty($functionName) && ! $signalSent) {
                            //     echo PHP_EOL;
                            //     echo "event: function_call\n";
                            //     echo 'data: ' . $functionName . "\n\n";
                            //     echo "\n\n";
                            //     $this->safeFlush();
                            //     $signalSent = true;
                            // }
                            $functionResponse = \App\Extensions\AIChatPro\System\Services\AiChatProService::callFunction($functionName, $argumentsString);
                            $output .= $functionResponse;
                            echo PHP_EOL;
                            echo "event: data\n";
                            echo 'data: ' . $functionResponse;
                            echo "\n\n";
                            $this->safeFlush();
                        }
                    }
                }
                if ((isset($response->response->delta) && $response->event === 'response.output_text.delta')) {
                    $text = $response->response->delta;
                    $messageFix = str_replace(["\r\n", "\r", "\n"], '<br/>', $text);
                    $output .= $messageFix;
                    $responsedText .= $text;
                    $total_used_tokens += countWords($text);
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
            $driver = $this->createDriver(EntityEnum::GPT_4_O);
        } else {
            $driver = $this->createDriver(EntityEnum::fromSlug($chat_bot));
        }
        $this->prepareStreamEnvironment();

        return response()->stream(function () use ($driver, $history, &$total_used_tokens, &$output, &$responsedText, $main_message, $contain_images) {
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
                $options['model'] = EntityEnum::GPT_4_O;
            } else {
                $options['tools'] = [
                    [
                        'type'             => 'file_search',
                        'vector_store_ids' => [$chat?->openai_vector_id],
                    ],
                ];
            }
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

            echo "event: stop\n";
            echo 'data: [DONE]';
            echo "\n\n";
            $this->safeFlush();
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

            $stream = OpenAI::responses()->createStreamed([
                'model'             => $driver->enum()->value,
                'input'             => $history,
                ...($driver->enum()->isReasoningModel() ? [
                    'reasoning' => [
                        'effort' => setting('openai_reasoning_models_effort', 'low'),
                    ],
                ] : []),
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

            $chat_id = $main_message->user_openai_chat_id;
            $chat = UserOpenaiChat::whereId($chat_id)->first();

            echo "event: message\n";
            echo 'data: ' . $main_message->id . "\n\n";

            if (! $contain_images) {
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
                        $jsonData = json_decode($chunk, false, 512, JSON_THROW_ON_ERROR);
                        if (isset($jsonData->delta->text)) {
                            $message = $jsonData->delta->text;
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
                }
            } else {
                ApiHelper::setOpenAiKey();
                $driver = $this->createDriver(EntityEnum::GPT_4_O);
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
                        echo PHP_EOL;
                        echo "event: data\n";
                        echo 'data: ' . $messageFix;
                        echo "\n\n";
                        $this->safeFlush();
                    }
                }
            }

            echo "event: stop\n";
            echo 'data: [DONE]';
            echo "\n\n";
            $this->safeFlush();

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

            echo "event: message\n";
            echo 'data: ' . $message_id . "\n\n";

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

        return response()->stream(function () use ($driver, $newhistory, &$total_used_tokens, &$output, &$responsedText, $main_message, $contain_images) {

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
            $response = $client
                ->setHistory($newhistory)
                ->streamGenerateContent($driver->enum()->value);

            while (! $response->getBody()->eof()) {
                $line = trim($client->readLine($response->getBody()));

                // Skip empty lines or JSON brackets
                if ($line === '' || $line === '[' || $line === ']') {
                    continue;
                }

                try {
                    $decodedLine = json_decode($line, true, 512, JSON_THROW_ON_ERROR);
                } catch (JsonException $e) {
                    Log::error('JSON decoding error: ' . $e->getMessage());
                    Log::error('Offending line: ' . $line);

                    continue;
                }

                // ✅ If it's an error object from Gemini
                if (isset($decodedLine['error'])) {
                    $errorMessage = $decodedLine['error']['message'] ?? 'Unknown error occurred.';
                    $formattedMessage = '⚠️ ' . $errorMessage;

                    echo PHP_EOL;
                    echo "event: data\n";
                    echo 'data: ' . $formattedMessage;
                    echo "\n\n";
                    $this->safeFlush();

                    // Optionally stop streaming here if it's fatal
                    break;
                }

                if (! isset($decodedLine['candidates'])) {
                    Log::info('Decoded line does not contain expected data: ' . json_encode($decodedLine));

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

                $line = $client->readLine($response->getBody());

                try {
                    $decodedLine = json_decode($line, true, 512, JSON_THROW_ON_ERROR);

                    if ($decodedLine === null || ! isset($decodedLine['candidates'])) {
                        Log::info('Decoded line does not contain expected data: ' . json_encode($decodedLine));

                        continue;
                    }
                } catch (JsonException $e) {
                    Log::error('JSON decoding error: ' . $e->getMessage());
                    Log::error('Offending line: ' . $line);

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

    private function saveStreamResponse($main_message, $chat, $responsedText, $output, $total_used_tokens, $driver): void
    {
        if ($this->tempChatActive) {
            // Add the assistant response to temp history
            if ($chat) {
                $this->addToTempHistory($chat->id, [
                    'role'    => 'assistant',
                    'content' => $responsedText,
                ]);

                // Delete from database to maintain temp nature
                $chat->messages()->delete();
            }

            return;
        }

        // Regular saving logic for non-temp chats
        $main_message->response = $responsedText;
        $main_message->output = $output;
        $main_message->credits = $total_used_tokens;
        $main_message->words = $total_used_tokens;
        $main_message->save();

        if ($chat) {
            $chat->total_credits += $total_used_tokens;
            $chat->save();
        }

        $driver->input($responsedText)->calculateCredit()->decreaseCredit();
        Usage::getSingle()->updateWordCounts($driver->calculate());
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
