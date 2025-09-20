<?php

namespace App\Http\Controllers\OpenAi;

use App\Domains\Engine\Enums\EngineEnum;
use App\Domains\Entity\Enums\EntityEnum;
use App\Domains\Entity\Facades\Entity as EntityFacade;
use App\Domains\Entity\Models\Entity;
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
use Illuminate\Support\Facades\Auth;
use Illuminate\Support\Facades\Http;
use Illuminate\Support\Str;
use Illuminate\View\View;
use JsonException;
use Random\RandomException;
use Symfony\Component\HttpFoundation\StreamedResponse;
use Throwable;

class GeneratorController extends Controller
{
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
            'chatbot', 'vision', 'chatPro' => $this->buildChatStreamedOutput($request),
            default => $this->buildOtherStreamedOutput($request),
        };
    }

    /**
     * @throws GuzzleException
     */
    public function buildChatStreamedOutput(Request $request): ?StreamedResponse
    {
        $chatParams = $this->extractChatParameters($request);
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

        $isFileSearch = setting('openai_file_search', 0) && $chatParams['chat']->openai_vector_id !== null;

        return $this->streamService->ChatStream(
            $chat_bot,
            $history,
            $message,
            $chatParams['chat_type'],
            $chatParams['contain_images'],
            $default_ai_engine,
            assistant: $chatParams['assistant'],
            openRouter: $chatParams['openRouter'],
            fileChat: $isFileSearch
        );
    }

    private function extractChatParameters(Request $request): array
    {
        $chat_id = $request->get('chat_id');
        $chat = UserOpenaiChat::with('category')->findOrFail($chat_id);

        return [
            'prompt'              => $request->get('prompt'),
            'realtime'            => $request->get('realtime', false),
            'chat_brand_voice'    => $request->get('chat_brand_voice'),
            'brand_voice_prod'    => $request->get('brand_voice_prod'),
            'chat_id'             => $chat_id,
            'chat_type'           => $request->get('template_type'),
            'images'              => $request->get('images', []),
            'pdfname'             => $request->get('pdfname', null),
            'pdfpath'             => $request->get('pdfpath', null),
            'assistant'           => $request->get('assistant', null),
            'chatbot_front_model' => $request->get('chatbot_front_model', null),
            'chat'                => $chat,
            'openRouter'          => $this->determineOpenRouter($request->get('chatbot_front_model', null)),
            'contain_images'      => false, // Will be determined later
        ];
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
            'images'              => $chatParams['images'],
            'pdfName'             => $chatParams['pdfname'],
            'pdfPath'             => $chatParams['pdfpath'],
        ];

        if (! empty($chatParams['shared_message_uuid'])) {
            $attributes['model_slug'] = $chatParams['chatbot_front_model'];
            $attributes['shared_uuid'] = $chatParams['shared_message_uuid'];
        }

        return UserOpenaiChatMessage::create($attributes);
    }

    private function buildChatHistory(array $chatParams, int $chat_id): array
    {
        $chat = $chatParams['chat'];
        $category = $chat->category;
        $systemRole = EntityEnum::fromSlug($this->determineChatBot($chatParams['chatbot_front_model']))->isBetaEntity() ? 'system' : 'user';

        $history = $this->initializeHistory($category, $systemRole);
        $history = $this->addFileOrInstructionsToHistory($history, $category, $chat_id, $chatParams['prompt'], $systemRole);
        $history = $this->addPreviousMessagesToHistory($history, $chat, $chatParams['assistant']);
        $history = $this->checkBrandVoice($chatParams['chat_brand_voice'], $chatParams['brand_voice_prod'], $history);
        $history = $this->addCurrentPromptToHistory($history, $chatParams, $systemRole);

        return $history;
    }

    private function initializeHistory($category, string $systemRole): array
    {
        if ($category->chat_completions) {
            $chat_completions = json_decode($category->chat_completions, true, 512, JSON_THROW_ON_ERROR);
            $history = [];
            foreach ($chat_completions as $item) {
                $history[] = [
                    'role'    => $item['role'],
                    'content' => $item['content'] ?? '',
                ];
            }

            return $history;
        }

        return [['role' => $systemRole, 'content' => 'You are a helpful assistant.']];
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
            }
        } elseif ($category && $category->instructions) {
            $history[] = ['role' => $systemRole, 'content' => $category->instructions];
        }

        return $history;
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

    private function addCurrentPromptToHistory(array $history, array $chatParams, string $systemRole): array
    {
        if (empty($chatParams['images']) && $chatParams['chat']->category->slug !== 'ai_vision') {
            if ($chatParams['realtime']) {
                $final_prompt = $this->getRealtimeEngine($chatParams);
                $history[] = ['role' => 'user', 'content' => $final_prompt ?? ''];
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

    private function getRealtimeEngine(array $chatParams): ?string
    {
        if (setting('default_realtime', 'serper') == 'serper' &&
            ! is_null($this->settings_two->serper_api_key)) {
            return $this->getRealtimePrompt($chatParams['prompt']);
        } elseif (setting('default_realtime') == 'perplexity' &&
            ! is_null(setting('perplexity_key'))) {
            return $this->realtimePromptPerplexity($chatParams['prompt']);
        }

        return null;
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
                        $imageData = file_get_contents(ltrim($image, '/'));
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
                    $imageData = file_get_contents(substr($item, 1, strlen($item) - 1));
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

    /**
     * @throws GuzzleException
     * @throws JsonException
     */
    private function getRealtimePrompt($realtimePrompt): ?string
    {
        $driver = EntityFacade::driver(EntityEnum::SERPER);

        if (! $driver->hasCreditBalance()) {
            echo PHP_EOL;
            echo "event: data\n";
            echo 'data: ' . __('You have no realtime search credits left. Please buy more credits to continue.');
            echo "\n\n";
            flush();
            echo "event: stop\n";
            echo 'data: [DONE]';
            echo "\n\n";
            flush();

            return null;
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

        return 'Prompt: ' . $realtimePrompt .
            '\n\nWeb search json results: '
            . json_encode($toGPT, JSON_THROW_ON_ERROR) .
            '\n\nInstructions: Based on the Prompt generate a proper response with help of Web search results(if the Web search results in the same context). Only if the prompt require links: (make curated list of links and descriptions using only the <a target="_blank">, write links with using <a target="_blank"> with mrgin Top of <a> tag is 5px and start order as number and write link first and then write description). Must not write links if its not necessary. Must not mention anything about the prompt text.';
    }

    public function realtimePromptPerplexity($realtimePrompt)
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

            } else {
                return response()->json([
                    'status'  => 'error',
                    'message' => $response->body(),
                ], 500);
            }
        } catch (Exception $e) {
            return response()->json([
                'status'  => 'error',
                'message' => $e->getMessage(),
            ], 500);
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
            $apikeyPart1 = base64_encode(rand(1, 100));
            $apikeyPart2 = base64_encode(rand(1, 100));
            $apikeyPart3 = base64_encode(rand(1, 100));
        } else {
            $apiKey = ApiHelper::setOpenAiKey();
            $len = strlen($apiKey);
            $len = max($len, 6);
            $parts[] = substr($apiKey, 0, $l[] = rand(1, $len - 5));
            $parts[] = substr($apiKey, $l[0], $l[] = rand(1, $len - $l[0] - 3));
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
