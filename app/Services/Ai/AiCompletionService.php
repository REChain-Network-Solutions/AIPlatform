<?php

declare(strict_types=1);

namespace App\Services\Ai;

use App\Domains\Engine\Enums\EngineEnum;
use App\Domains\Engine\Services\AnthropicService;
use App\Domains\Engine\Services\GeminiService;
use App\Helpers\Classes\ApiHelper;
use App\Helpers\Classes\Helper;
use GuzzleHttp\Client;
use OpenAI as OpenAIMain;
use OpenAI\Laravel\Facades\OpenAI;
use Illuminate\Support\Facades\Log;

class AiCompletionService
{
    public function __construct(
        private readonly ApiKeyResolver $keyResolver,
        private readonly DeviceCompletionService $deviceCompletion,
        private readonly LocalCompletionService $localCompletion,
    ) {}

    public function completeUserOnly(string $userContent, ?EngineEnum $engine = null): string
    {
        return $this->complete('You are a helpful assistant.', $userContent, $engine);
    }

    public function complete(string $systemPrompt, string $userContent, ?EngineEnum $engine = null): string
    {
        $engine ??= Helper::defaultEngine();

        foreach (config('ai.routing.order', ['device', 'local', 'cloud']) as $layer) {
            if ($layer === 'device') {
                $result = $this->deviceCompletion->complete($systemPrompt, $userContent);

                if ($result !== null) {
                    return $result;
                }
            }

            if ($layer === 'local') {
                $result = $this->localCompletion->complete($systemPrompt, $userContent);

                if ($result !== null) {
                    return $result;
                }
            }

            if ($layer === 'cloud') {
                return $this->completeViaCloud($systemPrompt, $userContent, $engine);
            }
        }

        Log::warning('AI completion fell through all routing layers, using cloud fallback');

        return $this->completeViaCloud($systemPrompt, $userContent, $engine);
    }

    private function completeViaCloud(string $systemPrompt, string $userContent, EngineEnum $engine): string
    {
        return match ($engine) {
            EngineEnum::ANTHROPIC => $this->viaAnthropic($systemPrompt, $userContent),
            EngineEnum::GEMINI    => $this->viaGemini($systemPrompt, $userContent, $engine),
            EngineEnum::DEEP_SEEK => $this->viaDeepSeek($systemPrompt, $userContent, $engine),
            EngineEnum::X_AI      => $this->viaXAi($systemPrompt, $userContent, $engine),
            default               => $this->viaOpenAi($systemPrompt, $userContent),
        };
    }

    private function viaOpenAi(string $systemPrompt, string $userContent): string
    {
        $this->keyResolver->applyOpenAiKey();

        $model = Helper::defaultWordModel();

        $result = OpenAI::responses()->create([
            'model' => $model->value,
            'input' => [
                ['role' => 'system', 'content' => $systemPrompt],
                ['role' => 'user', 'content' => $userContent],
            ],
        ]);

        return collect($result['output'] ?? [])
            ->flatMap(fn ($item) => $item['content'] ?? [])
            ->pluck('text')
            ->implode('');
    }

    private function viaAnthropic(string $systemPrompt, string $userContent): string
    {
        $this->keyResolver->applyAnthropicKey();

        $client = app(AnthropicService::class);

        $response = $client
            ->setSystem($systemPrompt)
            ->setMessages([
                ['role' => 'user', 'content' => $userContent],
            ])
            ->setStream(false)
            ->stream();

        $body = $response->json();

        return collect($body['content'] ?? [])
            ->where('type', 'text')
            ->pluck('text')
            ->implode('');
    }

    private function viaGemini(string $systemPrompt, string $userContent, EngineEnum $engine): string
    {
        $this->keyResolver->applyGeminiKey();

        $model = $engine->getDefaultWordModel(null);

        $client = app(GeminiService::class);

        $response = $client
            ->setHistory([
                ['role' => 'user', 'parts' => [['text' => $systemPrompt . "\n\n" . $userContent]]],
            ])
            ->generateContent($model->value);

        $body = $response->json();

        return collect($body['candidates'] ?? [])
            ->pluck('content.parts')
            ->flatten(1)
            ->pluck('text')
            ->implode('');
    }

    private function viaDeepSeek(string $systemPrompt, string $userContent, EngineEnum $engine): string
    {
        $this->keyResolver->applyDeepseekKey();

        $model = $engine->getDefaultWordModel(null);
        $apiKey = config('deepseek.api_key');

        $response = (new Client)->post('https://api.deepseek.com/chat/completions', [
            'headers' => [
                'Content-Type'  => 'application/json',
                'Accept'        => 'application/json',
                'Authorization' => "Bearer {$apiKey}",
            ],
            'json' => [
                'model'    => $model->value,
                'messages' => [
                    ['role' => 'system', 'content' => $systemPrompt],
                    ['role' => 'user', 'content' => $userContent],
                ],
                'stream' => false,
            ],
        ]);

        $body = json_decode($response->getBody()->getContents(), true);

        return $body['choices'][0]['message']['content'] ?? '';
    }

    private function viaXAi(string $systemPrompt, string $userContent, EngineEnum $engine): string
    {
        $this->keyResolver->applyXAiKey();

        $apiKey = ApiHelper::setXAiKey();
        $model = $engine->getDefaultWordModel(null);

        $client = OpenAIMain::factory()
            ->withApiKey($apiKey)
            ->withBaseUri('https://api.x.ai/v1')
            ->make();

        $result = $client->chat()->create([
            'model'    => $model->value,
            'messages' => [
                ['role' => 'system', 'content' => $systemPrompt],
                ['role' => 'user', 'content' => $userContent],
            ],
        ]);

        return $result->choices[0]->message->content ?? '';
    }
}
