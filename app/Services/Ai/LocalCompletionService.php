<?php

declare(strict_types=1);

namespace App\Services\Ai;

use Illuminate\Support\Facades\Http;
use Illuminate\Support\Facades\Log;

class LocalCompletionService
{
    public function enabled(): bool
    {
        return (bool) config('ai.routing.local_enabled', false);
    }

    public function complete(string $systemPrompt, string $userContent): ?string
    {
        if (! $this->enabled()) {
            return null;
        }

        $baseUrl = rtrim((string) config('services.ollama.base_url'), '/');

        if ($baseUrl === '') {
            Log::warning('LocalCompletionService enabled but no services.ollama.base_url configured');

            return null;
        }

        try {
            $response = Http::timeout(10)->post($baseUrl . '/api/chat', [
                'model'    => config('services.ollama.model', 'llama3'),
                'messages' => [
                    ['role' => 'system', 'content' => $systemPrompt],
                    ['role' => 'user', 'content' => $userContent],
                ],
            ]);

            if (! $response->ok()) {
                Log::warning('LocalCompletionService request failed', ['status' => $response->status()]);

                return null;
            }

            $data = $response->json();

            return data_get($data, 'message.content');
        } catch (\Throwable $e) {
            Log::warning('LocalCompletionService exception', ['error' => $e->getMessage()]);

            return null;
        }
    }
}
