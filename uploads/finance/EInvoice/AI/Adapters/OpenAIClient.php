<?php

namespace Modules\EInvoice\AI\Adapters;

use Modules\EInvoice\AI\ClientInterface;

class OpenAIClient implements ClientInterface
{
    protected array $config;

    public function __construct(array $config = [])
    {
        $this->config = $config ?: [];
    }

    protected function apiKey(): string
    {
        $key = $this->config['key'] ?? getenv('OPENAI_API_KEY') ?: '';
        if (!$key) {
            throw new \RuntimeException('OPENAI_API_KEY is missing. Add it to your .env');
        }
        return $key;
    }

    protected function model(): string
    {
        return $this->config['model'] ?? getenv('OPENAI_MODEL') ?: 'gpt-4o-mini';
    }

    public function complete(string $prompt, array $opts = []): string
    {
        // Minimal HTTP call using curl; no external SDK dependency.
        $payload = [
            'model' => $this->model(),
            'messages' => [
                ['role' => 'system', 'content' => $opts['system'] ?? 'You are a helpful assistant.'],
                ['role' => 'user', 'content' => $prompt],
            ],
            'temperature' => $opts['temperature'] ?? 0.2,
            'max_tokens' => $opts['max_tokens'] ?? 128,
        ];

        $ch = curl_init('https://api.openai.com/v1/chat/completions');
        curl_setopt_array($ch, [
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_HTTPHEADER => [
                'Content-Type: application/json',
                'Authorization: Bearer ' . $this->apiKey(),
            ],
            CURLOPT_POST => true,
            CURLOPT_POSTFIELDS => json_encode($payload),
            CURLOPT_TIMEOUT => 20,
        ]);
        $resp = curl_exec($ch);
        if ($resp === false) {
            $err = curl_error($ch);
            curl_close($ch);
            throw new \RuntimeException('OpenAI HTTP error: ' . $err);
        }
        $code = curl_getinfo($ch, CURLINFO_HTTP_CODE);
        curl_close($ch);
        $json = json_decode($resp, true);
        if ($code >= 400) {
            $msg = $json['error']['message'] ?? 'Unknown error';
            throw new \RuntimeException('OpenAI API error (' . $code . '): ' . $msg);
        }
        return $json['choices'][0]['message']['content'] ?? '';
    }

    public function health(): array
    {
        $ok = (bool) (($this->config['key'] ?? null) ?: getenv('OPENAI_API_KEY'));
        return [
            'provider' => 'openai',
            'model' => $this->model(),
            'has_key' => $ok,
            'status' => $ok ? 'ok' : 'missing_key'
        ];
    }
}
