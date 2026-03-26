<?php

namespace Modules\Feedback\AI;

use Illuminate\Support\Facades\Http;

class OpenAIClient implements ClientInterface
{
    protected string $apiKey;
    protected ?string $base;
    protected string $model;

    public function __construct()
    {
        $config = config('feedback.ai.openai');
        $this->apiKey = (string)($config['api_key'] ?? '');
        $this->base = $config['base'] ?? null;
        $this->model = $config['model'] ?? 'gpt-4o-mini';

        // Fallback: root config.php array (OPENAI_API_KEY/openai_api_key)
        if (empty($this->apiKey) && function_exists('base_path')) {
            $rootConfig = base_path('config.php');
            if (file_exists($rootConfig)) {
                $arr = include $rootConfig;
                if (is_array($arr)) {
                    $cand = $arr['OPENAI_API_KEY'] ?? $arr['openai_api_key'] ?? null;
                    if (!empty($cand)) {
                        $this->apiKey = (string)$cand;
                    }
                }
            }
        }
    }

    protected function endpoint(): string
    {
        $base = rtrim($this->base ?: 'https://api.openai.com', '/');
        return $base . '/v1/chat/completions';
    }

    protected function chat(array $messages): string
    {
        if (!$this->apiKey) {
            return '';
        }
        $resp = Http::withHeaders([
            'Authorization' => 'Bearer ' . $this->apiKey,
            'Content-Type'  => 'application/json',
        ])->post($this->endpoint(), [
            'model' => $this->model,
            'messages' => $messages,
            'temperature' => 0.4,
        ]);
        if (!$resp->ok()) {
            return '';
        }
        $data = $resp->json();
        return $data['choices'][0]['message']['content'] ?? '';
    }

    public function generateFeedback(array $context): array
    {
        $prompt = $context['prompt'] ?? '';
        $messages = [
            ['role'=>'system','content'=>'You generate concise feedback titles and clear descriptions.'],
            ['role'=>'user','content'=>"Create a feedback draft for: {$prompt}"],
        ];
        $content = $this->chat($messages);
        $parts = preg_split('/\n\n+/', trim($content));
        $title = trim($parts[0] ?? 'Auto-generated Feedback');
        $description = trim($parts[1] ?? $content);
        return compact('title','description');
    }

    public function suggestReply(array $context): array
    {
        $text = $context['text'] ?? ($context['feedback'] ?? '');
        $messages = [
            ['role'=>'system','content'=>'You craft polite, professional replies to customer feedback.'],
            ['role'=>'user','content'=>"Feedback: {$text}\n\nWrite a helpful reply."],
        ];
        $reply = $this->chat($messages);
        return ['reply' => trim($reply)];
    }

    public function smoke(): array
    {
        $ok = !empty($this->apiKey);
        return ['ok'=>$ok, 'details'=>$ok ? 'API key present' : 'Missing OPENAI_API_KEY or config.php fallback'];
    }
}
