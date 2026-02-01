<?php

namespace App\Extensions\Chatbot\System\Generators;

use App\Extensions\Chatbot\System\Generators\Contracts\Generator;
use App\Helpers\Classes\ApiHelper;
use Illuminate\Http\Client\PendingRequest;
use Illuminate\Support\Facades\Http;

class AnthropicGenerator extends Generator
{
    public const ENDPOINT = 'https://api.anthropic.com/v1/messages';

    public function generate(): string
    {
        $histories = array_values($this->modifyMessages());

        $response = $this->client()->post(self::ENDPOINT, [
            'model'      => $this->entity->value,
            'max_tokens' => 1024,
            'messages'   => $histories,
            'stream'     => false,
        ]);

        $message = $response->json('content.0.text');

        return $message ?: '';
    }

    public function client(): PendingRequest
    {
        return Http::withHeaders([
            'x-api-key'         => ApiHelper::setAnthropicKey(),
            'Accept'            => 'application/json',
            'Content-Type'      => 'application/json',
            'anthropic-version' => '2023-06-01',
        ]);
    }

    public function modifyMessages(): array
    {
        return $this->histories()
            ?->sortBy('id')
            ?->map(callback: function ($history) {
                return [
                    'role'    => $history->is_visitor ? 'user' : 'assistant',
                    'content' => $history->message,
                ];
            })?->toArray();
    }
}
