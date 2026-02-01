<?php

declare(strict_types=1);

namespace App\Extensions\Chatbot\System\Http\Resources\Api;

use Illuminate\Contracts\Support\Arrayable;
use Illuminate\Http\Request;
use Illuminate\Http\Resources\Json\JsonResource;
use JsonSerializable;

class ChatbotConversationResource extends JsonResource
{
    public function toArray(Request $request): array|Arrayable|JsonSerializable
    {
        return [
            'id'                => $this->getAttribute('id'),
            'ip_address'        => $this->getAttribute('ip_address'),
            'conversation_name' => $this->getAttribute('conversation_name'),
            'chatbot_id'        => $this->getAttribute('chatbot_id'),
            'session_id'        => $this->getAttribute('session_id'),
            'connect_agent_at'  => $this->getAttribute('connect_agent_at'),
            'last_message'      => $this->whenLoaded('lastMessage', function () {
                return $this->lastMessage?->getAttribute('message');
            }),
            'created_at' => $this->getAttribute('created_at')->timezone($this->timezone()),
            'routes'     => [
                'messages' => route('api.v2.chatbot.conversion.messages', [
                    'chatbot'             => $request->route('chatbot')?->getAttribute('uuid'),
                    'sessionId'           => $request->route('sessionId'),
                    'chatbotConversation' => $this->getAttribute('id'),
                ]),
            ],
        ];
    }

    public function timezone(): array|string
    {
        $timezone = request()->header('x-timezone');

        if (is_string($timezone)) {
            return $timezone;
        }

        return 'UTC';
    }
}
