<?php

declare(strict_types=1);

namespace App\Extensions\Chatbot\System\Http\Resources\Admin;

use App\Extensions\Chatbot\System\Http\Resources\Api\ChatbotHistoryResource;
use Illuminate\Contracts\Support\Arrayable;
use Illuminate\Http\Request;
use Illuminate\Http\Resources\Json\JsonResource;
use Illuminate\Support\Arr;
use JsonSerializable;

class ChatbotConversationForAblyResource extends JsonResource
{
    public function toArray(Request $request): array|Arrayable|JsonSerializable
    {
        return [
            'id'                => $this->getAttribute('id'),
            'chatbot_channel'   => $this->getAttribute('chatbot_channel'),
            'color'             => Arr::random(['#879EC4', '#018a1a', '#7f00c8', '#e633ec']),
            'ip_address'        => $this->getAttribute('ip_address'),
            'conversation_name' => $this->getAttribute('conversation_name'),
            'chatbot'           => ChatbotResource::make($this->getAttribute('chatbot')),
            'lastMessage'       => $this->getAttribute('lastMessage') ? ChatbotHistoryResource::make($this->getAttribute('lastMessage')) : [
                'message' => 'No message',
                'read_at' => now(),
            ],
            'chatbot_id'  => $this->getAttribute('chatbot_id'),
            'created_at'  => $this->getAttribute('created_at'),
            'histories'   => null,
        ];
    }
}
