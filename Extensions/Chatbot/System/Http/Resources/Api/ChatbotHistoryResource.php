<?php

declare(strict_types=1);

namespace App\Extensions\Chatbot\System\Http\Resources\Api;

use Illuminate\Contracts\Support\Arrayable;
use Illuminate\Http\Request;
use Illuminate\Http\Resources\Json\JsonResource;
use JsonSerializable;

class ChatbotHistoryResource extends JsonResource
{
    public function toArray(Request $request): array|Arrayable|JsonSerializable
    {
        return [
            'conversation_id'     => $this->getAttribute('conversation_id'),
            'role'                => $this->getAttribute('role'),
            'ip_address'          => $this->getAttribute('ip_address'),
            'message'             => $this->getAttribute('message'),
            'user'                => $this->getAttribute('user'),
            'created_at'          => $this->getAttribute('created_at')->timezone($this->timezone()),
            'read_at'          	  => $this->getAttribute('read_at'),
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
