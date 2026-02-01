<?php

declare(strict_types=1);

namespace App\Extensions\Chatbot\System\Http\Resources\Admin;

use Illuminate\Contracts\Support\Arrayable;
use Illuminate\Http\Request;
use Illuminate\Http\Resources\Json\JsonResource;
use Illuminate\Support\Facades\Route;
use JsonSerializable;

class ChatbotChannelResource extends JsonResource
{
    public function toArray(Request $request): array|Arrayable|JsonSerializable
    {
        return [
            'id'           => $this->id,
            'channel'      => $this->channel,
            'channel_id'   => $this->channelId(),
            'credentials'  => $this->credentials,
            'webhook'      => $this->getWebhook(),
        ];
    }

    private function getWebhook(): ?string
    {
        return match ($this->channel) {
            'whatsapp' => Route::has('api.v2.chatbot.channel.twilio.post.handle') ? route('api.v2.chatbot.channel.twilio.post.handle', [
                'chatbotId' => $this->chatbot_id,
                'channelId' => $this->id,
            ]) : null,
            'telegram' => Route::has('api.v2.chatbot.channel.telegram.post.handle') ? route('api.v2.chatbot.channel.telegram.post.handle', [
                'chatbotId' => $this->chatbot_id,
                'channelId' => $this->id,
            ]) : null,
            default    => null,
        };
    }

    private function channelId(): ?string
    {
        return match ($this->channel) {
            'whatsapp' => data_get($this->credentials, 'whatsapp_phone'),
            'telegram' => data_get($this->credentials, 'telegram_bot_name'),
            default    => null,
        };
    }
}
