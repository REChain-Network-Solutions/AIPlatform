<?php

namespace App\Extensions\Chatbot\System\Models;

use Illuminate\Database\Eloquent\Model;

class ChatbotChannel extends Model
{
    protected $table = 'ext_chatbot_channels';

    protected $fillable = [
        'user_id',
        'chatbot_id',
        'channel',
        'credentials',
        'payload',
        'connected_at',
    ];

    protected $casts = [
        'credentials'  => 'json',
        'payload'      => 'json',
        'connected_at' => 'datetime',
    ];

    public function isSandbox(): bool
    {
        return data_get($this->credentials, 'whatsapp_environment') === 'sandbox';
    }
}
