<?php

namespace App\Extensions\Chatbot\System\Models;

use Illuminate\Database\Eloquent\Model;

class ChatbotChannelWebhook extends Model
{
    public $timestamps = false;

    protected $table = 'ext_chatbot_channel_webhooks';

    protected $fillable = [
        'chatbot_id',
        'chatbot_channel_id',
        'payload',
        'created_at',
    ];

    protected $casts = [
        'payload'      => 'json',
    ];
}
