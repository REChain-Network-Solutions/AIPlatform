<?php

namespace App\Extensions\Chatbot\System\Models;

use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;
use Illuminate\Database\Eloquent\Relations\HasMany;
use Illuminate\Database\Eloquent\Relations\HasOne;

class ChatbotConversation extends Model
{
    protected $table = 'ext_chatbot_conversations';

    protected $fillable = [
        'chatbot_channel',
        'chatbot_channel_id',
        'customer_channel_id',
        'ip_address',
        'conversation_name',
        'chatbot_id',
        'session_id',
        'connect_agent_at',
        'customer_payload',
        'is_showed_on_history',
        'last_activity_at',
    ];

    protected $casts = [
        'chatbot_id'           => 'integer',
        'session_id'           => 'string',
        'customer_payload'     => 'json',
        'is_showed_on_history' => 'boolean',
        'last_activity_at'	    => 'datetime',
    ];

    public function chatbotChannel(): BelongsTo
    {
        return $this->belongsTo(ChatbotChannel::class, 'chatbot_channel_id');
    }

    public function sessionId(): string
    {
        return $this->session_id;
    }

    public function chatbot(): BelongsTo
    {
        return $this->belongsTo(Chatbot::class);
    }

    public function lastMessage(): HasOne
    {
        return $this->hasOne(ChatbotHistory::class, 'conversation_id')
            ->where('role', 'user')
            ->orderByDesc('id');
    }

    public function histories(): HasMany
    {
        return $this->hasMany(ChatbotHistory::class, 'conversation_id');
    }
}
