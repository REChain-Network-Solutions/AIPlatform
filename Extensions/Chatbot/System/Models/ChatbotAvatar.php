<?php

namespace App\Extensions\Chatbot\System\Models;

use App\Models\User;
use Illuminate\Database\Eloquent\Casts\Attribute;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;

class ChatbotAvatar extends Model
{
    public $timestamps = false;

    protected $table = 'ext_chatbot_avatars';

    protected $fillable = [
        'user_id',
        'avatar',
    ];

    protected $appends = [
        'avatar_url',
    ];

    public function user(): BelongsTo
    {
        return $this->belongsTo(User::class);
    }

    public function avatarUrl(): Attribute
    {
        return Attribute::make(
            get: fn () => asset($this->avatar),
        );
    }
}
