<?php

namespace App\Extensions\AISocialMedia\System\Models;

use App\Models\User;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;

class TwitterSettings extends Model
{
    protected $fillable = [
        'user_id',
        'consumer_key',
        'consumer_secret',
        'access_token',
        'access_token_secret',
        'bearer_token',
        'account_id',
    ];

    public function user(): BelongsTo
    {
        return $this->belongsTo(User::class);
    }
}
