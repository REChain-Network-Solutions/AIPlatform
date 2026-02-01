<?php

namespace App\Extensions\AiAvatar\System\Models;

use App\Models\User;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;

class AiAvatar extends Model
{
    protected $table = 'user_synthesia';

    protected $fillable = [
        'user_id',
        'avatar_id',
        'status',
    ];

    public function user(): BelongsTo
    {
        return $this->belongsTo(User::class, 'user_id', 'id');
    }
}
