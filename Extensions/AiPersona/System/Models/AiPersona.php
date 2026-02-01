<?php

namespace App\Extensions\AiPersona\System\Models;

use App\Models\User;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;

class AiPersona extends Model
{
    protected $table = 'user_heygen';

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
