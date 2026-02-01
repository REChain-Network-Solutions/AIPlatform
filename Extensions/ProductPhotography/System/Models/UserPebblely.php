<?php

namespace App\Extensions\ProductPhotography\System\Models;

use App\Models\User;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;

class UserPebblely extends Model
{
    protected $table = 'pebblely';

    protected $fillable = [
        'user_id',
        'image',
    ];

    public function user(): BelongsTo
    {
        return $this->belongsTo(User::class, 'user_id', 'id');
    }
}
