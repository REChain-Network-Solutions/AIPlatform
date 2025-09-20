<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;

class RecentSearchKey extends Model
{
    use HasFactory;

    protected $fillable = ['keyword'];

    // user for key
    public function user(): BelongsTo
    {
        return $this->belongsTo(User::class);
    }
}
