<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Concerns\HasUlids;
use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;

class UserUpload extends Model
{
    use HasFactory;
    use HasUlids;

    protected $fillable = [
        'user_id',
        'disk',
        'path',
        'original_name',
        'mime_type',
        'extension',
        'size',
        'sha256',
    ];

    protected $casts = [
        'size' => 'integer',
    ];

    public function user(): BelongsTo
    {
        return $this->belongsTo(User::class);
    }
}
