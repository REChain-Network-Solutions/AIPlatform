<?php

namespace App\Extensions\AIRealtimeImage\System\Models;

use App\Extensions\AIRealtimeImage\System\Enums\Status;
use App\Models\User;
use Illuminate\Database\Eloquent\Casts\Attribute;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;
use Illuminate\Support\Facades\Storage;

class RealtimeImage extends Model
{
    protected $table = 'ai_realtime_images';

    protected $fillable = [
        'user_id',
        'prompt',
        'disk',
        'image',
        'style',
        'payload',
        'response',
        'status',
        'model',
    ];

    protected $casts = [
        'payload'  => 'array',
        'response' => 'array',
        'status'   => Status::class,
    ];

    protected $appends = [
        'image_url',
    ];

    public function user(): BelongsTo
    {
        return $this->belongsTo(User::class);
    }

    public function imageUrl(): Attribute
    {
        return Attribute::make(function () {
            return $this->image ? str_replace('/public/', '/', Storage::disk($this->disk)->url($this->image)) : '';
        });
    }
}
