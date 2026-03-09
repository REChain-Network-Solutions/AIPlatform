<?php

namespace App\Models;

use App\Models\Concerns\HasCache;
use Illuminate\Database\Eloquent\Model;

class Extension extends Model
{
    use HasCache;

    public static int $cacheTtl = 3600 * 24; // 24h

    public static string $cacheKey = 'cache_extensions';

    protected $table = 'extensions';

    protected $guarded = [];

    protected static function booted(): void
    {
        static::saved(static fn () => static::forgetCache());
        static::deleted(static fn () => static::forgetCache());
    }
}
