<?php

namespace App\Models\Integration;

use App\Models\Concerns\HasCache;
use App\Models\Extension;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\HasOne;

class Integration extends Model
{
    use HasCache;

    public static int $cacheTtl = 3600 * 24;

    public static string $cacheKey = 'cache_integration';

    protected $table = 'integrations';

    protected $fillable = [
        'app',
        'description',
        'image',
        'slug',
        'status',
    ];

    protected static function boot(): void
    {
        parent::boot();

        static::saved(static function () {
            static::forgetCache();
        });

        static::deleted(static function () {
            static::forgetCache();
        });
    }

    public function hasExtension(): HasOne
    {
        return $this->hasOne(Extension::class, 'slug', 'slug')->where('installed', 1);
    }

    public function extension(): HasOne
    {
        return $this->hasOne(Extension::class, 'slug', 'slug');
    }

    public function getFormClassName(): string
    {
        return match ($this->slug) {
            'wordpress' => \App\Extensions\Wordpress\System\Services\Wordpress::class,
        };
    }
}
