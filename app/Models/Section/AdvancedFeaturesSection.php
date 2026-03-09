<?php

namespace App\Models\Section;

use App\Models\Concerns\HasCache;
use Illuminate\Database\Eloquent\Model;

class AdvancedFeaturesSection extends Model
{
    use HasCache;

    public static string $cacheKey = 'cache_advanced_section';

    public static int $cacheTtl = 3600 * 24;

    protected $table = 'advanced_features_section';

    protected $fillable = [
        'title',
        'description',
        'image',
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
}
