<?php

namespace App\Models\Section;

use App\Models\Concerns\HasCache;
use Illuminate\Database\Eloquent\Model;

class ComparisonSectionItems extends Model
{
    use HasCache;

    public static string $cacheKey = 'cache_comparison_section_items';

    public static int $cacheTtl = 3600 * 24;

    protected $table = 'comparison_section_items';

    protected $fillable = [
        'label', 'others', 'ours',
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
