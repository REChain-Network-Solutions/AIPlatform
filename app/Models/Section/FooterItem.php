<?php

namespace App\Models\Section;

use App\Models\Concerns\HasCache;
use Illuminate\Database\Eloquent\Model;

class FooterItem extends Model
{
    use HasCache;

    public static string $cacheKey = 'cache_footer_item';

    public static int $cacheTtl = 3600 * 24;

    protected $table = 'footer_items';

    protected $fillable = [
        'item',
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
