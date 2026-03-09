<?php

declare(strict_types=1);

namespace App\Helpers\Classes;

use Illuminate\Support\Facades\Cache;
use Illuminate\Support\Facades\Schema;

class TableSchema
{
    public static function hasTable(string $table, array $tables): bool
    {
        return in_array($table, $tables, true);
    }

    public function allTables(): array
    {
        return Cache::rememberForever('database_tables', function () {
            return Schema::getTableListing();
        });
    }
}
