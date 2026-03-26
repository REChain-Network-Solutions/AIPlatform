<?php

namespace Modules\Inspection\Support;

use Illuminate\Support\Facades\Schema;

/**
 * Lightweight schema guards used by views/controllers to avoid runtime errors
 * when a module is enabled but migrations haven't completed yet.
 */
final class SchemaGuards
{
    public static function hasCoreTables(): bool
    {
        return Schema::hasTable('inspection_schedule_recurring')
            && Schema::hasTable('inspection_schedule_recurring_items')
            && Schema::hasTable('inspection_schedules');
    }
}
