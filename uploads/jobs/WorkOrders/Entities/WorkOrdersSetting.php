<?php

namespace Modules\WorkOrders\Entities;

use Illuminate\Database\Eloquent\Model;

class WorkOrdersSetting extends Model
{
    protected $table = 'workorders_settings';
    protected $guarded = [];

    public static function getOrCreate(): self
    {
        return static::query()->first() ?? static::create([]);
    }
}
