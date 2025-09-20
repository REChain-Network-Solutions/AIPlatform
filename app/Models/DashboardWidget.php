<?php

namespace App\Models;

use App\Enums\DashboardWidget as EnumsDashboardWidget;
use Illuminate\Database\Eloquent\Model;

class DashboardWidget extends Model
{
    protected $table = 'dashboard_widgets';

    protected $fillable = ['name', 'enabled', 'order'];

    protected $casts = [
        'enabled' => 'boolean',
        'name'    => EnumsDashboardWidget::class,
    ];
}
