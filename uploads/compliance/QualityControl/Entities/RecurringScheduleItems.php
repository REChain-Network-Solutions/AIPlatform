<?php

namespace Modules\QualityControl\Entities;

use App\Models\BaseModel;

class RecurringScheduleItems extends BaseModel
{
    protected $guarded = ['id'];
    protected $table = 'inspection_schedule_recurring_items';



}
