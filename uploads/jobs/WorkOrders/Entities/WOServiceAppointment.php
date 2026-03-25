<?php

namespace Modules\WorkOrders\Entities;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;

class WOServiceAppointment extends Model
{
    use HasFactory;
    protected $fillable=[
        'wo_id',
        'start_date',
        'start_time',
        'end_date',
        'end_time',
        'status',
        'notes',
        'parent_id',
    ];

    public static $status=[
        'pending'=>'Pending',
        'schedule'=>'Schedule',
        'dispatched'=>'Dispatched',
        'on_hold'=>'On Hold',
        'completed'=>'Completed',
        'cancelled'=>'Cancelled',
        'reschedule'=>'Reschedule',
    ];
}
