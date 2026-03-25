<?php

namespace Workdo\Taskly\Entities;

use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Factories\HasFactory;

class Milestone extends Model
{
    use HasFactory;

    protected $fillable = [
        'id','project_id','title','status','cost','summary','progress','end_date','start_date'
    ];

    protected static function newFactory()
    {
        return \Workdo\Taskly\Database\factories\MilestoneFactory::new();
    }
}
