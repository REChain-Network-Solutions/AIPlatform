<?php

namespace Workdo\Taskly\Entities;

use Illuminate\Database\Eloquent\Model;

class Timesheet extends Model
{
    protected $fillable = [
        'project_id',
        'task_id',
        'date',
        'time',
        'description',
        'created_by',
    ];

    public function task()
    {
        return $this->hasOne('App\Models\Task', 'id', 'task_id');
    }
    public function project()
    {
        return $this->hasOne('App\Models\Project', 'id', 'project_id');
    }
}
