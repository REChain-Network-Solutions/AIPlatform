<?php

namespace Modules\WorkOrders\Entities;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;

class WorkOrder extends Model
{
    use HasFactory;
    protected $fillable=[
        'wo_id',
        'wo_detail',
        'type',
        'client',
        'asset',
        'due_date',
        'status',
        'priority',
        'notes',
        'assign',
        'preferred_date',
        'preferred_time',
        'preferred_note',
        'parent_id',
    ];

    public static $status=[
        'pending'=>'Pending',
        'approved'=>'Approved',
        'rejected'=>'Rejected',
        'on_hold'=>'On Hold',
        'cancelled'=>'Cancelled',
        'completed'=>'Completed',
    ];

    public function clients()
    {
        return $this->hasOne('Modules\WorkOrders\Entities\User','id','client');
    }
    public function assigned()
    {
        return $this->hasOne('Modules\WorkOrders\Entities\User','id','assign');
    }

    public function assets()
    {
        return $this->hasOne('Modules\WorkOrders\Entities\Asset','id','asset');
    }
    public function types()
    {
        return $this->hasOne('Modules\WorkOrders\Entities\WOType','id','type');
    }

    public function serviceParts()
    {
        return $this->hasMany('Modules\WorkOrders\Entities\WOServicePart','wo_id','id');
    }
    public function getWorkorderTotalAmount()
    {
        $woTotal = 0;
        foreach ($this->serviceParts as $serviceParts) {
            $woTotal += $serviceParts->amount;
        }
        return $woTotal;
    }
    public function services()
    {
        return $this->hasMany('Modules\WorkOrders\Entities\WOServicePart','wo_id','id')->where('type','service');
    }
    public function parts()
    {
        return $this->hasMany('Modules\WorkOrders\Entities\WOServicePart','wo_id','id')->where('type','part');
    }

    public function tasks()
    {
        return $this->hasMany('Modules\WorkOrders\Entities\WOServiceTask','wo_id','id');
    }

    public function appointments()
    {
        return $this->hasOne('Modules\WorkOrders\Entities\WOServiceAppointment','wo_id','id');
    }

    public function client()
    {
        $class = config('workorders.models.client', \App\Models\Client::class);
        return $this->belongsTo($class, 'client_id');
    }

    public function technician()
    {
        $class = config('workorders.models.user', \App\Models\User::class);
        return $this->belongsTo($class, 'technician_id');
    }

    public function appointments()
    {
        return $this->hasMany(Modules\WorkOrders\Entities\WOServiceAppointment::class, 'work_order_id');
    }

    public function tasks()
    {
        return $this->hasMany(Modules\WorkOrders\Entities\WOServiceTask::class, 'work_order_id');
    }

    public function parts()
    {
        return $this->hasMany(Modules\WorkOrders\Entities\WOServicePart::class, 'work_order_id');
    }
    

    public function convertToProject(): ?object
    {
        $projectClass = config('workorders.models.project', \App\Models\Project::class);
        $taskClass = config('workorders.models.task', \App\Models\Task::class);

        if (!class_exists($projectClass) || !class_exists($taskClass)) {
            return null;
        }

        $project = $projectClass::create([
            'project_name' => 'WO #'.$this->id.' — '.$this->status,
            'client_id' => $this->client_id,
            'start_date' => now(),
            'deadline' => $this->due_by,
        ]);

        foreach ($this->tasks as $line) {
            $taskClass::create([
                'name' => 'Task: '.$line->service_task_id,
                'project_id' => $project->id,
                'start_date' => now(),
                'due_date' => $this->due_by,
                'hourly_rate' => $line->rate,
                'task_hour' => $line->qty,
            ]);
        }

        return $project;
    }
    
}
