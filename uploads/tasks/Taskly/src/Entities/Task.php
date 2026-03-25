<?php

namespace Workdo\Taskly\Entities;

use App\Models\User;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Factories\HasFactory;

class   Task extends Model
{
    use HasFactory;



    protected static function newFactory()
    {
        return \Workdo\Taskly\Database\factories\TaskFactory::new();
    }
    protected $fillable = [
        'title',
        'priority',
        'description',
        'start_date',
        'due_date',
        'assign_to',
        'project_id',
        'milestone_id',
        'status',
        'order',
        'workspace',
    ];


    public function project()
    {
        return $this->hasOne('Workdo\Taskly\Entities\Project', 'id', 'project_id');
    }

    public function users()
    {
        return User::whereIn('id',explode(',',$this->assign_to))->get();
    }

    public function comments()
    {
        return $this->hasMany('Workdo\Taskly\Entities\Comment', 'task_id', 'id')->orderBy('id', 'DESC');
    }

    public function taskFiles()
    {
        return $this->hasMany('Workdo\Taskly\Entities\TaskFile', 'task_id', 'id')->orderBy('id', 'DESC');
    }

    public function milestones()
    {
        return $this->hasOne('Workdo\Taskly\Entities\Milestone', 'id', 'milestone_id');
    }

    public function milestone()
    {
        return $this->milestone_id ? Milestone::find($this->milestone_id) : null;
    }

    public function stage()
    {
        return $this->hasOne('Workdo\Taskly\Entities\Stage', 'id', 'status');
    }

    public function sub_tasks()
    {
        return $this->hasMany('Workdo\Taskly\Entities\SubTask', 'task_id', 'id')->orderBy('id', 'DESC');
    }

    public function taskCompleteSubTaskCount()
    {
        return $this->sub_tasks->where('status', '=', '1')->count();
    }

    public function taskTotalSubTaskCount()
    {
        return $this->sub_tasks->count();
    }

    public function subTaskPercentage()
    {
        $completedChecklist = $this->taskCompleteSubTaskCount();
        $allChecklist = max($this->taskTotalSubTaskCount(), 1);

        $percentageNumber = ceil(($completedChecklist / $allChecklist) * 100);
        $percentageNumber = $percentageNumber > 100 ? 100 : ($percentageNumber < 0 ? 0 : $percentageNumber);

        return (int) number_format($percentageNumber);
    }

    public static function getUsersData()
    {
        $zoommeetings = \DB::table('tasks')->get();

        $employeeIds = [];
        foreach ($zoommeetings as $item) {
            $employees = explode(',', $item->assign_to);
            foreach ($employees as $employee) {
                $employeeIds[] = $employee;
            }
        }
        $data = [];
        $users =  User::whereIn('id', array_unique($employeeIds))->get();
        foreach($users as $user)
        {

            $data[$user->id]['name']        = $user->name;
            $data[$user->id]['avatar']      = $user->avatar;
        }
        return $data;

    }
    public function signatures()
    {
        return $this->hasMany(\Workdo\Signature\Entities\TaskSignature::class, 'task_id', 'id');
    }
}
