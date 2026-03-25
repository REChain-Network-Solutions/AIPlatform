<?php

namespace Workdo\Taskly\Entities;

use App\Models\User;
use Carbon\Carbon;
use Carbon\CarbonPeriod;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Support\Facades\App;
use Illuminate\Support\Facades\Auth;
use Workdo\Taskly\Entities\Timesheet;

class Project extends Model
{
    use HasFactory;

    protected $fillable = [
        'name',
        'status',
        'description',
        'start_date',
        'end_date',
        'budget',
        'copylinksetting',
        'password',
        'workspace',
        'created_by',
        'is_active',
    ];


    protected static function newFactory()
    {
        return \Workdo\Taskly\Database\factories\ProjectFactory::new();
    }
    public function scopeProjectOnly($query)
    {
        return $query->where('type', 'project');
    }
    public function creater()
    {
        return $this->hasOne(\App\Models\User::class, 'id', 'created_by');
    }
    public function task()
    {
        return $this->hasMany('Workdo\Taskly\Entities\Task', 'project_id', 'id');
    }

    public function users()
    {
        return $this->belongsToMany('App\Models\User', 'user_projects', 'project_id', 'user_id')->withPivot('is_active')->orderBy('id', 'ASC');
    }

    public function clients()
    {
        return $this->belongsToMany('App\Models\User', 'client_projects', 'project_id', 'client_id')->withPivot('is_active')->orderBy('id', 'ASC');
    }

    public function venders()
    {
        return $this->belongsToMany('App\Models\User', 'vender_projects', 'project_id', 'vender_id')->withPivot('is_active')->orderBy('id', 'ASC');
    }

    public function countTask()
    {
        return Task::where('project_id', '=', $this->id)->count();
    }

    public function tasks()
    {
        return Task::where('project_id', '=', $this->id)->get();
    }

    public function user_tasks($user_id)
    {
        return Task::where('project_id', '=', $this->id)->where('assign_to', '=', $user_id)->get();
    }
    public function user_done_tasks($user_id)
    {
        return Task::join('stages', 'stages.id', '=', 'tasks.status')->where('project_id', '=', $this->id)->where('assign_to', '=', $user_id)->where('stages.complete', '=', '1')->get();
    }

    public function timesheet()
    {
        return Timesheet::where('project_id', '=', $this->id)->get();
    }


    public function countTaskComments()
    {
        return Task::join('comments', 'comments.task_id', '=', 'tasks.id')->where('project_id', '=', $this->id)->count();
    }

    public function getProgress()
    {

        $total     = Task::where('project_id', '=', $this->id)->count();
        $totalDone = Task::where('project_id', '=', $this->id)->where('status', '=', 'done')->count();
        if ($totalDone == 0) {
            return 0;
        }

        return round(($totalDone * 100) / $total);
    }

    public function milestones()
    {
        return $this->hasMany('Workdo\Taskly\Entities\Milestone', 'project_id', 'id');
    }

    public function files()
    {
        return $this->hasMany('Workdo\Taskly\Entities\ProjectFile', 'project_id', 'id');
    }

    public function activities()
    {
        return $this->hasMany('Workdo\Taskly\Entities\ActivityLog', 'project_id', 'id')->orderBy('id', 'desc');
    }
    public static function getFirstSeventhWeekDay($week)
    {
        $first_day = $seventh_day = null;

        if (isset($week)) {
            $first_day   = Carbon::now()->addWeeks($week)->startOfWeek();
            $seventh_day = Carbon::now()->addWeeks($week)->endOfWeek();
        }

        $dateCollection['first_day']   = $first_day;
        $dateCollection['seventh_day'] = $seventh_day;

        $period = CarbonPeriod::create($first_day, $seventh_day);

        foreach ($period as $key => $dateobj) {
            $dateCollection['datePeriod'][$key] = $dateobj;
        }

        return $dateCollection;
    }

    public function project_progress()
    {
        $stage    = \Workdo\Taskly\Entities\Stage::where('workspace_id', '=', getActiveWorkSpace())->where('complete', 1)->first();
        if (!empty($stage)) {
            $status = $stage->id;
        } else {
            $status = 0;
        }
        $total_task     = Task::where('project_id', '=', $this->id)->count();
        $completed_task =  Task::where('project_id', '=', $this->id)->where('status', '=', $status)->count();
        if ($total_task > 0) {
            $percentage = intval(($completed_task / $total_task) * 100);

            return [

                'percentage' => $percentage . '%',
            ];
        } else {
            return [

                'percentage' => 0,
            ];
        }
    }

    public function project_milestone_progress()
    {
        $total_milestone     = Milestone::where('project_id', '=', $this->id)->count();
        $total_progress_sum  = Milestone::where('project_id', '=', $this->id)->sum('progress');

        if ($total_milestone > 0) {
            $percentage = intval(($total_progress_sum / $total_milestone));


            return [

                'percentage' => $percentage . '%',
            ];
        } else {
            return [

                'percentage' => 0,
            ];
        }
    }

    public static function customerProject($customerId)
    {
        $project = '';
        if (module_is_active('Account')) {
            $customer                   = \Workdo\Account\Entities\Customer::find($customerId);
            $project = ClientProject::where('client_id', $customer->user_id)->get();
        }
        return $project;
    }

    public static function vendorProject($vendorId)
    {
        $project = '';
        if (module_is_active('Account')) {
            $vendor                   = \Workdo\Account\Entities\Vender::find($vendorId);
            $project = VenderProject::where('vender_id', $vendor->user_id)->get();
        }
        return $project;
    }
}
