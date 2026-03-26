<?php

namespace Modules\AuditLog\Exports;

use App\LogActivity;
use Carbon\Carbon;
use Illuminate\Contracts\View\View;
use Maatwebsite\Excel\Concerns\FromView;
use Illuminate\Contracts\Queue\ShouldQueue;
use Illuminate\Bus\Queueable;

class AttendanceExport implements FromView,ShouldQueue
{
    use Queueable;

    public function view(): View
    {
        $logActivity = LogActivity::leftJoin('users', 'users.id', '=', 'log_activities.causer_id')
        ->leftJoin('attendances', 'attendances.id', '=', 'log_activities.subject_id')
        ->leftJoin('users as whom_user', function ($join) {
            $join->on('whom_user.id', '=', 'attendances.user_id');})
        ->select('log_activities.*','users.name as name','users.id as user_id','whom_user.name as whom_user_name','whom_user.id as whom_user_id','attendances.id as attendance_id')
        ->where('log_activities.subject_type','App\Attendance');

       if(request()->daterange)
       {
        $dates = explode(' - ', request()->daterange);
        $startDate = Carbon::create($dates[0] ?? date('Y-m-d'));
        $endDate = Carbon::create($dates[1] ?? date('Y-m-d'));

        $logActivity = $logActivity->whereBetween('log_activities.created_at', [$startDate->toDateString().' 00:00:00', $endDate->toDateString().' 23:59:59']);
       }

        $data['logActivities'] =  $logActivity->get();

        return view('auditlog::attendance.export',$data);
    }
}