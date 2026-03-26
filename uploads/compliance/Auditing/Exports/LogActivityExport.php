<?php

namespace Modules\AuditLog\Exports;

use App\LogActivity;
use Carbon\Carbon;
use Illuminate\Contracts\View\View;
use Maatwebsite\Excel\Concerns\FromView;
use Illuminate\Contracts\Queue\ShouldQueue;
use Illuminate\Bus\Queueable;

class LogActivityExport implements FromView,ShouldQueue
{
    use Queueable;

    public function view(): View
    {
        $logActivity = LogActivity::leftJoin('users', 'users.id', '=', 'log_activities.causer_id')
        ->select('log_activities.*','users.name as name');

       if(request()->daterange)
       {
        $dates = explode(' - ', request()->daterange);
        $startDate = Carbon::create($dates[0] ?? date('Y-m-d'));
        $endDate = Carbon::create($dates[1] ?? date('Y-m-d'));
        $logActivity = $logActivity->whereBetween('log_activities.created_at', [$startDate->toDateString().' 00:00:00', $endDate->toDateString().' 23:59:59']);
       }

       if(request()->model_name)
         $logActivity = $logActivity->where('log_activities.subject_type',request()->model_name);

        $data['logActivities'] =  $logActivity->get();

        return view('auditlog::log-activities-exports',$data);
    }
}