<?php
namespace Modules\WorkOrders\Http\Controllers; use Illuminate\Routing\Controller; use Illuminate\Support\Facades\DB;
class DashboardController extends Controller{
  public function index(){
    $today = now()->toDateString();
    $stats = [
      'wo_open'=> DB::table('work_orders')->where('status','!=','completed')->count(),
      'wo_today'=> DB::table('work_orders')->whereDate('created_at',$today)->count(),
      'sla_breach'=> DB::table('work_orders')->where('breach_flag',true)->count(),
    ];
    $dispatch = [
      'today_assignments'=> DB::table('work_order_contractor_assignments')->whereDate('scheduled_at',$today)->count(),
      'unassigned'=> DB::table('work_orders')->whereNull('scheduled_at')->where('status','!=','completed')->count(),
    ];
    return view('workorders::dashboard.index', compact('stats','dispatch'));
  }
}