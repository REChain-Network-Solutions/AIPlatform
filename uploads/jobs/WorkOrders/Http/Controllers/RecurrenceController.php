<?php
namespace Modules\WorkOrders\Http\Controllers; use Illuminate\Routing\Controller; use Illuminate\Http\Request; use Illuminate\Support\Facades\DB;
class RecurrenceController extends Controller{
  public function store(Request $r,$workOrderId){
    $data=$r->validate(['rrule'=>'required|string','starts_at'=>'nullable|date','ends_at'=>'nullable|date']);
    DB::table('work_order_recurrences')->insert(['work_order_id'=>$workOrderId,'rrule'=>$data['rrule'],'starts_at'=>$data['starts_at']??null,'ends_at'=>$data['ends_at']??null,'active'=>true,'created_at'=>now(),'updated_at'=>now()]);
    return back()->with('status','Recurrence saved');
  }
}