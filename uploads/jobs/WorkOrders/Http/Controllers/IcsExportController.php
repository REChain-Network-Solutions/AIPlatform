<?php
namespace Modules\WorkOrders\Http\Controllers; use Illuminate\Routing\Controller; use Illuminate\Http\Request; use Illuminate\Support\Facades\DB;
class IcsExportController extends Controller{
  protected function H(){return "BEGIN:VCALENDAR\nVERSION:2.0\nPRODID:-//FieldServiceSuite//EN\nCALSCALE:GREGORIAN\nMETHOD:PUBLISH\n";}
  protected function F(){return "END:VCALENDAR\n";}
  protected function L($k,$v){return $k.':'.str_replace(['\n','\r'],['\\n',''],$v)."\n";}
  public function workOrder($id){
    $wo=DB::table('work_orders')->where('id',$id)->first(); abort_if(!$wo,404);
    $dt=$wo->scheduled_at ?? $wo->created_at;
    $ics=$this->H()."BEGIN:VEVENT\n".$this->L('UID','wo-'.$wo->id.'@fieldservicesuite').$this->L('DTSTAMP',now()->format('Ymd\THis\Z')).$this->L('DTSTART',date('Ymd\THis',strtotime($dt))).$this->L('SUMMARY','Work Order #'.$wo->id)."END:VEVENT\n".$this->F();
    return response($ics,200,['Content-Type'=>'text/calendar; charset=utf-8','Content-Disposition'=>'attachment; filename="wo_'.$id.'.ics"']);
  }
  public function contractor(Request $r){
    $userId=(int)$r->input('user_id',auth()->id()); $date=$r->input('date',now()->toDateString());
    $rows=DB::table('work_order_contractor_assignments')->whereDate('scheduled_at',$date)->where('user_id',$userId)->orderBy('scheduled_at')->get();
    $ics=$this->H(); foreach($rows as $a){ $ics.="BEGIN:VEVENT\n".$this->L('UID','a-'.$a->id.'@fieldservicesuite').$this->L('DTSTAMP',now()->format('Ymd\THis\Z')).$this->L('DTSTART',date('Ymd\THis',strtotime($a->scheduled_at))).$this->L('SUMMARY','WO #'.$a->work_order_id.' (User #'.$a->user_id))."END:VEVENT\n"; } $ics.=$this->F();
    return response($ics,200,['Content-Type'=>'text/calendar; charset=utf-8','Content-Disposition'=>'attachment; filename="schedule_'.$userId.'_'.$date.'.ics"']);
  }
}