<?php
namespace Modules\WorkOrders\Http\Controllers; use Illuminate\Routing\Controller; use Illuminate\Http\Request; use Illuminate\Support\Facades\DB;
use Modules\WorkOrders\Entities\{ChecklistTemplate,ChecklistItem,WorkOrderCheck,WorkOrderCheckItem};
class ChecklistController extends Controller{
  public function picker($woId){
    $vertical = optional(DB::table('fsm_settings')->orderBy('id','desc')->first())->vertical ?? 'general_trades';
    $templates = ChecklistTemplate::where('vertical',$vertical)->orderBy('label')->get();
    return view('workorders::checklists.picker', compact('woId','templates'));
  }
  public function attach(Request $r,$woId){
    $data=$r->validate(['template_id'=>'required|integer']);
    $tpl=ChecklistTemplate::findOrFail($data['template_id']);
    $wc=WorkOrderCheck::firstOrCreate(['work_order_id'=>$woId,'template_id'=>$tpl->id]);
    if(!WorkOrderCheckItem::where('work_check_id',$wc->id)->exists()){
      $items=ChecklistItem::where('template_id',$tpl->id)->orderBy('order')->get();
      foreach($items as $i){ WorkOrderCheckItem::create(['work_check_id'=>$wc->id,'template_item_id'=>$i->id,'text'=>$i->text,'required'=>$i->required,'status'=>'pending']); }
    }
    return redirect()->route('workorders.checklists.view',$woId)->with('status','Checklist attached.');
  }
  public function view($woId){
    $wc=WorkOrderCheck::where('work_order_id',$woId)->first();
    $items=$wc?WorkOrderCheckItem::where('work_check_id',$wc->id)->orderBy('id')->get():collect();
    return view('workorders::checklists.view', compact('woId','wc','items'));
  }
  public function updateItem(Request $r,$woId,$itemId){
    $data=$r->validate(['status'=>'required|in:pending,pass,fail,na','notes'=>'nullable|string']);
    WorkOrderCheckItem::where('id',$itemId)->update(['status'=>$data['status'],'notes'=>$data['notes']??null,'updated_at'=>now()]);
    return back();
  }
}