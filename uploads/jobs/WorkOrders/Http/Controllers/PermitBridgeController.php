<?php
namespace Modules\WorkOrders\Http\Controllers;
use Illuminate\Http\Request; use Illuminate\Routing\Controller;
use Modules\WorkOrders\Entities\WOPermit;
class PermitBridgeController extends Controller {
  public function index($id){ $rows=WOPermit::where('work_order_id',$id)->latest()->get();
    return view('workorders::widgets.permits',['work_order_id'=>$id,'rows'=>$rows]); }
  public function store(Request $r,$id){ $data=$r->validate(['type'=>['nullable','string','max:120'],
    'status'=>['nullable','string','in:pending,approved,rejected,expired'],'permit_number'=>['nullable','string','max:120'],
    'valid_from'=>['nullable','date'],'valid_to'=>['nullable','date']]);
    $data['work_order_id']=(int)$id; WOPermit::create($data); return back()->with('status','Permit updated.'); } }