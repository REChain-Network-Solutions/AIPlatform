<?php
namespace Modules\WorkOrders\Http\Controllers;
use Illuminate\Http\Request; use Illuminate\Routing\Controller;
use Modules\WorkOrders\Entities\WOInspection;
class InspectionBridgeController extends Controller {
  public function index($id){ $rows=WOInspection::where('work_order_id',$id)->latest()->get();
    return view('workorders::widgets.inspections',['work_order_id'=>$id,'rows'=>$rows]); }
  public function store(Request $r,$id){ $data=$r->validate(['inspection_id'=>['nullable','integer'],
    'template_name'=>['nullable','string','max:255'],'completed_by'=>['nullable','integer'],
    'completed_at'=>['nullable','date'],'pdf_path'=>['nullable','string','max:255']]);
    $data['work_order_id']=(int)$id; WOInspection::create($data);
    return back()->with('status','Inspection recorded.'); } }