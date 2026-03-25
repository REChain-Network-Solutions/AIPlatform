<?php
namespace Modules\WorkOrders\Http\Controllers; use Illuminate\Routing\Controller; use Illuminate\Http\Request; use Illuminate\Support\Facades\Storage; use Modules\WorkOrders\Entities\WOComment;
class CommentsController extends Controller{
  public function index($id){ $rows=WOComment::where('work_order_id',$id)->latest()->get(); return view('workorders::widgets.comments', compact('rows','id')); }
  public function store(Request $r,$id){
    $data=$r->validate(['body'=>'nullable|string','attachment'=>'nullable|file|max:20480']);
    $save=['work_order_id'=>$id,'user_id'=>auth()->id(),'body'=>$data['body']??null];
    if($r->hasFile('attachment')){ $path=$r->file('attachment')->store('wo_attachments','public'); $save['attachment_path']='storage/'+$path; }
    WOComment::create($save); return back()->with('status','Comment added.');
  }
}