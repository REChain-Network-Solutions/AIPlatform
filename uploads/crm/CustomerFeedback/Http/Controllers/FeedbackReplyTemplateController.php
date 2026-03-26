<?php
namespace Modules\CustomerFeedback\Http\Controllers;
use App\Http\Controllers\AccountBaseController; use Illuminate\Http\Request; use Modules\CustomerFeedback\Entities\FeedbackReplyTemplate;
class FeedbackReplyTemplateController extends AccountBaseController {
 public function index(){ $this->items=FeedbackReplyTemplate::where('company_id', company()->id)->latest('id')->paginate(20); return view('customer-feedback::settings.index', array_merge($this->data,['section'=>'templates','items'=>$this->items,'title'=>'Reply templates','fields'=>['name','description','message','reply_type','status'],'storeRoute'=>'feedback.settings.templates.store','deleteRouteBase'=>'/customer-feedback/settings/templates'])); }
 public function store(Request $request){ $data=$request->validate(['name'=>'required|string|max:191','description'=>'nullable|string','message'=>'required|string','reply_type'=>'required|in:auto,manual','status'=>'nullable|boolean']); $data['company_id']=company()->id; FeedbackReplyTemplate::updateOrCreate(['id'=>$request->id],$data); return redirect()->back()->with('success','Template saved.'); }
 public function destroy(FeedbackReplyTemplate $item){ $item->delete(); return redirect()->back()->with('success','Template deleted.'); }
}
