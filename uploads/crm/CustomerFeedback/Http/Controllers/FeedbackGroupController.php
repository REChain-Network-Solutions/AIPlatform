<?php
namespace Modules\CustomerFeedback\Http\Controllers;
use App\Http\Controllers\AccountBaseController; use Illuminate\Http\Request; use Modules\CustomerFeedback\Entities\FeedbackGroup;
class FeedbackGroupController extends AccountBaseController {
 public function index(){ $this->items=FeedbackGroup::where('company_id', company()->id)->latest('id')->paginate(20); return view('customer-feedback::settings.index', array_merge($this->data,['section'=>'groups','items'=>$this->items,'title'=>'Feedback groups','fields'=>['name','description','status'],'storeRoute'=>'feedback.settings.groups.store','deleteRouteBase'=>'/customer-feedback/settings/groups'])); }
 public function store(Request $request){ $data=$request->validate(['name'=>'required|string|max:191','description'=>'nullable|string','status'=>'nullable|boolean']); $data['company_id']=company()->id; FeedbackGroup::updateOrCreate(['id'=>$request->id],$data); return redirect()->back()->with('success','Group saved.'); }
 public function destroy(FeedbackGroup $item){ $item->delete(); return redirect()->back()->with('success','Group deleted.'); }
}
