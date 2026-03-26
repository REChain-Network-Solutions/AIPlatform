<?php
namespace Modules\CustomerFeedback\Http\Controllers;
use App\Http\Controllers\AccountBaseController; use Illuminate\Http\Request; use Modules\CustomerFeedback\Entities\FeedbackType;
class FeedbackTypeController extends AccountBaseController {
 public function index(){ $this->items=FeedbackType::where('company_id', company()->id)->latest('id')->paginate(20); return view('customer-feedback::settings.index', array_merge($this->data,['section'=>'types','items'=>$this->items,'title'=>'Feedback types','fields'=>['name','slug','description','type_category','status'],'storeRoute'=>'feedback.settings.types.store','deleteRouteBase'=>'/customer-feedback/settings/types'])); }
 public function store(Request $request){ $data=$request->validate(['name'=>'required|string|max:191','slug'=>'required|string|max:191','description'=>'nullable|string','type_category'=>'required|in:complaint,feedback,survey','status'=>'nullable|boolean']); $data['company_id']=company()->id; FeedbackType::updateOrCreate(['id'=>$request->id],$data); return redirect()->back()->with('success','Type saved.'); }
 public function destroy(FeedbackType $item){ $item->delete(); return redirect()->back()->with('success','Type deleted.'); }
}
