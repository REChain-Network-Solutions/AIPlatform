<?php
namespace Modules\CustomerFeedback\Http\Controllers;
use App\Http\Controllers\AccountBaseController; use Illuminate\Http\Request; use Modules\CustomerFeedback\Entities\FeedbackCustomForm;
class FeedbackCustomFormController extends AccountBaseController {
 public function index(){ $this->items=FeedbackCustomForm::where('company_id', company()->id)->latest('id')->paginate(20); return view('customer-feedback::settings.index', array_merge($this->data,['section'=>'forms','items'=>$this->items,'title'=>'Custom forms','fields'=>['name','description','fields','status'],'storeRoute'=>'feedback.settings.forms.store','deleteRouteBase'=>'/customer-feedback/settings/forms'])); }
 public function store(Request $request){ $data=$request->validate(['name'=>'required|string|max:191','description'=>'nullable|string','fields'=>'nullable','status'=>'nullable|boolean']); $data['company_id']=company()->id; $data['fields']=is_array($request->fields)?$request->fields:json_decode($request->fields?:'[]',true); FeedbackCustomForm::updateOrCreate(['id'=>$request->id],$data); return redirect()->back()->with('success','Form saved.'); }
 public function destroy(FeedbackCustomForm $item){ $item->delete(); return redirect()->back()->with('success','Form deleted.'); }
}
