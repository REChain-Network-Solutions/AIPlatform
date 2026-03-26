<?php
namespace Modules\CustomerFeedback\Http\Controllers;
use App\Http\Controllers\AccountBaseController; use Illuminate\Http\Request; use Modules\CustomerFeedback\Entities\FeedbackChannel;
class FeedbackChannelController extends AccountBaseController {
 public function index(){ $this->items=FeedbackChannel::where('company_id', company()->id)->latest('id')->paginate(20); return view('customer-feedback::settings.index', array_merge($this->data,['section'=>'channels','items'=>$this->items,'title'=>'Feedback channels','fields'=>['name','slug','description','icon','status'],'storeRoute'=>'feedback.settings.channels.store','deleteRouteBase'=>'/customer-feedback/settings/channels'])); }
 public function store(Request $request){ $data=$request->validate(['name'=>'required|string|max:191','slug'=>'required|string|max:191','description'=>'nullable|string','icon'=>'nullable|string|max:191','status'=>'nullable|boolean']); $data['company_id']=company()->id; FeedbackChannel::updateOrCreate(['id'=>$request->id],$data); return redirect()->back()->with('success','Channel saved.'); }
 public function destroy(FeedbackChannel $item){ $item->delete(); return redirect()->back()->with('success','Channel deleted.'); }
}
