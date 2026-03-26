<?php
namespace Modules\CustomerFeedback\Http\Controllers;
use App\Http\Controllers\AccountBaseController; use Illuminate\Http\Request; use Modules\CustomerFeedback\Entities\FeedbackAgentGroups; use Modules\CustomerFeedback\Entities\FeedbackGroup; use App\Models\User;
class FeedbackAgentGroupController extends AccountBaseController {
 public function index(){ $this->groups=FeedbackGroup::where('company_id',company()->id)->get(); $this->agents=User::query()->where('company_id',company()->id)->limit(200)->get(); $this->links=FeedbackAgentGroups::with(['group','agent','addedBy'])->where('company_id',company()->id)->latest('id')->paginate(20); return view('customer-feedback::settings.agents',$this->data); }
 public function store(Request $request){ $data=$request->validate(['group_id'=>'required|exists:feedback_groups,id','agent_id'=>'required|exists:users,id']); $data['company_id']=company()->id; $data['added_by']=user()->id; FeedbackAgentGroups::updateOrCreate(['company_id'=>company()->id,'group_id'=>$data['group_id'],'agent_id'=>$data['agent_id']],$data); return redirect()->back()->with('success','Agent assignment saved.'); }
 public function destroy(FeedbackAgentGroups $item){ $item->delete(); return redirect()->back()->with('success','Agent assignment deleted.'); }
}
