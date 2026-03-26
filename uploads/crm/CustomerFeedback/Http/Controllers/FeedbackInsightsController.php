<?php
namespace Modules\CustomerFeedback\Http\Controllers;
use App\Http\Controllers\AccountBaseController; use Modules\CustomerFeedback\Entities\FeedbackInsight; use Modules\CustomerFeedback\Entities\FeedbackTicket; use Modules\CustomerFeedback\Services\FeedbackAiService;
class FeedbackInsightsController extends AccountBaseController {
 public function __construct(private FeedbackAiService $aiService){ parent::__construct(); $this->pageTitle='customer-feedback::modules.analytics'; }
 public function getTicketInsights(FeedbackTicket $ticket){ return response()->json(['data'=>$this->aiService->getInsights($ticket)]); }
 public function analyzeTicket(FeedbackTicket $ticket){ $analysis=$this->aiService->analyzeTicket($ticket); $ticket->update(['ai_metadata'=>$analysis]); if(request()->wantsJson()){ return response()->json(['message'=>'Analysis complete.','data'=>$analysis]); } return redirect()->back()->with('success','Analysis complete.'); }
 public function getSentiment(FeedbackTicket $ticket){ return response()->json($this->aiService->getSentiment($ticket)); }
 public function getSuggestedCategory(FeedbackTicket $ticket){ return response()->json(['category'=>$this->aiService->suggestCategory($ticket)]); }
 public function getSuggestedPriority(FeedbackTicket $ticket){ return response()->json(['priority'=>$this->aiService->suggestPriority($ticket)]); }
 public function getSuggestedResponse(FeedbackTicket $ticket){ return response()->json(['suggested_response'=>$this->aiService->suggestResponse($ticket)]); }
 public function dashboard(){ $this->insightSummary=FeedbackInsight::selectRaw('insight_type, COUNT(*) as count')->groupBy('insight_type')->get(); $this->recentInsights=FeedbackInsight::with('ticket')->latest()->limit(20)->get(); $this->highConfidenceInsights=FeedbackInsight::where('confidence_score','>=',0.85)->with('ticket')->latest()->limit(10)->get(); return view('customer-feedback::insights.dashboard',$this->data); }
}
