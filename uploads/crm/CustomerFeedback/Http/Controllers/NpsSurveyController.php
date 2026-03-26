<?php

namespace Modules\CustomerFeedback\Http\Controllers;

use App\Helper\Reply;
use Illuminate\Http\Request;
use App\Http\Controllers\AccountBaseController;
use Modules\CustomerFeedback\Entities\NpsSurvey;
use Modules\CustomerFeedback\Entities\NpsResponse;
use Modules\CustomerFeedback\Entities\FeedbackTicket;
use Modules\CustomerFeedback\Http\Requests\StoreNpsSurvey;
use Modules\CustomerFeedback\Events\NpsSurveyCreated;

class NpsSurveyController extends AccountBaseController
{
    public function __construct()
    {
        parent::__construct();
        $this->pageTitle = 'customer-feedback::modules.nps';
        $this->middleware(function ($request, $next) {
            abort_403(!in_array('customer-feedback', $this->user->modules));
            return $next($request);
        });
    }

    /**
     * List NPS surveys
     */
    public function index()
    {
        $this->surveys = NpsSurvey::where('company_id', company()->id)
            ->withCount('responses')
            ->paginate(20);

        return view('customer-feedback::surveys.nps.index', $this->data);
    }

    /**
     * Create survey form
     */
    public function create()
    {
        $this->pageTitle = __('customer-feedback::app.createNpsSurvey');
        return view('customer-feedback::surveys.nps.create', $this->data);
    }

    /**
     * Store NPS survey
     */
    public function store(StoreNpsSurvey $request)
    {
        $survey = new NpsSurvey();
        $survey->company_id = company()->id;
        $survey->title = $request->title;
        $survey->description = $request->description;
        $survey->question = $request->question ?? NpsSurvey::DEFAULT_QUESTION;
        $survey->meta = $request->meta ?? [];
        $survey->status = true;
        $survey->save();

        event(new NpsSurveyCreated($survey));

        return Reply::success(__('messages.recordSaved'));
    }

    /**
     * Show survey responses
     */
    public function show(NpsSurvey $survey)
    {
        $this->survey = $survey;
        $this->responses = $survey->responses()->with('user', 'ticket')->paginate(20);

        // Calculate NPS Score
        $total = $survey->responses()->count();
        if ($total > 0) {
            $promoters = $survey->responses()->where('score', '>=', 9)->count();
            $detractors = $survey->responses()->where('score', '<=', 6)->count();
            $this->npsScore = (($promoters - $detractors) / $total) * 100;
        } else {
            $this->npsScore = null;
        }

        return view('customer-feedback::surveys.nps.show', $this->data);
    }

    /**
     * Submit NPS response (public endpoint)
     */
    public function submitResponse(Request $request, NpsSurvey $survey)
    {
        $request->validate([
            'score' => 'required|integer|min:1|max:10',
            'feedback' => 'nullable|string|max:1000',
        ]);

        $response = new NpsResponse();
        $response->nps_survey_id = $survey->id;
        $response->user_id = auth()->id();
        $response->score = $request->score;
        $response->feedback = $request->feedback;
        $response->company_id = company()->id;

        // Create linked feedback ticket
        if ($request->score <= 6) {
            $ticket = FeedbackTicket::create([
                'company_id' => company()->id,
                'user_id' => auth()->id(),
                'title' => 'NPS Feedback: ' . $survey->title,
                'description' => $request->feedback ?? 'NPS Score: ' . $request->score,
                'feedback_type' => FeedbackTicket::TYPE_SURVEY_RESPONSE,
                'status' => FeedbackTicket::STATUS_OPEN,
                'priority' => FeedbackTicket::PRIORITY_HIGH,
            ]);
            $response->feedback_ticket_id = $ticket->id;
        }

        $response->save();

        return response()->json(['message' => __('customer-feedback::messages.thankYou')]);
    }

    /**
     * Delete survey
     */
    public function destroy(NpsSurvey $survey)
    {
        abort_403(user()->permission('delete_feedback') != 'all');

        $survey->delete();

        return Reply::success(__('messages.recordDeleted'));
    }
}
