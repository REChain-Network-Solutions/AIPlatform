<?php

namespace Modules\Feedback\Http\Controllers;

use Client;
use Carbon\Carbon;
use App\Models\User;
use App\Helper\Reply;
use App\Models\Country;
use Illuminate\Http\Request;
use Modules\Units\Entities\Unit;
use Illuminate\Support\Facades\DB;
use Modules\Feedback\Entities\Feedback;
use Modules\Feedback\Entities\FeedbackTag;
use Modules\Engineerings\Entities\WorkOrder;
use Modules\Feedback\Entities\FeedbackType;
use Modules\Feedback\Entities\FeedbackGroup;
use Modules\Feedback\Entities\FeedbackReply;
use Modules\Engineerings\Entities\WorkRequest;
use App\Http\Controllers\AccountBaseController;
use Modules\Feedback\Entities\FeedbackChannel;
use Modules\Feedback\Entities\FeedbackTagList;
use Modules\Feedback\Http\Requests\StoreFeedback;
use Modules\Feedback\DataTables\FeedbackDataTable;
use Modules\Feedback\Http\Requests\UpdateFeedback;
use Modules\Feedback\Entities\FeedbackReplyTemplate;
use Modules\Houses\Entities\Area;
use Modules\Houses\Entities\House;

class FeedbackController extends AccountBaseController
{
    public function __construct()
    {
        parent::__construct();
        $this->pageTitle = 'feedback::modules.feedback';
        $this->middleware(function ($request, $next) {
            abort_403(!in_array('feedback', $this->user->modules));
            return $next($request);
        });
    }

    public function index(FeedbackDataTable $dataTable)
    {
        $this->viewPermission = user()->permission('view_feedback');
        $managePermission     = user()->permission('view_feedback');

        if (!request()->ajax()) {
            $this->channels    = FeedbackChannel::all();
            $this->groups      =
                $managePermission == 'none'
                ? null
                : FeedbackGroup::with([
                    'enabledAgents' => function ($q) use ($managePermission) {
                        if ($managePermission == 'added') {
                            $q->where('added_by', user()->id);
                        } elseif ($managePermission == 'owned') {
                            $q->where('agent_id', user()->id);
                        } elseif ($managePermission == 'both') {
                            $q->where('agent_id', user()->id)->orWhere('added_by', user()->id);
                        } else {
                            $q->get();
                        }
                    },
                    'enabledAgents.user',
                ])->get();

            $this->types = FeedbackType::all();
            $this->tags  = FeedbackTagList::all();
        }

        return $dataTable->render('feedback::feedback.index', $this->data);
    }

    public function getItems($id)
    {
        $items = House::where('area_id', $id)->get();
        return response()->json($items);
    }

    protected function deleteRecords($request)
    {
        abort_403(user()->permission('delete_feedback') != 'all');

        Feedback::whereIn('id', explode(',', $request->row_ids))->delete();
    }

    protected function changeBulkStatus($request)
    {
        abort_403(user()->permission('edit_feedback') != 'all');

        Feedback::whereIn('id', explode(',', $request->row_ids))->update(['status' => $request->status]);
    }

    public function create()
    {
        $this->addPermission = user()->permission('add_feedback');
        abort_403(!in_array($this->addPermission, ['all', 'owned', 'none']));

        $this->pageTitle     = __('feedback::app.feedback.addFeedback');
        $this->groups        = FeedbackGroup::with('enabledAgents', 'enabledAgents.user')->get();
        $this->types         = FeedbackType::all();
        $this->areas         = Area::all();
        $this->channels      = FeedbackChannel::all();
        $this->templates     = FeedbackReplyTemplate::all();
        $this->employees     = User::allEmployees(null, true, $this->addPermission == 'all' ? 'all' : null);
        $this->clients       = User::allClients();
        $this->unit          = Unit::all();
        $this->countries     = countries();
        $this->lastFeedback = Feedback::orderBy('id', 'desc')->first();
        $ticket              = new Feedback();

        $this->fields = null;

        if (!empty($ticket->getCustomFieldGroupsWithFields())) {
            $this->fields = $ticket->getCustomFieldGroupsWithFields()->fields;
        }

        if (request()->default_client) {
            $this->client = User::find(request()->default_client);
        }

        if (request()->ajax()) {
            $html = view('feedback::feedback.ajax.create', $this->data)->render();

            return Reply::dataOnly(['status' => 'success', 'html' => $html, 'title' => $this->pageTitle]);
        }

        $this->view = 'feedback::feedback.ajax.create';

        return view('feedback::feedback.create', $this->data);
    }

    public function store(StoreFeedback $request)
    {
        if (empty($request->type)) {
            return Reply::error(__('messages.addItem'));
        }

        $feedback             = new Feedback();
        $feedback->subject    = $request->subject;
        $feedback->status     = 'open';
        $feedback->priority   = 'medium';
        $feedback->no_hp      = $request->no_hp;
        $feedback->house_id    = $request->house_id;
        $feedback->user_id    = ($request->requester_type == 'employee') ? $request->user_id : $request->client_id;
        $feedback->agent_id   = $request->agent_id;
        $feedback->type_id    = $request->type_id;
        $feedback->channel_id = $request->channel_id;
        $feedback->save();

        if ($request->type == 'send') {
            $this->number = WorkRequest::lastInvoiceNumber() + 1;
            $this->zero   = '';
            if (strlen($this->number) < 4) {
                for ($i = 0; $i < 4 - strlen($this->number); $i++) {
                    $this->zero = '0' . $this->zero;
                }
            }
            $this->nomor    = 'WR-' . Carbon::now()->format('ym') . '-' . $this->zero . $this->number;
            $wr             = new WorkRequest();
            $wr->feedback_id  = $feedback->id;
            $wr->wr_no      = $this->nomor;
            $wr->check_time = date('Y-m-d H:i:s');
            $wr->problem    = $feedback->subject;
            $wr->house_id    = $feedback->house_id;
            $wr->assign_to  = $feedback->agent_id;
            $wr->created_by = user()->id;
            $wr->save();
        }

        // Save first message
        $reply               = new FeedbackReply();
        $reply->message      = trim_editor($request->description);
        $reply->feedback_id = $feedback->id;
        $reply->user_id      = $this->user->id;                     // Current logged in user
        $reply->save();

        // To add custom fields data
        if ($request->custom_fields_data) {
            $feedback->updateCustomFieldData($request->custom_fields_data);
        }

        // Save tags
        $tags = collect(json_decode($request->tags))->pluck('value');

        foreach ($tags as $tag) {
            $tag = FeedbackTagList::firstOrCreate([
                'tag_name' => $tag,
            ]);
            $feedback->feedbackTags()->attach($tag);
        }

        // Log search
        $this->logSearchEntry($feedback->id, $feedback->subject, 'feedback.show', 'feedback');

        $redirectUrl = urldecode($request->redirect_url);

        if ($redirectUrl == '') {
            $redirectUrl = route('feedback.index');
        }

        return Reply::successWithData(__('messages.recordSaved'), ['replyID' => $reply->id, 'redirectUrl' => $redirectUrl]);
    }

    public function show($feedbackNumber)
    {
        $this->viewFeedbackPermission = user()->permission('view_feedback');
        $this->feedback               = Feedback::with('requester', 'requester.feedback', 'reply', 'reply.files', 'reply.user')
            ->where('feedback_number', $feedbackNumber)
            ->first();
        abort_if(!$this->feedback, 404);
        $this->feedback = $this->feedback->withCustomFields();
        $this->pageTitle = __('feedback::modules.feedback') . '#' . $this->feedback->feedback_number;
        $this->groups    = FeedbackGroup::with('enabledAgents', 'enabledAgents.user')->get();
        $this->types     = FeedbackType::all();
        $this->channels  = FeedbackChannel::all();
        $this->templates = FeedbackReplyTemplate::all();
        $this->fields    = null;
        $this->wr        = DB::table('workrequests')
            ->where('feedback_id', $this->feedback->id)
            ->get();
        $this->wo = DB::table('workorders')
            ->where('feedback_id', $this->feedback->id)
            ->get();
        $this->feedbackChart = $this->feedbackChartData($this->feedback->user_id);


        if (!empty($this->feedback->getCustomFieldGroupsWithFields())) {
            $this->fields = $this->feedback->getCustomFieldGroupsWithFields()->fields;
        }

        return view('feedback::feedback.edit', $this->data);
    }

    public function feedbackChartData($id)
    {
        $labels         = ['open', 'pending', 'resolved', 'closed'];
        $data['labels'] = [__('app.open'), __('app.pending'), __('app.resolved'), __('app.closed')];
        $data['colors'] = ['#D30000', '#FCBD01', '#2CB100', '#1d82f5'];
        $data['values'] = [];

        foreach ($labels as $label) {
            $data['values'][] = Feedback::where('user_id', $id)
                ->where('status', $label)
                ->count();
        }

        return $data;
    }

    public function update(UpdateFeedback $request, $id)
    {
        $feedback         = Feedback::findOrFail($id);
        $feedback->status = $request->status;
        $feedback->save();

        $message = trim_editor($request->message);

        if ($message != '') {
            $reply               = new FeedbackReply();
            $reply->message      = $request->message;
            $reply->feedback_id = $feedback->id;
            $reply->user_id      = $this->user->id;       // Current logged in user
            $reply->save();

            return Reply::successWithData(__('messages.ticketReplySuccess'), ['reply_id' => $reply->id]);
        }

        return Reply::dataOnly(['status' => 'success']);
    }

    public function destroy($id)
    {
        $feedback = Feedback::findOrFail($id);

        $this->deleteFeedbackPermission = user()->permission('delete_feedback');
        abort_403(!($this->deleteFeedbackPermission == 'all' || ($this->deleteFeedbackPermission == 'added' && user()->id == $feedback->added_by) || ($this->deleteFeedbackPermission == 'owned' && (user()->id == $feedback->agent_id || user()->id == $feedback->user_id)) || ($this->deleteFeedbackPermission == 'both' && (user()->id == $feedback->agent_id || user()->id == $feedback->added_by || user()->id == $feedback->user_id))));

        Feedback::destroy($id);

        return Reply::success(__('messages.deleteSuccess'));
    }

    public function refreshCount(Request $request)
    {
        $viewPermission = user()->permission('view_feedback');

        $feedbacks = Feedback::with('agent');

        if (!is_null($request->startDate) && $request->startDate != '') {
            $startDate = Carbon::createFromFormat($this->company->date_format, $request->startDate)->toDateString();
            $feedbacks->where(DB::raw('DATE(`updated_at`)'), '>=', $startDate);
        }

        if (!is_null($request->endDate) && $request->endDate != '') {
            $endDate = Carbon::createFromFormat($this->company->date_format, $request->endDate)->toDateString();
            $feedbacks->where(DB::raw('DATE(`updated_at`)'), '<=', $endDate);
        }

        if (!is_null($request->agentId) && $request->agentId != 'all') {
            $feedbacks->where('agent_id', '=', $request->agentId);
        }

        if (!is_null($request->priority) && $request->priority != 'all') {
            $feedbacks->where('priority', '=', $request->priority);
        }

        if (!is_null($request->channelId) && $request->channelId != 'all') {
            $feedbacks->where('channel_id', '=', $request->channelId);
        }

        if (!is_null($request->typeId) && $request->typeId != 'all') {
            $feedbacks->where('type_id', '=', $request->typeId);
        }

        if ($viewPermission == 'none') {
            $feedbacks->where('user_id', '=', user()->id);
        }

        if ($viewPermission == 'owned') {
            $feedbacks->where('user_id', '=', user()->id)->orWhere('feedback.agent_id', '=', user()->id);
        }

        $feedbacks = $feedbacks->get();

        $openFeedback & NPS = $feedbacks
            ->filter(function ($value, $key) {
                return $value->status == 'open';
            })
            ->count();

        $pendingFeedback & NPS = $feedbacks
            ->filter(function ($value, $key) {
                return $value->status == 'pending';
            })
            ->count();

        $resolvedFeedback & NPS = $feedbacks
            ->filter(function ($value, $key) {
                return $value->status == 'resolved';
            })
            ->count();

        $closedFeedback & NPS = $feedbacks
            ->filter(function ($value, $key) {
                return $value->status == 'closed';
            })
            ->count();

        $totalFeedback & NPS = $feedbacks->count();

        $feedbackData = [
            'totalFeedback & NPS'    => $totalFeedback & NPS,
            'closedFeedback & NPS'   => $closedFeedback & NPS,
            'openFeedback & NPS'     => $openFeedback & NPS,
            'pendingFeedback & NPS'  => $pendingFeedback & NPS,
            'resolvedFeedback & NPS' => $resolvedFeedback & NPS,
        ];

        return Reply::dataOnly($feedbackData);
    }

    public function createWR($id, $feedback_number)
    {
        $this->ticket = Feedback::with('requester', 'requester.feedback', 'reply', 'reply.files', 'reply.user')
            ->where('id', $id)
            ->first();
        $this->number = WorkRequest::lastInvoiceNumber() + 1;
        $this->zero   = '';
        if (strlen($this->number) < 4) {
            for ($i = 0; $i < 4 - strlen($this->number); $i++) {
                $this->zero = '0' . $this->zero;
            }
        }
        $this->nomor = 'WR-' . Carbon::now()->format('ym') . '-' . $this->zero . $this->number;

        $wr             = new WorkRequest();
        $wr->feedback_id  = $this->ticket->id;
        $wr->wr_no      = $this->nomor;
        $wr->check_time = date('Y-m-d H:i:s');
        $wr->problem    = $this->ticket->subject;
        $wr->house_id    = $this->ticket->house_id;
        $wr->assign_to  = $this->ticket->agent_id;
        $wr->created_by = user()->id;
        $wr->save();

        $redirectUrl = route('feedback.show', [$feedback_number]);

        return redirect($redirectUrl)->with('success', __('inventory::messages.updateMR'));
    }

    public function createWO($id, $feedback_number, $wr)
    {
        $this->ticket = Feedback::with('requester', 'requester.feedback', 'reply', 'reply.files', 'reply.user')
            ->where('id', $id)
            ->first();
        $this->number = WorkOrder::lastInvoiceNumber() + 1;
        $this->zero   = '';
        if (strlen($this->number) < 4) {
            for ($i = 0; $i < 4 - strlen($this->number); $i++) {
                $this->zero = '0' . $this->zero;
            }
        }
        $this->nomor = 'WO-' . Carbon::now()->format('ym') . '-' . $this->zero . $this->number;

        $wo                   = new WorkOrder();
        $wo->workrequest_id   = $wr;
        $wo->feedback_id        = $this->ticket->id;
        $wo->nomor_wo         = $this->nomor;
        $wo->problem          = $this->ticket->subject;
        $wo->house_id          = $this->ticket->house_id;
        $wo->created_by       = user()->id;
        $wo->save();

        $redirectUrl = route('feedback.show', [$feedback_number]);
        return redirect($redirectUrl)->with('success', __('inventory::messages.updateMR'));
    }

    public function updateOtherData(Request $request, $id)
    {
        $feedback             = Feedback::findOrFail($id);
        $feedback->agent_id   = $request->agent_id;
        $feedback->type_id    = $request->type_id;
        $feedback->priority   = $request->priority;
        $feedback->channel_id = $request->channel_id;
        $feedback->status     = $request->status;
        $feedback->save();

        // Save tags
        $tags = collect(json_decode($request->tags))->pluck('value');
        FeedbackTag::where('feedback_id', $feedback->id)->delete();

        foreach ($tags as $tag) {
            $tag = FeedbackTagList::firstOrCreate([
                'tag_name' => $tag,
            ]);
            $feedback->feedbackTags()->attach($tag);
        }

        return Reply::success(__('messages.updateSuccess'));
    }

    public function changeStatus(Request $request)
    {
        $feedback = Feedback::find($request->ticketId);
        $this->editPermission = user()->permission('edit_feedback');

        abort_403(!($this->editPermission == 'all' || ($this->editPermission == 'owned' && user()->id == $feedback->agent_id)
        ));

        $feedback->update(['status' => $request->status]);

        return Reply::successWithData(__('messages.updateSuccess'), ['status' => 'success']);
    }
}
