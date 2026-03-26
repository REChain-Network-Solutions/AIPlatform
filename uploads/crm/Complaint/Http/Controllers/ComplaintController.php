<?php

namespace Modules\Complaint\Http\Controllers;

use Client;
use Carbon\Carbon;
use App\Models\User;
use App\Helper\Reply;
use App\Models\Country;
use Illuminate\Http\Request;
use Modules\Units\Entities\Unit;
use Illuminate\Support\Facades\DB;
use Modules\Complaint\Entities\Complaint;
use Modules\Complaint\Entities\ComplaintTag;
use Modules\Engineerings\Entities\WorkOrder;
use Modules\Complaint\Entities\ComplaintType;
use Modules\Complaint\Entities\ComplaintGroup;
use Modules\Complaint\Entities\ComplaintReply;
use Modules\Engineerings\Entities\WorkRequest;
use App\Http\Controllers\AccountBaseController;
use Modules\Complaint\Entities\ComplaintChannel;
use Modules\Complaint\Entities\ComplaintTagList;
use Modules\Complaint\Http\Requests\StoreComplaint;
use Modules\Complaint\DataTables\ComplaintDataTable;
use Modules\Complaint\Http\Requests\UpdateComplaint;
use Modules\Complaint\Entities\ComplaintReplyTemplate;
use Modules\Houses\Entities\Area;
use Modules\Houses\Entities\House;

class ComplaintController extends AccountBaseController
{
    public function __construct()
    {
        parent::__construct();
        $this->pageTitle = 'complaint::modules.complaint';
        $this->middleware(function ($request, $next) {
            abort_403(!in_array('complaint', $this->user->modules));
            return $next($request);
        });
    }

    public function index(ComplaintDataTable $dataTable)
    {
        $this->viewPermission = user()->permission('view_complaint');
        $managePermission     = user()->permission('view_complaint');

        if (!request()->ajax()) {
            $this->channels    = ComplaintChannel::all();
            $this->groups      =
                $managePermission == 'none'
                ? null
                : ComplaintGroup::with([
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

            $this->types = ComplaintType::all();
            $this->tags  = ComplaintTagList::all();
        }

        return $dataTable->render('complaint::complaint.index', $this->data);
    }

    public function getItems($id)
    {
        $items = House::where('area_id', $id)->get();
        return response()->json($items);
    }

    protected function deleteRecords($request)
    {
        abort_403(user()->permission('delete_complaint') != 'all');

        Complaint::whereIn('id', explode(',', $request->row_ids))->delete();
    }

    protected function changeBulkStatus($request)
    {
        abort_403(user()->permission('edit_complaint') != 'all');

        Complaint::whereIn('id', explode(',', $request->row_ids))->update(['status' => $request->status]);
    }

    public function create()
    {
        $this->addPermission = user()->permission('add_complaint');
        abort_403(!in_array($this->addPermission, ['all', 'owned', 'none']));

        $this->pageTitle     = __('complaint::app.complaint.addComplaint');
        $this->groups        = ComplaintGroup::with('enabledAgents', 'enabledAgents.user')->get();
        $this->types         = ComplaintType::all();
        $this->areas         = Area::all();
        $this->channels      = ComplaintChannel::all();
        $this->templates     = ComplaintReplyTemplate::all();
        $this->employees     = User::allEmployees(null, true, $this->addPermission == 'all' ? 'all' : null);
        $this->clients       = User::allClients();
        $this->unit          = Unit::all();
        $this->countries     = countries();
        $this->lastComplaint = Complaint::orderBy('id', 'desc')->first();
        $ticket              = new Complaint();

        $this->fields = null;

        if (!empty($ticket->getCustomFieldGroupsWithFields())) {
            $this->fields = $ticket->getCustomFieldGroupsWithFields()->fields;
        }

        if (request()->default_client) {
            $this->client = User::find(request()->default_client);
        }

        if (request()->ajax()) {
            $html = view('complaint::complaint.ajax.create', $this->data)->render();

            return Reply::dataOnly(['status' => 'success', 'html' => $html, 'title' => $this->pageTitle]);
        }

        $this->view = 'complaint::complaint.ajax.create';

        return view('complaint::complaint.create', $this->data);
    }

    public function store(StoreComplaint $request)
    {
        if (empty($request->type)) {
            return Reply::error(__('messages.addItem'));
        }

        $complaint             = new Complaint();
        $complaint->subject    = $request->subject;
        $complaint->status     = 'open';
        $complaint->priority   = 'medium';
        $complaint->no_hp      = $request->no_hp;
        $complaint->house_id    = $request->house_id;
        $complaint->user_id    = ($request->requester_type == 'employee') ? $request->user_id : $request->client_id;
        $complaint->agent_id   = $request->agent_id;
        $complaint->type_id    = $request->type_id;
        $complaint->channel_id = $request->channel_id;
        $complaint->save();

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
            $wr->complaint_id  = $complaint->id;
            $wr->wr_no      = $this->nomor;
            $wr->check_time = date('Y-m-d H:i:s');
            $wr->problem    = $complaint->subject;
            $wr->house_id    = $complaint->house_id;
            $wr->assign_to  = $complaint->agent_id;
            $wr->created_by = user()->id;
            $wr->save();
        }

        // Save first message
        $reply               = new ComplaintReply();
        $reply->message      = trim_editor($request->description);
        $reply->complaint_id = $complaint->id;
        $reply->user_id      = $this->user->id;                     // Current logged in user
        $reply->save();

        // To add custom fields data
        if ($request->custom_fields_data) {
            $complaint->updateCustomFieldData($request->custom_fields_data);
        }

        // Save tags
        $tags = collect(json_decode($request->tags))->pluck('value');

        foreach ($tags as $tag) {
            $tag = ComplaintTagList::firstOrCreate([
                'tag_name' => $tag,
            ]);
            $complaint->complaintTags()->attach($tag);
        }

        // Log search
        $this->logSearchEntry($complaint->id, $complaint->subject, 'complaint.show', 'complaint');

        $redirectUrl = urldecode($request->redirect_url);

        if ($redirectUrl == '') {
            $redirectUrl = route('complaint.index');
        }

        return Reply::successWithData(__('messages.recordSaved'), ['replyID' => $reply->id, 'redirectUrl' => $redirectUrl]);
    }

    public function show($complaintNumber)
    {
        $this->viewComplaintPermission = user()->permission('view_complaint');
        $this->complaint               = Complaint::with('requester', 'requester.complaint', 'reply', 'reply.files', 'reply.user')
            ->where('complaint_number', $complaintNumber)
            ->first();
        abort_if(!$this->complaint, 404);
        $this->complaint = $this->complaint->withCustomFields();
        $this->pageTitle = __('complaint::modules.complaint') . '#' . $this->complaint->complaint_number;
        $this->groups    = ComplaintGroup::with('enabledAgents', 'enabledAgents.user')->get();
        $this->types     = ComplaintType::all();
        $this->channels  = ComplaintChannel::all();
        $this->templates = ComplaintReplyTemplate::all();
        $this->fields    = null;
        $this->wr        = DB::table('workrequests')
            ->where('complaint_id', $this->complaint->id)
            ->get();
        $this->wo = DB::table('workorders')
            ->where('complaint_id', $this->complaint->id)
            ->get();
        $this->complaintChart = $this->complaintChartData($this->complaint->user_id);


        if (!empty($this->complaint->getCustomFieldGroupsWithFields())) {
            $this->fields = $this->complaint->getCustomFieldGroupsWithFields()->fields;
        }

        return view('complaint::complaint.edit', $this->data);
    }

    public function complaintChartData($id)
    {
        $labels         = ['open', 'pending', 'resolved', 'closed'];
        $data['labels'] = [__('app.open'), __('app.pending'), __('app.resolved'), __('app.closed')];
        $data['colors'] = ['#D30000', '#FCBD01', '#2CB100', '#1d82f5'];
        $data['values'] = [];

        foreach ($labels as $label) {
            $data['values'][] = Complaint::where('user_id', $id)
                ->where('status', $label)
                ->count();
        }

        return $data;
    }

    public function update(UpdateComplaint $request, $id)
    {
        $complaint         = Complaint::findOrFail($id);
        $complaint->status = $request->status;
        $complaint->save();

        $message = trim_editor($request->message);

        if ($message != '') {
            $reply               = new ComplaintReply();
            $reply->message      = $request->message;
            $reply->complaint_id = $complaint->id;
            $reply->user_id      = $this->user->id;       // Current logged in user
            $reply->save();

            return Reply::successWithData(__('messages.ticketReplySuccess'), ['reply_id' => $reply->id]);
        }

        return Reply::dataOnly(['status' => 'success']);
    }

    public function destroy($id)
    {
        $complaint = Complaint::findOrFail($id);

        $this->deleteComplaintPermission = user()->permission('delete_complaint');
        abort_403(!($this->deleteComplaintPermission == 'all' || ($this->deleteComplaintPermission == 'added' && user()->id == $complaint->added_by) || ($this->deleteComplaintPermission == 'owned' && (user()->id == $complaint->agent_id || user()->id == $complaint->user_id)) || ($this->deleteComplaintPermission == 'both' && (user()->id == $complaint->agent_id || user()->id == $complaint->added_by || user()->id == $complaint->user_id))));

        Complaint::destroy($id);

        return Reply::success(__('messages.deleteSuccess'));
    }

    public function refreshCount(Request $request)
    {
        $viewPermission = user()->permission('view_complaint');

        $complaints = Complaint::with('agent');

        if (!is_null($request->startDate) && $request->startDate != '') {
            $startDate = Carbon::createFromFormat($this->company->date_format, $request->startDate)->toDateString();
            $complaints->where(DB::raw('DATE(`updated_at`)'), '>=', $startDate);
        }

        if (!is_null($request->endDate) && $request->endDate != '') {
            $endDate = Carbon::createFromFormat($this->company->date_format, $request->endDate)->toDateString();
            $complaints->where(DB::raw('DATE(`updated_at`)'), '<=', $endDate);
        }

        if (!is_null($request->agentId) && $request->agentId != 'all') {
            $complaints->where('agent_id', '=', $request->agentId);
        }

        if (!is_null($request->priority) && $request->priority != 'all') {
            $complaints->where('priority', '=', $request->priority);
        }

        if (!is_null($request->channelId) && $request->channelId != 'all') {
            $complaints->where('channel_id', '=', $request->channelId);
        }

        if (!is_null($request->typeId) && $request->typeId != 'all') {
            $complaints->where('type_id', '=', $request->typeId);
        }

        if ($viewPermission == 'none') {
            $complaints->where('user_id', '=', user()->id);
        }

        if ($viewPermission == 'owned') {
            $complaints->where('user_id', '=', user()->id)->orWhere('complaint.agent_id', '=', user()->id);
        }

        $complaints = $complaints->get();

        $openComplaints = $complaints
            ->filter(function ($value, $key) {
                return $value->status == 'open';
            })
            ->count();

        $pendingComplaints = $complaints
            ->filter(function ($value, $key) {
                return $value->status == 'pending';
            })
            ->count();

        $resolvedComplaints = $complaints
            ->filter(function ($value, $key) {
                return $value->status == 'resolved';
            })
            ->count();

        $closedComplaints = $complaints
            ->filter(function ($value, $key) {
                return $value->status == 'closed';
            })
            ->count();

        $totalComplaints = $complaints->count();

        $complaintData = [
            'totalComplaints'    => $totalComplaints,
            'closedComplaints'   => $closedComplaints,
            'openComplaints'     => $openComplaints,
            'pendingComplaints'  => $pendingComplaints,
            'resolvedComplaints' => $resolvedComplaints,
        ];

        return Reply::dataOnly($complaintData);
    }

    public function createWR($id, $complaint_number)
    {
        $this->ticket = Complaint::with('requester', 'requester.complaint', 'reply', 'reply.files', 'reply.user')
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
        $wr->complaint_id  = $this->ticket->id;
        $wr->wr_no      = $this->nomor;
        $wr->check_time = date('Y-m-d H:i:s');
        $wr->problem    = $this->ticket->subject;
        $wr->house_id    = $this->ticket->house_id;
        $wr->assign_to  = $this->ticket->agent_id;
        $wr->created_by = user()->id;
        $wr->save();

        $redirectUrl = route('complaint.show', [$complaint_number]);

        return redirect($redirectUrl)->with('success', __('inventory::messages.updateMR'));
    }

    public function createWO($id, $complaint_number, $wr)
    {
        $this->ticket = Complaint::with('requester', 'requester.complaint', 'reply', 'reply.files', 'reply.user')
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
        $wo->complaint_id        = $this->ticket->id;
        $wo->nomor_wo         = $this->nomor;
        $wo->problem          = $this->ticket->subject;
        $wo->house_id          = $this->ticket->house_id;
        $wo->created_by       = user()->id;
        $wo->save();

        $redirectUrl = route('complaint.show', [$complaint_number]);
        return redirect($redirectUrl)->with('success', __('inventory::messages.updateMR'));
    }

    public function updateOtherData(Request $request, $id)
    {
        $complaint             = Complaint::findOrFail($id);
        $complaint->agent_id   = $request->agent_id;
        $complaint->type_id    = $request->type_id;
        $complaint->priority   = $request->priority;
        $complaint->channel_id = $request->channel_id;
        $complaint->status     = $request->status;
        $complaint->save();

        // Save tags
        $tags = collect(json_decode($request->tags))->pluck('value');
        ComplaintTag::where('complaint_id', $complaint->id)->delete();

        foreach ($tags as $tag) {
            $tag = ComplaintTagList::firstOrCreate([
                'tag_name' => $tag,
            ]);
            $complaint->complaintTags()->attach($tag);
        }

        return Reply::success(__('messages.updateSuccess'));
    }

    public function changeStatus(Request $request)
    {
        $complaint = Complaint::find($request->ticketId);
        $this->editPermission = user()->permission('edit_complaint');

        abort_403(!($this->editPermission == 'all' || ($this->editPermission == 'owned' && user()->id == $complaint->agent_id)
        ));

        $complaint->update(['status' => $request->status]);

        return Reply::successWithData(__('messages.updateSuccess'), ['status' => 'success']);
    }
}
