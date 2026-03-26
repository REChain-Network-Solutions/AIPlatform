<?php

namespace Modules\CustomerFeedback\Http\Controllers;

use Carbon\Carbon;
use App\Models\User;
use App\Helper\Reply;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\DB;
use App\Http\Controllers\AccountBaseController;
use Modules\CustomerFeedback\Entities\FeedbackTicket;
use Modules\CustomerFeedback\Entities\FeedbackChannel;
use Modules\CustomerFeedback\Entities\FeedbackType;
use Modules\CustomerFeedback\Entities\FeedbackGroup;
use Modules\CustomerFeedback\Entities\FeedbackTagList;
use Modules\CustomerFeedback\Entities\FeedbackReplyTemplate;
use Modules\CustomerFeedback\DataTables\FeedbackTicketDataTable;
use Modules\CustomerFeedback\Http\Requests\StoreFeedbackTicket;
use Modules\CustomerFeedback\Http\Requests\UpdateFeedbackTicket;
use Modules\CustomerFeedback\Events\FeedbackTicketCreated;
use Modules\CustomerFeedback\Events\FeedbackTicketUpdated;

class FeedbackTicketController extends AccountBaseController
{
    public function __construct()
    {
        parent::__construct();
        $this->pageTitle = 'customer-feedback::modules.feedback';
        $this->middleware(function ($request, $next) {
            abort_403(!in_array('customer-feedback', $this->user->modules));
            return $next($request);
        });
    }

    /**
     * Display list of feedback tickets with filtering
     */
    public function index(FeedbackTicketDataTable $dataTable)
    {
        $this->viewPermission = user()->permission('view_feedback');
        $managePermission = user()->permission('view_feedback');

        if (!request()->ajax()) {
            $this->channels = FeedbackChannel::where('status', true)->get();
            $this->types = FeedbackType::where('status', true)->get();
            $this->groups = $managePermission == 'none'
                ? null
                : FeedbackGroup::with([
                    'enabledAgents' => function ($q) use ($managePermission) {
                        if ($managePermission == 'added') {
                            $q->where('added_by', user()->id);
                        } elseif ($managePermission == 'owned') {
                            $q->where('agent_id', user()->id);
                        } elseif ($managePermission == 'both') {
                            $q->where('agent_id', user()->id)->orWhere('added_by', user()->id);
                        }
                    },
                    'enabledAgents.user',
                ])->where('status', true)->get();

            $this->tags = FeedbackTagList::where('company_id', company()->id)->get();
        }

        return $dataTable->render('customer-feedback::tickets.index', $this->data);
    }

    /**
     * Show create ticket form
     */
    public function create()
    {
        $this->addPermission = user()->permission('add_feedback');
        abort_403(!in_array($this->addPermission, ['all', 'owned', 'none']));

        $this->pageTitle = __('customer-feedback::app.addTicket');
        $this->groups = FeedbackGroup::where('status', true)->with('enabledAgents', 'enabledAgents.user')->get();
        $this->types = FeedbackType::where('status', true)->get();
        $this->channels = FeedbackChannel::where('status', true)->get();
        $this->templates = FeedbackReplyTemplate::where('status', true)->get();
        $this->employees = User::allEmployees(null, true, $this->addPermission == 'all' ? 'all' : null);
        $this->clients = User::allClients();

        $ticket = new FeedbackTicket();
        $this->fields = null;

        if (!empty($ticket->getCustomFieldGroupsWithFields())) {
            $this->fields = $ticket->getCustomFieldGroupsWithFields()->fields;
        }

        if (request()->default_client) {
            $this->client = User::find(request()->default_client);
        }

        return view('customer-feedback::tickets.create', $this->data);
    }

    /**
     * Store new feedback ticket
     */
    public function store(StoreFeedbackTicket $request)
    {
        abort_403(user()->permission('add_feedback') == 'none');

        $ticket = new FeedbackTicket();
        $ticket->company_id = company()->id;
        $ticket->user_id = $request->user_id;
        $ticket->agent_id = $request->agent_id;
        $ticket->title = $request->title;
        $ticket->description = $request->description;
        $ticket->feedback_type = $request->feedback_type ?? FeedbackTicket::TYPE_COMPLAINT;
        $ticket->status = FeedbackTicket::STATUS_OPEN;
        $ticket->priority = $request->priority ?? FeedbackTicket::PRIORITY_MEDIUM;
        $ticket->channel_id = $request->channel_id;
        $ticket->group_id = $request->group_id;
        $ticket->type_id = $request->type_id;

        if ($request->nps_score) {
            $ticket->nps_score = $request->nps_score;
        }

        if ($request->csat_score) {
            $ticket->csat_score = $request->csat_score;
        }

        if ($request->custom_meta) {
            $ticket->custom_meta = $request->custom_meta;
        }

        $ticket->save();

        // Fire event for processing (email, AI analysis, etc.)
        event(new FeedbackTicketCreated($ticket));

        return Reply::success(__('messages.recordSaved'));
    }

    /**
     * Show single ticket with replies
     */
    public function show(FeedbackTicket $ticket)
    {
        abort_403(!$this->canViewTicket($ticket));

        $ticket->update(['read' => true]);

        $this->pageTitle = __('customer-feedback::app.viewTicket');
        $this->ticket = $ticket->load('replies', 'files', 'channel', 'group', 'ticketType', 'tags');
        $this->templates = FeedbackReplyTemplate::where('status', true)->get();

        return view('customer-feedback::tickets.show', $this->data);
    }

    /**
     * Show edit form
     */
    public function edit(FeedbackTicket $ticket)
    {
        abort_403(user()->permission('edit_feedback') == 'none');

        $this->pageTitle = __('customer-feedback::app.editTicket');
        $this->ticket = $ticket;
        $this->groups = FeedbackGroup::where('status', true)->with('enabledAgents', 'enabledAgents.user')->get();
        $this->types = FeedbackType::where('status', true)->get();
        $this->channels = FeedbackChannel::where('status', true)->get();
        $this->employees = User::allEmployees();

        return view('customer-feedback::tickets.edit', $this->data);
    }

    /**
     * Update ticket
     */
    public function update(UpdateFeedbackTicket $request, FeedbackTicket $ticket)
    {
        abort_403(user()->permission('edit_feedback') == 'none');

        $ticket->update([
            'title' => $request->title,
            'description' => $request->description,
            'status' => $request->status,
            'priority' => $request->priority,
            'agent_id' => $request->agent_id,
            'group_id' => $request->group_id,
            'type_id' => $request->type_id,
        ]);

        if ($request->status == FeedbackTicket::STATUS_RESOLVED || $request->status == FeedbackTicket::STATUS_CLOSED) {
            $ticket->update(['resolved_at' => Carbon::now()]);
        }

        event(new FeedbackTicketUpdated($ticket));

        return Reply::success(__('messages.recordUpdated'));
    }

    /**
     * Delete ticket
     */
    public function destroy(FeedbackTicket $ticket)
    {
        abort_403(user()->permission('delete_feedback') != 'all');

        $ticket->delete();

        return Reply::success(__('messages.recordDeleted'));
    }

    /**
     * Bulk actions
     */
    public function bulk(Request $request)
    {
        $ids = explode(',', $request->row_ids);

        if ($request->action == 'delete') {
            abort_403(user()->permission('delete_feedback') != 'all');
            FeedbackTicket::whereIn('id', $ids)->delete();
        } elseif ($request->action == 'status') {
            abort_403(user()->permission('edit_feedback') == 'none');
            FeedbackTicket::whereIn('id', $ids)->update(['status' => $request->status]);
        } elseif ($request->action == 'assign') {
            abort_403(user()->permission('edit_feedback') == 'none');
            FeedbackTicket::whereIn('id', $ids)->update(['agent_id' => $request->agent_id]);
        }

        return Reply::success(__('messages.recordUpdated'));
    }

    /**
     * Export tickets
     */
    public function export(Request $request)
    {
        abort_403(user()->permission('view_feedback') == 'none');

        $query = FeedbackTicket::query();

        if ($request->status) {
            $query->where('status', $request->status);
        }

        if ($request->priority) {
            $query->where('priority', $request->priority);
        }

        if ($request->type_id) {
            $query->where('type_id', $request->type_id);
        }

        $tickets = $query->get();

        // Generate CSV
        $filename = 'feedback_export_' . now()->format('Y-m-d_His') . '.csv';
        $headers = [
            'Content-Type' => 'text/csv; charset=utf-8',
            'Content-Disposition' => 'attachment; filename="' . $filename . '"',
        ];

        $callback = function () use ($tickets) {
            $file = fopen('php://output', 'w');
            fputcsv($file, [
                'ID', 'Title', 'Status', 'Priority', 'Type', 'Agent', 'Client', 'Created',
            ]);

            foreach ($tickets as $ticket) {
                fputcsv($file, [
                    $ticket->id,
                    $ticket->title,
                    $ticket->status,
                    $ticket->priority,
                    $ticket->ticketType?->name,
                    $ticket->agent?->name,
                    $ticket->requester?->name,
                    $ticket->created_at->format('Y-m-d H:i'),
                ]);
            }

            fclose($file);
        };

        return response()->stream($callback, 200, $headers);
    }

    /**
     * Check if user can view ticket
     */
    private function canViewTicket(FeedbackTicket $ticket): bool
    {
        if (user()->permission('view_feedback') == 'all') {
            return true;
        }

        if (user()->permission('view_feedback') == 'owned' && $ticket->agent_id == user()->id) {
            return true;
        }

        if (user()->permission('view_feedback') == 'added' && $ticket->requester_id == user()->id) {
            return true;
        }

        return false;
    }
}
