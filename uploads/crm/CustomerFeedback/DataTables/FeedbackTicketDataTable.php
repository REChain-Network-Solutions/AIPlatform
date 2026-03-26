<?php

namespace Modules\CustomerFeedback\DataTables;

use Illuminate\Http\Request;
use Modules\CustomerFeedback\Entities\FeedbackTicket;

class FeedbackTicketDataTable
{
    public function __construct(private Request $request)
    {
    }

    public function render(string $view, array $data = [])
    {
        $query = FeedbackTicket::query()->with(['requester', 'agent', 'channel', 'group', 'ticketType']);

        if (function_exists('company') && company()) {
            $query->where('company_id', company()->id);
        }

        foreach (['status', 'priority', 'feedback_type', 'channel_id', 'group_id', 'type_id', 'agent_id'] as $field) {
            if ($value = $this->request->get($field)) {
                $query->where($field, $value);
            }
        }

        if ($search = trim((string) $this->request->get('search'))) {
            $query->where(function ($q) use ($search) {
                $q->where('title', 'like', "%{$search}%")
                    ->orWhere('description', 'like', "%{$search}%")
                    ->orWhere('id', $search);
            });
        }

        $tickets = $query->latest('id')->paginate(20)->withQueryString();

        if ($this->request->ajax()) {
            return response()->json([
                'data' => $tickets->map(fn ($ticket) => [
                    'id' => $ticket->id,
                    'title' => $ticket->title,
                    'status' => $ticket->status,
                    'priority' => $ticket->priority,
                    'feedback_type' => $ticket->feedback_type,
                    'requester' => optional($ticket->requester)->name,
                    'agent' => optional($ticket->agent)->name,
                    'created_at' => optional($ticket->created_at)?->toDateTimeString(),
                    'show_url' => route('feedback.tickets.show', $ticket),
                ]),
                'meta' => [
                    'current_page' => $tickets->currentPage(),
                    'last_page' => $tickets->lastPage(),
                    'total' => $tickets->total(),
                ],
            ]);
        }

        $data['tickets'] = $tickets;
        $data['filters'] = $this->request->all();

        return view($view, $data);
    }
}
