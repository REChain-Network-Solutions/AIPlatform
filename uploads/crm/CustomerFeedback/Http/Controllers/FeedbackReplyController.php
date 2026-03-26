<?php

namespace Modules\CustomerFeedback\Http\Controllers;

use App\Helper\Reply;
use App\Models\User;
use Illuminate\Http\Request;
use App\Http\Controllers\AccountBaseController;
use Modules\CustomerFeedback\Entities\FeedbackTicket;
use Modules\CustomerFeedback\Entities\FeedbackReply;
use Modules\CustomerFeedback\Entities\FeedbackReplyTemplate;
use Modules\CustomerFeedback\Http\Requests\StoreFeedbackReply;
use Modules\CustomerFeedback\Events\FeedbackReplyAdded;

class FeedbackReplyController extends AccountBaseController
{
    public function __construct()
    {
        parent::__construct();
        $this->middleware(function ($request, $next) {
            abort_403(!in_array('customer-feedback', $this->user->modules));
            return $next($request);
        });
    }

    /**
     * Get all replies for a ticket
     */
    public function index(FeedbackTicket $ticket)
    {
        abort_403(!$this->canViewTicket($ticket));

        $replies = $ticket->replies()
            ->with('user', 'files')
            ->orderBy('created_at', 'asc')
            ->get();

        return response()->json(['data' => $replies]);
    }

    /**
     * Store new reply
     */
    public function store(StoreFeedbackReply $request, FeedbackTicket $ticket)
    {
        abort_403(user()->permission('add_feedback_reply') == 'none');

        $reply = new FeedbackReply();
        $reply->feedback_id = $ticket->id;
        $reply->user_id = user()->id;
        $reply->message = $request->message;
        $reply->message_html = $request->message_html ?? $request->message;
        $reply->is_internal = $request->is_internal ?? false;
        $reply->source_channel = $request->source_channel ?? FeedbackReply::SOURCE_PORTAL;
        $reply->company_id = company()->id;
        $reply->save();

        // Handle file attachments
        if ($request->has('files')) {
            foreach ($request->file('files') as $file) {
                $this->storeReplyFile($reply, $file);
            }
        }

        // Fire event to send notifications and email if needed
        event(new FeedbackReplyAdded($reply));

        // Update ticket status if needed
        if ($ticket->status == FeedbackTicket::STATUS_PENDING) {
            $ticket->update(['status' => FeedbackTicket::STATUS_IN_PROGRESS]);
        }

        return Reply::success(__('customer-feedback::messages.replyAdded'));
    }

    /**
     * Get reply template
     */
    public function template(FeedbackReplyTemplate $template)
    {
        return response()->json(['message' => $template->message]);
    }

    /**
     * Mark ticket as resolved with reply
     */
    public function resolve(Request $request, FeedbackTicket $ticket)
    {
        abort_403(user()->permission('edit_feedback') == 'none');

        // Add closing reply if message provided
        if ($request->message) {
            $reply = new FeedbackReply();
            $reply->feedback_id = $ticket->id;
            $reply->user_id = user()->id;
            $reply->message = $request->message;
            $reply->is_internal = false;
            $reply->source_channel = FeedbackReply::SOURCE_PORTAL;
            $reply->company_id = company()->id;
            $reply->save();

            event(new FeedbackReplyAdded($reply));
        }

        $ticket->update([
            'status' => FeedbackTicket::STATUS_RESOLVED,
            'resolved_at' => now(),
        ]);

        return Reply::success(__('customer-feedback::messages.ticketResolved'));
    }

    /**
     * Store file attachment for reply
     */
    private function storeReplyFile(FeedbackReply $reply, $file)
    {
        $path = $file->store('feedback-attachments/' . $reply->feedback_id, 'public');

        $reply->files()->create([
            'filename' => $file->getClientOriginalName(),
            'file_path' => $path,
            'file_size' => $file->getSize(),
            'mime_type' => $file->getMimeType(),
            'company_id' => company()->id,
        ]);
    }

    /**
     * Delete reply
     */
    public function destroy(FeedbackReply $reply)
    {
        abort_403(user()->permission('delete_feedback') != 'all');

        $ticketId = $reply->feedback_id;
        $reply->delete();

        return Reply::success(__('messages.recordDeleted'));
    }

    /**
     * Check permissions
     */
    private function canViewTicket(FeedbackTicket $ticket): bool
    {
        if (user()->permission('view_feedback') == 'all') {
            return true;
        }

        if (user()->permission('view_feedback') == 'owned' && $ticket->agent_id == user()->id) {
            return true;
        }

        return false;
    }
}
