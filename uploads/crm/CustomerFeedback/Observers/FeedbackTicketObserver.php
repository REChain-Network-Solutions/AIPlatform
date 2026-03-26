<?php

namespace Modules\CustomerFeedback\Observers;

use Modules\CustomerFeedback\Entities\FeedbackTicket;

class FeedbackTicketObserver
{
    /**
     * Handle ticket creation
     */
    public function creating(FeedbackTicket $ticket)
    {
        $ticket->company_id = company()->id;
    }

    /**
     * Handle ticket updates
     */
    public function updating(FeedbackTicket $ticket)
    {
        // Track status changes for resolution tracking
        if ($ticket->isDirty('status')) {
            $oldStatus = $ticket->getOriginal('status');
            $newStatus = $ticket->status;

            if (in_array($newStatus, [FeedbackTicket::STATUS_RESOLVED, FeedbackTicket::STATUS_CLOSED])) {
                $ticket->resolved_at = now();
            }
        }
    }

    /**
     * Handle ticket deletion
     */
    public function deleting(FeedbackTicket $ticket)
    {
        // Log or perform cleanup if needed
    }
}
