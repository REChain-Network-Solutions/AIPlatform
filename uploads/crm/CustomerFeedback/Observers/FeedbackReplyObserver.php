<?php

namespace Modules\CustomerFeedback\Observers;

use Modules\CustomerFeedback\Entities\FeedbackReply;

class FeedbackReplyObserver
{
    /**
     * Handle reply creation
     */
    public function creating(FeedbackReply $reply)
    {
        $reply->company_id = company()->id;
    }

    /**
     * Mark ticket as read when replied to
     */
    public function created(FeedbackReply $reply)
    {
        $reply->ticket->update(['read' => true]);

        // Update last activity
        $reply->ticket->update(['updated_at' => now()]);
    }
}
