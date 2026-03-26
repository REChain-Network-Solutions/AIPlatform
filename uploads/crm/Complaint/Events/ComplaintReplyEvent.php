<?php

namespace Modules\Complaint\Events;

use Modules\Complaint\Entities\ComplaintReply;
use Illuminate\Broadcasting\InteractsWithSockets;
use Illuminate\Foundation\Events\Dispatchable;
use Illuminate\Queue\SerializesModels;

class ComplaintReplyEvent
{

    use Dispatchable, InteractsWithSockets, SerializesModels;

    public $complaintReply;
    public $notifyUser;

    public function __construct(ComplaintReply $complaintReply, $notifyUser)
    {
        $this->complaintReply = $complaintReply;
        $this->notifyUser = $notifyUser;
    }

}
