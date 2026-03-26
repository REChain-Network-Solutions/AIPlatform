<?php

namespace Modules\Complaint\Events;

use Modules\Complaint\Entities\Complaint;
use Illuminate\Broadcasting\InteractsWithSockets;
use Illuminate\Foundation\Events\Dispatchable;
use Illuminate\Queue\SerializesModels;

class ComplaintRequesterEvent
{

    use Dispatchable, InteractsWithSockets, SerializesModels;

    public $complaint;
    public $notifyUser;

    public function __construct(Complaint $complaint, $notifyUser)
    {
        $this->complaint = $complaint;
        $this->notifyUser = $notifyUser;
    }

}
