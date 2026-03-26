<?php

namespace Modules\Complaint\Events;

use Modules\Complaint\Entities\Complaint;
use Illuminate\Broadcasting\InteractsWithSockets;
use Illuminate\Foundation\Events\Dispatchable;
use Illuminate\Queue\SerializesModels;

class ComplaintEvent
{

    use Dispatchable, InteractsWithSockets, SerializesModels;

    public $complaint;
    public $notificationName;

    public function __construct(Complaint $complaint, $notificationName)
    {
        $this->complaint = $complaint;
        $this->notificationName = $notificationName;
    }

}
