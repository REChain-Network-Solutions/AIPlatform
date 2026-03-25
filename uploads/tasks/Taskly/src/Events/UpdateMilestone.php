<?php

namespace Workdo\Taskly\Events;

use Illuminate\Queue\SerializesModels;

class UpdateMilestone
{
    use SerializesModels;

    /**
     * Create a new event instance.
     *
     * @return void
     */
    public $request;
    public $milestone;

    public function __construct($request,$milestone)
    {
        $this->request      = $request;
        $this->milestone    = $milestone;
    }

    /**
     * Get the channels the event should be broadcast on.
     *
     * @return array
     */
    public function broadcastOn()
    {
        return [];
    }
}
