<?php

namespace Workdo\Taskly\Events;

use Illuminate\Queue\SerializesModels;

class DestroyMilestone
{
    use SerializesModels;

    /**
     * Create a new event instance.
     *
     * @return void
     */
    public $milestone;

    public function __construct($milestone)
    {
        $this->milestone = $milestone;
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
