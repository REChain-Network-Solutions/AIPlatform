<?php

namespace Workdo\Taskly\Events;

use Illuminate\Queue\SerializesModels;

class DestroyTask
{
    use SerializesModels;

    /**
     * Create a new event instance.
     *
     * @return void
     */
    public $taskID;

    public function __construct($taskID)
    {
        $this->taskID = $taskID;
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
