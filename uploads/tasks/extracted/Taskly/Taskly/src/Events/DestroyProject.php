<?php

namespace Workdo\Taskly\Events;

use Illuminate\Queue\SerializesModels;

class DestroyProject
{
    use SerializesModels;

    /**
     * Create a new event instance.
     *
     * @return void
     */
    public $project;

    public function __construct($project)
    {
        $this->project = $project;
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
